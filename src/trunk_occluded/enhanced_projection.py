#!/usr/bin/env python3
"""
Tree Ground Projection Estimator (no-trunk cases) - V2 Enhanced
-------------------------------------------------
Advanced multi-modal approach for estimating tree trunk base position from
Street View images when the trunk is occluded by buildings or other obstacles.

Key Features:
- Robust crown disparity extraction using upper bbox portion with statistical filtering
- Dynamic building/occluder detection with high-percentile disparity analysis
- Intelligent scale calibration with per-image ground validation
- Conservative crown-to-ground projection with tree height estimation
- Advanced AUTO scale mode with soft/hard confidence gates and ratio validation

Scale Modes:
- 'global': Uses fixed scale from JSON file
- 'per_image_ground': Calibrates scale from ground pixels using camera geometry
- 'auto': Intelligent blending with confidence-based fallback and ratio guards

Building Detection:
- Wide-band search around tree location using 95th percentile disparity
- Guarded building overrule only when occluder is depth-proximate to crown
- Anti "fence-in-the-road" failure mode protection

Projection Method:
- Adaptive projection factor based on estimated tree height and bbox dimensions
- Building-constrained range ensures minimum clearance behind detected obstacles
- Confidence-weighted decision making between direct and building-constrained estimates

Target Use Case:
- Production-ready system requiring robust performance across diverse urban scenes
- Handles challenging cases with multiple occluders and scale uncertainty
- Optimized for Street View imagery with GPS/heading metadata embedded in filenames

Input Requirements:
- CSV/XLSX file containing bbox data (file_name, x_box, y_box, width_box, height_box)
- Image name and tree index (0-based) for selecting specific tree within image
- Disparity map (.npy) for each image and scale JSON with {"inverse_depth_scale": <float>}
- Street View image filenames containing GPS coordinates and heading/FOV metadata
"""

import os
import re
import math
import json
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------------
# Parsing helpers for file_name
# ------------------------------

def parse_gps_from_filename(filename: str) -> Tuple[float, float]:
    patterns = [
        r"location=([-0-9\.]+),([-0-9\.]+)",
        r"&location=([-0-9\.]+),([-0-9\.]+)",
        r"location%3D([-0-9\.]+)%2C([-0-9\.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise ValueError(f"Could not parse GPS from filename: {filename}")


def parse_heading_and_fov_from_filename(filename: str) -> Tuple[float, float]:
    h_pats = [r"heading=([-0-9\.]+)", r"&heading=([-0-9\.]+)", r"heading%3D([-0-9\.]+)"]
    f_pats = [r"fov=([-0-9\.]+)", r"&fov=([-0-9\.]+)", r"fov%3D([-0-9\.]+)"]
    heading = None
    fov = None
    for p in h_pats:
        m = re.search(p, filename)
        if m:
            heading = float(m.group(1)); break
    for p in f_pats:
        m = re.search(p, filename)
        if m:
            fov = float(m.group(1)); break
    if heading is None or fov is None:
        raise ValueError(f"Could not parse heading/fov from filename: {filename}")
    return heading, fov


# ------------------------------
# Camera / geometry helpers
# ------------------------------

def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    """Compute vertical FOV (degrees) from horizontal FOV and aspect ratio."""
    hf = math.radians(hfov_deg)
    vf = 2.0 * math.atan(math.tan(hf / 2.0) * (height / float(width)))
    return math.degrees(vf)


def intrinsics_from_fov(hfov_deg: float, vfov_deg: float, width: int, height: int) -> Tuple[float, float]:
    """Approximate pinhole intrinsics (fx, fy) from FOVs and image size."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return fx, fy


def gps_from_heading_and_distance(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """Forward geodesy: move distance_m at bearing from (lat,lon)."""
    R = 6371000.0
    th = math.radians(bearing_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    dr = distance_m / R
    p2 = math.asin(math.sin(p1) * math.cos(dr) + math.cos(p1) * math.sin(dr) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(dr) * math.cos(p1), math.cos(dr) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


# ------------------------------
# Table + bbox helpers
# ------------------------------

def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported table format: {ext}")
    required = {"file_name", "x_box", "y_box", "width_box", "height_box"}
    if not required.issubset(df.columns):
        raise ValueError(f"Table must contain columns: {required}")
    df["file_name"] = df["file_name"].astype(str)
    return df


def bbox_to_pixel(cx: float, cy: float, bw: float, bh: float,
                  orig_w: int, orig_h: int, disp_w: int, disp_h: int) -> Tuple[int, int, int, int, int, int]:
    """Map normalized YOLO bbox to pixel space of disparity map.
    Returns x_bc, y_bc, x1, y1, x2, y2 (integers)."""
    x0 = cx * orig_w
    y0 = cy * orig_h
    x_bc0 = x0
    y_bc0 = y0 + (bh * orig_h) / 2.0

    sx = disp_w / float(orig_w)
    sy = disp_h / float(orig_h)

    x_bc = int(round(x_bc0 * sx))
    y_bc = int(round(y_bc0 * sy))

    w_disp = bw * disp_w
    h_disp = bh * disp_h
    x1 = int(round(x_bc - w_disp / 2.0))
    y1 = int(round(y_bc - h_disp / 2.0))
    x2 = int(round(x_bc + w_disp / 2.0))
    y2 = int(round(y_bc + h_disp / 2.0))
    return x_bc, y_bc, x1, y1, x2, y2


# ------------------------------
# Crown disparity & occluder estimation
# ------------------------------

def robust_crown_disparity(disp_patch: np.ndarray, top_frac: float = 0.7, stat: str = "median") -> Tuple[float, float]:
    """Return (disparity, confidence) for the crown using the top part of the bbox.
    Confidence is based on valid coverage and MAD/median robustness.
    """
    H, W = disp_patch.shape[:2]
    end = max(1, int(H * max(0.1, min(top_frac, 0.95))))
    crown = disp_patch[:end, :]
    vals = crown[np.isfinite(crown)]
    if vals.size < 20:
        return float("nan"), 0.0
    if stat == "p90":
        disp = float(np.nanpercentile(vals, 90))
    else:
        disp = float(np.nanmedian(vals))
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med))) if vals.size > 0 else float("inf")
    # coverage of valid pixels
    cov = vals.size / float(crown.size)
    # map robustness to [0,1]; smaller MAD/med is better
    rob = 1.0 - min(1.0, (mad / (0.3 * (med + 1e-6))))
    conf = max(0.0, min(1.0, 0.5 * cov + 0.5 * rob))
    return disp, conf


def building_disparity_band(disp_map: np.ndarray, x_bc: int, y1: int, y2: int,
                            margin: int = 40, percentile: float = 95.0) -> Optional[float]:
    """Return a representative occluder disparity (p95) around the bbox column.
    We cut sky by starting at y_top = y1; extend a bit below to capture fences/fronts."""
    H, W = disp_map.shape[:2]
    x_left = max(0, x_bc - margin)
    x_right = min(W, x_bc + margin + 1)
    y_top = max(0, y1)
    y_bot = min(H, y2 + margin)
    region = disp_map[y_top:y_bot, x_left:x_right]
    vals = region[np.isfinite(region)]
    if vals.size < 100:
        return None
    return float(np.nanpercentile(vals, percentile))


# ------------------------------
# Scale estimation (monocular inverse-depth)
# ------------------------------

def ground_scale_with_confidence(disp_map: np.ndarray, hfov_deg: float, vfov_deg: float,
                                 cam_h: float, pitch_deg: float, phi_min_deg: float = 2.0) -> Tuple[Optional[float], float]:
    """Estimate per-image scale from the ground and return (scale, confidence).
    We use rows below ~70% height, enforce a minimum depression angle, and
    compute a robust median of (distance * disparity).
    """
    H, W = disp_map.shape[:2]
    fx, fy = intrinsics_from_fov(hfov_deg, vfov_deg, W, H)
    v0 = int(0.70 * H)
    ys, xs = np.mgrid[v0:H, 0:W]
    disp_vals = disp_map[ys, xs]
    vc = ys - (H / 2.0)
    phi = np.arctan2(vc, fy) + math.radians(pitch_deg)
    phi_min = math.radians(phi_min_deg)
    mask = (phi > phi_min) & np.isfinite(disp_vals) & (disp_vals > 1e-6)
    n_total = (H - v0) * W
    n_valid = int(np.count_nonzero(mask))
    if n_valid < 300:
        return None, 0.0
    d_ground = cam_h / np.tan(phi[mask])
    s_samples = (d_ground * disp_vals[mask]).astype(np.float64)
    s_samples = s_samples[np.isfinite(s_samples)]
    if s_samples.size < 300:
        return None, 0.0
    med = float(np.median(s_samples))
    mad = float(np.median(np.abs(s_samples - med)))
    if med <= 0:
        return None, 0.0
    coverage = n_valid / float(n_total)
    robustness = max(0.0, 1.0 - (mad / (0.25 * med)))
    # gentle squashing
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
    w_cov = _sigmoid((coverage - 0.05) / 0.03)
    w_rob = _sigmoid((robustness - 0.5) / 0.15)
    conf = float(np.clip(0.5 * w_cov + 0.5 * w_rob, 0.0, 1.0))
    return med, conf


# ------------------------------
# Main estimator
# ------------------------------

def main() -> int:
    p = argparse.ArgumentParser("Tree Ground Projection Estimator")
    p.add_argument('--csv_path', required=True)
    p.add_argument('--image_name', required=True)
    p.add_argument('--tree_index', type=int, required=True, help='0-based index for the tree within image')
    p.add_argument('--disp_path', required=True, help='Path to disparity .npy file OR a directory holding files')
    p.add_argument('--scale_path', required=True, help='JSON with {"inverse_depth_scale": float} for global mode')

    # image/disparity sizes used when mapping normalized bboxes
    p.add_argument('--orig_w', type=int, default=400)
    p.add_argument('--orig_h', type=int, default=400)
    p.add_argument('--disp_w', type=int, default=512)
    p.add_argument('--disp_h', type=int, default=256)

    # camera + ground geometry
    p.add_argument('--camera_height', type=float, default=2.5)
    p.add_argument('--pitch_deg', type=float, default=0.0)

    # scale handling
    p.add_argument('--scale_mode', choices=['global', 'per_image_ground', 'auto'], default='auto')
    p.add_argument('--auto_ratio_soft', type=float, default=1.35)
    p.add_argument('--auto_ratio_hard', type=float, default=2.0)
    p.add_argument('--auto_phi_min_deg', type=float, default=2.0)

    # building / occluder handling
    p.add_argument('--building_margin', type=int, default=40, help='Half-width (px) for the occluder search band')
    p.add_argument('--building_percentile', type=float, default=95.0, help='Percentile for occluder disparity')
    p.add_argument('--building_clearance', type=float, default=2.5, help='Meters added behind the occluder')
    p.add_argument('--building_gap_max', type=float, default=8.0, help='Only overrule if crown and occluder are within this depth gap (m)')

    # crown measurement options
    p.add_argument('--crown_top_frac', type=float, default=0.7)
    p.add_argument('--crown_stat', choices=['median', 'p90'], default='median')

    # projection tuning
    p.add_argument('--projection_factor', type=float, default=0.7, help='Fraction of (tree_height - camera_height) added to range')
    p.add_argument('--tree_height_cap', type=float, default=10.0)

    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    print("Tree Ground Projection Estimator")
    print("=" * 50)
    print(f"Image: {args.image_name}")
    print(f"Tree Index: {args.tree_index}")
    print(f"Scale mode: {args.scale_mode}")
    print(f"Camera height: {args.camera_height} m  |  Pitch: {args.pitch_deg}°")
    print(f"Building clearance: {args.building_clearance} m")
    print("=" * 50)

    # Load table & select row
    df = load_table(args.csv_path)
    rows = df[df['file_name'] == args.image_name]
    if rows.empty:
        raise ValueError(f"No rows found for image: {args.image_name}")
    if not (0 <= args.tree_index < len(rows)):
        raise ValueError(f"tree_index {args.tree_index} out of range (found {len(rows)})")
    r = rows.iloc[args.tree_index]
    cx, cy = float(r['x_box']), float(r['y_box'])
    bw, bh = float(r['width_box']), float(r['height_box'])

    # Find disparity file
    if os.path.isdir(args.disp_path):
        base, _ = os.path.splitext(args.image_name)
        cand1 = os.path.join(args.disp_path, base + '.npy')
        cand2 = os.path.join(args.disp_path, base + '_disp.npy')
        npy_path = cand1 if os.path.isfile(cand1) else (cand2 if os.path.isfile(cand2) else None)
        if npy_path is None:
            raise FileNotFoundError(f"Could not find disparity for {args.image_name}. Tried: {cand1} / {cand2}")
    else:
        npy_path = args.disp_path
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"Disparity path is not a file: {npy_path}")

    disp_map = np.load(npy_path)
    print(f"Loaded disparity map: {disp_map.shape}  from {npy_path}")

    # Bbox → disparity coords
    x_bc, y_bc, x1, y1, x2, y2 = bbox_to_pixel(
        cx, cy, bw, bh,
        orig_w=args.orig_w, orig_h=args.orig_h,
        disp_w=args.disp_w, disp_h=args.disp_h,
    )
    H, W = disp_map.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid bbox after mapping to disparity space")

    disp_patch = disp_map[y1:y2, x1:x2]

    # Crown disparity + confidence
    crown_disp, crown_conf = robust_crown_disparity(disp_patch, args.crown_top_frac, args.crown_stat)
    if not np.isfinite(crown_disp) or crown_disp <= 1e-6:
        raise RuntimeError("Could not estimate crown disparity")

    # Camera & FOVs from filename
    cam_lat, cam_lon = parse_gps_from_filename(args.image_name)
    heading, hfov = parse_heading_and_fov_from_filename(args.image_name)
    vfov = vfov_from_hfov(hfov, args.disp_w, args.disp_h)

    # Obtain scale
    with open(args.scale_path, 'r', encoding='utf-8') as f:
        global_scale = json.load(f)['inverse_depth_scale']

    if args.scale_mode == 'global':
        scale_value = global_scale
        print(f"Using global scale: {scale_value:.6f}")
    elif args.scale_mode == 'per_image_ground':
        s_est, conf = ground_scale_with_confidence(disp_map, hfov, vfov, args.camera_height, args.pitch_deg, args.auto_phi_min_deg)
        if s_est is None:
            print("Per-image scale calibration failed – using global scale.")
            scale_value = global_scale
        else:
            scale_value = float(s_est)
            print(f"Using per-image scale: {scale_value:.6f} (conf={conf:.2f})")
    else:  # auto
        s_est, conf = ground_scale_with_confidence(disp_map, hfov, vfov, args.camera_height, args.pitch_deg, args.auto_phi_min_deg)
        if s_est is None:
            scale_value = global_scale
            print("Auto scale: fallback to global (not enough ground).")
        else:
            ratio = s_est / global_scale
            dev = abs(math.log(max(ratio, 1e-9)))
            soft_dev = abs(math.log(args.auto_ratio_soft))
            hard_dev = abs(math.log(args.auto_ratio_hard))
            if dev >= hard_dev:
                scale_value = global_scale
                print(f"Auto scale: ratio={ratio:.2f} ≥ hard({args.auto_ratio_hard:.2f}) → fallback to global {global_scale:.6f}")
            else:
                if dev > soft_dev:
                    penalty = (dev - soft_dev) / max(hard_dev - soft_dev, 1e-6)
                    conf *= max(0.0, 1.0 - penalty)
                scale_value = float(conf * s_est + (1.0 - conf) * global_scale)
                print(f"Auto scale: s_ground={s_est:.6f}, ratio={ratio:.2f}, conf={conf:.2f} → used={scale_value:.6f}")

    # Distances
    crown_distance = scale_value / crown_disp

    # Estimate a rough tree height from bbox size (bounded), then project to ground
    bbox_height_rel = (y2 - y1) / float(args.disp_h)
    est_tree_h = max(5.0, min(args.tree_height_cap, 8.0 + (bbox_height_rel - 0.20) * 20.0))
    eff_above_cam = max(0.0, est_tree_h - args.camera_height)
    proj_delta = crown_distance * min(args.projection_factor, eff_above_cam / max(0.5, args.camera_height))
    candidate_crown = crown_distance + proj_delta

    # Occluder (building/fence) candidate
    bld_disp = building_disparity_band(
        disp_map, x_bc, y1, y2, margin=args.building_margin, percentile=args.building_percentile
    )
    has_building = False
    candidate_building = None
    if bld_disp is not None and bld_disp > crown_disp * 1.3:
        bld_distance = scale_value / bld_disp
        candidate_building = bld_distance + args.building_clearance
        has_building = True

    # Decide which ground distance to use
    method = 'direct_ground_projection'
    ratio_used = None
    if has_building and candidate_building is not None:
        gap = crown_distance - (scale_value / bld_disp)
        ratio_used = (scale_value / global_scale) if global_scale > 0 else None
        # Overrule only if occluder and crown are at similar depth (gap small) OR crown_conf low
        if (gap <= args.building_gap_max) or (crown_conf < 0.6 and args.scale_mode != 'global'):
            ground_distance = candidate_building
            method = 'building_overrule'
        else:
            ground_distance = max(candidate_building, candidate_crown)
            method = 'building_constrained'
    else:
        ground_distance = candidate_crown

    # Bearing from x offset in disparity space
    offset_px = x_bc - (args.disp_w / 2.0)
    angle_offset = (offset_px / args.disp_w) * hfov
    bearing = (heading + angle_offset) % 360.0

    tree_lat, tree_lon = gps_from_heading_and_distance(cam_lat, cam_lon, bearing, ground_distance)

    # Report
    print("\nGround Projection Results:")
    print("=" * 50)
    print(f"Method: {method}")
    print(f"Confidence (crown): {crown_conf:.3f}")
    print(f"Has building: {has_building}")
    print(f"Crown range: {crown_distance:.2f} m")
    print(f"Ground range: {ground_distance:.2f} m")
    print(f"Range difference: {ground_distance - crown_distance:.2f} m")
    print(f"Pixel: ({x_bc}, {y_bc}) | Bearing: {bearing:.1f}°")
    print(f"Camera GPS: ({cam_lat:.6f}, {cam_lon:.6f})")
    print(f"Tree   GPS: ({tree_lat:.6f}, {tree_lon:.6f})")

    # Save JSON
    out = {
        'image_name': args.image_name,
        'tree_index': args.tree_index,
        'method_used': method,
        'crown_disp': float(crown_disp),
        'crown_confidence': float(crown_conf),
        'crown_distance_m': float(crown_distance),
        'ground_distance_m': float(ground_distance),
        'building_detected': bool(has_building),
        'building_disp': float(bld_disp) if bld_disp is not None else None,
        'projection_factor': float(args.projection_factor),
        'tree_height_est_m': float(est_tree_h),
        'scale_mode': args.scale_mode,
        'scale_value_used': float(scale_value),
        'global_scale': float(global_scale),
        'auto_ratio_used': float(ratio_used) if ratio_used is not None else None,
        'camera_height_m': float(args.camera_height),
        'pitch_deg': float(args.pitch_deg),
        'bearing_deg': float(bearing),
        'pixel_x': int(x_bc),
        'pixel_y': int(y_bc),
        'camera_lat': float(cam_lat),
        'camera_lon': float(cam_lon),
        'tree_lat': float(tree_lat),
        'tree_lon': float(tree_lon),
    }
    safe_name = args.image_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('&', '_').replace('=', '_')
    out_path = f"ground_projection_result_{safe_name}_{args.tree_index}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
