#!/usr/bin/env python3
"""
Tree Ground Projection Estimator (no-trunk cases) - V4 Fixed-Only
-------------------------------------------------
Optimized single-method approach for estimating tree trunk base position using
consistent fixed projection from crown to ground with building constraint validation.

Key Features:
- Robust crown disparity extraction using median statistics from upper bbox region
- Fixed projection method with consistent crown-to-ground range multiplication
- Simple building detection using 95th percentile disparity in search bands
- Blended scale estimation combining per-image ground calibration with global fallback
- Streamlined codebase focused on reliable single-method performance

Scale Modes:
- Intelligent blending of per-image ground scale with global scale when available
- Automatic ratio validation ensures reasonable scale estimates within bounds
- Fallback to global scale when per-image calibration yields unrealistic values

Building Detection:
- Horizontal band search using configurable margin around tree center column
- Threshold-based detection requiring 30% higher disparity than crown measurement
- Fixed clearance margin applied uniformly behind all detected building surfaces

Projection Method:
- Single fixed projection factor consistently applied to crown distance
- Building constraint overrides projection when occluder requires greater clearance
- No adaptive projection - uses constant multiplication factor for all scenarios

Target Use Case:
- High-throughput processing systems requiring consistent and predictable behavior
- Minimal complexity deployment suitable for embedded or resource-constrained environments
- Applications where projection method consistency is prioritized over scene adaptivity

Input Requirements:
- CSV/XLSX file containing bbox data (file_name, x_box, y_box, width_box, height_box)
- Image name and tree index (0-based) for selecting specific tree within image
- Disparity map (.npy) for each image and scale JSON with {"inverse_depth_scale": <float>}
- Street View image filenames containing GPS coordinates and heading/FOV metadata
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------
# Utilities
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


def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    hf = math.radians(hfov_deg)
    vf = 2.0 * math.atan(math.tan(hf / 2.0) * (height / float(width)))
    return math.degrees(vf)


def intrinsics_from_fov(hfov_deg: float, vfov_deg: float, width: int, height: int) -> Tuple[float, float]:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return fx, fy


def gps_from_heading_and_distance(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    R = 6371000.0
    th = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    dR = distance_m / R
    phi2 = math.asin(math.sin(phi1) * math.cos(dR) + math.cos(phi1) * math.sin(dR) * math.cos(th))
    lam2 = lam1 + math.atan2(math.sin(th) * math.sin(dR) * math.cos(phi1),
                             math.cos(dR) - math.sin(phi1) * math.sin(phi2))
    return math.degrees(phi2), math.degrees(lam2)


# ------------------------------
# IO helpers
# ------------------------------

def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported table format: {ext}")
    # Normalize headers if necessary
    df.columns = [str(c).strip() for c in df.columns]
    # Expect columns: file_name, x_box, y_box, width_box, height_box
    required = {"file_name", "x_box", "y_box", "width_box", "height_box"}
    cols_lower = {c.lower() for c in df.columns}
    if not required.issubset(cols_lower):
        raise ValueError(f"Table must contain columns: {required}. Found: {list(df.columns)}")
    # ensure column names as expected (lowercase)
    mapping = {c: c.lower() for c in df.columns}
    df = df.rename(columns=mapping)
    df["file_name"] = df["file_name"].astype(str)
    return df


def bbox_to_pixel(cx: float, cy: float, bw: float, bh: float,
                  orig_w: int, orig_h: int, disp_w: int, disp_h: int) -> Tuple[int, int, int, int, int, int]:
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
# Scale estimation (per-image ground)
# ------------------------------

def estimate_per_image_scale(disp_map: np.ndarray, hfov_deg: float, vfov_deg: float,
                             cam_h: float = 2.5, pitch_deg: float = 0.0,
                             min_points: int = 300) -> Optional[float]:
    H, W = disp_map.shape[:2]
    fx, fy = intrinsics_from_fov(hfov_deg, vfov_deg, W, H)
    v0 = int(0.70 * H)
    ys, xs = np.mgrid[v0:H, 0:W]
    disp_vals = disp_map[ys, xs]
    vc = ys - (H / 2.0)
    phi = np.arctan2(vc, fy) + math.radians(pitch_deg)
    mask = (phi > math.radians(1.0)) & np.isfinite(disp_vals) & (disp_vals > 1e-6)
    if np.count_nonzero(mask) < min_points:
        return None
    d_ground = cam_h / np.tan(phi[mask])
    s_samples = (d_ground * disp_vals[mask]).astype(np.float64)
    s_samples = s_samples[np.isfinite(s_samples)]
    if s_samples.size < min_points:
        return None
    med = float(np.median(s_samples))
    return med if med > 0 else None


# ------------------------------
# Crown disparity extraction and building detection
# ------------------------------

def extract_crown_disparity_simple(disp_patch: np.ndarray, top_frac: float = 0.7) -> Tuple[float, float]:
    H, W = disp_patch.shape[:2]
    crown_end = max(1, int(H * top_frac))
    crown = disp_patch[:crown_end, :]
    vals = crown[np.isfinite(crown)]
    if vals.size < 10:
        return float("nan"), 0.0
    disp = float(np.median(vals))
    coverage = vals.size / float(crown.size)
    mad = float(np.median(np.abs(vals - disp)))
    consistency = max(0.0, 1.0 - min(1.0, mad / (0.3 * (disp + 1e-9))))
    confidence = 0.4 * coverage + 0.6 * consistency
    return disp, float(np.clip(confidence, 0.0, 1.0))


def detect_building_disparity(disp_map: np.ndarray, x_bc: int, y1: int, y2: int,
                              margin: int = 40, min_vals: int = 100) -> Optional[float]:
    H, W = disp_map.shape[:2]
    x_left = max(0, x_bc - margin)
    x_right = min(W, x_bc + margin + 1)
    y_top = max(0, y1)
    y_bot = min(H, y2 + margin)
    region = disp_map[y_top:y_bot, x_left:x_right]
    vals = region[np.isfinite(region)]
    if vals.size < min_vals:
        return None
    return float(np.nanpercentile(vals, 95))


# ------------------------------
# Fixed projection logic
# ------------------------------

def fixed_projection_method(crown_disp: float, scale: float,
                            bld_disp: Optional[float] = None,
                            projection_factor: float = 0.6,
                            building_clearance: float = 2.5) -> Tuple[float, str]:
    crown_distance = scale / crown_disp
    projected_distance = crown_distance * (1.0 + projection_factor)
    method = "fixed_projection"
    if bld_disp is not None and bld_disp > crown_disp * 1.3:
        building_distance = scale / bld_disp
        building_constrained_distance = building_distance + building_clearance
        if building_constrained_distance > projected_distance:
            projected_distance = building_constrained_distance
            method = "building_constrained"
    return projected_distance, method


# ------------------------------
# Main estimator (fixed-only)
# ------------------------------

class FixedOnlyEstimator:
    def __init__(self, camera_height: float = 2.5, projection_factor: float = 0.6,
                 building_clearance: float = 2.5):
        self.camera_height = camera_height
        self.projection_factor = projection_factor
        self.building_clearance = building_clearance

    def estimate_tree_position(self, disp_map: np.ndarray, tree_row: pd.Series,
                               hfov_deg: float, vfov_deg: float, global_scale: float,
                               orig_size: Tuple[int, int] = (400, 400),
                               disp_size: Tuple[int, int] = (512, 256)) -> dict:
        cx, cy = float(tree_row["x_box"]), float(tree_row["y_box"])
        bw, bh = float(tree_row["width_box"]), float(tree_row["height_box"])
        x_bc, y_bc, x1, y1, x2, y2 = bbox_to_pixel(cx, cy, bw, bh, orig_size[0], orig_size[1], disp_size[0], disp_size[1])
        H, W = disp_map.shape[:2]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid bbox after mapping to disparity coordinates")

        disp_patch = disp_map[y1:y2, x1:x2]
        crown_disp, crown_conf = extract_crown_disparity_simple(disp_patch)
        if not np.isfinite(crown_disp) or crown_disp <= 1e-9:
            raise ValueError("Failed to extract valid crown disparity")

        bld_disp = detect_building_disparity(disp_map, x_bc, y1, y2)

        per_image_scale = estimate_per_image_scale(disp_map, hfov_deg, vfov_deg, self.camera_height)
        if per_image_scale is not None and 0.5 < (per_image_scale / global_scale) < 2.0:
            # blend per-image with global (simple weighted blend)
            # weight depends slightly on how different the scene is (here just moderate mixing)
            conf = 0.6
            scale = conf * per_image_scale + (1 - conf) * global_scale
            scale_method = "blended"
        else:
            scale = global_scale
            scale_method = "global"

        ground_distance, method = fixed_projection_method(
            crown_disp, scale, bld_disp,
            projection_factor=self.projection_factor,
            building_clearance=self.building_clearance
        )

        offset_px = x_bc - (disp_size[0] / 2.0)
        angle_offset = (offset_px / disp_size[0]) * hfov_deg

        return {
            "method_family": "fixed_only",
            "method_detail": method,
            "crown_disparity": float(crown_disp),
            "crown_confidence": float(crown_conf),
            "crown_distance_m": float(scale / crown_disp),
            "ground_distance_m": float(ground_distance),
            "building_detected": bld_disp is not None,
            "building_disparity": float(bld_disp) if bld_disp is not None else None,
            "scale_used": float(scale),
            "scale_method": scale_method,
            "bearing_offset_deg": float(angle_offset),
            "pixel_x": int(x_bc),
            "pixel_y": int(y_bc),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        }


# ------------------------------
# CLI / main
# ------------------------------

def find_npy_for_image(image_name: str, disp_dir: str) -> Tuple[Optional[str], Tuple[str, str]]:
    base, _ = os.path.splitext(str(image_name))
    p_base = os.path.join(disp_dir, base + ".npy")
    p_disp = os.path.join(disp_dir, base + "_disp.npy")
    if os.path.isfile(p_base):
        return p_base, (p_base, p_disp)
    if os.path.isfile(p_disp):
        return p_disp, (p_base, p_disp)
    return None, (p_base, p_disp)


def main() -> int:
    p = argparse.ArgumentParser(description="Fixed-only Tree Ground Projection Estimator")
    p.add_argument('--csv_path', required=True)
    p.add_argument('--image_name', required=True)
    p.add_argument('--tree_index', type=int, required=True)
    p.add_argument('--disp_path', required=True)
    p.add_argument('--scale_path', required=True)
    p.add_argument('--orig_w', type=int, default=400)
    p.add_argument('--orig_h', type=int, default=400)
    p.add_argument('--disp_w', type=int, default=512)
    p.add_argument('--disp_h', type=int, default=256)
    p.add_argument('--projection_factor', type=float, default=0.6)
    p.add_argument('--building_clearance', type=float, default=2.5)
    p.add_argument('--camera_height', type=float, default=2.5)
    args = p.parse_args()

    print("Fixed-only Tree Ground Projection Estimator")
    print("=" * 50)
    print(f"Image: {args.image_name}")
    print(f"Tree Index: {args.tree_index}")
    print(f"Projection factor: {args.projection_factor}")
    print("=" * 50)

    df = load_table(args.csv_path)
    rows = df[df['file_name'].astype(str) == str(args.image_name)].reset_index(drop=True)
    if rows.empty:
        raise RuntimeError(f"No rows found for image_name: {args.image_name}")
    if args.tree_index < 0 or args.tree_index >= len(rows):
        raise RuntimeError(f"tree_index {args.tree_index} out of range (found {len(rows)} rows)")
    tree_row = rows.iloc[args.tree_index]

    if os.path.isfile(args.disp_path):
        disp_file = args.disp_path
    elif os.path.isdir(args.disp_path):
        disp_file, candidates = find_npy_for_image(args.image_name, args.disp_path)
        if disp_file is None:
            print("ERROR: Could not find disparity file. Tried:")
            print(f"  {candidates[0]}\n  {candidates[1]}")
            return 1
    else:
        print(f"ERROR: disp_path does not exist: {args.disp_path}")
        return 1

    disp_map = np.load(disp_file)
    print(f"Loaded disparity map: {disp_map.shape} from {disp_file}")

    with open(args.scale_path, 'r', encoding='utf-8') as f:
        global_scale = json.load(f)['inverse_depth_scale']

    heading_deg, hfov_deg = parse_heading_and_fov_from_filename(args.image_name)
    vfov_deg = vfov_from_hfov(hfov_deg, args.disp_w, args.disp_h)

    estimator = FixedOnlyEstimator(
        camera_height=args.camera_height,
        projection_factor=args.projection_factor,
        building_clearance=args.building_clearance
    )

    result = estimator.estimate_tree_position(
        disp_map, tree_row, hfov_deg, vfov_deg, global_scale,
        orig_size=(args.orig_w, args.orig_h),
        disp_size=(args.disp_w, args.disp_h)
    )

    cam_lat, cam_lon = parse_gps_from_filename(args.image_name)
    bearing = (heading_deg + result['bearing_offset_deg']) % 360.0
    tree_lat, tree_lon = gps_from_heading_and_distance(cam_lat, cam_lon, bearing, result['ground_distance_m'])

    print("\nEstimation Results:")
    print("=" * 50)
    print(f"Method detail: {result['method_detail']}")
    print(f"Crown confidence: {result['crown_confidence']:.3f}")
    print(f"Building detected: {result['building_detected']}")
    print(f"Crown distance: {result['crown_distance_m']:.2f} m")
    print(f"Ground distance: {result['ground_distance_m']:.2f} m")
    print(f"Tree GPS: ({tree_lat:.6f}, {tree_lon:.6f})")

    out = {
        "image_name": args.image_name,
        "tree_index": args.tree_index,
        "result": result,
        "bearing_deg": bearing,
        "camera_lat": cam_lat,
        "camera_lon": cam_lon,
        "tree_lat": tree_lat,
        "tree_lon": tree_lon
    }

    safe_name = args.image_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    out_path = f"fixed_result_{safe_name}_{args.tree_index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
