#!/usr/bin/env python3
"""
Tree Ground Projection Estimator (no-trunk cases) - V3 Simplified
-------------------------------------------------
Streamlined approach for estimating tree trunk base position from Street View
images when the trunk is occluded, emphasizing precision and computational efficiency.

Key Features:
- Robust crown disparity extraction using upper bbox portion with trimmed statistics
- Precise bearing computation using pinhole camera model derived from HFOV/VFOV
- Streamlined building detection with 95th percentile disparity in search bands
- Conservative building-constrained projection with fixed clearance margins
- Simplified scale mode selection between global and per-image calibration

Scale Modes:
- 'global': Uses fixed scale from JSON configuration file
- 'per_image_ground': Calibrates scale from ground pixels using camera height and geometry
- Automatic fallback to global scale when per-image calibration fails

Building Detection:
- Horizontal band search around tree column using configurable margin width
- Simple threshold-based detection with 30% disparity difference requirement
- Fixed clearance margin applied behind detected building surfaces

Projection Method:
- Direct crown-to-ground estimation using crown disparity (p70/median) without adaptive projection
- Building constraint ensures minimum safe distance behind detected obstacles
- Single-pass decision between direct crown range and building-constrained estimates

Target Use Case:
- Balanced performance system suitable for research and development workflows
- Cleaner codebase with reduced complexity compared to advanced variants
- Effective for standard urban scenes with moderate occluder complexity

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
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Filename parsers
# --------------------------------------------------------------------------------------

def parse_gps_from_filename(filename: str) -> Tuple[float, float]:
    """Parse camera GPS from an image name.

    Supports plain and URL-encoded variants like:
      ...location=32.123,-117.456&...
      ...location%3D32.123%2C-117.456...
    """
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
    """Parse heading (deg) and horizontal FOV (deg) from filename."""
    heading_patterns = [r"heading=([-0-9\.]+)", r"&heading=([-0-9\.]+)", r"heading%3D([-0-9\.]+)"]
    fov_patterns = [r"fov=([-0-9\.]+)", r"&fov=([-0-9\.]+)", r"fov%3D([-0-9\.]+)"]
    heading = None
    fov = None
    for pat in heading_patterns:
        m = re.search(pat, filename)
        if m:
            heading = float(m.group(1))
            break
    for pat in fov_patterns:
        m = re.search(pat, filename)
        if m:
            fov = float(m.group(1))
            break
    if heading is None or fov is None:
        raise ValueError(f"Could not parse heading/fov from filename: {filename}")
    return heading, fov


# --------------------------------------------------------------------------------------
# Camera model & geometry helpers
# --------------------------------------------------------------------------------------

def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    """Compute vertical FOV (deg) from horizontal FOV (deg) and aspect ratio.

    Uses the pinhole relationship: tan(HFOV/2) / (W/2) = tan(VFOV/2) / (H/2).
    """
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (height / float(width)))
    return math.degrees(vfov)


def intrinsics_from_fov(hfov_deg: float, vfov_deg: float, width: int, height: int) -> Tuple[float, float]:
    """Return (fx, fy) for a centered pinhole camera."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
    return fx, fy


def bbox_to_pixel(cx: float, cy: float, bw: float, bh: float,
                  orig_w: int, orig_h: int, disp_w: int, disp_h: int) -> Tuple[int, int, int, int, int, int]:
    """Map YOLO-normalized bbox (center cx,cy & size bw,bh) in the original grid
    to bottom-center and bbox corners in the disparity grid.
    Returns integers: x_bc, y_bc, x1, y1, x2, y2.
    """
    # original (normalized → pixels in original grid)
    x0 = cx * orig_w
    y0 = cy * orig_h
    x_bc0 = x0
    y_bc0 = y0 + (bh * orig_h) / 2.0

    # scale to disparity resolution
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


def robust_quantile(values: np.ndarray, q: float = 0.7) -> float:
    """Return a robust quantile (ignoring NaN) with simple sanity checks."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return float("nan")
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(vals, q))


def calibrate_scale_from_ground(
    disp_map: np.ndarray,
    hfov_deg: float,
    vfov_deg: float,
    cam_h: float = 2.5,
    pitch_deg: float = 0.0,
    ground_band_min_frac: float = 0.70,
    min_points: int = 500,
) -> Optional[float]:
    """Estimate a per-image scale from ground pixels.

    Idea: for a pixel row v, vertical angle to ray is atan((v - cy)/fy) + pitch.
    For ground points (phi > 0), the range to ground along the ray satisfies
      d_ground = cam_h / tan(phi).
    If disparity ~ (scale / depth), then scale ≈ median(d_ground * disparity).
    Returns None if not enough reliable points are found.
    """
    H, W = disp_map.shape[:2]
    fx, fy = intrinsics_from_fov(hfov_deg, vfov_deg, W, H)
    # ground band: bottom portion of the image
    v0 = int(ground_band_min_frac * H)
    ys, xs = np.mgrid[v0:H, 0:W]
    disp_vals = disp_map[ys, xs]

    # compute vertical angle (downwards positive)
    vc = ys - (H / 2.0)
    phi = np.arctan2(vc, fy) + math.radians(pitch_deg)

    mask = (phi > math.radians(0.5)) & np.isfinite(disp_vals) & (disp_vals > 1e-6)
    if np.count_nonzero(mask) < min_points:
        return None

    d_ground = cam_h / np.tan(phi[mask])
    s_samples = d_ground * disp_vals[mask]

    s = np.median(s_samples[np.isfinite(s_samples)])
    if not np.isfinite(s) or s <= 0:
        return None
    return float(s)


def find_npy_for_image(image_name: str, disp_dir: str) -> Tuple[Optional[str], Tuple[str, str]]:
    """Try to locate the disparity file for an image using two conventions:
       <base>.npy  or  <base>_disp.npy
    Returns (path or None, (candidate1, candidate2)).
    """
    base, _ = os.path.splitext(str(image_name))
    p_base = os.path.join(disp_dir, base + ".npy")
    p_disp = os.path.join(disp_dir, base + "_disp.npy")
    if os.path.isfile(p_base):
        return p_base, (p_base, p_disp)
    if os.path.isfile(p_disp):
        return p_disp, (p_base, p_disp)
    return None, (p_base, p_disp)


# --------------------------------------------------------------------------------------
# Core estimation class
# --------------------------------------------------------------------------------------

class GroundProjectionEstimator:
    """Estimate trunk base range/bearing from a bbox when the trunk is occluded."""

    def __init__(self,
                 camera_height: float = 2.5,
                 building_clearance: float = 2.5,
                 crown_top_frac: float = 0.7,
                 crown_stat: str = "p70",
                 debug: bool = False):
        self.camera_height = camera_height
        self.building_clearance = building_clearance
        self.crown_top_frac = crown_top_frac
        self.crown_stat = crown_stat  # "median" or "p70"
        self.debug = debug
        self.results = {}

    # --------- disparity measurement ---------
    def _crown_disparity(self, disp_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
        """Measure disparity in the crown area (upper part of bbox). Returns (disp, confidence)."""
        H, W = disp_map.shape[:2]
        x1c = max(0, min(x1, W-1)); x2c = max(0, min(x2, W))
        y1c = max(0, min(y1, H-1)); y2c = max(0, min(y2, H))
        if x1c >= x2c or y1c >= y2c:
            return float("nan"), 0.0

        patch = disp_map[y1c:y2c, x1c:x2c]
        # Crown = top portion of bbox
        h = patch.shape[0]
        crown_end = int(max(1, round(self.crown_top_frac * h)))
        crown_patch = patch[0:crown_end, :]

        vals = crown_patch[np.isfinite(crown_patch)]
        if vals.size < 10:
            return float("nan"), 0.0

        if self.crown_stat == "median":
            d = float(np.median(vals))
        else:  # p70 gives a bias toward nearer structures (higher disparity)
            d = robust_quantile(vals, 0.70)

        # confidence from sample count and dispersion
        iqr = np.subtract(*np.percentile(vals, [75, 25])) if vals.size >= 20 else np.nan
        conf = 0.3 + 0.7 * (1.0 / (1.0 + (iqr if np.isfinite(iqr) else 1.0)))
        conf = float(max(0.0, min(1.0, conf)))
        return d, conf

    def _building_disparity(self, disp_map: np.ndarray, x_bc: int, y1: int, y2: int, margin: int = 40) -> Optional[float]:
        """Stronger building detection: wider horizontal band, cut sky, use high percentile.
        Returns a representative building disparity (p95) or None.
        """
        H, W = disp_map.shape[:2]
        x_left = max(0, x_bc - margin)
        x_right = min(W, x_bc + margin + 1)
        y_top = max(0, y1)
        y_bot = min(H, y2 + margin)
        region = disp_map[y_top:y_bot, x_left:x_right]
        vals = region[np.isfinite(region)]
        if vals.size < 100:
            return None
        return float(np.nanpercentile(vals, 95))

    # --------- projection / bearing ---------
    @staticmethod
    def bearing_and_range_from_pixel(u: int, disp: float, scale: float,
                                     hfov_deg: float, vfov_deg: float,
                                     W: int, H: int) -> Tuple[float, float]:
        """Compute (bearing_offset_deg, horizontal_range_m) from pixel u and disparity.
        Uses a centered pinhole model.
        """
        fx, fy = intrinsics_from_fov(hfov_deg, vfov_deg, W, H)
        uc = u - W/2.0
        Z = scale / max(disp, 1e-6)
        X = Z * (uc / fx)
        R = math.hypot(X, Z)  # horizontal range in the ground plane
        alpha_deg = math.degrees(math.atan2(X, Z))
        return alpha_deg, R

    # --------- main API ---------
    def estimate(self,
                 disp_map: np.ndarray,
                 tree_row: pd.Series,
                 hfov_deg: float,
                 vfov_deg: float,
                 scale_value: float,
                 orig_size: Tuple[int, int] = (400, 400),
                 disp_size: Tuple[int, int] = (512, 256)) -> Tuple[int, int, float, dict]:
        """Estimate (x_disp, y_disp, ground_range_m, details_dict)."""
        cx = float(tree_row["x_box"]); cy = float(tree_row["y_box"])
        bw = float(tree_row["width_box"]); bh = float(tree_row["height_box"])
        x_bc, y_bc, x1, y1, x2, y2 = bbox_to_pixel(cx, cy, bw, bh,
                                                   orig_size[0], orig_size[1],
                                                   disp_size[0], disp_size[1])
        H, W = disp_map.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid bbox after clipping")

        # 1) disparity of the crown
        crown_disp, crown_conf = self._crown_disparity(disp_map, x1, y1, x2, y2)
        if not np.isfinite(crown_disp) or crown_disp <= 0:
            raise ValueError("Failed to estimate crown disparity")

        # 2) bearing & range from crown disparity
        alpha_deg, R_crown = self.bearing_and_range_from_pixel(x_bc, crown_disp, scale_value,
                                                               hfov_deg, vfov_deg, W, H)

        # 3) detect near building and apply a back-off margin
        bld_disp = self._building_disparity(disp_map, x_bc, y1, y2)
        has_building = False
        R_building = None
        if bld_disp is not None and bld_disp > crown_disp * 1.3:  # 30% closer than crown
            R_building = (scale_value / bld_disp)
            has_building = True

        if has_building:
            ground_range = max(R_crown, R_building + self.building_clearance)
            method = "building_constrained"
        else:
            ground_range = R_crown
            method = "direct_from_crown"

        details = {
            "method": method,
            "confidence": float(crown_conf * (0.8 if has_building else 1.0)),
            "x_bc": int(x_bc),
            "y_bc": int(y_bc),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "crown_disparity": float(crown_disp),
            "range_from_crown_m": float(R_crown),
            "has_building": bool(has_building),
            "building_disparity": float(bld_disp) if bld_disp is not None else None,
            "building_range_m": float(R_building) if R_building is not None else None,
            "bearing_offset_deg": float(alpha_deg),
        }
        self.results = details
        return x_bc, y_bc, ground_range, details


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------

def _clean_header(s: str) -> str:
    s = str(s)
    # remove zero-width & BOM, normalize spacing
    s = re.sub(r"[\u200b\u200e\u200f\ufeff]+", "", s)
    s = s.strip().lower().replace("-", "_").replace(" ", "_")
    return s


def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _clean_header(c) for c in df.columns})
    aliases = {
        "filename": "file_name",
        "file": "file_name",
        "image": "file_name",
        "image_name": "file_name",
        "img_name": "file_name",
        "x": "x_box",
        "y": "y_box",
        "w": "width_box",
        "h": "height_box",
    }
    for a, b in aliases.items():
        if a in df.columns and b not in df.columns:
            df = df.rename(columns={a: b})
    required = {"file_name", "x_box", "y_box", "width_box", "height_box"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}. Got: {list(df.columns)}")
    return df


def load_table_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        data = pd.read_excel(path, sheet_name=None)
        for sheet_name, df in data.items():
            try:
                return unify_columns(df)
            except KeyError:
                continue
        raise KeyError("No sheet with required columns found in the Excel file.")
    elif ext == ".csv":
        return unify_columns(pd.read_csv(path))
    else:
        raise ValueError(f"Unsupported table format: {ext}")


# --------------------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------------------

def gps_from_heading_and_distance(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    R = 6371000.0
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    dR = distance_m / R
    phi2 = math.asin(
        math.sin(phi1) * math.cos(dR) + math.cos(phi1) * math.sin(dR) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(dR) * math.cos(phi1),
        math.cos(dR) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), math.degrees(lam2)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Tree Ground Projection Estimator (trunk occluded)")
    p.add_argument("--csv_path", required=True, help="Path to CSV/XLSX with bbox rows")
    p.add_argument("--image_name", required=True, help="Exact 'file_name' as in the table")
    p.add_argument("--tree_index", type=int, required=True, help="0-based index within that image")
    p.add_argument("--disp_path", required=True, help="Path to disparity .npy or a directory containing it")
    p.add_argument("--scale_path", required=True, help="Path to global scale JSON (fallback)")
    p.add_argument("--scale_mode", choices=["global", "per_image_ground", "auto"], default="auto",
                   help="How to obtain metric scale: use global JSON or calibrate from ground pixels")
    p.add_argument("--orig_w", type=int, default=400)
    p.add_argument("--orig_h", type=int, default=400)
    p.add_argument("--disp_w", type=int, default=512)
    p.add_argument("--disp_h", type=int, default=256)
    p.add_argument("--camera_height", type=float, default=2.5, help="Camera height above ground (m)")
    p.add_argument("--pitch_deg", type=float, default=0.0, help="Camera pitch, positive = looking down (deg)")
    p.add_argument("--building_clearance", type=float, default=2.5, help="Meters behind the detected building")
    p.add_argument("--crown_top_frac", type=float, default=0.7, help="Top fraction of bbox used as crown area")
    p.add_argument("--crown_stat", choices=["median", "p70"], default="p70")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()

    print("Tree Ground Projection Estimator")
    print("=" * 50)
    print(f"Image: {args.image_name}")
    print(f"Tree Index: {args.tree_index}")
    print(f"Scale mode: {args.scale_mode}")
    print(f"Camera height: {args.camera_height} m  |  Pitch: {args.pitch_deg}°")
    print(f"Building clearance: {args.building_clearance} m")
    print("=" * 50)

    # Load table and select row
    df = load_table_any(args.csv_path)
    rows = df[df["file_name"].astype(str) == str(args.image_name)].reset_index(drop=True)
    if rows.empty:
        raise RuntimeError(f"No rows found for image_name: {args.image_name}")
    if args.tree_index < 0 or args.tree_index >= len(rows):
        raise RuntimeError(f"tree_index {args.tree_index} out of range (found {len(rows)} rows)")
    tree_row = rows.iloc[args.tree_index]

    # Locate disparity file
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
    print(f"Loaded disparity map: {disp_map.shape}  from {disp_file}")

    # Parse camera intrinsics
    heading_deg, hfov_deg = parse_heading_and_fov_from_filename(args.image_name)
    vfov_deg = vfov_from_hfov(hfov_deg, args.disp_w, args.disp_h)

        # Obtain scale
    with open(args.scale_path, "r", encoding="utf-8") as f:
        global_scale = json.load(f)["inverse_depth_scale"]

    if args.scale_mode == "global":
        scale_value = global_scale
        print(f"Using global scale: {scale_value:.6f}")
    elif args.scale_mode == "per_image_ground":
        s_est = calibrate_scale_from_ground(
            disp_map, hfov_deg, vfov_deg, cam_h=args.camera_height, pitch_deg=args.pitch_deg
        )
        if s_est is None:
            print("Per-image scale calibration failed – using global scale.")
            scale_value = global_scale
        else:
            scale_value = s_est
            print(f"Using per-image scale: {scale_value:.6f}")
    else:
        # AUTO mode: compute per-image ground scale with confidence and blend
        def _sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))

        def ground_scale_with_confidence(disp_map, hfov_deg, vfov_deg, cam_h, pitch_deg):
            H, W = disp_map.shape[:2]
            fx, fy = intrinsics_from_fov(hfov_deg, vfov_deg, W, H)
            v0 = int(0.70 * H)
            ys, xs = np.mgrid[v0:H, 0:W]
            disp_vals = disp_map[ys, xs]
            vc = ys - (H / 2.0)
            phi = np.arctan2(vc, fy) + math.radians(pitch_deg)
            mask = (phi > math.radians(0.5)) & np.isfinite(disp_vals) & (disp_vals > 1e-6)
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
            w_cov = _sigmoid((coverage - 0.05) / 0.03)
            w_rob = _sigmoid((robustness - 0.5) / 0.15)
            conf = float(np.clip(0.5 * w_cov + 0.5 * w_rob, 0.0, 1.0))
            return med, conf

        s_est, conf = ground_scale_with_confidence(disp_map, hfov_deg, vfov_deg,
                                                   args.camera_height, args.pitch_deg)
        if s_est is None:
            scale_value = global_scale
            print("Auto scale: fallback to global (not enough ground).")
        else:
            scale_value = float(conf * s_est + (1.0 - conf) * global_scale)
            print(f"Auto scale: s_ground={s_est:.6f}, conf={conf:.2f}, global={global_scale:.6f} → used={scale_value:.6f}")

    # Create estimator
    estimator = GroundProjectionEstimator(
        camera_height=args.camera_height,
        building_clearance=args.building_clearance,
        crown_top_frac=args.crown_top_frac,
        crown_stat=args.crown_stat,
        debug=args.debug,
    )

    x_disp, y_disp, ground_range_m, details = estimator.estimate(
        disp_map=disp_map,
        tree_row=tree_row,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        scale_value=scale_value,
        orig_size=(args.orig_w, args.orig_h),
        disp_size=(args.disp_w, args.disp_h),
    )

    # Bearing and GPS
    bearing = (heading_deg + details["bearing_offset_deg"]) % 360.0
    cam_lat, cam_lon = parse_gps_from_filename(args.image_name)
    tree_lat, tree_lon = gps_from_heading_and_distance(cam_lat, cam_lon, bearing, ground_range_m)

    print("\nGround Projection Results:")
    print("=" * 50)
    print(f"Method: {details['method']}")
    print(f"Confidence: {details['confidence']:.3f}")
    print(f"Has building: {details['has_building']}")
    crown_range = details.get("range_from_crown_m", None)
    if crown_range is not None:
        print(f"Crown range: {crown_range:.2f} m")
    print(f"Ground range: {ground_range_m:.2f} m")
    if crown_range is not None:
        print(f"Range difference: {ground_range_m - crown_range:.2f} m")
    print(f"Pixel: ({x_disp}, {y_disp}) | Bearing: {bearing:.1f}°")
    print(f"Camera GPS: ({cam_lat:.6f}, {cam_lon:.6f})")
    print(f"Tree   GPS: ({tree_lat:.6f}, {tree_lon:.6f})")

    # Save JSON
    out = {
        "image_name": args.image_name,
        "tree_index": args.tree_index,
        "method": details["method"],
        "confidence": details["confidence"],
        "has_building": details["has_building"],
        "crown_disparity": details["crown_disparity"],
        "crown_range_m": crown_range,
        "ground_range_m": ground_range_m,
        "bearing_deg": bearing,
        "camera_lat": cam_lat,
        "camera_lon": cam_lon,
        "tree_lat": tree_lat,
        "tree_lon": tree_lon,
        "scale_value": scale_value,
        "scale_mode": args.scale_mode,
        "building_disparity": details.get("building_disparity"),
        "building_range_m": details.get("building_range_m"),
        "x_disp": x_disp,
        "y_disp": y_disp,
        "bbox": details["bbox"],
        "params": {
            "orig_w": args.orig_w,
            "orig_h": args.orig_h,
            "disp_w": args.disp_w,
            "disp_h": args.disp_h,
            "camera_height": args.camera_height,
            "pitch_deg": args.pitch_deg,
            "building_clearance": args.building_clearance,
            "crown_top_frac": args.crown_top_frac,
            "crown_stat": args.crown_stat,
            "hfov_deg": hfov_deg,
            "vfov_deg": vfov_deg,
        },
    }

    safe_name = (
        args.image_name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("&", "_")
        .replace("=", "_")
        .replace(" ", "_")
    )
    out_path = f"ground_projection_result_{safe_name}_{args.tree_index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Optional debug viz
    if args.debug:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            im = ax.imshow(disp_map, cmap="plasma")
            x1, y1, x2, y2 = details["bbox"]
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec="w", lw=2))
            ax.plot(details["x_bc"], details["y_bc"], "wo", ms=6, label="bottom-center")
            ax.set_title("Disparity map with bbox")
            ax.legend(loc="lower right")
            fig.colorbar(im, ax=ax)
            debug_png = f"ground_projection_analysis_{safe_name}_{args.tree_index}.png"
            plt.tight_layout(); plt.savefig(debug_png, dpi=160)
            print(f"Debug figure saved to: {debug_png}")
        except Exception as e:  # matplotlib not installed etc.
            print(f"[debug] Skipped debug figure: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
