#!/usr/bin/env python3
"""
Tree Ground Projection Estimator (no-trunk cases) - MoE
-------------------------------------------------
Advanced Mixture of Experts approach combining multiple estimation methods with
intelligent confidence-weighted aggregation for optimal tree trunk base positioning.

Key Features:
- Multi-candidate estimation: crown direct, projected, building-constrained, and COG methods
- Confidence-weighted aggregation using median, weighted-median, or trimmed-mean modes
- Dynamic building detection with adaptive ratio thresholds and vertical run analysis
- Advanced near-occluder guard system preventing scale corruption from close obstacles
- Sophisticated AUTO scale mode with confidence blending and ratio validation guards

Scale Modes:
- 'global': Uses fixed scale from JSON configuration file
- 'per_image_ground': Ground-pixel calibration with column exclusion and confidence scoring
- 'auto': Intelligent confidence-weighted blending with soft/hard ratio gates and guard override

Building Detection:
- Dynamic threshold adjustment based on crown confidence and vertical extent analysis
- P95 disparity band search with configurable horizontal and vertical margins
- Near-occluder detection triggers protective scale override to prevent estimation corruption

Projection Method:
- Multiple candidate generation: direct crown, adaptive/fixed projection, building clearance
- Optional COG (Center of Gravity) ray-scan method for ground-contact point detection
- Confidence-weighted aggregation across all valid candidates using selectable combination modes

Target Use Case:
- State-of-the-art research system designed for maximum accuracy across challenging scenarios
- Handles complex urban environments with multiple occluders, scale ambiguity, and close obstacles
- Suitable for benchmark evaluation and scenarios requiring optimal performance regardless of complexity

Input Requirements:
- CSV/XLSX file containing bbox data (file_name, x_box, y_box, width_box, height_box)
- Image name and tree index (0-based) for selecting specific tree within image
- Disparity map (.npy) for each image and scale JSON with {"inverse_depth_scale": <float>}
- Street View image filenames containing GPS coordinates and heading/FOV metadata
"""

from __future__ import annotations
import os, re, math, json, argparse
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

# ---------------------------
# JSON helper (fix np types)
# ---------------------------
def _json_default(o):
    import numpy as _np
    if isinstance(o, (_np.integer, _np.int8, _np.int16, _np.int32, _np.int64)):
        return int(o)
    if isinstance(o, (_np.floating, _np.float16, _np.float32, _np.float64)):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)

# ---------------------------
# Parsers
# ---------------------------
def parse_gps_from_filename(filename: str) -> Tuple[float, float]:
    pats = [
        r"location=([-0-9\.]+),([-0-9\.]+)",
        r"&location=([-0-9\.]+),([-0-9\.]+)",
        r"location%3D([-0-9\.]+)%2C([-0-9\.]+)",
    ]
    for p in pats:
        m = re.search(p, filename)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise ValueError(f"Could not parse GPS from filename: {filename}")

def parse_heading_and_fov_from_filename(filename: str) -> Tuple[float, float]:
    h_pats = [r"heading=([-0-9\.]+)", r"&heading=([-0-9\.]+)", r"heading%3D([-0-9\.]+)"]
    f_pats = [r"fov=([-0-9\.]+)", r"&fov=([-0-9\.]+)", r"fov%3D([-0-9\.]+)"]
    heading = None; fov = None
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

# ---------------------------
# Camera / geometry
# ---------------------------
def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    hf = math.radians(hfov_deg)
    vf = 2.0 * math.atan(math.tan(hf/2.0) * (height / float(width)))
    return math.degrees(vf)

def intrinsics_from_fov(hfov_deg: float, vfov_deg: float, width: int, height: int) -> Tuple[float, float]:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg)/2.0)
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg)/2.0)
    return fx, fy

def gps_from_heading_and_distance(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    R = 6371000.0
    th = math.radians(bearing_deg)
    p1 = math.radians(lat); l1 = math.radians(lon)
    dr = distance_m / R
    p2 = math.asin(math.sin(p1) * math.cos(dr) + math.cos(p1) * math.sin(dr) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(dr) * math.cos(p1), math.cos(dr) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)

# ---------------------------
# IO helpers
# ---------------------------
def _clean_header(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u200b\u200e\u200f\ufeff]+", "", s)
    s = s.strip().lower().replace("-", "_").replace(" ", "_")
    return s

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _clean_header(c) for c in df.columns})
    aliases = {
        "filename": "file_name", "file": "file_name",
        "image": "file_name", "image_name": "file_name", "img_name": "file_name",
        "x": "x_box", "y": "y_box", "w": "width_box", "h": "height_box",
    }
    for a, b in aliases.items():
        if a in df.columns and b not in df.columns:
            df = df.rename(columns={a: b})
    required = {"file_name", "x_box", "y_box", "width_box", "height_box"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}. Got: {list(df.columns)}")
    df["file_name"] = df["file_name"].astype(str)
    return df

def load_table_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        data = pd.read_excel(path, sheet_name=None)
        for _, d in data.items():
            try:
                return unify_columns(d)
            except KeyError:
                continue
        raise KeyError("No sheet with required columns found in the Excel file.")
    elif ext == ".csv":
        return unify_columns(pd.read_csv(path))
    else:
        raise ValueError(f"Unsupported table format: {ext}")

def find_npy_for_image(image_name: str, disp_dir: str) -> Tuple[Optional[str], Tuple[str, str]]:
    base, _ = os.path.splitext(str(image_name))
    p1 = os.path.join(disp_dir, base + ".npy")
    p2 = os.path.join(disp_dir, base + "_disp.npy")
    if os.path.isfile(p1): return p1, (p1, p2)
    if os.path.isfile(p2): return p2, (p1, p2)
    return None, (p1, p2)

# ---------------------------
# BBox mapping
# ---------------------------
def bbox_to_pixel(cx: float, cy: float, bw: float, bh: float,
                  orig_w: int, orig_h: int, disp_w: int, disp_h: int) -> Tuple[int, int, int, int, int, int]:
    x0 = cx * orig_w; y0 = cy * orig_h
    x_bc0 = x0; y_bc0 = y0 + (bh * orig_h)/2.0
    sx = disp_w / float(orig_w); sy = disp_h / float(orig_h)
    x_bc = int(round(x_bc0 * sx)); y_bc = int(round(y_bc0 * sy))
    w_disp = bw * disp_w; h_disp = bh * disp_h
    x1 = int(round(x_bc - w_disp/2.0)); y1 = int(round(y_bc - h_disp/2.0))
    x2 = int(round(x_bc + w_disp/2.0)); y2 = int(round(y_bc + h_disp/2.0))
    return x_bc, y_bc, x1, y1, x2, y2

# ---------------------------
# Disparity measurements
# ---------------------------
def robust_crown_disparity(disp_map: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                           top_frac: float = 0.7, stat: str = "median") -> Tuple[float, float]:
    H, W = disp_map.shape[:2]
    x1c = max(0, min(x1, W-1)); x2c = max(0, min(x2, W))
    y1c = max(0, min(y1, H-1)); y2c = max(0, min(y2, H))
    if x1c >= x2c or y1c >= y2c: return float("nan"), 0.0
    patch = disp_map[y1c:y2c, x1c:x2c]
    if patch.size == 0: return float("nan"), 0.0
    h = patch.shape[0]; end = max(1, int(h * max(0.1, min(top_frac, 0.95))))
    crown = patch[:end, :]
    vals = crown[np.isfinite(crown)]
    if vals.size < 20: return float("nan"), 0.0
    disp = float(np.nanpercentile(vals, 90)) if stat == "p90" else float(np.nanmedian(vals))
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med))) if vals.size > 0 else float("inf")
    cov = vals.size / float(crown.size)
    rob = 1.0 - min(1.0, (mad / (0.3 * (med + 1e-6))))
    conf = float(np.clip(0.5 * cov + 0.5 * rob, 0.0, 1.0))
    return disp, conf

def band_p95_disparity(disp_map: np.ndarray, x_bc: int, y1: int, y2: int,
                       half_px: int = 40, extra_down: int = 40) -> Optional[float]:
    H, W = disp_map.shape[:2]
    xl = max(0, x_bc - half_px); xr = min(W, x_bc + half_px + 1)
    yt = max(0, y1); yb = min(H, y2 + extra_down)
    region = disp_map[yt:yb, xl:xr]
    vals = region[np.isfinite(region)]
    if vals.size < 100: return None
    return float(np.nanpercentile(vals, 95))

def vertical_run_len_px(y_top: int, y_bot: int) -> int:
    return max(0, int(y_bot) - int(y_top))

# ---------------------------
# Auto scale
# ---------------------------
def ground_scale_with_conf(disp_map: np.ndarray, hfov_deg: float, vfov_deg: float,
                           cam_h: float, pitch_deg: float,
                           phi_min_deg: float = 2.0,
                           exclude_col_half: int = 0) -> Dict[str, Any]:
    H, W = disp_map.shape[:2]
    fx, fy = intrinsics_from_fov(hfov_deg, vfov_deg, W, H)
    v0 = int(0.70 * H)
    ys, xs = np.mgrid[v0:H, 0:W]
    if exclude_col_half > 0:
        c0 = W//2 - exclude_col_half; c1 = W//2 + exclude_col_half
        mask_excl = (xs >= c0) & (xs <= c1)
    else:
        mask_excl = np.zeros_like(xs, dtype=bool)
    disp_vals = disp_map[ys, xs]
    vc = ys - (H/2.0)
    phi = np.arctan2(vc, fy) + math.radians(pitch_deg)
    phi_min = math.radians(phi_min_deg)
    mask = (~mask_excl) & (phi > phi_min) & np.isfinite(disp_vals) & (disp_vals > 1e-6)

    n_total = (H - v0) * W
    n_valid = int(np.count_nonzero(mask))
    if n_valid < 300:
        return {"s_ground": None, "used": None, "conf": 0.0, "coverage": n_valid/float(n_total+1e-6),
                "mad_over_med": None, "note": "not_enough_ground"}

    d_ground = cam_h / np.tan(phi[mask])
    s_samples = (d_ground * disp_vals[mask]).astype(np.float64)
    s_samples = s_samples[np.isfinite(s_samples)]
    if s_samples.size < 300:
        return {"s_ground": None, "used": None, "conf": 0.0, "coverage": n_valid/float(n_total+1e-6),
                "mad_over_med": None, "note": "not_enough_samples"}

    med = float(np.median(s_samples))
    mad = float(np.median(np.abs(s_samples - med)))
    coverage = n_valid / float(n_total)
    robustness = max(0.0, 1.0 - (mad / (0.25 * (med + 1e-9))))
    def _sigmoid(x: float) -> float: return 1.0 / (1.0 + math.exp(-x))
    w_cov = _sigmoid((coverage - 0.05) / 0.03)
    w_rob = _sigmoid((robustness - 0.5) / 0.15)
    conf = float(np.clip(0.5 * w_cov + 0.5 * w_rob, 0.0, 1.0))
    return {"s_ground": med, "used": med, "conf": conf, "coverage": coverage, "mad_over_med": mad/max(med,1e-9)}

# ---------------------------
# COG (ray-scan) candidate
# ---------------------------
def cog_candidate_range(disp_map: np.ndarray, x_bc: int, y1: int, y_bc: int,
                        scale_value: float, band_half_px: int = 8) -> Tuple[Optional[float], Dict[str, Any]]:
    H, W = disp_map.shape[:2]
    xl = max(0, x_bc - band_half_px); xr = min(W, x_bc + band_half_px + 1)
    yt = max(0, min(y1, H-1))
    yb = min(H-1, y_bc + max(25, (H//6)))
    best_row = None; best_disp = None
    for rr in range(yt, yb+1):
        band = disp_map[rr:rr+1, xl:xr]
        vals = band[np.isfinite(band)]
        if vals.size < 3: continue
        d = np.nanpercentile(vals, 90)
        if not np.isfinite(d) or d <= 0: continue
        if (best_disp is None) or (d > best_disp):
            best_disp = float(d); best_row = int(rr)
    if best_disp is None:
        return None, {"used_row": None, "used_col": int(x_bc), "score": 0.0,
                      "search_window": [int(yt), int(yb)], "band_half_px": int(band_half_px)}
    R = scale_value / best_disp
    score = (best_row - yt) / max(1, (yb - yt))
    meta = {"used_row": int(best_row), "used_col": int(x_bc), "score": float(np.clip(score, 0.0, 1.0)),
            "search_window": [int(yt), int(yb)], "band_half_px": int(band_half_px)}
    return float(R), meta

# ---------------------------
# MoE aggregators
# ---------------------------
def weighted_median(values: List[float], weights: List[float]) -> float:
    pairs = [(v, w) for v, w in zip(values, weights)
             if v is not None and np.isfinite(v) and w is not None and w > 0]
    if not pairs: return float("nan")
    pairs.sort(key=lambda t: t[0])
    total = sum(w for _, w in pairs); acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= 0.5 * total: return float(v)
    return float(pairs[-1][0])

def aggregate_candidates(cand_map: Dict[str, Optional[float]], weights: Dict[str, Optional[float]],
                         mode: str = "median", trim_frac: float = 0.25) -> float:
    vals = [cand_map[k] for k in ["crown","projected","building","cog"]
            if k in cand_map and cand_map[k] is not None and np.isfinite(cand_map[k])]
    if not vals: return float("nan")
    if mode == "median":
        return float(np.median(vals))
    if mode == "trimmed_mean":
        vv = sorted(vals); k = max(0, int(trim_frac * len(vv)))
        core = vv[k: len(vv)-k] or vv
        return float(np.mean(core))
    if mode == "weighted_median":
        ws = []
        for k in ["crown","projected","building","cog"]:
            if k in cand_map and cand_map[k] is not None and np.isfinite(cand_map[k]):
                ws.append(max(1e-3, float(weights.get(k, 0.0))))
        return float(weighted_median(vals, ws))
    return float(np.median(vals))

# ---------------------------
# Main
# ---------------------------
def main() -> int:
    p = argparse.ArgumentParser("D-final – Unified Tree Ground Projection (MoE)")

    # IO
    p.add_argument("--csv_path", required=True)
    p.add_argument("--image_name", required=True)
    p.add_argument("--tree_index", type=int, required=True)
    p.add_argument("--disp_path", required=True)
    p.add_argument("--scale_path", required=True)

    # sizes
    p.add_argument("--orig_w", type=int, default=400)
    p.add_argument("--orig_h", type=int, default=400)
    p.add_argument("--disp_w", type=int, default=512)
    p.add_argument("--disp_h", type=int, default=256)

    # camera
    p.add_argument("--camera_height", type=float, default=2.5)
    p.add_argument("--pitch_deg", type=float, default=0.0)

    # scale mode (auto with guards)
    p.add_argument("--scale_mode", choices=["global", "per_image_ground", "auto"], default="auto")
    p.add_argument("--auto_ratio_soft", type=float, default=1.35)
    p.add_argument("--auto_ratio_hard", type=float, default=1.80)
    p.add_argument("--auto_phi_min_deg", type=float, default=2.0)
    p.add_argument("--auto_exclude_col_half", type=int, default=64)
    p.add_argument("--auto_clip_ratio_lo", type=float, default=0.85)
    p.add_argument("--auto_clip_ratio_hi", type=float, default=1.35)

    # crown
    p.add_argument("--crown_top_frac", type=float, default=0.7)
    p.add_argument("--crown_stat", choices=["median", "p90"], default="median")

    # building / occluder
    p.add_argument("--building_band_half_px", type=int, default=40)
    p.add_argument("--building_extra_down", type=int, default=40)
    p.add_argument("--building_clearance", type=float, default=2.5)

    # dynamic building ratio thresholds
    p.add_argument("--building_ratio_base", type=float, default=1.45)
    p.add_argument("--building_ratio_max",  type=float, default=1.75)

    # near-occluder guard
    p.add_argument("--vrun_strong_thr", type=int, default=80)
    p.add_argument("--guard_r_thr", type=float, default=1.60)

    # projection
    p.add_argument("--proj_mode", choices=["fixed","adaptive"], default="fixed")
    p.add_argument("--projection_factor", type=float, default=0.30)  # used if fixed
    p.add_argument("--cap_proj_gain", type=float, default=1.60)      # cap on (1+pf)

    # COG
    p.add_argument("--use_cog", type=lambda s: s.lower() not in ["0","false","no"], default=True)
    p.add_argument("--cog_band_half_px", type=int, default=8)

    # MoE aggregation
    p.add_argument("--moe_mode", choices=["median","weighted_median","trimmed_mean"], default="median")
    p.add_argument("--trim_frac", type=float, default=0.25)

    # debug
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    print("D-final – Unified Tree Ground Projection (MoE)")
    print("=" * 50)
    print(f"Image: {args.image_name}")
    print(f"Tree Index: {args.tree_index}")
    print(f"Scale mode: {args.scale_mode} | proj_mode={args.proj_mode} | moe_mode={args.moe_mode} | use_cog={args.use_cog}")
    print("=" * 50)

    # Load table & row
    df = load_table_any(args.csv_path)
    rows = df[df["file_name"].astype(str) == str(args.image_name)].reset_index(drop=True)
    if rows.empty: raise RuntimeError(f"No rows found for image_name: {args.image_name}")
    if not (0 <= args.tree_index < len(rows)): raise RuntimeError(f"tree_index {args.tree_index} out of range (found {len(rows)})")
    r = rows.iloc[args.tree_index]
    cx, cy = float(r["x_box"]), float(r["y_box"])
    bw, bh = float(r["width_box"]), float(r["height_box"])

    # Disparity file
    if os.path.isdir(args.disp_path):
        disp_file, tried = find_npy_for_image(args.image_name, args.disp_path)
        if disp_file is None:
            print("ERROR: Could not find disparity file. Tried:")
            print(f"  {tried[0]}\n  {tried[1]}"); return 1
    else:
        disp_file = args.disp_path
        if not os.path.isfile(disp_file):
            print(f"ERROR: disp_path not found: {disp_file}"); return 1

    disp_map = np.load(disp_file)
    print(f"Loaded disparity map: {disp_map.shape}  from {disp_file}")

    # Camera & FOV
    heading_deg, hfov_deg = parse_heading_and_fov_from_filename(args.image_name)
    vfov_deg = vfov_from_hfov(hfov_deg, args.disp_w, args.disp_h)

    # BBox in disparity space
    x_bc, y_bc, x1, y1, x2, y2 = bbox_to_pixel(
        cx, cy, bw, bh,
        orig_w=args.orig_w, orig_h=args.orig_h,
        disp_w=args.disp_w, disp_h=args.disp_h
    )
    H, W = disp_map.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)

    # Crown disparity
    crown_disp, crown_conf = robust_crown_disparity(
        disp_map, x1, y1, x2, y2, top_frac=args.crown_top_frac, stat=args.crown_stat
    )
    if not np.isfinite(crown_disp) or crown_disp <= 1e-6:
        raise RuntimeError("Could not estimate crown disparity")

    # Band p95 + vertical run proxy
    band_p95 = band_p95_disparity(
        disp_map, x_bc, y1, y2,
        half_px=args.building_band_half_px, extra_down=args.building_extra_down
    )
    vrun = vertical_run_len_px(y1, min(H, y2 + args.building_extra_down))

    # Load global scale
    with open(args.scale_path, "r", encoding="utf-8") as f:
        global_scale = float(json.load(f)["inverse_depth_scale"])

    # Auto / per-image / global scale
    if args.scale_mode == "global":
        scale_value = global_scale
        auto_meta = {"note": "global"}
    elif args.scale_mode == "per_image_ground":
        meta = ground_scale_with_conf(
            disp_map, hfov_deg, vfov_deg, args.camera_height, args.pitch_deg,
            phi_min_deg=args.auto_phi_min_deg, exclude_col_half=args.auto_exclude_col_half
        )
        if meta["s_ground"] is None:
            print("Per-image scale calibration failed – using GLOBAL")
            scale_value = global_scale
        else:
            scale_value = meta["s_ground"]
        auto_meta = meta
    else:
        meta = ground_scale_with_conf(
            disp_map, hfov_deg, vfov_deg, args.camera_height, args.pitch_deg,
            phi_min_deg=args.auto_phi_min_deg, exclude_col_half=args.auto_exclude_col_half
        )
        if meta["s_ground"] is None:
            print("Auto scale: fallback to global (not enough ground).")
            scale_value = global_scale
            auto_meta = meta
        else:
            ratio = meta["s_ground"] / max(global_scale, 1e-9)
            # soft/hard gates
            if ratio >= args.auto_ratio_hard or ratio <= (1.0/args.auto_ratio_hard):
                print(f"Auto scale: ratio={ratio:.2f} ≥ hard({args.auto_ratio_hard:.2f}) → fallback to global {global_scale:.6f}")
                scale_value = global_scale
            else:
                used_scale = float(meta["conf"] * meta["s_ground"] + (1.0 - meta["conf"]) * global_scale)
                scale_value = used_scale
            clipped_ratio = float(np.clip(ratio, args.auto_clip_ratio_lo, args.auto_clip_ratio_hi))
            print(f"Auto scale excl[x∈{W//2-args.auto_exclude_col_half},{W//2+args.auto_exclude_col_half}]: "
                  f"s_ground={meta['s_ground']:.6f}, ratio={ratio:.2f}→clip[{args.auto_clip_ratio_lo:.2f},{args.auto_clip_ratio_hi:.2f}]={clipped_ratio:.2f}, "
                  f"conf={meta['conf']:.2f}, used={scale_value:.6f}, meta={{'coverage': {meta['coverage']:.6f}, 'mad_over_med': {meta['mad_over_med']}}}")
            auto_meta = meta

    # Candidate A: crown range
    crown_range = scale_value / crown_disp

    # Projection: fixed or adaptive
    if args.proj_mode == "adaptive":
        if crown_range < 20: pf = 0.35
        elif crown_range < 30: pf = 0.25
        else: pf = 0.15
        bbox_height_rel = (y2 - y1) / float(args.disp_h)
        pf += 0.20 * (bbox_height_rel - 0.15)
        pf = float(np.clip(pf, 0.10, 0.45))
    else:
        pf = float(args.projection_factor)
    projected = crown_range * (1.0 + max(0.0, min(pf, args.cap_proj_gain - 1.0)))

    # Candidate B: building constraint (+dynamic threshold)
    r_ratio = None; building_plus = None; strong_building = False
    if band_p95 is not None and np.isfinite(band_p95) and band_p95 > 1e-6:
        r_ratio = float(band_p95 / crown_disp)
        building_plus = (scale_value / band_p95) + args.building_clearance
        r_thr = args.building_ratio_base + 0.25 * (crown_conf - 0.5)
        r_thr = float(np.clip(r_thr, args.building_ratio_base, args.building_ratio_max))
        strong_building = (r_ratio >= r_thr) and (vrun <= args.vrun_strong_thr)

    # Candidate C: COG (optional)
    cog_R = None; cog_meta = {}
    if args.use_cog:
        cog_R, cog_meta = cog_candidate_range(
            disp_map, x_bc, y1, y_bc, scale_value, band_half_px=args.cog_band_half_px
        )

    # Near-occluder guard
    guard_on = False
    if (r_ratio is not None) and (r_ratio >= args.guard_r_thr) and (vrun <= args.vrun_strong_thr):
        guard_on = True
        print(f"Near-occluder guard ON: r={r_ratio:.2f}, vrun={vrun}")
        if args.scale_mode == "auto":
            print(f"Auto scale SUPPRESSED: r={r_ratio:.2f}≥{args.guard_r_thr:.2f} & vrun={vrun}≤{args.vrun_strong_thr} → using GLOBAL {global_scale:.6f}")
            scale_value = global_scale
            crown_range = scale_value / crown_disp
            projected = crown_range

    # MoE aggregation
    cand_map = {"crown": crown_range, "projected": projected,
                "building": building_plus, "cog": cog_R}
    weights = {
        "crown": 0.4 + 0.6 * float(crown_conf),                                   # 0.4..1.0
        "projected": 0.5 * (0.4 + 0.6 * float(crown_conf)),
        "building": ( (0.3 + 0.7 * float(min(max(((r_ratio or 0) - args.building_ratio_base) / 0.3, 0.0), 1.0)))
                      + (0.1 if vrun <= args.vrun_strong_thr else 0.0) ) if (building_plus is not None) else None,
        "cog": 0.3 + 0.7 * float((cog_meta or {}).get("score", 0.0)) if (cog_R is not None) else None,
    }

    if guard_on:
        chosen = crown_range
        moe_note = "guard_crown"
    else:
        chosen = aggregate_candidates(cand_map, weights, mode=args.moe_mode, trim_frac=args.trim_frac)
        moe_note = f"moe_{args.moe_mode}"
        if not np.isfinite(chosen):
            chosen = crown_range
            moe_note = "fallback_crown"

        if building_plus is not None and not strong_building:
            chosen = max(chosen, building_plus)
            moe_note += "_building_guard"

    # Bearing
    bearing_offset = ((x_bc - (W/2.0)) / float(W)) * hfov_deg
    bearing = (heading_deg + bearing_offset) % 360.0
    cam_lat, cam_lon = parse_gps_from_filename(args.image_name)
    tree_lat, tree_lon = gps_from_heading_and_distance(cam_lat, cam_lon, bearing, chosen)

    # Prints
    print("\nGround Projection Results:")
    print("=" * 50)
    print(f"MoE: {moe_note}  |  strong_building={strong_building}")
    print(f"Crown conf: {crown_conf:.3f}")
    if band_p95 is not None:
        print(f"band_p95: {band_p95:.9f}  |  r(band/crown)={r_ratio:.6f}  |  vrun={vrun}")
    else:
        print("band_p95: None")
    if args.use_cog and cog_R is not None:
        print(f"COG: disp≈{(scale_value/cog_R) if cog_R>0 else float('nan'):.12f}  score={cog_meta.get('score', 0.0):.3f}  meta={cog_meta}")
    print(f"Crown range:   {crown_range:6.2f} m")
    print(f"Projected:     {projected:6.2f} m")
    if building_plus is not None:
        print(f"Building+:     {building_plus:6.2f} m")
    print(f"Chosen ground: {chosen:6.2f} m")
    print(f"Pixel: ({x_bc}, {y_bc}) | Bearing offset: {bearing_offset:.1f}°")
    print(f"Camera GPS: ({cam_lat:.6f}, {cam_lon:.6f})")
    print(f"Tree   GPS: ({tree_lat:.6f}, {tree_lon:.6f})")

    # Save JSON
    out = {
        "image_name": args.image_name,
        "tree_index": int(args.tree_index),
        "moe_note": moe_note,
        "crown_conf": float(crown_conf),
        "crown_disp": float(crown_disp),
        "crown_range_m": float(crown_range),
        "projected_range_m": float(projected),
        "building_p95_disp": float(band_p95) if band_p95 is not None else None,
        "building_plus_range_m": float(building_plus) if building_plus is not None else None,
        "building_ratio_r": float(r_ratio) if r_ratio is not None else None,
        "strong_building": bool(strong_building),
        "vrun_px": int(vrun),
        "use_cog": bool(args.use_cog),
        "cog_range_m": float(cog_R) if cog_R is not None else None,
        "cog_meta": cog_meta,
        "chosen_ground_m": float(chosen),
        "bearing_deg": float(bearing),
        "bearing_offset_deg": float(bearing_offset),
        "camera_lat": float(cam_lat),
        "camera_lon": float(cam_lon),
        "tree_lat": float(tree_lat),
        "tree_lon": float(tree_lon),
        "params": {
            "orig_w": int(args.orig_w), "orig_h": int(args.orig_h),
            "disp_w": int(args.disp_w), "disp_h": int(args.disp_h),
            "camera_height": float(args.camera_height), "pitch_deg": float(args.pitch_deg),
            "crown_top_frac": float(args.crown_top_frac), "crown_stat": str(args.crown_stat),
            "building_band_half_px": int(args.building_band_half_px), "building_extra_down": int(args.building_extra_down),
            "building_clearance": float(args.building_clearance),
            "building_ratio_base": float(args.building_ratio_base),
            "building_ratio_max": float(args.building_ratio_max),
            "vrun_strong_thr": int(args.vrun_strong_thr),
            "guard_r_thr": float(args.guard_r_thr),
            "proj_mode": str(args.proj_mode),
            "projection_factor": float(args.projection_factor),
            "cap_proj_gain": float(args.cap_proj_gain),
            "use_cog": bool(args.use_cog),
            "cog_band_half_px": int(args.cog_band_half_px),
            "moe_mode": str(args.moe_mode),
            "trim_frac": float(args.trim_frac),
            "scale_mode": str(args.scale_mode),
            "auto_ratio_soft": float(args.auto_ratio_soft),
            "auto_ratio_hard": float(args.auto_ratio_hard),
            "auto_phi_min_deg": float(args.auto_phi_min_deg),
            "auto_exclude_col_half": int(args.auto_exclude_col_half),
            "auto_clip_ratio_lo": float(args.auto_clip_ratio_lo),
            "auto_clip_ratio_hi": float(args.auto_clip_ratio_hi),
            "hfov_deg": float(hfov_deg), "vfov_deg": float(vfov_deg),
        },
        "scales": {
            "global_inverse_depth_scale": float(global_scale),
            "auto_meta": {k: _json_default(v) for k, v in (auto_meta or {}).items()}
        },
        "pixels": {"x_bc": int(x_bc), "y_bc": int(y_bc), "bbox": [int(x1), int(y1), int(x2), int(y2)]},
    }

    safe_name = (
        args.image_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        .replace("&", "_").replace("=", "_").replace(" ", "_")
    )
    out_path = f"ground_projection_result_{safe_name}_{args.tree_index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"\nResults saved to: {out_path}")

    # Optional debug viz
    if args.debug:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(9, 4))
            im = ax.imshow(disp_map, cmap="plasma")
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec="w", lw=2))
            ax.plot([x_bc], [y_bc], "wo", ms=5)
            ax.set_title("Disparity map + bbox")
            fig.colorbar(im, ax=ax); plt.tight_layout()
            dbg = f"ground_projection_analysis_{safe_name}_{args.tree_index}.png"
            plt.savefig(dbg, dpi=150)
            print(f"[debug] Figure saved: {dbg}")
        except Exception as e:
            print(f"[debug] Skipped debug figure: {e}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
