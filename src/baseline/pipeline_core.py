import os
import re
import math
import numpy as np
import pandas as pd

# ------------------------------
# Geometry helpers
# ------------------------------

def bbox_to_pixel(cx, cy, bw, bh,
                  orig_width=400, orig_height=400,
                  disp_width=512, disp_height=256):
    """
    Map normalized YOLO bbox (cx, cy, bw, bh) in original image (orig_w x orig_h)
    to bottom-center and corners in disparity map (disp_w x disp_h).
    Returns: x_bc, y_bc, x1, y1, x2, y2  (integers)
    """
    x0 = cx * orig_width
    y0 = cy * orig_height
    x_bc0 = x0
    y_bc0 = y0 + (bh * orig_height) / 2.0

    sx = disp_width / float(orig_width)
    sy = disp_height / float(orig_height)

    x_bc = int(round(x_bc0 * sx))
    y_bc = int(round(y_bc0 * sy))

    w_disp = bw * disp_width
    h_disp = bh * disp_height
    x1 = int(round(x_bc - w_disp / 2.0))
    y1 = int(round(y_bc - h_disp / 2.0))
    x2 = int(round(x_bc + w_disp / 2.0))
    y2 = int(round(y_bc + h_disp / 2.0))

    return x_bc, y_bc, x1, y1, x2, y2


def get_disparity_window(disp_map, x_bd, y_bd,
                         x1, y1, x2, y2,
                         win_x=2, win_y=4):
    """
    Extract an asymmetric window around (x_bd, y_bd) in the disparity map:
    horizontally: [x_bd - win_x, x_bd + win_x]
    vertically:   only upwards by win_y (no downward), and not above y1.
    Returns mean disparity in the patch (nanmean).
    """
    H, W = disp_map.shape[:2]

    x_min = max(x_bd - win_x, 0)
    x_max = min(x_bd + win_x + 1, W)

    y_min = max(y_bd - win_y, 0 if y1 is None else y1)
    y_max = min(y_bd + 1, H)

    if y_min >= y_max or x_min >= x_max:
        return float("nan")

    patch = disp_map[y_min:y_max, x_min:x_max]
    return float(np.nanmean(patch))


# ------------------------------
# Parsing helpers
# ------------------------------

def parse_gps_from_filename(filename):
    m = re.search(r"location=([-0-9\.]+),([-0-9\.]+)", filename)
    if not m:
        raise ValueError(f"Could not parse GPS from filename: {filename}")
    return float(m.group(1)), float(m.group(2))

def parse_heading_and_fov_from_filename(filename):
    mh = re.search(r"heading=([-0-9\.]+)", filename)
    mf = re.search(r"fov=([-0-9\.]+)", filename)
    if not (mh and mf):
        raise ValueError(f"Could not parse heading/fov from filename: {filename}")
    return float(mh.group(1)), float(mf.group(1))


# ------------------------------
# Geodesy
# ------------------------------

def gps_from_heading_and_distance(lat, lon, bearing_deg, distance_m):
    R = 6371000.0
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    dR = distance_m / R

    phi2 = math.asin(
        math.sin(phi1) * math.cos(dR) +
        math.cos(phi1) * math.sin(dR) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(dR) * math.cos(phi1),
        math.cos(dR) - math.sin(phi1) * math.sin(phi2)
    )
    return math.degrees(phi2), math.degrees(lam2)


# ------------------------------
# Scale loader
# ------------------------------

def load_scale(scale_path):
    import json
    with open(scale_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["inverse_depth_scale"]


# ------------------------------
# Core pipeline steps (in-memory)
# ------------------------------

def compute_disparity_stats_for_rows(rows_df, disp_map,
                                     orig_size=(400, 400),
                                     disp_size=(512, 256),
                                     win_x=2, win_y=4):
    """
    rows_df must have columns: file_name, x_box, y_box, width_box, height_box
    Returns a DataFrame with: file_name, x_disp, y_disp, disp_avg
    """
    out = []
    for _, r in rows_df.iterrows():
        fname = r["file_name"]
        cx, cy = float(r["x_box"]), float(r["y_box"])
        bw, bh = float(r["width_box"]), float(r["height_box"])

        x_bd, y_bd, x1, y1, x2, y2 = bbox_to_pixel(
            cx, cy, bw, bh,
            orig_width=orig_size[0], orig_height=orig_size[1],
            disp_width=disp_size[0], disp_height=disp_size[1]
        )
        disp_avg = get_disparity_window(
            disp_map, x_bd, y_bd, x1, y1, x2, y2,
            win_x=win_x, win_y=win_y
        )
        out.append({
            "file_name": fname,
            "x_disp": x_bd,
            "y_disp": y_bd,
            "disp_avg": disp_avg
        })
    return pd.DataFrame(out)


def estimate_distances_df(df, scale_value):
    """
    Adds estimated_distance_m = scale_value / disp_avg
    """
    out = df.copy()
    if "disp_avg" not in out.columns:
        raise ValueError("Missing 'disp_avg' column")
    out["estimated_distance_m"] = scale_value / out["disp_avg"]
    return out


def estimate_tree_coords_df(df, disp_width):
    """
    Input df must have: file_name, x_disp, estimated_distance_m
    Returns df with tree_lat, tree_lon appended.
    """
    required = {"file_name", "x_disp", "estimated_distance_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"df must contain columns: {required}")

    lats, lons = [], []
    for _, r in df.iterrows():
        fname = r["file_name"]
        x_disp = float(r["x_disp"])
        dist_m = float(r["estimated_distance_m"])

        cam_lat, cam_lon = parse_gps_from_filename(fname)
        heading, fov = parse_heading_and_fov_from_filename(fname)

        offset_px = x_disp - (disp_width / 2.0)
        angle_offset = (offset_px / disp_width) * fov
        bearing = (heading + angle_offset) % 360.0

        lat_t, lon_t = gps_from_heading_and_distance(
            cam_lat, cam_lon, bearing, dist_m
        )
        lats.append(lat_t)
        lons.append(lon_t)

    out = df.copy()
    out["tree_lat"] = lats
    out["tree_lon"] = lons
    return out
