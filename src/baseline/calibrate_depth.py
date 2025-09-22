import json
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def calibrate_inverse_depth_scale(camera_coords, tree_coords, disp_avg, output_path="depth_scale.json"):
    """
    Calibrates scale factor for inverse-depth disparity (monodepth).
    Computes: distance = scale / disp_avg  → scale = distance * disp_avg
    Saves the scale to a JSON file.
    """
    lat_cam, lon_cam = camera_coords
    lat_tree, lon_tree = tree_coords

    distance_meters = haversine_distance(lat_cam, lon_cam, lat_tree, lon_tree)

    if disp_avg == 0:
        raise ValueError("disp_avg cannot be zero")

    scale = distance_meters * disp_avg

    scale_data = {
        "inverse_depth_scale": scale,
        "calibration": {
            "camera_lat": lat_cam,
            "camera_lon": lon_cam,
            "tree_lat": lat_tree,
            "tree_lon": lon_tree,
            "distance_meters": distance_meters,
            "disp_avg": disp_avg,
            "depth_model": "monodepth (inverse depth)"
        }
    }

    with open(output_path, "w") as f:
        json.dump(scale_data, f, indent=4)

    print(f"Calibration complete. Saved to {output_path}")
    print(f"Scale (inverse depth): {scale:.3f} → distance = scale / disp_avg")


if __name__ == "__main__":
    camera_coords = (32.05332566340407, 34.81022952939507)
    tree_coords = (32.0532986, 34.8101803)
    disp_avg = 0.034628998

    calibrate_inverse_depth_scale(camera_coords, tree_coords, disp_avg)
