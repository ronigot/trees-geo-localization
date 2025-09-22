import os
import argparse
import numpy as np
import pandas as pd
from src.baseline.pipeline_core import (
    compute_disparity_stats_for_rows,
    estimate_distances_df,
    estimate_tree_coords_df,
    load_scale,
)

DISP_DIR_DEFAULT = "data/disparities"
SCALE_FILE_DEFAULT = "scale/depth_scale.json"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ensure_file_exists(path: str, desc: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {desc}: {path}")

def load_bboxes_table(path: str, csv_sep: str = ",", csv_encoding: str = "utf-8"):
    """
    Load the annotations table as DataFrame from either .xlsx/.xls or .csv.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path, sep=csv_sep, encoding=csv_encoding)
    else:
        raise ValueError(f"Unsupported table format: {ext} (expected .xlsx/.xls/.csv)")

def sanitize_for_filename(name: str) -> str:
    """
    Create a file-system friendly stem from a potentially long/complex image name.
    """
    bad = '<>:"/\\|?*'
    stem = name
    for ch in bad:
        stem = stem.replace(ch, "_")
    # Also compress spaces
    stem = "_".join(stem.split())
    return stem

def find_npy_for_image(image_name: str, disp_dir: str):
    """
    Try both naming conventions:
      - <base>.npy       (pipeline-friendly)
      - <base>_disp.npy  (monodepth_simple.py default)
    Returns:
      (found_path or None, (candidate_base, candidate_disp))
    """
    base, _ = os.path.splitext(str(image_name))
    p_base = os.path.join(disp_dir, base + ".npy")
    p_disp = os.path.join(disp_dir, base + "_disp.npy")

    if os.path.isfile(p_base):
        return p_base, (p_base, p_disp)
    if os.path.isfile(p_disp):
        return p_disp, (p_base, p_disp)
    return None, (p_base, p_disp)



def run_single_image(excel_path, scale_file,
                     image_name, npy_path,
                     orig_w, orig_h, disp_w, disp_h,
                     win_x, win_y,
                     output_csv,
                     save_intermediate=False,
                     intermediate_dir="results/intermediate",
                     quiet=False):
    """
    Single-image pipeline: filter rows for image_name, use its .npy, produce CSV with tree coords.
    Optionally save intermediate distances CSV for this image only.
    """
    df_all = load_bboxes_table(excel_path)
    df_all["file_name"] = df_all["file_name"].astype(str)
    image_name = str(image_name) # make sure types match
    cols_needed = {"file_name", "x_box", "y_box", "width_box", "height_box"}
    if not cols_needed.issubset(df_all.columns):
        raise ValueError(f"Excel must contain columns: {cols_needed}")

    rows = df_all[df_all["file_name"] == image_name].copy()
    if rows.empty:
        raise ValueError(f"No rows found in Excel for file_name: {image_name}")

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Disparity .npy not found: {npy_path}")
    disp_map = np.load(npy_path)

    # Step 1: disparity stats for these rows
    dstats = compute_disparity_stats_for_rows(
        rows, disp_map,
        orig_size=(orig_w, orig_h),
        disp_size=(disp_w, disp_h),
        win_x=win_x, win_y=win_y
    )

    # Step 2: distances
    scale_value = load_scale(scale_file)
    dists = estimate_distances_df(dstats, scale_value)
    if dists["estimated_distance_m"].isna().any() and not quiet:
        n_nan = dists["estimated_distance_m"].isna().sum()
        print(f"[warn] {n_nan} rows have NaN distances (likely empty/invalid disparity window).")

    # Optional: save intermediate distances for this image
    if save_intermediate:
        ensure_dir(intermediate_dir)
        stem = sanitize_for_filename(image_name)
        inter_path = os.path.join(intermediate_dir, f"{stem}_distances.csv")
        dists.to_csv(inter_path, index=False)
        if not quiet:
            print(f"[single] Saved intermediate distances: {inter_path}")

    # Step 3: GPS
    trees = estimate_tree_coords_df(dists, disp_width=disp_w)

    if output_csv:
        out_dir = os.path.dirname(output_csv) or "."
        ensure_dir(out_dir)
        trees.to_csv(output_csv, index=False)
        if not quiet:
            print(f"[single] Saved final: {output_csv}")

    return trees

def run_batch(excel_path, scale_file,
              disp_dir,
              orig_w, orig_h, disp_w, disp_h,
              win_x, win_y,
              output_csv,
              save_intermediate=False,
              intermediate_dir="results/intermediate",
              quiet=False):
    """
    Batch pipeline: iterate unique file_name in Excel, find matching .npy in disp_dir,
    compute all trees, and aggregate to one CSV.
    Optionally save ONE aggregated distances.csv containing all rows.
    """
    df_all = load_bboxes_table(excel_path)
    df_all["file_name"] = df_all["file_name"].astype(str)
    cols_needed = {"file_name", "x_box", "y_box", "width_box", "height_box"}
    if not cols_needed.issubset(df_all.columns):
        raise ValueError(f"Excel must contain columns: {cols_needed}")

    scale_value = load_scale(scale_file)
    all_results = []
    all_distances = []  # collect optional intermediate distances

    for image_name, rows in df_all.groupby("file_name"):
        image_name_str = str(image_name)
        npy_path, candidates = find_npy_for_image(image_name_str, disp_dir)
        if npy_path is None:
            if not quiet:
                print(f"[batch][WARN] Missing .npy for {image_name_str}. "
                      f"Tried: {candidates[0]} and {candidates[1]}. Skipping.")
            continue
        disp_map = np.load(npy_path)

        dstats = compute_disparity_stats_for_rows(
            rows, disp_map,
            orig_size=(orig_w, orig_h),
            disp_size=(disp_w, disp_h),
            win_x=win_x, win_y=win_y
        )
        dists = estimate_distances_df(dstats, scale_value)
        if dists["estimated_distance_m"].isna().any() and not quiet:
            n_nan = dists["estimated_distance_m"].isna().sum()
            print(f"[warn] {n_nan} rows have NaN distances (likely empty/invalid disparity window).")

        trees = estimate_tree_coords_df(dists, disp_width=disp_w)

        all_results.append(trees)
        if save_intermediate:
            all_distances.append(dists)

        if not quiet:
            print(f"[batch] processed: {image_name_str} ({len(rows)} trees)")

    if not all_results:
        raise RuntimeError("No results produced in batch mode.")

    final_df = pd.concat(all_results, ignore_index=True)

    out_dir = os.path.dirname(output_csv) or "."
    ensure_dir(out_dir)
    final_df.to_csv(output_csv, index=False)
    if not quiet:
        print(f"[batch] Saved final: {output_csv}")

    # Save one aggregated distances.csv (optional)
    if save_intermediate and all_distances:
        ensure_dir(intermediate_dir)
        distances_path = os.path.join(intermediate_dir, "distances.csv")
        pd.concat(all_distances, ignore_index=True).to_csv(distances_path, index=False)
        if not quiet:
            print(f"[batch] Saved intermediate distances: {distances_path}")

    return final_df

def main():
    ap = argparse.ArgumentParser(
        description="Tree mapping pipeline (single image or batch)."
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # ---- single image mode ----
    sp1 = sub.add_parser("single", help="Run pipeline for one image (one .npy)")
    sp1.add_argument("--excel", required=True, help="Path to Excel with bbox rows")
    sp1.add_argument("--scale_file", default=SCALE_FILE_DEFAULT,
                     help=f"Path to calibration scale JSON (default: {SCALE_FILE_DEFAULT})")
    sp1.add_argument("--image_name", required=True, help="file_name value as in Excel")
    sp1.add_argument("--npy_path", default=None,
                     help="Explicit path to disparity .npy. If omitted, will auto-locate in data/disparities")
    sp1.add_argument("--output_csv", required=True, help="Output CSV path for this image")
    sp1.add_argument("--orig_w", type=int, default=400)
    sp1.add_argument("--orig_h", type=int, default=400)
    sp1.add_argument("--disp_w", type=int, default=512)
    sp1.add_argument("--disp_h", type=int, default=256)
    sp1.add_argument("--win_x", type=int, default=2)
    sp1.add_argument("--win_y", type=int, default=4)
    sp1.add_argument("--save_intermediate", action="store_true",
                     help="Save intermediate distances CSV for this image")
    sp1.add_argument("--intermediate_dir", default="results/intermediate",
                     help="Directory for intermediate CSVs (if saving)")
    sp1.add_argument("--quiet", action="store_true")

    # ---- batch mode ----
    sp2 = sub.add_parser("batch", help="Run pipeline for all images listed in Excel")
    sp2.add_argument("--excel", required=True, help="Path to Excel with bbox rows")
    sp2.add_argument("--scale_file", default=SCALE_FILE_DEFAULT,
                     help=f"Path to calibration scale JSON (default: {SCALE_FILE_DEFAULT})")
    sp2.add_argument("--disp_dir", required=False, default=DISP_DIR_DEFAULT,
                     help=f"Directory with disparity .npy files (default: {DISP_DIR_DEFAULT})")
    sp2.add_argument("--output_csv", default="results/tree_locations.csv",
                     help="Final aggregated CSV (default: results/tree_locations.csv)")
    sp2.add_argument("--orig_w", type=int, default=400)
    sp2.add_argument("--orig_h", type=int, default=400)
    sp2.add_argument("--disp_w", type=int, default=512)
    sp2.add_argument("--disp_h", type=int, default=256)
    sp2.add_argument("--win_x", type=int, default=2)
    sp2.add_argument("--win_y", type=int, default=4)
    sp2.add_argument("--save_intermediate", action="store_true",
                     help="Save one aggregated distances.csv for all images")
    sp2.add_argument("--intermediate_dir", default="results/intermediate",
                     help="Directory for intermediate CSVs (if saving)")
    sp2.add_argument("--quiet", action="store_true")

    args = ap.parse_args()

    if args.mode == "single":
        ensure_file_exists(args.scale_file, "scale file")
        npy_path = args.npy_path
        if npy_path:
            ensure_file_exists(npy_path, "--npy_path")
        else:
            npy_found, candidates = find_npy_for_image(args.image_name, DISP_DIR_DEFAULT)
            if npy_found is None:
                raise FileNotFoundError(
                    f"Missing .npy for {args.image_name} under {DISP_DIR_DEFAULT}. "
                    f"Tried: {candidates[0]} and {candidates[1]}"
                )
            npy_path = npy_found
            if not args.quiet:
                print(f"[single] Using disparity: {npy_path}")

        run_single_image(
            excel_path=args.excel,
            scale_file=args.scale_file,
            image_name=args.image_name,
            npy_path=npy_path,
            orig_w=args.orig_w, orig_h=args.orig_h,
            disp_w=args.disp_w, disp_h=args.disp_h,
            win_x=args.win_x, win_y=args.win_y,
            output_csv=args.output_csv,
            save_intermediate=args.save_intermediate,
            intermediate_dir=args.intermediate_dir,
            quiet=args.quiet
        )

    else:
        ensure_file_exists(args.scale_file, "scale file")
        run_batch(
            excel_path=args.excel,
            scale_file=args.scale_file,
            disp_dir=args.disp_dir,
            orig_w=args.orig_w, orig_h=args.orig_h,
            disp_w=args.disp_w, disp_h=args.disp_h,
            win_x=args.win_x, win_y=args.win_y,
            output_csv=args.output_csv,
            save_intermediate=args.save_intermediate,
            intermediate_dir=args.intermediate_dir,
            quiet=args.quiet
        )

if __name__ == "__main__":
    main()
