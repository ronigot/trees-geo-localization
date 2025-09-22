import os
import argparse
from typing import Tuple, List

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_table(path: str, csv_sep: str = ",", csv_encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load annotations table from .csv or .xlsx/.xls into a DataFrame.
    Expected columns: file_name, x_box, y_box, width_box, height_box
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, sep=csv_sep, encoding=csv_encoding)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported table format: {ext} (expected .csv/.xlsx/.xls)")
    required = {"file_name", "x_box", "y_box", "width_box", "height_box"}
    if not required.issubset(df.columns):
        raise ValueError(f"Table must contain columns: {required}")
    df["file_name"] = df["file_name"].astype(str)
    return df


def norm_bbox_to_xyxy(
    x_c_norm: float, y_c_norm: float, w_norm: float, h_norm: float,
    W: int, H: int
) -> Tuple[int, int, int, int]:
    """
    Convert normalized center-based bbox (YOLO style) to pixel corner bbox (x1,y1,x2,y2).
    Clamps to image bounds and guarantees at least 1px size.
    """
    x_c = x_c_norm * W
    y_c = y_c_norm * H
    bw = w_norm * W
    bh = h_norm * H

    x1 = int(round(x_c - bw / 2.0))
    y1 = int(round(y_c - bh / 2.0))
    x2 = int(round(x_c + bw / 2.0))
    y2 = int(round(y_c + bh / 2.0))

    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))

    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def _pick_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Try to load a TrueType font with given size. Fallback to PIL default font.
    """
    # Common fonts to try (Windows, Linux)
    candidates: List[str] = [
        "arial.ttf",
        "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_bboxes_on_image(
    img: Image.Image,
    rows_df: pd.DataFrame,
    thickness: int = None,
    font_size: int = None,
    mark_bottom_center: bool = False
) -> Image.Image:
    """
    Draw all bboxes from rows_df on a copy of img.
    Numbering is 1..N in the order rows appear (group order preserved).
    """
    W, H = img.size
    t = thickness if thickness is not None else max(2, int(round(0.002 * (W + H))))
    fs = font_size if font_size is not None else max(12, int(round(0.02 * (W + H))))
    font = _pick_font(fs)

    # Distinct colors (RGB)
    palette = [
        (36, 255, 12),   # green
        (255, 165, 0),   # orange
        (0, 128, 255),   # blue-ish
        (255, 0, 255),   # magenta
        (255, 255, 0),   # yellow
        (255, 0, 0),     # red
        (0, 255, 255),   # cyan
    ]

    out = img.copy()
    draw = ImageDraw.Draw(out)

    for idx, row in enumerate(rows_df.itertuples(index=False), start=1):
        try:
            x1, y1, x2, y2 = norm_bbox_to_xyxy(
                float(row.x_box), float(row.y_box),
                float(row.width_box), float(row.height_box),
                W, H
            )
        except Exception:
            continue

        color = palette[(idx - 1) % len(palette)]

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=t)

        # label background
        label = str(idx)
        tw, th = draw.textlength(label, font=font), font.size
        pad = max(2, t // 2)
        bg_x2 = min(x1 + int(tw) + 2 * pad, W - 1)
        bg_y2 = min(y1 + th + 2 * pad, H - 1)
        draw.rectangle([x1, y1, bg_x2, bg_y2], fill=color)
        # label text (black)
        draw.text((x1 + pad, y1 + pad), label, fill=(0, 0, 0), font=font)

        if mark_bottom_center:
            # bottom-center of the bbox
            x_bc = (x1 + x2) // 2
            y_bc = y2
            L = max(3, t * 2)
            draw.line([(x_bc - L, y_bc), (x_bc + L, y_bc)], fill=color, width=t)
            draw.line([(x_bc, y_bc - L), (x_bc, y_bc + L)], fill=color, width=t)

    return out


def sanitize_stem(name: str) -> str:
    """Make a filesystem-friendly stem from name."""
    bad = '<>:"/\\|?*'
    stem = name
    for ch in bad:
        stem = stem.replace(ch, "_")
    stem = "_".join(stem.split())
    return stem


def main():
    ap = argparse.ArgumentParser(
        description="Visualize normalized YOLO-style bboxes on images using Pillow."
    )
    ap.add_argument("--table", required=True,
                    help="Path to CSV/XLSX/XLS with columns: file_name,x_box,y_box,width_box,height_box")
    ap.add_argument("--images_dir", required=True,
                    help="Directory containing images referenced by 'file_name'")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to save visualizations")
    ap.add_argument("--csv_sep", default=",", help="CSV separator (default: ',')")
    ap.add_argument("--csv_encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    ap.add_argument("--ext_out", default=None,
                    help="Optional output extension override (e.g. jpg/png). If omitted, keeps original extension.")
    ap.add_argument("--thickness", type=int, default=None, help="Line thickness override")
    ap.add_argument("--font_size", type=int, default=None, help="Font size override")
    ap.add_argument("--mark_bottom_center", action="store_true",
                    help="Also mark the bottom-center point used for disparity sampling")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = ap.parse_args()

    df = load_table(args.table, csv_sep=args.csv_sep, csv_encoding=args.csv_encoding)
    groups = df.groupby("file_name", sort=False)

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok, n_missing = 0, 0

    for file_name, rows in groups:
        img_path = os.path.join(args.images_dir, file_name)
        if not os.path.isfile(img_path):
            n_missing += 1
            if not args.quiet:
                print(f"[WARN] Missing image for '{file_name}' → {img_path}. Skipping.")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            n_missing += 1
            if not args.quiet:
                print(f"[WARN] Failed to open image: {img_path}. Skipping.")
            continue

        out_img = draw_bboxes_on_image(
            img,
            rows,
            thickness=args.thickness,
            font_size=args.font_size,
            mark_bottom_center=args.mark_bottom_center
        )

        base, orig_ext = os.path.splitext(os.path.basename(file_name))
        if args.ext_out:
            ext = "." + args.ext_out.lstrip(".")
        else:
            ext = orig_ext if orig_ext else ".jpg"

        out_path = os.path.join(args.out_dir, f"{sanitize_stem(base)}_boxed{ext}")
        try:
            out_img.save(out_path)
            n_ok += 1
            if not args.quiet:
                print(f"[OK] Saved: {out_path}  ({len(rows)} boxes)")
        except Exception as e:
            if not args.quiet:
                print(f"[WARN] Failed to save {out_path}: {e}. Trying JPG fallback.")
            out_path = os.path.join(args.out_dir, f"{sanitize_stem(base)}_boxed.jpg")
            out_img.save(out_path, format="JPEG")
            n_ok += 1
            if not args.quiet:
                print(f"[OK] Saved (fallback): {out_path}  ({len(rows)} boxes)")

    if not args.quiet:
        print(f"Done. Wrote {n_ok} images. Missing/failed: {n_missing}")


if __name__ == "__main__":
    main()
