# Copyright UCL Business plc 2017. Patent Pending.
# Licensed for non-commercial use under UCLB ACP-A.
# This helper script wraps the original monodepth test_simple to run over a folder.

from __future__ import absolute_import, division, print_function
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # keep TF messages visible if needed

import argparse
import glob
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim  # noqa: F401 (import side-effects in original code)
import matplotlib.pyplot as plt
from PIL import Image

from monodepth_model import MonodepthModel, monodepth_parameters

def post_process_disparity(disp):
    """
    Same as in monodepth_simple.py: left/right flip, average, feathered blending.
    disp: shape [2, H, W]
    """
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def load_and_prepare_image(path, input_width, input_height):
    """
    Read RGB image, remember original size, resize to network size (Lanczos), normalize to [0,1].
    Returns:
        input_pair: np.array of shape [2, H, W, 3]  (orig + flipped)
        orig_size: (H_orig, W_orig)
    """
    im = Image.open(path).convert("RGB")
    orig_w, orig_h = im.size
    im_resized = im.resize((input_width, input_height), resample=Image.LANCZOS)
    arr = np.asarray(im_resized, dtype=np.float32) / 255.0
    pair = np.stack([arr, np.fliplr(arr)], axis=0)
    return pair, (orig_h, orig_w)

def list_images(images_dir, patterns):
    """
    Collect and sort image paths by given glob patterns.
    """
    all_paths = []
    for pat in patterns:
        all_paths.extend(glob.glob(os.path.join(images_dir, pat)))
    all_paths = sorted(all_paths)
    return all_paths

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Run MonoDepth on a folder of images (batch).")
    ap.add_argument("--images_dir", required=True, help="Directory with input images")
    ap.add_argument("--checkpoint_path", required=True, help="Path to checkpoint to load (no extension)")
    ap.add_argument("--encoder", default="vgg", choices=["vgg", "resnet50"], help="Encoder type")
    ap.add_argument("--input_height", type=int, default=256, help="Network input height")
    ap.add_argument("--input_width",  type=int, default=512, help="Network input width")

    ap.add_argument("--patterns", default="*.jpg,*.jpeg,*.png",
                    help="Comma-separated glob patterns (default: *.jpg,*.jpeg,*.png)")
    ap.add_argument("--output_dir", default=None,
                    help="Directory to write outputs. If omitted, saves next to each image.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing outputs if present")

    ap.add_argument("--npy_name_mode", choices=["base", "disp_suffix", "both"], default="base",
                    help="How to name .npy: base.npy (pipeline-friendly), base_disp.npy, or both")
    ap.add_argument("--save_png", action="store_true",
                    help="Also save a pseudo-color disparity PNG (like monodepth_simple)")

    args = ap.parse_args()

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    image_paths = list_images(args.images_dir, patterns)
    if not image_paths:
        print("[ERR] No images found in:", args.images_dir)
        return

    # Build graph once
    left = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False,
    )
    model = MonodepthModel(params, "test", left, None)

    # Session & restore
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # checkpoint path without extension, same as original
    restore_path = args.checkpoint_path.split(".")[0]
    print(f"[INFO] Restoring checkpoint: {restore_path}")
    saver.restore(sess, restore_path)

    # Output dir
    if args.output_dir is not None:
        ensure_dir(args.output_dir)

    t0 = time.time()
    print(f"[INFO] Found {len(image_paths)} images. Starting inference...")

    for idx, img_path in enumerate(image_paths, 1):
        try:
            input_pair, (orig_h, orig_w) = load_and_prepare_image(
                img_path, args.input_width, args.input_height
            )

            # Run inference
            disp = sess.run(model.disp_left_est[0], feed_dict={left: input_pair})
            disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

            # Resolve output base and directory
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_dir = args.output_dir if args.output_dir else os.path.dirname(img_path)
            ensure_dir(out_dir)

            # npy outputs
            npy_base = os.path.join(out_dir, f"{base}.npy")
            npy_disp = os.path.join(out_dir, f"{base}_disp.npy")

            if not args.overwrite and (
                (args.npy_name_mode in ["base", "both"] and os.path.isfile(npy_base)) or
                (args.npy_name_mode in ["disp_suffix", "both"] and os.path.isfile(npy_disp))
            ):
                print(f"[SKIP] {base}: npy exists. Use --overwrite to replace.")
            else:
                if args.npy_name_mode in ["base", "both"]:
                    np.save(npy_base, disp_pp)
                if args.npy_name_mode in ["disp_suffix", "both"]:
                    np.save(npy_disp, disp_pp)

            # Optional PNG (for visualization only)
            if args.save_png:
                # resize to original size for nicer visualization (like monodepth_simple)
                disp_vis = np.array(
                    Image.fromarray(disp_pp.squeeze()).resize((orig_w, orig_h), resample=Image.BILINEAR)
                )
                png_path = os.path.join(out_dir, f"{base}_disp.png")
                plt.imsave(png_path, disp_vis, cmap="plasma")

            if idx % 10 == 0 or idx == len(image_paths):
                print(f"[{idx}/{len(image_paths)}] done: {base}")

        except Exception as e:
            print(f"[ERR] Failed on {img_path}: {e}")

    print(f"[INFO] Finished {len(image_paths)} images in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
