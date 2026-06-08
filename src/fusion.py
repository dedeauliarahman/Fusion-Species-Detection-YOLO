import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT        = Path(__file__).resolve().parent.parent
DEFAULT_RGB      = REPO_ROOT / "data" / "rgb"
DEFAULT_THERMAL  = REPO_ROOT / "data" / "thermal"
DEFAULT_OUTPUT   = REPO_ROOT / "data" / "fusion"


DEFAULT_SCALE    = 136
DEFAULT_OFFSET_X = 520
DEFAULT_OFFSET_Y = 205
DEFAULT_OPACITY  = 0.35
DEFAULT_COLORMAP = "inferno"
OUTPUT_FORMAT    = "jpg"


def apply_colormap(thermal_raw: np.ndarray, colormap: str):

    try:
        import matplotlib.cm as mcm
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    t_min, t_max = thermal_raw.min(), thermal_raw.max()
    if t_max == t_min:
        t_norm = np.zeros_like(thermal_raw, dtype=np.float32)
    else:
        t_norm = (thermal_raw - t_min) / (t_max - t_min)

    try:
        cmap = mcm.colormaps[colormap]      # Matplotlib >= 3.7
    except AttributeError:
        cmap = mcm.get_cmap(colormap)       # Matplotlib < 3.7

    colored = (cmap(t_norm) * 255).astype(np.uint8)
    return Image.fromarray(colored, "RGBA")


def compute_thermal_box(thermal_w, thermal_h, scale, offset_x, offset_y):

    new_w = int(thermal_w * scale / 100)
    new_h = int(thermal_h * scale / 100)
    return offset_x, offset_y, new_w, new_h


def validate_crop(crop_left, crop_upper, crop_right, crop_lower):
    if crop_right <= crop_left or crop_lower <= crop_upper:
        raise ValueError(
            f"Crop area is empty or invalid: "
            f"({crop_left},{crop_upper}) → ({crop_right},{crop_lower}). "
            f"Check --offset-x, --offset-y, and --scale values."
        )

def fuse_image(rgb_path, thermal_path, output_path,
               scale, offset_x, offset_y, opacity, colormap):

    from PIL import Image
    import tifffile

    rgb_img      = Image.open(rgb_path).convert("RGBA")
    thermal_raw  = tifffile.imread(thermal_path).astype(np.float32)
    thermal_rgba = apply_colormap(thermal_raw, colormap)

    rgb_w, rgb_h = rgb_img.size
    orig_tw, orig_th = thermal_rgba.size

    left, upper, new_w, new_h = compute_thermal_box(
        orig_tw, orig_th, scale, offset_x, offset_y
    )

    crop_left  = max(0, left)
    crop_upper = max(0, upper)
    crop_right  = min(rgb_w, left + new_w)
    crop_lower  = min(rgb_h, upper + new_h)
    validate_crop(crop_left, crop_upper, crop_right, crop_lower)

    thermal_scaled = thermal_rgba.resize((new_w, new_h), Image.LANCZOS)

    r, g, b, a = thermal_scaled.split()
    a_overlay   = a.point(lambda x: int(x * opacity))
    thermal_overlay = thermal_scaled.copy()
    thermal_overlay.putalpha(a_overlay)

    canvas = rgb_img.copy()
    canvas.paste(thermal_overlay, (left, upper), thermal_overlay)

    fused = canvas.crop((crop_left, crop_upper, crop_right, crop_lower))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fused.convert("RGB").save(output_path, quality=95)


def find_pairs(rgb_root, thermal_root):

    rgb_files = sorted(set(
        glob.glob(os.path.join(glob.escape(rgb_root), "**", "*.JPG"), recursive=True)
        + glob.glob(os.path.join(glob.escape(rgb_root), "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(glob.escape(rgb_root), "*.JPG"))
        + glob.glob(os.path.join(glob.escape(rgb_root), "*.jpg"))
    ))

    pairs, missing = [], []

    for rgb_path in rgb_files:
        rel_path     = os.path.relpath(rgb_path, rgb_root)
        base         = os.path.splitext(rel_path)[0]
        thermal_rel  = base + ".tiff"
        thermal_path = os.path.join(thermal_root, thermal_rel)

        if os.path.exists(thermal_path):
            pairs.append((rgb_path, thermal_path, rel_path))
        else:
            missing.append((rgb_path, thermal_path))

    return pairs, missing


def build_output_path(output_root, rel_path):

    rel_dir   = os.path.dirname(rel_path)
    base_name = os.path.splitext(os.path.basename(rel_path))[0]
    fname     = f"{base_name}_fusion.{OUTPUT_FORMAT}"
    return os.path.join(output_root, rel_dir, fname)

def parse_args():
    parser = argparse.ArgumentParser(
        description="RGB-Thermal image fusion pipeline for wildlife species detection"
    )
    parser.add_argument(
        "--rgb", type=str, default=str(DEFAULT_RGB),
        help=f"Folder containing RGB images (*_W.JPG) (default: {DEFAULT_RGB})",
    )
    parser.add_argument(
        "--thermal", type=str, default=str(DEFAULT_THERMAL),
        help=f"Folder containing thermal TIFF images (*_T.tiff) (default: {DEFAULT_THERMAL})",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output folder for fused images (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--scale", type=int, default=DEFAULT_SCALE,
        help=f"Thermal size as %% of original (default: {DEFAULT_SCALE})",
    )
    parser.add_argument(
        "--offset-x", type=int, default=DEFAULT_OFFSET_X,
        help=f"Horizontal offset of thermal on RGB canvas in px (default: {DEFAULT_OFFSET_X})",
    )
    parser.add_argument(
        "--offset-y", type=int, default=DEFAULT_OFFSET_Y,
        help=f"Vertical offset of thermal on RGB canvas in px (default: {DEFAULT_OFFSET_Y})",
    )
    parser.add_argument(
        "--opacity", type=float, default=DEFAULT_OPACITY,
        help=f"Thermal overlay opacity, 0.0–1.0 (default: {DEFAULT_OPACITY})",
    )
    parser.add_argument(
        "--colormap", type=str, default=DEFAULT_COLORMAP,
        help=f"Matplotlib colormap name (default: {DEFAULT_COLORMAP})",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip pairs whose output file already exists",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    SEP  = "=" * 65
    SEP2 = "-" * 65

    print(SEP)
    print("  Fusion-Species-Detection-YOLO — RGB-Thermal Fusion")
    print(SEP)
    print(f"  RGB input   : {os.path.abspath(args.rgb)}")
    print(f"  Thermal in  : {os.path.abspath(args.thermal)}")
    print(f"  Output      : {os.path.abspath(args.output)}")
    print(f"  Scale       : {args.scale}%  |  Offset: ({args.offset_x}, {args.offset_y}) px")
    print(f"  Opacity     : {args.opacity}  |  Colormap: {args.colormap}")
    print(f"  Skip exist  : {args.skip_existing}")
    print(SEP)

    for path, label in [(args.rgb, "RGB"), (args.thermal, "Thermal")]:
        if not os.path.isdir(path):
            print(f"{label} folder not found: {os.path.abspath(path)}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    pairs, missing = find_pairs(args.rgb, args.thermal)

    if not pairs:
        print("No matching RGB + thermal pairs found.")
        print(f"   Make sure RGB files end with _W.JPG and thermal files end with _T.tiff")
        sys.exit(1)

    print(f"  Pairs found    : {len(pairs)}")

    if missing:
        print(f"    Missing thermal: {len(missing)} file(s)")
        print(SEP2)
        for rgb_p, tiff_p in missing:
            print(f"    RGB     : {rgb_p}")
            print(f"    Missing : {tiff_p}")
        print()

    print(SEP)

    success = 0
    failed  = 0
    skipped = 0

    for i, (rgb_path, thermal_path, rel_path) in enumerate(pairs, 1):
        out_path = build_output_path(args.output, rel_path)

        print(f"[{i:>3}/{len(pairs)}] {rel_path}")
        print(f"  RGB     : {rgb_path}")
        print(f"  Thermal : {thermal_path}")
        print(f"  Output  : {out_path}")

        if args.skip_existing and os.path.exists(out_path):
            print("  ⏭️  Skipped (already exists)\n")
            skipped += 1
            continue

        try:
            fuse_image(
                rgb_path, thermal_path, out_path,
                scale    = args.scale,
                offset_x = args.offset_x,
                offset_y = args.offset_y,
                opacity  = args.opacity,
                colormap = args.colormap,
            )
            print("  Done\n")
            success += 1
        except Exception as e:
            print(f"  Failed: {e}\n")
            failed += 1

    print(SEP)
    print("  SUMMARY")
    print(SEP2)
    print(f"  Success : {success}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped : {skipped}")
    print(SEP2)
    print(f"  Fused images saved to: {os.path.abspath(args.output)}")
    print(SEP)


if __name__ == "__main__":
    main()
