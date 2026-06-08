

import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "best.pt"
DEFAULT_SOURCE  = REPO_ROOT / "data"   / "fusion"
DEFAULT_OUTPUT  = REPO_ROOT / "results"


DEFAULT_IMGSZ      = 1280
DEFAULT_CONF_THRES = 0.50
DEFAULT_IOU_THRES  = 0.50

CLASSES = [
    "Banteng",
    "Green peacock",
    "Javan deer",
    "Long-tailed macaque",
    "Water buffalo",
    "Wildboar",
]

RENAME_MAP = {
    "Bull"         : "Banteng",
    "Deer"         : "Javan deer",
    "Monkey"       : "Long-tailed macaque",
    "Peacock"      : "Green peacock",
    "Pig"          : "Wildboar",
    "Water Buffalo": "Water buffalo",
}

WARNA_KELAS = [
    (  0, 200, 255),  # orange-yellow  — Banteng
    ( 50, 220,  50),  # green          — Green peacock
    (255, 100,  30),  # sky-blue       — Javan deer
    (220,  60, 220),  # purple         — Long-tailed macaque
    ( 30, 230, 230),  # yellow         — Water buffalo
    ( 40,  40, 230),  # red            — Wildboar
]
FALLBACK_WARNA = [(255,255,0),(0,255,255),(255,0,255),(128,255,0),(0,128,255)]

BOX_THICKNESS   = 2
FONT_SCALE      = 0.45
LABEL_THICKNESS = 1
ALPHA_FILL      = 0.12


def gambar_bbox_bersih(
    img, boxes, classes, warna_kelas,
    box_thick=2, font_scale=0.45, label_thick=1, alpha_fill=0.12,
    show_label=True,
):
    overlay = img.copy()

    if boxes is None or len(boxes) == 0:
        return img

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cid   = int(box.cls[0])
        warna = warna_kelas[cid % len(warna_kelas)]
        label = f"{classes[cid]}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), warna, -1)

        cv2.rectangle(img, (x1, y1), (x2, y2), warna, box_thick)

        if show_label:
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thick
            )
            pad    = 4
            chip_y = max(y1 - th - pad * 2, 0)
            chip_x2 = min(x1 + tw + pad * 2, img.shape[1])
            cv2.rectangle(img, (x1, chip_y), (chip_x2, y1), warna, -1)

            cv2.putText(
                img, label,
                (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), label_thick, cv2.LINE_AA,
            )

    img = cv2.addWeighted(overlay, alpha_fill, img, 1 - alpha_fill, 0)
    return img


def gambar_panel_info(img, cls_count):

    h, w = img.shape[:2]

    ref_w    = 1920
    skala    = w / ref_w
    fs_judul = round(0.60 * skala, 2)
    fs_isi   = round(0.50 * skala, 2)
    thick    = max(1, int(1.5 * skala))
    pad      = int(14 * skala)
    lh_judul = int(30 * skala)
    lh_isi   = int(26 * skala)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    total_obj = sum(cls_count.values())

    if cls_count:
        baris_judul = f"Total: {total_obj} individuals"
        baris_isi   = [
            f"  {cls}: {n} individuals"
            for cls, n in sorted(cls_count.items())
        ]
    else:
        baris_judul = "No detection"
        baris_isi   = []

    semua_baris = [baris_judul] + baris_isi

    lebar_maks = max(
        cv2.getTextSize(
            b, font, fs_judul if i == 0 else fs_isi, thick
        )[0][0]
        for i, b in enumerate(semua_baris)
    )
    panel_w = lebar_maks + pad * 2
    panel_h = pad + lh_judul + len(baris_isi) * lh_isi + pad

    margin = int(16 * skala)
    px1, py1 = margin, margin
    px2, py2 = px1 + panel_w, py1 + panel_h

    overlay = img.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)

    accent_x = px1 + int(5 * skala)
    cv2.line(
        img,
        (accent_x, py1 + pad // 2),
        (accent_x, py2 - pad // 2),
        (0, 220, 255),
        max(2, int(3 * skala)),
    )

    cv2.putText(
        img, baris_judul,
        (px1 + pad + int(6 * skala), py1 + pad + lh_judul - int(6 * skala)),
        font, fs_judul, (0, 220, 255), thick, cv2.LINE_AA,
    )

    for j, baris in enumerate(baris_isi):
        y_txt = py1 + pad + lh_judul + (j + 1) * lh_isi - int(4 * skala)
        cv2.putText(
            img, baris,
            (px1 + pad + int(6 * skala), y_txt),
            font, fs_isi, (240, 240, 240), thick, cv2.LINE_AA,
        )

    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 inference for wildlife species detection (RGB-thermal fusion)"
    )
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS),
        help=f"Path to YOLOv8 weights file (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--source", type=str, default=str(DEFAULT_SOURCE),
        help=f"Input image directory (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output directory for annotated images (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help=f"YOLO input image size (default: {DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF_THRES,
        help=f"Confidence threshold (default: {DEFAULT_CONF_THRES})",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU_THRES,
        help=f"NMS IoU threshold (default: {DEFAULT_IOU_THRES})",
    )
    parser.add_argument(
        "--no-label", action="store_true",
        help="Hide class label chips on bounding boxes",
    )
    parser.add_argument(
        "--no-panel", action="store_true",
        help="Hide the detection summary panel on output images",
    )
    return parser.parse_args()


def collect_images(source_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    images = []
    for ext in exts:
        images.extend(glob.glob(f"{source_dir}/{ext}"))
        images.extend(glob.glob(f"{source_dir}/{ext.upper()}"))
    return sorted(set(images))


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        print(f" Model weights not found: {args.weights}")
        sys.exit(1)

    if not os.path.isdir(args.source):
        print(f" Source directory not found: {args.source}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    all_images = collect_images(args.source)
    if not all_images:
        print(f" No images found in: {args.source}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Fusion-Species-Detection-YOLO — Inference")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Source  : {args.source}  ({len(all_images)} images)")
    print(f"  Output  : {args.output}")
    print(f"  imgsz   : {args.imgsz}  |  conf: {args.conf}  |  iou: {args.iou}")
    print("=" * 60 + "\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print(" ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.weights)

    global CLASSES, WARNA_KELAS
    CLASSES = [RENAME_MAP.get(name, name) for name in model.names.values()]

    # Extend color list if model has more classes than defaults
    while len(WARNA_KELAS) < len(CLASSES):
        WARNA_KELAS.append(FALLBACK_WARNA[len(WARNA_KELAS) % len(FALLBACK_WARNA)])

    print(f" Model loaded — {len(CLASSES)} classes detected:")
    for i, cls in enumerate(CLASSES):
        print(f"   ID {i} → {cls}")
    print()

    print(f"Running inference on {len(all_images)} image(s)...\n")
    pred_results = model.predict(
        source   = args.source,
        imgsz    = args.imgsz,
        conf     = args.conf,
        iou      = args.iou,
        save     = False,
        save_txt = False,
        verbose  = False,
    )

    rekap_global = []

    for i, result in enumerate(pred_results):
        img = result.orig_img.copy()

        img = gambar_bbox_bersih(
            img,
            result.boxes,
            CLASSES,
            WARNA_KELAS,
            box_thick  = BOX_THICKNESS,
            font_scale = FONT_SCALE,
            label_thick= LABEL_THICKNESS,
            alpha_fill = ALPHA_FILL,
            show_label = not args.no_label,
        )

        det_cls = []
        if result.boxes is not None and len(result.boxes) > 0:
            for cid in result.boxes.cls.cpu().numpy().astype(int):
                if cid < len(CLASSES):
                    det_cls.append(CLASSES[cid])
        cls_count = Counter(det_cls)

        rekap_global.append({
            "file"  : Path(result.path).name,
            "total" : sum(cls_count.values()),
            "detail": dict(cls_count),
        })

        if not args.no_panel:
            img = gambar_panel_info(img, cls_count)

        nama_file = Path(result.path).name
        out_path  = os.path.join(args.output, nama_file)
        cv2.imwrite(out_path, img)

        n_obj = sum(cls_count.values())
        print(f"  [{i+1:>3}/{len(pred_results)}]  {nama_file:<35} → {n_obj} object(s) detected")

    print("\n" + "=" * 65)
    print("                  INFERENCE SUMMARY")
    print("=" * 65)
    print(f"{'No':<5} {'File':<34} {'Total':>5}  Detail")
    print("-" * 65)

    grand_total   = 0
    grand_counter = Counter()

    for idx, row in enumerate(rekap_global, 1):
        detail_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(row["detail"].items())
        ) or "-"
        print(f"  {idx:<4} {row['file']:<34} {row['total']:>5}  {detail_str}")
        grand_total += row["total"]
        grand_counter.update(row["detail"])

    print("=" * 65)
    print(f"  GRAND TOTAL: {grand_total} individuals")
    for cls, n in sorted(grand_counter.items()):
        print(f"    → {cls}: {n}")
    print("=" * 65)
    print(f"\n🎉 Done! Annotated images saved to: {args.output}\n")


if __name__ == "__main__":
    main()
