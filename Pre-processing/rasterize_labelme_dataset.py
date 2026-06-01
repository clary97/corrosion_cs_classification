"""Rasterize a labelme-polygon dataset into the (Train|Test) x (Images|Masks)
pixel-mask layout that ``Training - Testing/main_plus.py`` consumes.

Output layout:

    <output>/
      Train/Images/<stem>.png      # resized to --image-size square
      Train/Masks/<stem>.png       # 4-class BGR mask, INTER_NEAREST resize
      Test/Images/...
      Test/Masks/...

Class color mapping for masks (BGR, matches Training - Testing/datahandler_plus.py):

    Good (background) = (0,   0,   0)
    Fair              = (0,   0,   128)
    Poor              = (0,   128, 0)
    Severe            = (0,   128, 128)

Configure the ``SOURCE_ROOT`` / ``SPLIT_SPECS`` / ``LABEL_TO_BGR`` block below
to match your source dataset, then run:

    python Pre-processing/rasterize_labelme_dataset.py \\
        --output /path/to/rasterized_dataset \\
        --image-size 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Source dataset configuration — edit to match your dataset.
# ---------------------------------------------------------------------------
SOURCE_ROOT = "/workspace/nas_200/minkyung/corrosion_images"

# (image_dir_relative, json_dir_relative, output_split)
# Multiple entries can map into the same output split (e.g. merge val -> Train).
SPLIT_SPECS: list[tuple[str, str, str]] = [
    ("images/train/images",      "Annotation_json/train/label",      "Train"),
    ("images/Validation/images", "Annotation_json/validation/label", "Train"),
    ("images/test/images",       "Annotation_json/test/label",       "Test"),
]

# labelme `label` string -> BGR pixel color for the mask.
# Anything not listed here is silently treated as background (0,0,0).
LABEL_TO_BGR: dict[str, tuple[int, int, int]] = {
    "Fair":   (0, 0,   128),
    "Poor":   (0, 128, 0),
    "Severe": (0, 128, 128),
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# ---------------------------------------------------------------------------


def rasterize_mask(json_path: Path, fallback_hw: tuple[int, int]) -> np.ndarray:
    """Read a labelme JSON and return a BGR mask of shape (H, W, 3)."""
    with json_path.open() as f:
        data = json.load(f)

    h = data.get("imageHeight") or fallback_hw[0]
    w = data.get("imageWidth") or fallback_hw[1]
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    skipped_labels = set()
    skipped_types = set()
    for shape in data.get("shapes", []):
        label = shape.get("label")
        bgr = LABEL_TO_BGR.get(label)
        if bgr is None:
            skipped_labels.add(label)
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            skipped_types.add(shape.get("shape_type"))
            continue
        pts = np.asarray(shape["points"], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=bgr)

    return mask, skipped_labels, skipped_types


def index_by_stem(directory: Path, exts: set[str] | None = None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        out[p.stem] = p
    return out


def process(args: argparse.Namespace) -> None:
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    target_size = (args.image_size, args.image_size)  # (W, H) for cv2.resize
    source = Path(SOURCE_ROOT)

    grand = {"ok": 0, "fail": 0, "orphan_img": 0, "orphan_json": 0}
    all_skipped_labels: set[str] = set()
    all_skipped_types: set[str] = set()

    for img_rel, json_rel, split in SPLIT_SPECS:
        img_dir = source / img_rel
        json_dir = source / json_rel
        if not img_dir.is_dir() or not json_dir.is_dir():
            print(f"  [skip] {split}: missing {img_dir} or {json_dir}")
            continue

        out_img_dir = output_root / split / "Images"
        out_msk_dir = output_root / split / "Masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_msk_dir.mkdir(parents=True, exist_ok=True)

        imgs = index_by_stem(img_dir, IMAGE_EXTS)
        jsons = index_by_stem(json_dir, {".json"})

        common = sorted(set(imgs) & set(jsons))
        only_img = sorted(set(imgs) - set(jsons))
        only_json = sorted(set(jsons) - set(imgs))

        ok = fail = 0
        for stem in common:
            try:
                img = cv2.imread(str(imgs[stem]), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"  [warn] unreadable image: {imgs[stem]}")
                    fail += 1
                    continue
                mask, skip_labels, skip_types = rasterize_mask(jsons[stem], img.shape[:2])
                all_skipped_labels |= {l for l in skip_labels if l is not None}
                all_skipped_types |= {t for t in skip_types if t is not None}

                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(str(out_img_dir / f"{stem}.png"), img_resized)
                cv2.imwrite(str(out_msk_dir / f"{stem}.png"), mask_resized)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [err] {stem}: {exc}")
                fail += 1

        grand["ok"] += ok
        grand["fail"] += fail
        grand["orphan_img"] += len(only_img)
        grand["orphan_json"] += len(only_json)
        print(
            f"  [{img_rel} -> {split}] ok={ok} fail={fail} "
            f"orphan_img={len(only_img)} orphan_json={len(only_json)}"
        )

    print(
        f"\nDone. ok={grand['ok']} fail={grand['fail']} "
        f"orphan_img={grand['orphan_img']} orphan_json={grand['orphan_json']}"
    )
    if all_skipped_labels:
        print(f"  unmapped labels (treated as background): {sorted(all_skipped_labels)}")
    if all_skipped_types:
        print(f"  skipped shape_types (non-polygon): {sorted(all_skipped_types)}")
    print(f"Output: {output_root}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", required=True, help="Output dataset directory.")
    ap.add_argument(
        "--image-size", type=int, default=512,
        help="Resize images and masks to this side length (default 512).",
    )
    args = ap.parse_args()

    if not SPLIT_SPECS:
        sys.exit("SPLIT_SPECS is empty — edit rasterize_labelme_dataset.py.")
    if not LABEL_TO_BGR:
        sys.exit("LABEL_TO_BGR is empty — edit rasterize_labelme_dataset.py.")

    process(args)


if __name__ == "__main__":
    main()
