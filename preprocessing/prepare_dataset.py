"""Unify one or more corrosion-segmentation datasets into the layout that
``Training - Testing/main_plus.py`` expects:

    <output>/
      Train/Images/<prefix>__<stem>.<ext>
      Train/Masks/<prefix>__<stem>.<ext>
      Test/Images/...
      Test/Masks/...

Add or remove sources by editing the ``DATASETS`` list below — each entry
points at a source dataset's train/test image and mask folders and assigns
a short ``prefix`` that is prepended to every filename so multiple
datasets can be merged without collisions.

Class color mapping (BGR; must match Training - Testing/datahandler_plus.py):

    Good (background) = (0,   0,   0)
    Fair              = (0,   0,   128)   # red in RGB
    Poor              = (0,   128, 0)     # green
    Severe            = (0,   128, 128)   # yellow

Usage:
    python prepare_dataset.py --output /path/to/unified
    python prepare_dataset.py --output /path/to/unified --mode copy
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset registry.  Add another dict here when a new dataset arrives.
# `root` is an absolute path; the four subdir fields are relative to it.
# `prefix` must be unique across datasets (used to namespace filenames).
# ---------------------------------------------------------------------------
DATASETS: list[dict] = [
    {
        "name": "CCSC",  # Bianchi et al., 2021 — Corrosion Condition State Classification
        "root": "/mnt/nas_200/Corrosion_Condition_State_Classification/512x512",
        "train_images": "Train/images_512",
        "train_masks":  "Train/mask_512",
        "test_images":  "Test/images_512",
        "test_masks":   "Test/mask_512",
        "prefix": "ccsc",
    },
    {
        "name": "CIR",  # Corrosion Images Rasterized (labelme JSONs -> pixel masks
                        # via preprocessing/rasterize_labelme_dataset.py)
        "root": "/mnt/nas_200/corrosion_images_512",
        "train_images": "Train/Images",
        "train_masks":  "Train/Masks",
        "test_images":  "Test/Images",
        "test_masks":   "Test/Masks",
        "prefix": "cir",
    },
]


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def files_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


def process_split(
    split: str,
    datasets: list[dict],
    output_root: Path,
    mode: str,
    key_images: str,
    key_masks: str,
) -> int:
    out_images = output_root / split / "Images"
    out_masks = output_root / split / "Masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    total = 0
    for ds in datasets:
        src_img_dir = Path(ds["root"]) / ds[key_images]
        src_msk_dir = Path(ds["root"]) / ds[key_masks]
        if not src_img_dir.is_dir() or not src_msk_dir.is_dir():
            print(f"  [skip] {ds['name']}/{split}: missing {src_img_dir} or {src_msk_dir}")
            continue

        imgs = files_in(src_img_dir)
        masks = files_in(src_msk_dir)
        mask_by_stem = {p.stem: p for p in masks}

        paired = [(im, mask_by_stem[im.stem]) for im in imgs if im.stem in mask_by_stem]
        orphan_images = len(imgs) - len(paired)
        orphan_masks = len(masks) - len(paired)

        for im, mk in paired:
            new_stem = f"{ds['prefix']}__{im.stem}"
            link_or_copy(im, out_images / f"{new_stem}{im.suffix}", mode)
            link_or_copy(mk, out_masks / f"{new_stem}{mk.suffix}", mode)
        total += len(paired)

        print(
            f"  [{ds['name']}/{split}] paired={len(paired)} "
            f"orphan_images={orphan_images} orphan_masks={orphan_masks}"
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory — pass this to main_plus.py via -data_directory.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="symlink (default, saves disk) or copy (portable).",
    )
    args = parser.parse_args()

    if not DATASETS:
        sys.exit("DATASETS list is empty — edit prepare_dataset.py and add at least one entry.")

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Unifying {len(DATASETS)} dataset(s) -> {output_root}  (mode={args.mode})")
    train_total = process_split("Train", DATASETS, output_root, args.mode, "train_images", "train_masks")
    test_total = process_split("Test", DATASETS, output_root, args.mode, "test_images", "test_masks")

    print(f"\nDone.  Train pairs: {train_total}   Test pairs: {test_total}")
    print(f"Pass this to main_plus.py:  -data_directory '{output_root}'")


if __name__ == "__main__":
    main()
