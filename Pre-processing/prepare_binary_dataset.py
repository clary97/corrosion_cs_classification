"""이진 분류용 데이터셋 생성 스크립트

기존 4클래스 마스크를 2클래스로 재매핑:
  Good (배경)           BGR (0, 0,   0) → class 0  (유지)
  Fair + Poor + Severe  BGR (0, 0, 128) → class 1  (Corrosion으로 통합)

기존 unified_corrosion 데이터셋을 그대로 활용하며
BGR 색상만 교체한 새 마스크를 저장합니다.

Usage:
    python prepare_binary_dataset.py \
        --input  /home/ldh/minkyung/unified_corrosion \
        --output /home/ldh/minkyung/unified_corrosion_binary
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# 4클래스 BGR 원본 → 이진 BGR 목적지
REMAP: dict[tuple, tuple] = {
    (0,   0,   0): (0,   0,   0),    # Good     → Good       (유지)
    (0,   0, 128): (0,   0, 128),    # Fair     → Corrosion
    (0, 128,   0): (0,   0, 128),    # Poor     → Corrosion
    (0, 128, 128): (0,   0, 128),    # Severe   → Corrosion
}


def remap_mask(src_path: Path, dst_path: Path) -> None:
    mask = cv2.imread(str(src_path))          # BGR
    out  = np.zeros_like(mask)
    for src_bgr, dst_bgr in REMAP.items():
        px = np.array(src_bgr, dtype=np.uint8)
        match = np.all(mask == px, axis=2)
        out[match] = dst_bgr
    cv2.imwrite(str(dst_path), out)


def process_split(split: str, input_root: Path, output_root: Path) -> None:
    src_img_dir  = input_root  / split / 'Images'
    src_msk_dir  = input_root  / split / 'Masks'
    dst_img_dir  = output_root / split / 'Images'
    dst_msk_dir  = output_root / split / 'Masks'

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_msk_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(src_img_dir.iterdir())
    print(f"[{split}] {len(images)}개 처리 중...")

    for img_path in tqdm(images, desc=split):
        # 이미지는 심볼릭 링크 or 복사
        dst_img = dst_img_dir / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # 마스크는 재매핑
        msk_path = src_msk_dir / img_path.name
        if msk_path.exists():
            remap_mask(msk_path, dst_msk_dir / img_path.name)

    print(f"  → {dst_msk_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',  required=True, help='기존 unified_corrosion 경로')
    parser.add_argument('--output', required=True, help='이진 데이터셋 저장 경로')
    args = parser.parse_args()

    input_root  = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"입력: {input_root}")
    print(f"출력: {output_root}")
    print(f"매핑: Fair/Poor/Severe → Corrosion (BGR 0,0,128)\n")

    for split in ['Train', 'Test']:
        process_split(split, input_root, output_root)

    print("\n완료.")
    print(f"학습 명령어:")
    print(f"  python main_plus.py \\")
    print(f"    -data_directory '{output_root}' \\")
    print(f"    -exp_directory  './stored_weights/finetune_binary_v1' \\")
    print(f"    --epochs 40 --batchsize 2 --output_stride 8 --channels 2 \\")
    print(f"    --loss cross_entropy \\")
    print(f"    --class_weights 0.15 0.85")


if __name__ == '__main__':
    main()
