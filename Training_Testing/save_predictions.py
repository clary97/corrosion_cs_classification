"""
Test set 인퍼런스 결과 이미지 저장 스크립트

각 이미지마다 4-panel 그림 저장:
  [원본 이미지] | [GT 마스크] | [예측 마스크] | [오버레이]
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import datahandler_plus
from datahandler_plus import SegDataset, ToTensor

# ── 설정 ──────────────────────────────────────────────────────────────────────
EXP_DIR      = './stored_weights/finetune_wce_merged_v1'
DATA_DIR     = '/home/ldh/minkyung/unified_corrosion'
WEIGHTS_FILE = os.path.join(EXP_DIR, 'weights.pt')
SAVE_DIR     = os.path.join(EXP_DIR, 'predictions')
CLASS_NAMES  = ['Good', 'Fair', 'Poor', 'Severe']

# BGR → RGB 색상 (datahandler의 BGR 매핑을 RGB로 변환)
# Good=(0,0,0) Fair=(0,0,128) Poor=(0,128,0) Severe=(0,128,128) in BGR
CLASS_COLORS_RGB = np.array([
    [  0,   0,   0],   # Good    — 검정
    [128,   0,   0],   # Fair    — 빨강
    [  0, 128,   0],   # Poor    — 초록
    [128, 128,   0],   # Severe  — 노랑
], dtype=np.uint8)
OVERLAY_ALPHA = 0.45
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"디바이스: {device}")

model = torch.load(WEIGHTS_FILE, map_location=device, weights_only=False)
model.eval()

data_transforms = transforms.Compose([ToTensor()])
test_ds = SegDataset(root_dir=os.path.join(DATA_DIR, 'Test'),
                     transform=data_transforms,
                     imageFolder='Images', maskFolder='Masks')
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=2, drop_last=False)

print(f"Test 이미지 수: {len(test_ds)}")


def label_to_color(label_map):
    """(H, W) 클래스 인덱스 → (H, W, 3) RGB 컬러 이미지"""
    h, w = label_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in enumerate(CLASS_COLORS_RGB):
        color[label_map == cls_idx] = rgb
    return color


legend_patches = [
    mpatches.Patch(color=CLASS_COLORS_RGB[i] / 255.0, label=CLASS_NAMES[i])
    for i in range(len(CLASS_NAMES))
]

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        images     = batch['image'].to(device)           # (1, H, W, 3)
        true_masks = batch['mask'].cpu().numpy()[0]      # (H, W)

        inp = images.permute(0, 3, 1, 2).contiguous()
        out = model(inp)
        pred = torch.argmax(out, dim=1).cpu().numpy()[0]  # (H, W)

        # 원본 이미지 복원 (ToTensor는 /255 없이 float 변환만 하므로 바로 uint8)
        img_np = images.cpu().numpy()[0]                  # (H, W, 3) float 0~255
        img_np = img_np.clip(0, 255).astype(np.uint8)

        gt_color   = label_to_color(true_masks)
        pred_color = label_to_color(pred)

        # 오버레이: 원본 + 예측 마스크 블렌딩
        overlay = (img_np * (1 - OVERLAY_ALPHA) +
                   pred_color * OVERLAY_ALPHA).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ['Input Image', 'Ground Truth', 'Prediction', 'Overlay']
        imgs   = [img_np, gt_color, pred_color, overlay]

        for ax, title, im in zip(axes, titles, imgs):
            ax.imshow(im)
            ax.set_title(title, fontsize=12)
            ax.axis('off')

        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout()

        save_path = os.path.join(SAVE_DIR, f'pred_{idx:04d}.png')
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_ds):
            print(f"  [{idx+1}/{len(test_ds)}] 저장: {save_path}")

print(f"\n완료. 저장 위치: {SAVE_DIR}")
