"""
이진 분류 Test set 인퍼런스 결과 이미지 저장

각 이미지마다 4-panel:
  [원본 이미지] | [GT 마스크] | [예측 마스크] | [오버레이]

색상:
  Good      — 검정 (0, 0, 0)
  Corrosion — 빨강 (128, 0, 0) in RGB
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from datahandler_plus import SegDataset, ToTensor

# ── 설정 ──────────────────────────────────────────────────────────────────────
EXP_DIR      = './stored_weights/finetune_binary_v1'
DATA_DIR     = '/home/ldh/minkyung/unified_corrosion_binary'
WEIGHTS_FILE = os.path.join(EXP_DIR, 'weights.pt')
SAVE_DIR     = os.path.join(EXP_DIR, 'predictions')
CLASS_NAMES  = ['Good', 'Corrosion']
CLASS_COLORS = np.array([
    [  0,   0,   0],   # Good      — 검정
    [200,  50,  50],   # Corrosion — 빨강
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

legend_patches = [
    mpatches.Patch(color=CLASS_COLORS[i]/255.0, label=CLASS_NAMES[i])
    for i in range(len(CLASS_NAMES))
]


def label_to_color(label_map):
    h, w = label_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in enumerate(CLASS_COLORS):
        color[label_map == cls_idx] = rgb
    return color


with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        images     = batch['image'].to(device)
        true_masks = batch['mask'].cpu().numpy()[0]

        inp  = images.permute(0, 3, 1, 2).contiguous()
        out  = model(inp)
        pred = torch.argmax(out, dim=1).cpu().numpy()[0]

        img_np = images.cpu().numpy()[0].clip(0, 255).astype(np.uint8)
        gt_color   = label_to_color(true_masks)
        pred_color = label_to_color(pred)
        overlay = (img_np * (1 - OVERLAY_ALPHA) +
                   pred_color * OVERLAY_ALPHA).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, title, im in zip(axes,
                                  ['Input Image', 'Ground Truth', 'Prediction', 'Overlay'],
                                  [img_np, gt_color, pred_color, overlay]):
            ax.imshow(im); ax.set_title(title, fontsize=12); ax.axis('off')

        fig.legend(handles=legend_patches, loc='lower center',
                   ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout()
        save_path = os.path.join(SAVE_DIR, f'pred_{idx:04d}.png')
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_ds):
            print(f"  [{idx+1}/{len(test_ds)}] {save_path}")

print(f"\n완료. 저장 위치: {SAVE_DIR}")
