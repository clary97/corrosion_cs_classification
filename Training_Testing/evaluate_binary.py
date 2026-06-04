"""
이진 분류 (Good vs Corrosion) 학습 결과 분석

- 학습 곡선 (Loss / F1 / IoU)
- 최종 모델 지표 (Precision, Recall, F1, IoU, FPR)
- 4클래스 결과와 나란히 비교
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
import datahandler_plus
from datahandler_plus import SegDataset, ToTensor

# ── 설정 ──────────────────────────────────────────────────────────────────────
EXP_DIR_BINARY = './stored_weights/finetune_binary_v1'
EXP_DIR_4CLS   = './stored_weights/finetune_wce_merged_v1'
DATA_DIR       = '/home/ldh/minkyung/unified_corrosion_binary'
WEIGHTS_FILE   = os.path.join(EXP_DIR_BINARY, 'weights.pt')
LOG_FILE       = os.path.join(EXP_DIR_BINARY, 'log_3.csv')
SAVE_DIR       = os.path.join(EXP_DIR_BINARY, 'plots')
CLASS_NAMES    = ['Good', 'Corrosion']
BATCH_SIZE     = 4
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

cols = ['epoch', 'Train_loss', 'Test_loss',
        'Train_f1', 'Train_iou', 'Train_spectrum',
        'Test_f1',  'Test_iou',  'Test_spectrum']

# ── 1. 학습 곡선 ──────────────────────────────────────────────────────────────
print("[1] 학습 곡선 생성")
df_bin = pd.read_csv(LOG_FILE, header=None, names=cols)

# 4클래스 로그가 있으면 비교 곡선 추가
df_4cls = None
log_4cls = os.path.join(EXP_DIR_4CLS, 'log_3.csv')
if os.path.exists(log_4cls):
    df_4cls = pd.read_csv(log_4cls, header=None, names=cols)

fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

for ax_idx, (metric, title) in enumerate([
    ('loss', 'Loss'),
    ('f1',   'F1 Score (weighted)'),
    ('iou',  'IoU / Jaccard (weighted)'),
]):
    ax = fig.add_subplot(gs[ax_idx])
    ax.plot(df_bin.epoch, df_bin[f'Train_{metric}'], label='Binary Train', color='steelblue')
    ax.plot(df_bin.epoch, df_bin[f'Test_{metric}'],  label='Binary Test',  color='steelblue', linestyle='--')
    if df_4cls is not None:
        ax.plot(df_4cls.epoch, df_4cls[f'Train_{metric}'], label='4-class Train', color='coral', alpha=0.7)
        ax.plot(df_4cls.epoch, df_4cls[f'Test_{metric}'],  label='4-class Test',  color='coral', linestyle='--', alpha=0.7)
    ax.set_title(title); ax.set_xlabel('Epoch')
    if metric != 'loss':
        ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(True)

fig.suptitle('Binary vs 4-class Training Curves', fontsize=13)
path = os.path.join(SAVE_DIR, 'training_curves_binary.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: {path}")

# ── 2. 최종 모델 추론 ─────────────────────────────────────────────────────────
print("\n[2] 최종 모델로 Test set 추론")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"  디바이스: {device}")

model = torch.load(WEIGHTS_FILE, map_location=device, weights_only=False)
model.eval()

data_transforms = transforms.Compose([ToTensor()])
test_ds = SegDataset(root_dir=os.path.join(DATA_DIR, 'Test'),
                     transform=data_transforms,
                     imageFolder='Images', maskFolder='Masks')
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4, drop_last=False)

all_true, all_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device, dtype=torch.long)
        images = images.permute(0, 3, 1, 2).contiguous()
        out    = model(images)
        pred   = torch.argmax(out, dim=1)
        all_true.append(masks.cpu().numpy().ravel())
        all_pred.append(pred.cpu().numpy().ravel())

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)

# ── 3. 지표 계산 ──────────────────────────────────────────────────────────────
print("\n[3] 지표 계산")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
n  = 2

precision, recall, f1, iou, fpr = [], [], [], [], []
for i in range(n):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f   = 2*p*r / (p+r)  if (p+r)   > 0 else 0.0
    iou_v = tp / (tp + fp + fn) if (tp+fp+fn) > 0 else 0.0
    fpr_v = fp / (fp + tn)      if (fp + tn)  > 0 else 0.0
    precision.append(p); recall.append(r)
    f1.append(f);        iou.append(iou_v); fpr.append(fpr_v)

support = cm.sum(axis=1)
weights = support / support.sum()
w_p  = np.dot(weights, precision)
w_r  = np.dot(weights, recall)
w_f1 = np.dot(weights, f1)
w_iou= np.dot(weights, iou)
w_fpr= np.dot(weights, fpr)

# ── 4. 결과 출력 ──────────────────────────────────────────────────────────────
header = f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'FPR':>10} {'Support':>10}"
print("\n" + "=" * 74)
print(header)
print("-" * 74)
for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls:<12} {precision[i]:>10.4f} {recall[i]:>10.4f} "
          f"{f1[i]:>10.4f} {iou[i]:>10.4f} {fpr[i]:>10.4f} {support[i]:>10,}")
print("-" * 74)
print(f"{'Weighted':<12} {w_p:>10.4f} {w_r:>10.4f} "
      f"{w_f1:>10.4f} {w_iou:>10.4f} {w_fpr:>10.4f} {support.sum():>10,}")
print("=" * 74)

# ── 5. 시각화 ─────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))

# 혼동 행렬
im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_title('Confusion Matrix')
plt.colorbar(im, ax=axes[0])
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(CLASS_NAMES)
axes[0].set_yticks([0,1]); axes[0].set_yticklabels(CLASS_NAMES)
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

# 지표 바 차트
x = np.arange(n)
w = 0.15
axes[1].bar(x - 2*w, precision, w, label='Precision')
axes[1].bar(x - 1*w, recall,    w, label='Recall')
axes[1].bar(x,       f1,        w, label='F1')
axes[1].bar(x + 1*w, iou,       w, label='IoU')
axes[1].bar(x + 2*w, fpr,       w, label='FPR')
axes[1].set_xticks(x); axes[1].set_xticklabels(CLASS_NAMES)
axes[1].set_ylim(0, 1.05); axes[1].set_title('Metrics per Class')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.4)

fig2.suptitle('Binary Classification — Evaluation on Test Set', fontsize=13)
path2 = os.path.join(SAVE_DIR, 'evaluation_metrics_binary.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  저장: {path2}")
print("\n완료.")
