"""
학습 결과 분석 스크립트
- 학습 곡선 (Loss / F1 / IoU)
- 최종 모델 전체 지표 (Precision, Recall, F1, IoU, FPR) per class + weighted avg
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(__file__))
import datahandler_plus

# ── 설정 ──────────────────────────────────────────────────────────────────────
EXP_DIR      = './stored_weights/finetune_wce_merged_v1'
DATA_DIR     = '/home/ldh/minkyung/unified_corrosion'
WEIGHTS_FILE = os.path.join(EXP_DIR, 'weights.pt')   # 최종 저장 모델
LOG_FILE     = os.path.join(EXP_DIR, 'log_3.csv')
SAVE_DIR     = os.path.join(EXP_DIR, 'plots')
CLASS_NAMES  = ['Good', 'Fair', 'Poor', 'Severe']
BATCH_SIZE   = 4
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. 학습 곡선 ──────────────────────────────────────────────────────────────
print("=" * 60)
print("[1] 학습 곡선 생성")
cols = ['epoch', 'Train_loss', 'Test_loss',
        'Train_f1', 'Train_iou', 'Train_spectrum',
        'Test_f1',  'Test_iou',  'Test_spectrum']
df = pd.read_csv(LOG_FILE, header=None, names=cols)

fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(df.epoch, df.Train_loss, label='Train')
ax1.plot(df.epoch, df.Test_loss,  label='Test')
ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)

ax2 = fig.add_subplot(gs[1])
ax2.plot(df.epoch, df.Train_f1, label='Train')
ax2.plot(df.epoch, df.Test_f1,  label='Test')
ax2.set_title('F1 Score (weighted)'); ax2.set_xlabel('Epoch')
ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True)

ax3 = fig.add_subplot(gs[2])
ax3.plot(df.epoch, df.Train_iou, label='Train')
ax3.plot(df.epoch, df.Test_iou,  label='Test')
ax3.set_title('IoU / Jaccard (weighted)'); ax3.set_xlabel('Epoch')
ax3.set_ylim(0, 1); ax3.legend(); ax3.grid(True)

fig.suptitle('Training Curves — finetune_wce_merged_v1', fontsize=13)
path = os.path.join(SAVE_DIR, 'training_curves.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  저장: {path}")

# ── 2. 최종 모델 추론 ─────────────────────────────────────────────────────────
print("\n[2] 최종 모델로 Test set 추론")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"  디바이스: {device}")

model = torch.load(WEIGHTS_FILE, map_location=device, weights_only=False)
model.eval()

from torchvision import transforms
from datahandler_plus import SegDataset, ToTensor

data_transforms = transforms.Compose([ToTensor()])
test_ds  = SegDataset(root_dir=os.path.join(DATA_DIR, 'Test'),
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

# ── 3. 혼동 행렬 기반 지표 계산 ───────────────────────────────────────────────
print("\n[3] 지표 계산")
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
n  = len(CLASS_NAMES)

precision, recall, f1, iou, fpr = [], [], [], [], []
for i in range(n):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp        # 다른 클래스인데 i로 예측
    fn = cm[i, :].sum() - tp        # i인데 다른 클래스로 예측
    tn = cm.sum() - tp - fp - fn

    p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f   = 2*p*r / (p+r)  if (p+r)   > 0 else 0.0
    iou_val = tp / (tp + fp + fn)   if (tp+fp+fn) > 0 else 0.0
    fpr_val = fp / (fp + tn)        if (fp + tn)  > 0 else 0.0

    precision.append(p); recall.append(r)
    f1.append(f);        iou.append(iou_val); fpr.append(fpr_val)

# 클래스별 픽셀 수 (가중치)
support = cm.sum(axis=1)
weights = support / support.sum()

w_precision = np.dot(weights, precision)
w_recall    = np.dot(weights, recall)
w_f1        = np.dot(weights, f1)
w_iou       = np.dot(weights, iou)
w_fpr       = np.dot(weights, fpr)

# ── 4. 결과 출력 ──────────────────────────────────────────────────────────────
header = f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'FPR':>10} {'Support':>10}"
print("\n" + "=" * 72)
print(header)
print("-" * 72)
for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls:<10} {precision[i]:>10.4f} {recall[i]:>10.4f} "
          f"{f1[i]:>10.4f} {iou[i]:>10.4f} {fpr[i]:>10.4f} {support[i]:>10,}")
print("-" * 72)
print(f"{'Weighted':<10} {w_precision:>10.4f} {w_recall:>10.4f} "
      f"{w_f1:>10.4f} {w_iou:>10.4f} {w_fpr:>10.4f} {support.sum():>10,}")
print("=" * 72)

# ── 5. 혼동 행렬 시각화 ───────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# 혼동 행렬 (count)
im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0].set_title('Confusion Matrix (counts)')
plt.colorbar(im1, ax=axes[0])
tick_marks = np.arange(n)
axes[0].set_xticks(tick_marks); axes[0].set_xticklabels(CLASS_NAMES, rotation=45)
axes[0].set_yticks(tick_marks); axes[0].set_yticklabels(CLASS_NAMES)
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f'{cm[i,j]:,}',
                     ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=9)

# 지표 바 차트
x = np.arange(n)
w = 0.18
axes[1].bar(x - 2*w, precision, w, label='Precision')
axes[1].bar(x - 1*w, recall,    w, label='Recall')
axes[1].bar(x,       f1,        w, label='F1')
axes[1].bar(x + 1*w, iou,       w, label='IoU')
axes[1].bar(x + 2*w, fpr,       w, label='FPR')
axes[1].set_xticks(x); axes[1].set_xticklabels(CLASS_NAMES)
axes[1].set_ylim(0, 1.05); axes[1].set_title('Metrics per Class')
axes[1].legend(loc='lower right'); axes[1].grid(axis='y', alpha=0.4)

fig2.suptitle('Evaluation on Test Set — finetune_wce_merged_v1', fontsize=13)
path2 = os.path.join(SAVE_DIR, 'evaluation_metrics.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  저장: {path2}")
print("\n완료.")
