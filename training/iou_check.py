import sys, os, torch, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
sys.path.insert(0, os.path.dirname(__file__))
from datahandler_plus import SegDataset, ToTensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# ── 4-class ────────────────────────────────────────────────────────────
model4 = torch.load('./stored_weights/finetune_wce_merged_v1/weights.pt',
                    map_location=device, weights_only=False)
model4.eval()
ds4 = SegDataset(root_dir='/home/ldh/minkyung/unified_corrosion/Test',
                 transform=transforms.Compose([ToTensor()]),
                 imageFolder='Images', maskFolder='Masks')
t4, p4 = [], []
with torch.no_grad():
    for b in DataLoader(ds4, batch_size=4, num_workers=0):
        imgs = b['image'].to(device).permute(0,3,1,2).contiguous()
        masks = b['mask'].to(device, dtype=torch.long)
        pred = torch.argmax(model4(imgs), dim=1)
        t4.append(masks.cpu().numpy().ravel())
        p4.append(pred.cpu().numpy().ravel())
cm4 = confusion_matrix(np.concatenate(t4), np.concatenate(p4), labels=[0,1,2,3])

print("\n=== 4-class 클래스별 IoU ===")
ious4 = []
for i, name in enumerate(['Good','Fair','Poor','Severe']):
    tp = int(cm4[i,i]); fp = int(cm4[:,i].sum()-tp); fn = int(cm4[i,:].sum()-tp)
    iou = tp/(tp+fp+fn)
    ious4.append(iou)
    print(f"  {name:<8}: {iou:.4f} ({iou*100:.1f}%)")

# Fair+Poor+Severe → binary 관점의 Corrosion IoU
tp_c = int(sum(cm4[i,i] for i in [1,2,3]))
fp_c = int(sum(cm4[0,j] for j in [1,2,3]))   # Good → 부식 오분류
fn_c = int(sum(cm4[i,0] for i in [1,2,3]))   # 부식 → Good 오분류
iou_c4 = tp_c / (tp_c + fp_c + fn_c)
print(f"\n  [4-class] 부식(Fair+Poor+Severe) 통합 IoU = {iou_c4:.4f} ({iou_c4*100:.1f}%)")
print(f"  (부식끼리 헷갈린 픽셀은 TP로 처리 — binary 관점)")

support4 = cm4.sum(axis=1)
w4 = support4 / support4.sum()
print(f"\n  4-class Weighted IoU = {np.dot(w4, ious4):.4f}  ← 보고서 수치")
print(f"  (Good {w4[0]*100:.1f}% / Fair {w4[1]*100:.1f}% / Poor {w4[2]*100:.1f}% / Severe {w4[3]*100:.1f}% 가중)")

# ── Binary ─────────────────────────────────────────────────────────────
model2 = torch.load('./stored_weights/finetune_binary_v1/weights.pt',
                    map_location=device, weights_only=False)
model2.eval()
ds2 = SegDataset(root_dir='/home/ldh/minkyung/unified_corrosion_binary/Test',
                 transform=transforms.Compose([ToTensor()]),
                 imageFolder='Images', maskFolder='Masks')
t2, p2 = [], []
with torch.no_grad():
    for b in DataLoader(ds2, batch_size=4, num_workers=0):
        imgs = b['image'].to(device).permute(0,3,1,2).contiguous()
        masks = b['mask'].to(device, dtype=torch.long)
        pred = torch.argmax(model2(imgs), dim=1)
        t2.append(masks.cpu().numpy().ravel())
        p2.append(pred.cpu().numpy().ravel())
cm2 = confusion_matrix(np.concatenate(t2), np.concatenate(p2), labels=[0,1])

print("\n=== Binary 클래스별 IoU ===")
ious2 = []
for i, name in enumerate(['Good','Corrosion']):
    tp = int(cm2[i,i]); fp = int(cm2[:,i].sum()-tp); fn = int(cm2[i,:].sum()-tp)
    iou = tp/(tp+fp+fn)
    ious2.append(iou)
    print(f"  {name:<10}: {iou:.4f} ({iou*100:.1f}%)")

support2 = cm2.sum(axis=1)
w2 = support2 / support2.sum()
print(f"\n  Binary Weighted IoU = {np.dot(w2, ious2):.4f}  ← 보고서 수치")
print(f"  (Good {w2[0]*100:.1f}% / Corrosion {w2[1]*100:.1f}% 가중)")

print("\n=== 부식 검출 관점 직접 비교 ===")
print(f"  4-class 부식 통합 IoU : {iou_c4:.4f} ({iou_c4*100:.1f}%)")
print(f"  Binary  Corrosion IoU : {ious2[1]:.4f} ({ious2[1]*100:.1f}%)")
print(f"  향상                  : +{(ious2[1]-iou_c4)*100:.1f}%p")
