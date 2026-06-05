# 부식 세그멘테이션 실험 결과 비교 보고서

## 1. 실험 개요

본 보고서는 교량 부식 상태 의미론적 분할(Semantic Segmentation) 모델에 대해 두 가지 학습 설정의 성능을 비교한다.

| 항목 | 실험 A (4-class) | 실험 B (Binary) |
|---|---|---|
| 모델 | DeepLabV3+ / ResNet-50 | DeepLabV3+ / ResNet-50 |
| 클래스 수 | 4 (Good / Fair / Poor / Severe) | 2 (Good / Corrosion) |
| 에포크 | 40 | 40 |
| 배치 크기 | 2 | 2 |
| Output stride | 8 | 8 |
| 손실 함수 | Cross-Entropy | Cross-Entropy |
| 클래스 가중치 | 0.1 / 0.3 / 0.3 / 0.3 | 0.15 / 0.85 |
| 데이터셋 | unified\_corrosion | unified\_corrosion\_binary |
| 입력 해상도 | 512 × 512 | 512 × 512 |

이진 분류 실험은 Fair / Poor / Severe 세 클래스를 하나의 **Corrosion** 클래스로 통합하여, 부식 유무 자체에 집중하는 설정이다.

> 평가 지표는 최종 저장 모델(`weights.pt`)을 테스트셋 전체(총 픽셀 14,680,064개)에 적용하여 혼동 행렬(Confusion Matrix)로부터 직접 계산하였다.

---

## 2. 실험 A — 4-class 세그멘테이션 결과

| Class | Precision | Recall | F1 | IoU | FPR | Support (pixels) |
|---|---|---|---|---|---|---|
| Good | 0.9383 | 0.9593 | 0.9487 | 0.9023 | 0.2036 | 11,206,383 |
| Fair | 0.6248 | 0.5629 | 0.5922 | 0.4207 | 0.0451 | 1,728,971 |
| Poor | 0.6643 | 0.6913 | 0.6775 | 0.5123 | 0.0349 | 1,334,125 |
| Severe | 0.8323 | 0.5605 | 0.6699 | 0.5036 | 0.0033 | 410,585 |
| **Weighted Avg** | **0.8735** | **0.8771** | **0.8742** | **0.7990** | **0.1640** | 14,680,064 |

**클래스별 관찰:**
- **Good**: 샘플이 압도적으로 많아(76.3%) 모든 지표가 높으나, FPR이 0.20으로 다른 클래스에서 Good으로 잘못 분류되는 비율이 상대적으로 높다.
- **Fair**: Recall 0.56으로 가장 낮음. 경계 부식 특성상 Good/Poor와 혼동이 많아 검출 누락이 빈번하다.
- **Poor**: Fair와 외관이 유사하여 Precision/Recall 모두 0.66~0.69 수준.
- **Severe**: Precision은 0.83으로 높지만 Recall이 0.56으로 낮아, 심각 부식의 상당 부분이 다른 클래스로 분류된다.

---

## 3. 실험 B — Binary 세그멘테이션 결과

| Class | Precision | Recall | F1 | IoU | FPR | Support (pixels) |
|---|---|---|---|---|---|---|
| Good | 0.9565 | 0.9492 | 0.9529 | 0.9100 | 0.1393 | 11,206,383 |
| Corrosion | 0.8402 | 0.8607 | 0.8503 | 0.7396 | 0.0508 | 3,473,681 |
| **Weighted Avg** | **0.9290** | **0.9283** | **0.9286** | **0.8696** | **0.1184** | 14,680,064 |

**클래스별 관찰:**
- **Good**: Precision과 Recall이 모두 0.95 수준으로 균형 잡힌 성능. FPR도 4-class 대비 0.21 → 0.14로 크게 감소.
- **Corrosion**: Fair/Poor/Severe 통합으로 Recall이 0.86까지 향상. 부식 픽셀의 검출 누락이 대폭 줄었다.

---

## 4. 두 실험 비교 (Weighted Average 기준)

| 지표 | 실험 A (4-class) | 실험 B (Binary) | 향상 |
|---|---|---|---|
| Precision | 0.8735 | **0.9290** | **+5.55%p** |
| Recall | 0.8771 | **0.9283** | **+5.12%p** |
| F1-score | 0.8742 | **0.9286** | **+5.44%p** |
| IoU (Jaccard) | 0.7990 | **0.8696** | **+7.06%p** |
| FPR | 0.1640 | **0.1184** | **−4.56%p** ↓ |

모든 지표에서 Binary 모델이 우세하며, 특히 IoU(+7.06%p)와 FPR 감소(-4.56%p)가 두드러진다.

---

## 5. 과적합(Overfitting) 분석

학습 로그 기반 최종 에포크(Epoch 40) 지표:

| 항목 | 실험 A (4-class) | 실험 B (Binary) |
|---|---|---|
| Train F1 (Epoch 40) | 98.17% | 94.68% |
| Test F1 (Epoch 40 로그) | 85.94% | 89.37% |
| **Train–Test 격차** | **12.23%p** | **5.31%p** |
| Test Loss (Epoch 40) | 0.883 | 0.275 |

- **4-class 모델**은 Train F1이 98%를 넘지만 Test F1이 86% 수준으로, Train–Test 격차가 약 **12%p**에 달한다.
- **Binary 모델**은 Train–Test 격차가 약 **5%p**로 감소하여 일반화 성능이 크게 향상되었다.
- Test Loss 역시 0.883 → 0.275로 대폭 낮아져 모델의 불확실성이 감소하였다.

---

## 6. 해석 및 논의

### 성능 향상 원인
1. **결정 경계 단순화**: Fair/Poor/Severe 간 경계는 시각적으로 유사하여 4-class 모델이 혼동하기 쉽다. 이진 분류로 통합하면 결정 경계가 명확해진다.
2. **클래스 불균형 완화**: 세 클래스를 합치면 Corrosion 샘플(3,473,681 픽셀, 23.7%)이 충분히 확보되어 학습이 안정화된다. 4-class에서 Severe는 전체 픽셀의 2.8%에 불과했다.
3. **FPR 감소**: 부식 클래스가 단일화되어, 모델이 부식 픽셀을 Good으로 잘못 분류하는 오류가 감소하였다.

### 한계 및 주의사항
- Binary 모델은 **부식의 심각도(Fair/Poor/Severe)를 구분하지 않는다**. 교량 유지보수 계획 수립에는 등급 정보가 필수적이므로, 심각도 세분화가 필요한 경우 4-class 모델을 병행 사용해야 한다.
- Binary 모델의 높은 지표는 "부식 존재 여부 검출"이라는 더 쉬운 태스크에 특화된 결과이므로, 조건 상태 평가(Condition State Assessment) 기준으로 직접 비교할 때는 해석에 주의가 필요하다.
- 원저자(Bianchi et al.) 보고 성능(F1 86.67%)은 원본 데이터셋 기준이므로, 본 실험(통합 데이터셋 사용)과의 직접 비교 시 주의가 필요하다.

---

## 7. 결론

| 목적 | 권장 모델 |
|---|---|
| 부식 유무 스크리닝 | **Binary 모델** (F1 92.9%, IoU 87.0%, FPR 11.8%) |
| 부식 심각도 세분화 | **4-class 모델** (F1 87.4%, IoU 79.9%) |
| 일반화 성능 우선 | **Binary 모델** (Train–Test 격차 5.3%p) |

Binary 분류 실험을 통해 Weighted 기준 F1 **+5.44%p**, IoU **+7.06%p**, FPR **−4.56%p**의 성능 향상이 확인되었다. 두 모델을 상호 보완적으로 활용하는 2단계 파이프라인(Binary → 4-class)을 향후 연구 방향으로 제안한다.

---

## 부록: 실험 환경

| 항목 | 내용 |
|---|---|
| Framework | PyTorch 2.x, CUDA 13.2 |
| Backbone | ResNet-50 (ImageNet pretrained) |
| Decoder | DeepLabV3+ |
| Optimizer | Adam |
| 데이터 증강 | HorizontalFlip, Affine (rotate ±0.1 rad, translate ±2.5%, shear ±2.5%, scale 97.5–102.5%) |
| 평가 방식 | Test set 전체 픽셀 혼동 행렬 기반 (14,680,064 pixels) |
| 학습 로그 | `stored_weights/finetune_wce_merged_v1/log_3.csv` (4-class) |
|  | `stored_weights/finetune_binary_v1/log_3.csv` (Binary) |
| 평가 스크립트 | `training/evaluate_results.py` (4-class) |
|  | `training/evaluate_binary.py` (Binary) |
