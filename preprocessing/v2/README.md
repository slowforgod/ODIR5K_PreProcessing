# preprocessing/v2/ — Data Augmentation + LSE Oversampling + Loss 비교

**역할**: 학습 시 on-the-fly random augmentation과 LSE(Label-Space Expansion) 오버샘플링을 결합하여 클래스 불균형을 보정하고, 5가지 loss function을 순차 비교한다.

**담당자**: TBD

---

## 구현 파일

| 파일 | 역할 |
|------|------|
| `augmentation.py` | `V2Preprocessor` — albumentations 기반 학습 전용 augmentation 파이프라인 |
| `oversampling.py` | `expand_with_lse` — 소수 클래스 greedy 오버샘플링 |
| `losses.py` | 5가지 loss 구현 및 `make_criterion` 팩토리 |

---

## 1. Augmentation (`augmentation.py`)

학습 시 DataLoader가 batch를 생성할 때마다 random transform을 적용한다(on-the-fly). 검증 시에는 적용하지 않는다.

### 파이프라인 순서

```
Resize(224, 224)
→ HorizontalFlip(p=0.5)
→ VerticalFlip(p=0.1)           ← 안저 특성상 낮은 확률 유지
→ OneOf([Rotate, Affine, RandomResizedCrop], p=0.8)
→ ColorJitter(p=0.3)            ← OneOf 외부 — 독립 적용
→ GaussianBlur(p=0.2)           ← OneOf 외부 — 독립 적용
```

### 파라미터 기본값 (configs/v2.yaml에서 주입)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `hflip_p` | 0.5 | HorizontalFlip 확률 |
| `vflip_p` | 0.1 | VerticalFlip 확률 |
| `geo_p` | 0.8 | 기하 변환 그룹(OneOf) 확률 |
| `rotate_limit` | 20 | Rotate 최대 각도 (±°) |
| `affine_scale` | (0.85, 1.15) | Affine scale 범위 |
| `affine_translate` | 0.07 | Affine translate_percent |
| `affine_rotate` | 15 | Affine 최대 회전 (±°) |
| `affine_shear` | 5 | Affine shear — x축만, y=0 고정 |
| `crop_scale` | (0.82, 1.0) | RandomResizedCrop scale 범위 |
| `color_jitter_p` | 0.3 | ColorJitter 확률 |
| `brightness` | 0.1 | ColorJitter brightness 강도 |
| `contrast` | 0.1 | ColorJitter contrast 강도 |
| `saturation` | 0.05 | ColorJitter saturation 강도 |
| `blur_p` | 0.2 | GaussianBlur 확률 |
| `blur_limit` | (3, 5) | GaussianBlur kernel size 범위 |

### 설계 원칙

- **hue=0 고정**: 안저 이미지에서 색상(hue)은 병변 구분의 핵심 신호 — 변형 금지
- **Affine y-shear=0**: fundus disc·vessel 구조를 수직으로 왜곡하면 병리 특징이 손상됨
- **VerticalFlip p=0.1**: 위아래 반전은 임상적으로 비자연적이나 소량은 정규화에 기여

### 클래스 명세

```python
class V2Preprocessor:
    def __init__(self, img_size=224, hflip_p=0.5, vflip_p=0.1, ...): ...

    def apply(self, image: np.ndarray, label=None) -> tuple:
        # image: (H, W, 3) RGB uint8
        # 변환 후 동일 dtype·shape 반환; label은 그대로 통과
        ...

    def is_train_only(self) -> bool:
        return True  # 검증 시 적용 안 함
```

---

## 2. LSE 오버샘플링 (`oversampling.py`)

학습 데이터에서 소수 클래스의 양성 샘플 수를 `target_count`까지 증가시키는 **greedy per-class 오버샘플링**.

### 알고리즘

1. 전체 원본 row를 records 리스트에 추가 (aug_idx=0)
2. 클래스 c별로 양성 count < target_count인 경우:
   - c가 양성인 source row를 추출 → 무작위 셔플
   - source row를 순환하며 augmented copy 추가 (aug_idx ≥ 1)
   - 각 source row의 복제 수는 `max_aug_per_source`로 제한
   - multi-label: copy 추가 시 해당 row의 **모든 양성 클래스** count를 동시 증가 (cross-class bookkeeping)
3. 검증셋은 일절 손대지 않음

### 핵심 파라미터 (configs/v2.yaml)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `lse_custom_target_count` | 850 | 클래스별 목표 양성 샘플 수 |
| `lse_max_aug_per_source` | 12 | source row당 최대 augmented copy 수 |

### CB weight와의 관계 — 중요

CB(Class-Balanced) weight는 반드시 **LSE 적용 전 원본 train_df**에서 계산해야 한다:

```python
# train_v2.py 핵심 로직
CLASS_WEIGHT_REFERENCE_DF = train_df.copy()          # LSE 전 원본
cb_weights = effective_number_class_weights(
    CLASS_WEIGHT_REFERENCE_DF[class_cols].values, ...
)
expanded_records = expand_with_lse(train_df, ...)    # LSE는 이후에 적용
```

LSE 후 분포로 CB weight를 계산하면 오버샘플링이 되돌려져 의미가 없어진다.

---

## 3. Loss 함수 (`losses.py`)

5가지 loss를 순차적으로 비교한다. loss별로 동일한 ImageNet pretrained ResNet-50을 fresh start한다.

### Loss 목록

| 이름 | 설명 | 주요 하이퍼파라미터 |
|------|------|---------------------|
| `bce_with_logits_loss` | `nn.BCEWithLogitsLoss()` baseline | — |
| `focal_loss` | Multi-label Focal Loss (Lin et al. 2017) | alpha=0.75, gamma=2.0 |
| `robust_asymmetric_loss` | ASL + Hill polynomial 변형 (Ben-Baruch et al. 2020) | gamma_neg=2.0, gamma_pos=1.0, clip=0.05, neg_terms=2 |
| `cb_bce_with_logits_loss` | CB-weighted BCE (Cui et al. 2019) | beta=0.9999, normalize="mean_one" |
| `cb_focal_loss` | CB-weighted Focal | alpha=0.75, gamma=2.0 + CB weight |

### 예상 CB 가중치 (sanity check)

```
N≈0.62  D≈0.62  G≈1.35  C≈1.27  A≈1.54  M≈1.55  H≈2.05
```

희귀 클래스(H, M, A)일수록 높은 가중치 → 소수 클래스 학습 신호 증폭.

### Focal Loss 수식

```
α_t = α·y + (1-α)·(1-y)
p_t = p·y + (1-p)·(1-y)        (p = sigmoid(logits))
loss = -α_t · (1 - p_t)^γ · log(p_t)
```

### Robust Asymmetric Loss — Negative 처리

```
p_neg = clamp(1 - p + clip, max=1)
loss_neg = (1-y) · ((1 - p_neg)^neg_terms) / neg_terms   ← Hill polynomial
```
표준 log 대신 다항식으로 근사 → 레이블 노이즈에 강건한 gradient bound.

### make_criterion 팩토리

```python
from preprocessing.v2.losses import make_criterion

criterion = make_criterion(
    "cb_bce_with_logits_loss",
    cb_weights=cb_weights,   # (C,) float32 tensor, pre-LSE 계산
    params=cfg["losses"],    # yaml에서 읽은 dict
).to(device)
```

---

## 4. 학습 설정 요약 (`configs/v2.yaml`)

| 항목 | 값 |
|------|-----|
| 모델 | ResNet-50, ImageNet pretrained, dropout=0.5 |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) — AdamW 아님 |
| Scheduler | CosineAnnealingLR (T_max=num_epochs) |
| Epochs | 20 (patience=8, early stop 기준: val_macro_auc) |
| Batch size | 32 |
| AMP | CUDA 사용 시 활성화 |
| TTA (검증) | Horizontal flip 평균 (프로브 2회) |
| Threshold | Per-class 최적화 (val F1 최대화, `thresholds.npy` 저장) |
| 출력 | `analysis/v2_analysis/{loss_name}/` |

---

## 5. 실험 결과 (`analysis/v2_analysis/v2_resnet50_seed42.json`)

| Loss | Macro AUC | Macro F1 | Kappa | Best Epoch |
|------|-----------|----------|-------|------------|
| bce_with_logits_loss | 0.8582 | 0.6171 | 0.5114 | 4 |
| focal_loss | **0.8639** | 0.6282 | 0.5320 | 4 |
| robust_asymmetric_loss | 0.8597 | 0.6117 | 0.5142 | 4 |
| cb_bce_with_logits_loss | 0.8634 | **0.6432** | **0.5524** | 7 |
| cb_focal_loss | 0.8621 | 0.6179 | 0.5259 | 12 |

- **AUC 최고**: focal_loss (0.8639)
- **F1·Kappa 최고**: cb_bce_with_logits_loss (F1=0.6432, Kappa=0.5524)
- CB 계열은 H·A 등 소수 클래스의 sensitivity를 끌어올려 F1 개선

---

## ⚠ 인터페이스 안정성

V2는 V4·V6에서 재사용된다.

- 클래스명 `V2Preprocessor` 고정
- `apply(image, label) -> (image, label)` 시그니처 고정
- `is_train_only()` 반환값 `True` 고정 (검증에 적용하면 평가 오염)
- `make_criterion` 팩토리의 5가지 loss 이름 고정

## 입력·출력 계약

| 항목 | 규칙 |
|------|------|
| 입력 image dtype | `np.uint8`, shape `(H, W, 3)`, **RGB** |
| 출력 image dtype | `np.uint8`, shape `(H, W, 3)`, **RGB** |
| label | `apply()` 내부에서 변형 금지. 받은 그대로 반환. |
| `is_train_only()` | `True` — 검증/추론에 적용 금지 |
