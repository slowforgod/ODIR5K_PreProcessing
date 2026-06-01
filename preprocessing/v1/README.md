# preprocessing/v1/ — Multi-Variant CLAHE + Per-Class Routing

**역할**: LAB 색공간에서 선택된 채널에 CLAHE를 적용하는 5가지 Variant를 독립 학습한 뒤, 클래스별 라우팅으로 확률을 결합하여 최적 조합을 탐색한다.

**담당자**: TBD

---

## 구현 파일

| 파일 | 역할 |
|------|------|
| `clahe.py` | `V1Preprocessor` — 5가지 CLAHE variant 선택 가능 |
| `routing.py` | `hard_route`, `soft_route_uniform` 및 기본 라우팅 맵 |

---

## 1. CLAHE Variant (`clahe.py`)

### 5가지 Variant

| Variant | CLAHE 적용 채널 | 설명 |
|---------|----------------|------|
| `V0` | 없음 | Identity passthrough — 기준선(baseline) |
| `V1_L` | L | 명도 채널만 (기존 V1) |
| `V1_LA` | L + A | 명도 + 녹적 대비 채널 |
| `V1_LB` | L + B | 명도 + 황청 대비 채널 |
| `V1_LAB` | L + A + B | 전체 3채널 |

### 알고리즘 (V0 제외)

```
RGB → LAB (cv2.COLOR_RGB2LAB)
  → 지정 채널(L / A / B)에 cv2.CLAHE.apply()
LAB → RGB (cv2.COLOR_LAB2RGB)
```

- CLAHE 객체(`cv2.createCLAHE`)는 `__init__`에서 **한 번만** 생성하고 `apply()` 호출 시 재사용 (per-image 생성 금지 — 효율성)
- A·B 채널 CLAHE는 색 대비를 강화해 혈관·출혈 영역의 경계를 뚜렷하게 만들 수 있으나, 과도하면 색 왜곡 위험이 있어 실험으로 검증

### CLAHE 파라미터 (configs/v1.yaml)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `clip_limit` | 2.0 | 히스토그램 클리핑 한계 — 높을수록 대비 강조, 노이즈 위험 |
| `tile_grid_size` | (8, 8) | 타일 분할 크기 — 지역 대비 적응 범위 |

### `is_train_only()` = False

CLAHE는 학습과 **검증 모두** 적용한다. V1의 핵심 가정은 "이미지 전처리 변형 자체의 효과를 평가"이므로, 검증에도 동일하게 적용해야 일관된 비교가 가능하다.

### 클래스 명세

```python
class V1Preprocessor:
    def __init__(
        self,
        variant: str = "V1_L",        # "V0" | "V1_L" | "V1_LA" | "V1_LB" | "V1_LAB"
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ): ...

    def apply(self, image: np.ndarray, label=None) -> tuple:
        # image: (H, W, 3) RGB uint8 → 동일 dtype·shape 반환
        # label: 변형 없이 그대로 반환
        ...

    def is_train_only(self) -> bool:
        return False  # 학습·검증 양쪽 적용
```

### 입력·출력 계약

| 항목 | 규칙 |
|------|------|
| 입력 image dtype | `np.uint8`, shape `(H, W, 3)`, **RGB** |
| 출력 image dtype | `np.uint8`, shape `(H, W, 3)`, **RGB** |
| label | `apply()` 내부에서 변형 금지. 받은 그대로 반환. |
| V0 | image를 copy 없이 그대로 반환 |

---

## 2. 학습 전략

5가지 Variant를 각각 **독립적인 ResNet-50**으로 순차 학습한다. 라벨이 동일하므로 `WeightedRandomSampler` 가중치와 `pos_weight`는 첫 번째 Variant 학습 시 **한 번만 계산**하고 이후 재사용한다.

| 항목 | 값 (configs/v1.yaml) |
|------|----------------------|
| 모델 | ResNet-50, ImageNet pretrained, dropout=0.5 |
| 옵티마이저 | Adam (lr=1e-4, weight_decay=1e-4) |
| 스케줄러 | CosineAnnealingLR |
| Epochs | 20 (patience=8) |
| 클래스 불균형 보정 | `WeightedRandomSampler` + `BCEWithLogitsLoss(pos_weight=...)` |
| 검증 TTA | Horizontal flip 평균 (2회 forward) |
| Threshold | Per-class 최적화 (val F1 최대화) |

---

## 3. 라우팅 전략 (`routing.py`)

5개 Variant를 각각 독립 학습한 뒤 validation set에서 얻은 확률(`probs_by_variant`)을 두 가지 방식으로 결합한다. 결합·평가는 `train/train_v1.py`에서 수행한다.

### 3-1. Hard Routing

**각 클래스별로 지정된 단일 Variant의 확률만 사용.**

```python
# 클래스 c의 최종 확률 = routing_map[c] Variant가 예측한 c 클래스 확률
out[:, col] = probs_by_variant[routing_map[cls]][:, col]
```

기본 라우팅 맵 (`DEFAULT_HARD_ROUTING`, configs/v1.yaml의 `routing.hard`에서 오버라이드 가능):

| 클래스 | 배정 Variant | 배정 근거 |
|--------|-------------|-----------|
| N (Normal) | V1_LA | L+A 대비 강화 → 정상 안저 패턴 구별 |
| D (Diabetes) | V1_LB | L+B → 황반 부근 황색 병변 대비 강화 |
| G (Glaucoma) | V0 | 시신경 구조는 원본 색상 정보가 중요 — CLAHE 불필요 |
| C (Cataract) | V1_LAB | 전체 채널 대비 강화 → 흐린 수정체 경계 탐지 |
| A (AMD) | V1_L | 명도 대비만 — 드루젠 경계 탐지 |
| H (Hypertension) | V1_LB | L+B → 혈관 직경 이상 탐지 |
| M (Myopia) | V1_L | 명도 대비 → 맥락막 패턴 강조 |

**특성**: 각 클래스에 도메인 지식으로 선정한 "최적" Variant를 할당. 최종 확률 행렬은 클래스마다 서로 다른 Variant에서 조립되므로 클래스 간 일관성은 없지만, 클래스별 최적화가 가능하다.

```python
from preprocessing.v1.routing import hard_route, DEFAULT_HARD_ROUTING

probs_hard = hard_route(
    probs_by_variant,   # {"V0": (N,7), "V1_L": (N,7), ...}
    class_names,        # ["N", "D", "G", "C", "A", "H", "M"]
    routing_map=DEFAULT_HARD_ROUTING,
)   # → (N, 7)
```

### 3-2. Soft Routing (Uniform Average)

**모든 5개 Variant의 확률을 균등 평균.**

```python
# stack → (5, N, 7), mean over axis=0 → (N, 7)
stacked = np.stack(list(probs_by_variant.values()), axis=0)
probs_soft = stacked.mean(axis=0)
```

**특성**: 도메인 지식 없이 앙상블 효과를 얻는 방식. 어느 Variant도 특별히 신뢰하지 않고 5개 모델의 판단을 고르게 반영한다. 단일 Variant의 오류를 평균화하여 감소시키는 효과가 있다.

```python
from preprocessing.v1.routing import soft_route_uniform

probs_soft = soft_route_uniform(
    probs_by_variant,
    class_names,
)   # → (N, 7)
```

### 두 라우팅 방식 비교

| | Hard Routing | Soft Routing |
|---|---|---|
| 아이디어 | 클래스별 최적 Variant 선택 | 모든 Variant 균등 앙상블 |
| 도메인 지식 필요 | 있음 (라우팅 맵) | 없음 |
| 클래스 간 일관성 | 없음 (클래스별 다른 source) | 있음 (동일 평균) |
| 오류 분산 | 배정된 Variant의 오류 그대로 | 5개 평균으로 감소 |

---

## 4. 실험 결과 (`analysis/v1_analysis/v1_resnet50_seed42.json`)

### 4-1. Per-Variant 결과

| Variant | Macro AUC | Macro F1 | Kappa | Best Epoch |
|---------|-----------|----------|-------|------------|
| **V0** | **0.8632** | 0.6120 | 0.5067 | 3 |
| V1_L | 0.8521 | 0.6265 | 0.5263 | 6 |
| V1_LA | 0.8560 | 0.6193 | 0.5119 | 3 |
| **V1_LB** | **0.8613** | 0.6252 | 0.5317 | 7 |
| V1_LAB | 0.8559 | **0.6392** | **0.5350** | 11 |

- AUC 기준: V0(0.8632) ≥ V1_LB(0.8613) > V1_LA ≈ V1_LAB > V1_L
- F1 기준: V1_LAB(0.6392) > V1_L(0.6265) > V1_LB > V1_LA > V0
- CLAHE가 항상 유리하지 않음 — V0 baseline이 AUC에서 최고

### 4-2. 라우팅 결과

| 라우팅 방식 | Macro AUC | Macro F1 | Kappa |
|-------------|-----------|----------|-------|
| Hard Routing | 0.8584 | 0.6370 | 0.5342 |
| **Soft Routing (Uniform)** | **0.8822** | **0.6512** | **0.5501** |

**Soft Routing이 Hard Routing 대비 AUC +0.024, F1 +0.014 우위.**

Soft Routing의 H 클래스 sensitivity가 0.667로 Hard(0.452)의 1.5배 — 앙상블 효과가 희귀 클래스에서 두드러짐.

### 4-3. 클래스별 AUC — Soft vs Hard

| 클래스 | Hard | Soft | Δ |
|--------|------|------|---|
| N | 0.7421 | **0.7801** | +0.038 |
| D | 0.7876 | **0.8044** | +0.017 |
| G | 0.8922 | **0.9134** | +0.021 |
| C | **0.9373** | 0.9361 | -0.001 |
| A | 0.8712 | **0.8773** | +0.006 |
| H | 0.8217 | **0.8947** | +0.073 |
| M | 0.9566 | **0.9691** | +0.013 |

H(hypertension) 클래스는 Hard 배정(V1_LB)보다 5개 앙상블이 압도적으로 유리 (+0.073).

---

## ⚠ 인터페이스 안정성

V1의 결과는 **V4 / V5 / V6에서 재사용**된다.

- 클래스명 `V1Preprocessor` 고정
- `apply(image, label) -> (image, label)` 시그니처 고정
- `variant="V1_L"`이 기본값 — 기존 단일-variant 코드와 호환
- `DEFAULT_HARD_ROUTING`, `DEFAULT_VARIANTS` 딕셔너리 키 순서 고정

임의로 변경하면 V4·V5·V6 파이프라인이 깨진다.
