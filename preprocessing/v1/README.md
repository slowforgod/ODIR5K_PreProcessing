# preprocessing/v1/ — Multi-Variant CLAHE + Per-Class Routing

**역할**: LAB 색공간에서 선택된 채널에 CLAHE를 적용하는 5가지 변형(Variant)을 제공하고,
클래스별 라우팅을 통해 최적 변형 조합을 탐색한다.

**담당자**: TBD

## 구현 파일

- `clahe.py` — `V1Preprocessor` (5가지 variant 선택 가능)
- `routing.py` — `hard_route`, `soft_route_uniform` 함수 및 기본 라우팅 맵

---

## 5가지 CLAHE Variant

| Variant | CLAHE 적용 채널 |
|---------|----------------|
| `V0`    | NONE (identity passthrough, CLAHE 없음) |
| `V1_L`  | L 채널만 (기존 V1) |
| `V1_LA` | L + A 채널 |
| `V1_LB` | L + B 채널 |
| `V1_LAB`| L + A + B 채널 (전체) |

비-V0 변형: RGB → LAB → 지정 채널에 CLAHE → LAB → RGB → uint8

---

## 알고리즘 (비-V0)

1. `cv2.cvtColor(image, cv2.COLOR_RGB2LAB)` — RGB → LAB
2. 지정된 채널(`L`, `A`, `B` 중 선택)에 CLAHE 적용
3. `cv2.cvtColor(..., cv2.COLOR_LAB2RGB)` — LAB → RGB로 복원

CLAHE 객체(`cv2.createCLAHE(...)`)는 `__init__`에서 **한 번만** 생성하고
`apply()` 호출 시 재사용한다 (per-image 생성 금지 — 효율성).

---

## 권장 파라미터

- `clipLimit = 2.0`
- `tileGridSize = (8, 8)`

두 값은 `configs/v1.yaml`의 `preprocessing` 섹션에서 읽는다.

---

## 클래스 명세 (`clahe.py`)

```python
class V1Preprocessor:
    def __init__(
        self,
        variant: str = "V1_L",       # "V0" | "V1_L" | "V1_LA" | "V1_LB" | "V1_LAB"
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ): ...

    def apply(self, image: np.ndarray, label=None) -> tuple:
        # image: (H, W, 3) RGB uint8
        # label: 변형 없이 그대로 반환
        ...

    def is_train_only(self) -> bool:
        return False  # 학습/검증 모두 적용
```

### BasePreprocessor 인터페이스 계약

| 항목 | 규칙 |
|------|------|
| 입력 `image` dtype | `np.uint8`, shape `(H, W, 3)`, **RGB** |
| 출력 `image` dtype | `np.uint8`, shape `(H, W, 3)`, **RGB** |
| `label` | `apply()` 내부에서 변형 금지. 받은 그대로 반환. |
| `is_train_only()` | `False` — train·val 양쪽 적용 |

> V0 variant는 image를 그대로 반환(copy 없음). 나머지는 uint8 dtype과 HWC RGB 레이아웃을 반드시 유지.

---

## 라우팅 전략 (`routing.py`)

5개 variant를 각각 독립적으로 학습한 후, validation set에서 두 가지 방식으로 확률을 결합한다.
실제 결합 및 평가는 `train/train_v1.py`에서 수행된다.

### Hard Routing (per-class 고정 배정)

각 클래스마다 가장 적합하다고 판단된 variant의 확률만 사용:

| 클래스 | 배정 Variant |
|--------|-------------|
| N      | V1_LA       |
| D      | V1_LB       |
| G      | V0          |
| C      | V1_LAB      |
| A      | V1_L        |
| H      | V1_LB       |
| M      | V1_L        |

```python
prob_hard[:, c] = probs_by_variant[routing_map[c]][:, c]
```

### Soft Routing (uniform average)

모든 5개 variant의 확률을 균등 평균:

```python
prob_soft = np.stack(list(probs_by_variant.values())).mean(axis=0)
```

두 라우팅 모드 각각에 대해 독립적으로 `find_optimal_thresholds` → `compute_metrics`를 수행하고
결과를 JSON에 별도 저장한다.

---

## ⚠ 인터페이스 안정성

V1의 결과는 **V4 / V5 / V6에서 재사용**된다.
- 클래스명 `V1Preprocessor` 고정
- `apply(image, label) -> (image, label)` 시그니처 고정
- `variant="V1_L"`이 기본값이므로 기존 단일-variant 사용 코드와 호환됨

마음대로 바꾸면 V4/V5/V6가 깨진다.
