# model/

이 폴더의 모든 모델 파일이 지켜야 할 **공통 인터페이스 계약**.
백본별 구현 상세(ResNet-50 / DenseNet / EfficientNet)는 각 `.py` 파일 상단 docstring에 기술한다.

---

## 1. 팩토리 함수 — `build_model`

`train/`은 항상 이 함수를 통해 모델을 생성한다. 직접 클래스를 import하지 않는다.

```python
# model/base.py
def build_model(name       : str,
                num_classes: int   = 7,
                pretrained : bool  = True,
                dropout    : float = 0.5) -> nn.Module:
    """
    Args:
        name        : "resnet50" | "densenet" | "efficientnet"
        num_classes : 7  (N, D, G, C, A, H, M)
        pretrained  : torchvision ImageNet 가중치 사용 여부
        dropout     : 마지막 FC 직전 Dropout 비율

    Returns:
        nn.Module. forward 시그니처는 아래 섹션 2 참고.
    """
```

---

## 2. forward 공통 시그니처

모든 모델은 아래 시그니처를 그대로 구현한다.
`train/`과 `analysis/metrics.py`가 이 형태로 모델을 호출한다.

```python
def forward(self,
            x            : torch.Tensor,        # (B, 3, H, W)
            y            : torch.Tensor = None,  # (B, 7) multi-hot label
            alpha        : float        = 0.0,
            mixup_layers : list         = None
           ) -> torch.Tensor | tuple:
    """
    y=None (일반 forward):
        반환 → logits  shape (B, 7)

    y가 주어진 경우 (Manifold Mixup):
        반환 → (logits, label_a, label_b, lam)
               logits  : (B, 7)
               label_a : (B, 7)  원본 label
               label_b : (B, 7)  셔플된 label
               lam     : float
    """
```

### 반환값 타입 요약

| 호출 방식 | 반환 |
|-----------|------|
| `model(x)` | `logits` — `(B, 7)` Tensor |
| `model(x, y=labels, alpha=a, mixup_layers=layers)` | `(logits, label_a, label_b, lam)` |

---

## 3. 출력 계약

- **sigmoid를 forward 내부에서 붙이지 않는다.**
  학습 시 `BCEWithLogitsLoss`는 logits를 직접 받아야 한다.
- 추론·평가 시에는 호출자(`train/`, `analysis/`)가 `torch.sigmoid(logits)`를 직접 호출한다.

---

## 4. 아키텍처 공통 규칙

| 항목 | 규칙 |
|------|------|
| 사전학습 | torchvision pretrained 가중치 사용 (`pretrained=True` 기본) |
| 분류 헤드 | 원본 마지막 FC → `nn.Dropout(dropout)` + `nn.Linear(?, 7)` 로 교체 |
| Dropout | FC 바로 **앞**에 1개만 |
| 출력 크기 | 항상 `(B, 7)` |
| 클래스 순서 | `[N, D, G, C, A, H, M]` 고정 — `configs/`의 `class_names`와 동일 순서 |

---

## 5. `preprocessing/`과의 연결 계약

Manifold Mixup(`y` 인자가 주어진 경우) 사용 시, 모델 forward 내부에서
`preprocessing/v3/` 의 `V3Preprocessor.apply(feat, label, alpha)`를 호출한다.

`V3Preprocessor.apply()`의 반환:
```
(mixed_feat, label_a, label_b, lam)
```
을 받아서 이후 레이어를 통과시키고, 최종 `(logits, label_a, label_b, lam)`을 반환한다.

> V3 시그니처 상세는 `preprocessing/v3/README.md` 참고.
> Manifold Mixup을 지원하지 않는 백본은 `y`가 주어지면 무시하고 logits만 반환해도 된다.
