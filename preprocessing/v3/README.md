# preprocessing/v3/ — Manifold Mixup

**역할**: 모델 forward 도중 feature를 두 샘플끼리 섞어 정규화 효과를 주는 기법.
이미지 변형이 아니라 **feature 레벨**에서 작동한다.

**담당자**: TBD

## 구현해야 할 파일

- `manifold_mixup.py`

## 다른 V와 다른 점

- dataset 단계가 아니라 **학습 루프(`train/mixup_trainer.py`)에서 호출**됨
- 시그니처가 다른 V와 다름 — feature·label·alpha를 입력으로 받음

## 클래스 명세

```python
class V3Preprocessor:
    """
    NOTE: BasePreprocessor와 시그니처가 다름.
    학습 루프에서 직접 호출되며, 모델 forward 중간에 끼어든다.
    """

    def __init__(self, alpha: float = 0.2,
                 mixup_layers: list = [0, 1, 2, 3]):
        self.alpha = alpha
        self.mixup_layers = mixup_layers

    def apply(self, feat: torch.Tensor,
              label: torch.Tensor,
              alpha: float):
        """
        Args:
            feat:  (B, ...) tensor — 섞을 feature
            label: (B, num_classes) multi-hot tensor
            alpha: Beta(alpha, alpha) 파라미터

        Returns:
            mixed_feat: (B, ...) — lam * feat + (1-lam) * feat[perm]
            label_a:    원본 label
            label_b:    label[perm]
            lam:        float — Beta(alpha, alpha)에서 샘플링
        """
        ...
```

## 알고리즘

1. `lam = np.random.beta(alpha, alpha)` (alpha 기본 0.2)
2. `perm = torch.randperm(batch_size)`
3. `mixed_feat = lam * feat + (1 - lam) * feat[perm]`
4. `label_a, label_b = label, label[perm]` (라벨은 섞지 않고 두 개를 함께 반환)
5. 반환: `(mixed_feat, label_a, label_b, lam)`

## mixup_layers 선택 규칙

- 후보: `[0, 1, 2, 3]`
  - `0`: pixel-level (입력 이미지 자체에서 mixup)
  - `1`: ResNet layer1 직후
  - `2`: ResNet layer2 직후
  - `3`: ResNet layer3 직후
- **epoch마다** `mixup_layers`에서 무작위 1개 선택
- 선택된 레이어 위치에서 한 번만 mixup 호출

## 모델 forward와의 합의

`model/`의 forward는 아래 형태로 호출 가능해야 한다:

```python
forward(x, y=None, alpha=0.2, mixup_layers=[0, 1, 2, 3])
```

`y=None`이면 일반 forward (mixup 없음).
`y`가 주어지면 forward 중간에 V3Preprocessor.apply()를 호출하여 mixup 수행.

자세한 모델 인터페이스는 `model/README.md` 참고.

## 손실 계산

학습 루프에서:

```python
loss = lam * BCE(logits, label_a) + (1 - lam) * BCE(logits, label_b)
```

간단히 하려면 `mixed_label = lam*label_a + (1-lam)*label_b`로 만들어
`BCE(logits, mixed_label)` 한 번 계산해도 된다 (BCE는 라벨에 대해 선형이므로
수학적으로 동일).

## ⚠ 인터페이스 안정성

V3는 V5/V6에서 재사용된다. 클래스명·시그니처를 임의로 바꾸지 말 것.
