# preprocessing/v6/ — V1 + V2 + V3 (CLAHE + Augmentation + Manifold Mixup)

**역할**: 이미지 단계에서 CLAHE → augmentation, 그리고 학습 루프에서 Mixup.

## 구현해야 할 파일

- `preprocess.py`

## 알고리즘

1. **dataset 단계**: V1 → V2 (V4와 동일)
2. **학습 루프 단계**: V3 (Manifold Mixup) — trainer가 호출

## 클래스 명세

```python
from preprocessing.v1.clahe import V1Preprocessor
from preprocessing.v2.augmentation import V2Preprocessor

class V6Preprocessor(BasePreprocessor):
    def __init__(self, v1_kwargs: dict, v2_kwargs: dict):
        self.v1 = V1Preprocessor(**v1_kwargs)
        self.v2 = V2Preprocessor(**v2_kwargs)

    def apply(self, image, label=None):
        image, label = self.v1.apply(image, label)
        image, label = self.v2.apply(image, label)
        return image, label

    def is_train_only(self) -> bool:
        return True  # V2 포함이므로
```

## 학습 루프

- entrypoint: `train/train_v6.py`
- trainer: `MixupTrainer` (V3 적용)
- V3는 trainer가 직접 인스턴스화하므로 V6Preprocessor에서는 다루지 않는다

## ⚠ 주의

- `is_train_only=True` — V2가 포함되어 학습 전용
- 검증 시에는 augmentation 없이 CLAHE만 적용해야 하므로, dataset 구성 시
  train/val에 다른 preprocessor를 붙이는 로직을 trainer/dataset이 처리

## config

`configs/v6.yaml`:
- `preprocessing.v1`: clip_limit, tile_grid_size
- `preprocessing.v2`: augmentation 확률들
- `preprocessing.v3`: mixup_alpha, mixup_layers
