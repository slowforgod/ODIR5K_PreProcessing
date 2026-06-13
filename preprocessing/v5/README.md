# preprocessing/v5/ — V1 + V3 (CLAHE + Manifold Mixup)

**역할**: 이미지 단계에서는 CLAHE만, Mixup은 학습 루프에서 별도로 적용.

## 구현해야 할 파일

- `preprocess.py`

## 핵심 포인트

- V3(Manifold Mixup)은 dataset이 아니라 **학습 루프(`train/mixup_trainer.py`)에서 호출**된다
- 따라서 V5의 dataset 측 변환은 **V1과 동일**
- Mixup 적용은 trainer가 V3Preprocessor를 직접 인스턴스화해서 처리

## 클래스 명세

```python
from preprocessing.v1.clahe import V1Preprocessor

class V5Preprocessor(BasePreprocessor):
    def __init__(self, v1_kwargs: dict):
        self.v1 = V1Preprocessor(**v1_kwargs)

    def apply(self, image, label=None):
        return self.v1.apply(image, label)

    def is_train_only(self) -> bool:
        return False  # CLAHE는 검증에도 적용
```

## ⚠ 주의

- `is_train_only=False` — CLAHE는 검증에서도 적용되어야 학습/검증 분포가 맞는다
- V3 호출은 trainer 책임. V5Preprocessor가 알 필요 없음.
- 학습 entrypoint는 `train/train_v5.py` (V3 적용 위해 `MixupTrainer` 사용)

## config

`configs_ResNet/v5.yaml`에는:
- `preprocessing:` 아래 V1 파라미터 (`clip_limit`, `tile_grid_size`)
- 별도로 V3 파라미터 (`mixup_alpha`, `mixup_layers`) — trainer가 읽음
