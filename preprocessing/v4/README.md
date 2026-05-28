# preprocessing/v4/ — V1 + V2 (CLAHE + Augmentation)

**역할**: CLAHE 적용 후 데이터 증강.

## 구현해야 할 파일

- `preprocess.py`

## 알고리즘

1. V1Preprocessor (CLAHE) 적용
2. V2Preprocessor (augmentation) 적용

순서 고정. CLAHE는 결정적(deterministic) 연산이고, augmentation은 확률적이므로
CLAHE를 먼저 거는 것이 자연스럽다.

## 클래스 명세

```python
from preprocessing.v1.clahe import V1Preprocessor
from preprocessing.v2.augmentation import V2Preprocessor

class V4Preprocessor(BasePreprocessor):
    def __init__(self, v1_kwargs: dict, v2_kwargs: dict):
        self.v1 = V1Preprocessor(**v1_kwargs)
        self.v2 = V2Preprocessor(**v2_kwargs)

    def apply(self, image, label=None):
        image, label = self.v1.apply(image, label)
        image, label = self.v2.apply(image, label)
        return image, label

    def is_train_only(self) -> bool:
        return True  # V2가 학습 전용이므로 V4도 학습 전용
```

## 자동 동작 원칙

V1·V2 본체가 완성되면 V4는 **import 기반 조합**으로 자동 동작한다.
독자적으로 새 변환을 추가하지 말 것.

## config

`configs/v4.yaml`의 `preprocessing:` 섹션에는 V1·V2 두 묶음의 파라미터가 들어간다.
예시는 `configs/README.md` 참고.
