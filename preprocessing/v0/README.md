# preprocessing/v0/ — Baseline (identity)

**역할**: 입력 이미지·라벨을 변형 없이 그대로 반환하는 baseline.

## 구현해야 할 파일

- `preprocess.py`

## 클래스 명세

```python
class V0Preprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        ...

    def apply(self, image, label=None):
        # 그대로 반환
        return image, label

    def is_train_only(self) -> bool:
        return False
```

## 비고

- 어떤 변형도 하지 않는다
- 학습/검증 모두에 동일하게 적용 (`is_train_only=False`)
- V0의 결과는 다른 V와의 비교 baseline으로 사용된다
