# preprocessing/v2/ — Data Augmentation

**역할**: 학습 시에만 적용되는 데이터 증강. albumentations 권장.

**담당자**: TBD

## 구현해야 할 파일

- `augmentation.py`

## 권장 변환 목록

안저 영상에 **안전한** 변형만 사용한다.

| 변환 | 권장값 | 비고 |
|------|--------|------|
| HorizontalFlip | p=0.5 | 좌우 반전 (안전) |
| Rotate | limit=±15° | 너무 큰 회전 금지 |
| ShiftScaleRotate | shift=0.05, scale=0.05, rotate=10 | 약하게 |
| ColorJitter | brightness=0.1, contrast=0.1 | 색상은 보수적 |
| RandomBrightnessContrast | p=0.5 | brightness/contrast ±0.1 |

각 변환의 **확률(p)·강도(limit)는 config에서 받기**.

## ❌ 금지

- **수직 반전(VerticalFlip)** — 안저 영상의 위/아래는 의미가 다름
- **과한 색상 변화** — hue shift, saturation 큰 변화 등 (병변의 색 정보 손상)
- **너무 큰 회전 (>20°)**
- Cutout / CoarseDropout — 병변을 가릴 수 있으므로 보수적으로 사용 (필요 시 작은 영역만)

## 클래스 명세

```python
class V2Preprocessor(BasePreprocessor):
    def __init__(self, **aug_probs):
        # 예: hflip_p=0.5, rotate_limit=15, ...
        # albumentations Compose 빌드
        ...

    def apply(self, image, label=None):
        # image: (H, W, 3) RGB uint8
        # albumentations transform 호출
        return augmented_image, label

    def is_train_only(self) -> bool:
        return True
```

## 입력·출력

- 입력 image: (H, W, 3) **RGB uint8** numpy array
- 출력 image: (H, W, 3) **RGB uint8** numpy array (dtype 유지)
- label: 그대로 통과 (multi-label classification이라 라벨도 그대로)

## ⚠ 인터페이스 안정성

V2는 V4/V6에서 재사용된다.
- 클래스명 `V2Preprocessor` 고정
- 시그니처 `apply(image, label) -> (image, label)` 고정
