# preprocessing/v1/ — CLAHE

**역할**: LAB 색공간에서 L 채널에만 CLAHE를 적용해 안저 영상의 국소 대비를 강화.

**담당자**: TBD

## 구현해야 할 파일

- `clahe.py`

## 알고리즘

1. `cv2.cvtColor(image, cv2.COLOR_RGB2LAB)` — RGB → LAB
2. L 채널 분리
3. `cv2.createCLAHE(clipLimit, tileGridSize)`로 L 채널에만 CLAHE 적용
4. L 채널 재결합
5. `cv2.cvtColor(..., cv2.COLOR_LAB2RGB)` — LAB → RGB로 복원

A, B 채널은 건드리지 말 것 (색 왜곡 방지).

## 권장 파라미터

- `clipLimit = 2.0`
- `tileGridSize = (8, 8)`

두 값은 모두 **config(`configs/v1.yaml`)에서 읽어오도록** 만들 것.

## 클래스 명세

```python
class V1Preprocessor(BasePreprocessor):
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        ...

    def apply(self, image, label=None):
        # image: (H, W, 3) RGB uint8
        # → CLAHE 적용된 이미지 반환
        return clahe_image, label

    def is_train_only(self) -> bool:
        return False  # 학습/검증 모두 적용
```

## ⚠ 인터페이스 안정성

V1의 결과는 **V4 / V5 / V6에서 재사용**된다.
- 클래스명 `V1Preprocessor` 고정
- `apply(image, label) -> (image, label)` 시그니처 고정
- `__init__` 파라미터는 config 키와 1:1 매칭

마음대로 바꾸면 V4/V5/V6가 깨진다. 변경이 필요하면 팀에 먼저 공유할 것.

## 입력·출력

- 입력 image: (H, W, 3) **RGB uint8** numpy array
- 출력 image: (H, W, 3) **RGB uint8** numpy array (dtype 유지)
- label: 그대로 통과
