# preprocessing/

이 폴더의 모든 전처리 클래스가 지켜야 할 **공통 인터페이스 계약**.
각 V의 구현 상세는 해당 `v{N}/README.md`를 볼 것.

---

## 1. 표준 인터페이스 — `BasePreprocessor`

V0, V1, V2, V4, V5, V6의 클래스는 아래 두 메서드를 반드시 구현한다.

```python
class V{N}Preprocessor:

    def apply(self, image: np.ndarray, label: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image : (H, W, 3) RGB uint8 numpy array
            label : (7,) multi-hot float32 numpy array. None일 수 있음.
        Returns:
            (transformed_image, label) — label은 변형 없이 그대로 통과.
        """

    def is_train_only(self) -> bool:
        """
        True  → 학습 DataLoader에만 적용 (augmentation 계열)
        False → 학습·검증 DataLoader 양쪽에 모두 적용 (CLAHE 계열)
        """
```

### I/O 타입 계약

| 항목 | 규칙 |
|------|------|
| 입력 `image` dtype | `np.uint8`, shape `(H, W, 3)`, **RGB** 채널 순서 |
| 출력 `image` dtype | `np.uint8`, shape `(H, W, 3)`, **RGB** 채널 순서 유지 |
| `label` | `apply()` 내부에서 절대 변형하지 않는다. 받은 그대로 반환. |

> OpenCV 함수를 쓸 경우 내부에서 BGR 변환 후 처리하고 반환 전에 반드시 RGB로 복원할 것.

### 클래스 명명 규칙

```
V0Preprocessor, V1Preprocessor, V2Preprocessor,
V4Preprocessor, V5Preprocessor, V6Preprocessor
```

V4/V5/V6는 V1/V2를 **import해 조합**한다.
V1/V2의 클래스명·메서드 시그니처를 임의로 바꾸면 조합 클래스가 깨진다.

---

## 2. V3 예외 — Manifold Mixup

V3(`V3Preprocessor`)는 **이미지가 아닌 모델 feature를 섞는** 기법이므로
dataset 단계가 아니라 **학습 루프**(`train/mixup_trainer.py`)에서 호출된다.
`apply()` 시그니처가 표준과 다르고, `is_train_only()`는 구현하지 않는다.

```python
class V3Preprocessor:

    def apply(self,
              feat  : torch.Tensor,   # (B, ...) 모델 중간 feature
              label : torch.Tensor,   # (B, 7)   multi-hot label
              alpha : float           # Beta(alpha, alpha) 파라미터
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            mixed_feat : lam * feat + (1-lam) * feat[perm]
            label_a    : 원본 label
            label_b    : label[perm]
            lam        : float, Beta(alpha, alpha) 샘플링 결과
        """
```

`train/mixup_trainer.py`는 V3를 직접 인스턴스화하여 호출한다.
`model/`의 forward 시그니처와도 합의 필요 — `model/README.md` 참고.

---

## 3. `train/`과의 연결 규칙

`train/base_trainer.py`(또는 `dataset` 클래스)는 preprocessor를 다음 방식으로 사용한다:

```python
preprocessor = V{N}Preprocessor(**cfg.preprocessing)

if preprocessor.is_train_only():
    train_dataset.transform = preprocessor.apply
    val_dataset.transform   = None          # 검증엔 적용 안 함
else:
    train_dataset.transform = preprocessor.apply
    val_dataset.transform   = preprocessor.apply
```

이 흐름이 성립하려면 `is_train_only()`의 반환값이 정확해야 한다.
