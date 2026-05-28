import numpy as np
 
 
class V0Preprocessor:
    """
    V0 Baseline Preprocessor.
    이미지·라벨을 변형 없이 그대로 반환하는 identity 전처리.
    다른 V와의 비교 baseline으로 사용된다.
    """
 
    def __init__(self, **kwargs):
        pass
 
    def apply(self, image: np.ndarray, label: np.ndarray = None):
        """
        Args:
            image : (H, W, 3) RGB uint8 numpy array
            label : (7,) multi-hot float32 numpy array. None일 수 있음.
        Returns:
            (image, label) — 변형 없이 그대로 반환.
        """
        return image, label
 
    def is_train_only(self) -> bool:
        """
        False → 학습·검증 DataLoader 양쪽에 모두 적용.
        V0는 identity이므로 양쪽 적용해도 무방.
        """
        return False