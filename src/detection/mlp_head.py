"""
MLP Head
- 입력: YAMNet의 521차원 클래스 score (평균했든, 1프레임이든)
- 출력: 우리 문제용 siren probability (0~1)
- 실제 학습 모델이 있으면 같은 인터페이스로 교체하면 됨
"""

import os
import numpy as np


class MLPHead:
    def __init__(self, weight_dir: str = "weights", input_dim: int = 521,
                 hidden_dim: int = 128, out_dim: int = 1):
        self.weight_dir = weight_dir
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.W1 = self._load_np("W1.npy")
        self.b1 = self._load_np("b1.npy")
        self.W2 = self._load_np("W2.npy")
        self.b2 = self._load_np("b2.npy")

    def _load_np(self, name: str) -> np.ndarray:
        path = os.path.join(self.weight_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[MLPHead] {path} 를 찾을 수 없습니다.\n"
                f"→`python src/detection/generate_weights.py` 실행해서 weight를 만들어 주세요."
            )
        return np.load(path)

    def __call__(self, x: np.ndarray) -> float:
        """
        x: (521,) 또는 (T, 521)
        - (T, 521)이면 평균해서 씀
        return: float, 0~1
        """
        if x.ndim == 2:
            x = x.mean(axis=0)  # (521,)
        if x.shape[0] != self.input_dim:
            raise ValueError(f"[MLPHead] expected dim {self.input_dim}, got {x.shape[0]}")

        # 1층
        h = x @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        # 2층
        out = h @ self.W2 + self.b2
        prob = 1 / (1 + np.exp(-out))
        return float(prob.squeeze())
