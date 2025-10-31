"""
YAMNet wrapper
- 16kHz mono float32 입력을 받아서
  (frame_scores, embeddings, labels)를 돌려준다.
- head가 없을 때를 위해 siren/alarm 계열만 묶어서 clip-level 점수도 계산한다.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import medfilt

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
SR = 16000


class YamNetWrapper:
    def __init__(self, class_map_path: str = "./yamnet_class_map.csv"):
        # 모델 로드
        self.model = hub.load(YAMNET_HANDLE)

        # 클래스 맵
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(
                f"YAMNet 클래스 맵을 찾을 수 없습니다: {class_map_path}\n"
                f"→ https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv 내려서 루트에 두세요."
            )

        class_map = pd.read_csv(class_map_path)
        self.labels = class_map["display_name"].tolist()

        # siren/alarm 라벨 인덱스 미리 뽑아두기
        self._siren_mask = np.array(
            [any(k in name.lower() for k in ["siren", "alarm", "ambulance", "fire engine"])
             for name in self.labels]
        )

    def infer(self, wav_16k: np.ndarray):
        """
        wav_16k: (N,) float32, 16kHz mono
        return: frame_scores (T, 521), embeddings (T, 1024)
        """
        if wav_16k.ndim != 1:
            raise ValueError(f"YamNetWrapper expects 1-D audio, got shape {wav_16k.shape}")

        # 텐서로 변환해서 추론
        scores, embeddings, _ = self.model(tf.constant(wav_16k, dtype=tf.float32))
        return scores.numpy(), embeddings.numpy()

    def siren_fallback_score(self, frame_scores: np.ndarray) -> float:
        """
        MLP head가 없을 때 쓸 수 있는 간단한 clip-level 점수.
        YAMNet이 예측한 frame_scores에서 siren/alarm 계열만 골라
        - 프레임별 max
        - median filter로 살짝 부드럽게
        - 상위 10% 평균
        으로 만든다.
        """
        if not self._siren_mask.any():
            return 0.0

        per_frame = frame_scores[:, self._siren_mask].max(axis=1)  # (T,)
        # 스파이크 제거
        per_frame = medfilt(per_frame, kernel_size=5)

        k = max(1, int(len(per_frame) * 0.1))
        top_mean = np.sort(per_frame)[-k:].mean()

        return float(top_mean)

    @property
    def label_list(self):
        return self.labels
