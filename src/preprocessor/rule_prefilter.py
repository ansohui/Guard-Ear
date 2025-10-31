"""
Rule-based prefilter
- 너무 작은 소리나 관심 대역이 아닌 건 detection model로 안 보내도 되게 1차 점수화
- detection에서 이 점수를 ML 점수와 fuse(가중합)해서 사용
"""

import numpy as np
from scipy.signal import welch


def rule_prefilter(wav: np.ndarray,
                   sr: int = 16000,
                   min_db: float = -35.0,
                   band: tuple = (600, 2500)) -> float:
    """
    return: 0~1 사이 점수
    """
    # loudness
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-12
    db = 20 * np.log10(rms)
    if db < min_db:    # 너무 작으면 아예 0
        return 0.0

    # band energy
    freqs, psd = welch(wav, sr, nperseg=1024)
    bmask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.sum(psd[bmask])
    total_power = np.sum(psd) + 1e-9
    ratio = band_power / total_power   # 0~1

    # 0.05보다 작으면 작은 값, 그 이상이면 0.5 이상으로 올려줌
    if ratio < 0.05:
        return float(ratio / 0.05 * 0.3)
    else:
        return float(min(1.0, 0.3 + (ratio - 0.05) * 3.0))
