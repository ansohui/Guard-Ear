"""
Rule-based prefilter 
- 단순 대역비만 보는 게 아니라
  1) 음량
  2) 사이렌 핵심 대역(600~2500Hz) 세분
  3) 화재경보 대역(2.5~4.5kHz)
  4) 주기성 약식 검사
  을 합쳐서 0~1 점수로 만든다.
- detection 단계에서 ML 점수와 가중합으로 fuse해서 사용한다.
"""

import numpy as np
from scipy.signal import welch


def _loudness_score(wav: np.ndarray, min_db: float = -35.0, max_db: float = -5.0) -> float:
    """RMS → dB → 0~0.4로 매핑"""
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-12
    db = 20 * np.log10(rms)

    if db <= min_db:
        return 0.0

    
    norm = (db - min_db) / (max_db - min_db)
    norm = float(np.clip(norm, 0.0, 1.0))
    return 0.4 * norm  # 최대 0.4


def _band_energy_ratio(psd, freqs, band):
    bmask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = float(np.sum(psd[bmask]))
    total_power = float(np.sum(psd)) + 1e-9
    return band_power / total_power


def _periodicity_hint(wav: np.ndarray, sr: int = 16000) -> float:
    """
    신호의 저주파 대역에서 반복성이 존재하는지를 간단히 평가하는 보조 지표.

    - 사이렌·경보음은 보통 수백 ms 단위로 에너지 패턴이 반복되는 특성이 있음
    - 전체 자기상관을 계산하지 않고, 0.2~0.6초 구간만 제한적으로 확인해 연산량을 줄임
    - 프리필터 단계이므로 이 값은 최종 점수에 최대 0.05까지만 반영함
    """
    max_lag_sec = 0.6  
    max_lag = int(sr * max_lag_sec)
    if len(wav) < max_lag * 2:
        return 0.0

    segment = wav[: sr * 2]
    ac = np.correlate(segment, segment, mode="full")
    ac = ac[len(ac)//2:]

    # 0.2~0.6초 사이의 최대값
    lo = int(sr * 0.2)
    hi = int(sr * 0.6)
    if hi > len(ac):
        return 0.0

    win = ac[lo:hi]
    if ac[0] <= 0:
        return 0.0
    peak = float(np.max(win) / (ac[0] + 1e-9))

    return float(np.clip(peak, 0.0, 1.0)) * 0.05


def rule_prefilter(
    wav: np.ndarray,
    sr: int = 16000,
    min_db: float = -35.0,
) -> float:
    """
    구성:
      score = loud(0~0.4)
            + core_band(0~0.4)
            + alarm_band(0~0.15)
            + periodicity(0~0.05)
    """

    # 1) loudness
    score = _loudness_score(wav, min_db=min_db)

    # 2) 주파수 분석 
    freqs, psd = welch(wav, sr, nperseg=1024)

    # 2-1) 사이렌 핵심 대역 3구간
    core_bands = [(600, 900), (900, 1500), (1500, 2500)]
    core_score = 0.0
    for low, high in core_bands:
        r = _band_energy_ratio(psd, freqs, (low, high))
        # 각 구간별로 0~0.15 (0.4로 클립)
        core_score += min(0.15, r * 1.5)
    core_score = min(core_score, 0.4)
    score += core_score

    # 3) 화재경보 대역 (2.5~4.5kHz)
    alarm_ratio = _band_energy_ratio(psd, freqs, (2500, 4500))
    alarm_score = min(0.15, alarm_ratio * 1.5)
    score += alarm_score

    # 4) 주기성
    score += _periodicity_hint(wav, sr=sr)

    # 5) 최종 클립
    score = float(np.clip(score, 0.0, 1.0))
    return score
