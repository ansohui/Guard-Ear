import numpy as np

def rule_based_prefilter(wav, sr=16000,
                         min_db=-35,
                         min_freq=400,
                         max_freq=2500):
    """
    freq, decibel 기반 간단 프리필터
    - 너무 작은 소음이면 통과 안 시킴
    - 사이렌 대역이 아니면 통과 안 시킴
    """
    # 에너지 검사
    rms = np.sqrt(np.mean(wav ** 2))
    db = 20 * np.log10(rms + 1e-6)
    if db < min_db:
        return False

    # 주파수 대역 검사 
    fft = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(len(wav), d=1.0/sr)
    band_power = np.mean(np.abs(fft[(freqs > min_freq) & (freqs < max_freq)]))
    if band_power < 0.001: 
        return False

    return True
