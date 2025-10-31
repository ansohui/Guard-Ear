import numpy as np
import librosa

def to_16khz_mono(audio, target_sr=16000):
    """
    audio: np.ndarray (mono or stereo, any sr)
    return: np.ndarray (mono, target_sr)
    """

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # 16k로 리샘플
    y = librosa.resample(audio, orig_sr=target_sr, target_sr=target_sr)
    return y.astype(np.float32)
