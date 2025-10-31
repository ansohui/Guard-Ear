"""
src/main.py

Frame Buffer 
→ Preprocessor (input_formatting + rule_prefilter)
→ Detection (YAMNet + MLP Head fallback)
→ Postprocessor (Hysteresis)
→ Server (notification) 훅만 호출
"""

import os
import soundfile as sf
import numpy as np

from src.preprocessor import rule_prefilter as rp
from src.detection.yamnet_wrapper import YamNetWrapper
from src.detection.mlp_head import MLPHead
from src.postprocessor.hysteresis_rules import HysteresisDetector


# === 1. 입력 포맷터 ===
def _load_and_format(path: str) -> np.ndarray:
    """16kHz mono + RMS 정규화"""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"expected 16kHz, got {sr}")
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    y = y / max(rms, 1e-4)
    return y.astype("float32")


# === 2. 메인 함수 ===
def main():
    audio_path = "./dataset.wav"   # 데모용

    # 1) 입력
    y = _load_and_format(audio_path)

    # 2) rule 프리필터
    p_rule = rp.rule_prefilter(y)

    # 3) detection
    yam = YamNetWrapper("./yamnet_class_map.csv")
    frame_scores, embeddings = yam.infer(y)

    
    use_head = os.path.exists("weights/W1.npy")
    if use_head:
        head = MLPHead("weights")
        p_ml = head(frame_scores)   
    else:
        p_ml = yam.siren_fallback_score(frame_scores)

    # 4) 점수 fuse
    ALPHA = 0.6
    p_fused = ALPHA * p_ml + (1 - ALPHA) * p_rule

    # 5) 히스테리시스 
    hyster = HysteresisDetector()
    alerts = []
    for _ in range(10):
        alert = hyster.update(p_fused)
        alerts.append(alert)

    final_alert = any(alerts)

    # 6) 결과 출력 
    print({
        "file": os.path.basename(audio_path),
        "p_rule": round(p_rule, 3),
        "p_ml": round(p_ml, 3),
        "p_fused": round(p_fused, 3),
        "alert": bool(final_alert),
        "last_state": hyster.state,
    })


# === 3. 실행 엔트리 ===
if __name__ == "__main__":
    main()
