from frame_buffer.frame_buffer import FrameBuffer
from preprocessor.input_formatting import to_16khz_mono
from preprocessor.rule_prefilter import rule_based_prefilter
from detection.yamnet_wrapper import YamNetDetector
from detection.mlp_head import MLPHead
from postprocessor.hysteresis_rules import Hysteresis
from server.api import send_event

def main():
    # 1) 버퍼 준비
    buffer = FrameBuffer(window_sec=0.96, hop_sec=0.48, sample_rate=16000)

    # 2) 모델 준비
    yamnet = YamNetDetector()
    mlp = MLPHead(input_dim=521, hidden_dim=128, num_classes=1)  # siren prob
    post = Hysteresis(enter_th=0.7, exit_th=0.3, hold_frames=3)

    # 외부에서 오디오 프레임이 들어온다고 가정
    for audio_chunk in buffer.stream_from_mic():   # generator
        # (a) 16kHz mono 맞추기
        wav16 = to_16khz_mono(audio_chunk, target_sr=16000)

        # (b) 룰 프리필터
        passed = rule_based_prefilter(wav16, sr=16000)
        if not passed:
            post.update(False)
            continue

        # (c) YamNet → 521D prob
        yamnet_probs = yamnet.predict(wav16, sr=16000)

        # (d) MLP head → siren prob
        siren_prob = mlp.predict(yamnet_probs)

        # (e) 후처리 (히스테리시스 + 룰)
        is_siren = post.update(siren_prob > 0.5)

        # (f) 서버 알림
        if is_siren:
            send_event({"type": "siren_detected", "prob": float(siren_prob)})
            print("[INFO] siren detected!")

if __name__ == "__main__":
    main()
