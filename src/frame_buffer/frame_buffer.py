import sounddevice as sd
import numpy as np

class FrameBuffer:
    def __init__(self, window_sec=0.96, hop_sec=0.48, sample_rate=16000):
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.sr = sample_rate
        self.window_len = int(window_sec * sample_rate)
        self.hop_len = int(hop_sec * sample_rate)
        self.buffer = np.zeros(0, dtype=np.float32)

    def stream_from_mic(self):
        with sd.InputStream(channels=1, samplerate=self.sr, callback=None):
            while True:
                in_data = sd.rec(self.hop_len, samplerate=self.sr,
                                 channels=1, dtype="float32")
                sd.wait()
                in_data = in_data.flatten()
                self.buffer = np.concatenate([self.buffer, in_data])
                if len(self.buffer) >= self.window_len:
                    frame = self.buffer[:self.window_len]
                    self.buffer = self.buffer[self.hop_len:]
                    yield frame
