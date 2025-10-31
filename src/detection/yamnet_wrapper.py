import numpy as np

class YamNetDetector:
    def __init__(self, model_path=None):
        
        # tf.lite / tf savedmodel / onnx
        self.num_classes = 521

    def predict(self, wav, sr=16000):
        # log-mel → YamNet 호출
        probs = np.random.rand(self.num_classes).astype(np.float32)
        probs = probs / probs.sum()
        return probs
