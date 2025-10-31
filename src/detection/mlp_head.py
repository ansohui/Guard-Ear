import numpy as np
import os

class MLPHead:
    def __init__(self, input_dim=521, hidden_dim=128, num_classes=1, weight_dir="weights"):
        self.W1 = np.load(os.path.join(weight_dir, "W1.npy"))
        self.b1 = np.load(os.path.join(weight_dir, "b1.npy"))
        self.W2 = np.load(os.path.join(weight_dir, "W2.npy"))
        self.b2 = np.load(os.path.join(weight_dir, "b2.npy"))

    def predict(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(h, 0)
        out = h @ self.W2 + self.b2
        prob = 1 / (1 + np.exp(-out))
        return float(prob.squeeze())
