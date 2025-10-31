import numpy as np

class MLPHead:
    def __init__(self, input_dim=521, hidden_dim=128, num_classes=1):

        rng = np.random.default_rng(0)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, num_classes)) * 0.01
        self.b2 = np.zeros(num_classes)

    def predict(self, x):
        """
        x: (521,) yamnet prob
        return: scalar siren prob (0~1)
        """
        h = x @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        out = h @ self.W2 + self.b2
        # sigmoid
        prob = 1 / (1 + np.exp(-out))
        return float(prob.squeeze())
