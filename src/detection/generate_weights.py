import numpy as np
import os

os.makedirs("weights", exist_ok=True)

W1 = np.random.randn(521, 128).astype(np.float32) * 0.01
b1 = np.zeros(128, dtype=np.float32)
W2 = np.random.randn(128, 1).astype(np.float32) * 0.01
b2 = np.zeros(1, dtype=np.float32)

np.save("weights/W1.npy", W1)
np.save("weights/b1.npy", b1)
np.save("weights/W2.npy", W2)
np.save("weights/b2.npy", b2)

print("✅weights saved under ./weights/")
