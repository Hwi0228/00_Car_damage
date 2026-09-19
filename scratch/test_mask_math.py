import numpy as np

# Proto masks: (32, 80, 80) -> reshape to (32, 6400)
proto = np.random.randn(32, 80, 80)
proto_flat = proto.reshape(32, 80 * 80)

# Mask coefficients: (32,)
coeff = np.random.randn(32)

# Matrix multiplication: (1, 32) @ (32, 6400) -> (1, 6400) -> reshape to (80, 80)
mask_logits = np.matmul(coeff, proto_flat).reshape(80, 80)
mask_sigmoid = 1.0 / (1.0 + np.exp(-mask_logits))

print("Mask sigmoid min, max:", mask_sigmoid.min(), mask_sigmoid.max())
print("Mask shape:", mask_sigmoid.shape)
