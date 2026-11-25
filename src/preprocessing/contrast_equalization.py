import numpy as np


def normalize_contrast(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    alpha, tau, epsilon = 0.1, 10.0, 1e-8
    image = image / max(float(np.mean(np.abs(image) ** alpha)) ** (1 / alpha), epsilon)
    image = image / max(
        float(np.mean(np.minimum(tau, np.abs(image)) ** alpha)) ** (1 / alpha), epsilon
    )
    return (127.5 * (np.tanh(image / tau) + 1)).astype(np.uint8)
