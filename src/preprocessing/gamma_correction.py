import numpy as np


def correct_gamma(image: np.ndarray, gamma: float = 0.2) -> np.ndarray:
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be finite and positive.")
    return 255.0 * (image.astype(np.float32) / 255.0) ** gamma
