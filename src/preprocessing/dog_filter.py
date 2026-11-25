import cv2
import numpy as np


def filter_dog(image: np.ndarray, sigma1: float = 1.0, sigma2: float = 4.0) -> np.ndarray:
    if not (np.isfinite(sigma1) and np.isfinite(sigma2) and 0 < sigma1 < sigma2):
        raise ValueError("DoG requires 0 < sigma1 < sigma2, both finite.")
    size = 4 * int(np.ceil(sigma2)) + 1
    image = image.astype(np.float32)
    return cv2.GaussianBlur(image, (size, size), sigma1) - cv2.GaussianBlur(
        image, (size, size), sigma2
    )
