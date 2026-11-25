import cv2
import numpy as np

HOG_FEATURE_SIZE = 1764
HOG_FEATURE_VERSION = "hog128-cell16-block32-stride16-bins9-l2hys-no-gamma-v1"


def extract_hog(image: np.ndarray) -> np.ndarray:
    if image.shape != (128, 128) or image.dtype != np.uint8:
        raise ValueError("HOG expects a 128x128 uint8 grayscale image.")
    descriptor = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9,
        _gammaCorrection=False,
    )
    return descriptor.compute(image).ravel()
