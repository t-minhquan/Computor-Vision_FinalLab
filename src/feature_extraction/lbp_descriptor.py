import cv2
import numpy as np

# Keep each uniform pattern separate: 58 patterns plus one non-uniform bin.
UNIFORM_LOOKUP = np.full(256, 58, dtype=np.uint8)
_bin = 0
for _code in range(256):
    _bits = [(_code >> p) & 1 for p in range(8)]
    if sum(_bits[p] != _bits[(p + 1) % 8] for p in range(8)) <= 2:
        UNIFORM_LOOKUP[_code] = _bin
        _bin += 1
FEATURE_SIZE = 64 * 59
FEATURE_VERSION = "gamma02-dog1-4-contrast01-10-lbp8r1-grid8-l1-v1"


def extract_lbp(image: np.ndarray) -> np.ndarray:
    if image.shape != (128, 128) or not np.isfinite(image).all():
        raise ValueError("LBP expects a finite 128x128 grayscale image.")
    image = image.astype(np.float32)
    yy, xx = np.mgrid[:128, :128].astype(np.float32)
    codes = np.zeros(image.shape, dtype=np.uint8)
    for p in range(8):
        angle = 2 * np.pi * p / 8
        neighbor = cv2.remap(
            image,
            xx + np.float32(np.cos(angle)),
            yy - np.float32(np.sin(angle)),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        codes |= (neighbor >= image).astype(np.uint8) << p
    mapped = UNIFORM_LOOKUP[codes]
    histograms = [
        np.bincount(mapped[y : y + 16, x : x + 16].ravel(), minlength=59)
        for y in range(0, 128, 16)
        for x in range(0, 128, 16)
    ]
    features = np.concatenate(histograms).astype(np.float32)
    return features / max(float(features.sum()), 1.0)
