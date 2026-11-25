import warnings
from io import BytesIO
from threading import Lock

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000


def decode_image(content: bytes) -> np.ndarray:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Image must contain 1 byte to 5 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as header:
                if header.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported.")
                if header.width * header.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 12 megapixel limit.")
                header.verify()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("Invalid image data.") from exc
    try:
        image = cv2.imdecode(
            np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
    except cv2.error as exc:
        raise ValueError("Invalid image data.") from exc
    if image is None:
        raise ValueError("Invalid image data.")
    return image


def prepare_cropped_face(image: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (128, 128))


class FaceDetector:
    def __init__(self):
        self.lock = Lock()
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.cascade.empty():
            raise RuntimeError("Cannot load the Haar face detector.")

    def crop(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect on a bounded image to keep large uploads from slowing the detector.
        scale = min(1.0, 960 / max(gray.shape))
        small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
        with self.lock:
            faces = self.cascade.detectMultiScale(
                small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
        if len(faces) != 1:
            raise ValueError("Expected exactly one face; found %d." % len(faces))
        x, y, width, height = (int(round(int(v) / scale)) for v in faces[0])
        height = min(height, gray.shape[0] - y)
        width = min(width, gray.shape[1] - x)
        face = cv2.resize(gray[y : y + height, x : x + width], (128, 128))
        return face, (x, y, width, height)
