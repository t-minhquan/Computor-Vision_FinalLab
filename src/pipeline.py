import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.emotion_recognition.svm_fer_classifier import (
    DEFAULT_MODEL_PATH,
    FEATURE_SPECS,
    load_classifier,
    predict_emotion,
)
from src.feature_extraction.hog_descriptor import extract_hog
from src.feature_extraction.lbp_descriptor import extract_lbp
from src.preprocessing.contrast_equalization import normalize_contrast
from src.preprocessing.dog_filter import filter_dog
from src.preprocessing.face_crop import FaceDetector, decode_image, prepare_cropped_face
from src.preprocessing.gamma_correction import correct_gamma


def extract_features(face: np.ndarray, *, feature_type: str = "lbp") -> np.ndarray:
    if (
        face.shape != (128, 128)
        or not np.isfinite(face).all()
        or face.min() < 0
        or face.max() > 255
    ):
        raise ValueError("Expected a finite 128x128 grayscale face in [0, 255].")
    if feature_type == "hog":
        return extract_hog(face)
    if feature_type != "lbp":
        raise ValueError(f"Unknown feature type: {feature_type}")
    return extract_lbp(normalize_contrast(filter_dog(correct_gamma(face))))


class EmotionPipeline:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, *, feature_type: str = "lbp"):
        if feature_type not in FEATURE_SPECS:
            raise ValueError(f"Unknown feature type: {feature_type}")
        self.feature_type = feature_type
        self.detector = FaceDetector()
        self.classifier = (
            load_classifier(model_path, feature_type=feature_type) if model_path.is_file() else None
        )

    @property
    def ready(self) -> bool:
        return self.classifier is not None

    def predict(self, content: bytes, *, cropped_faces: bool = False) -> dict:
        if not self.ready:
            raise RuntimeError("No trained model. Train a model and restart the backend.")
        start = perf_counter()
        image = decode_image(content)
        if cropped_faces:
            face = prepare_cropped_face(image)
            height, width = image.shape[:2]
            box = (0, 0, width, height)
        else:
            face, box = self.detector.crop(image)
        emotion = predict_emotion(
            self.classifier, extract_features(face, feature_type=self.feature_type)
        )
        return {
            "emotion": emotion,
            "face_box": dict(zip(("x", "y", "width", "height"), box)),
            "latency_ms": round((perf_counter() - start) * 1000, 3),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize one facial expression.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-type", choices=FEATURE_SPECS, default="lbp")
    parser.add_argument(
        "--cropped-faces", action="store_true", help="Input is already a cropped face."
    )
    args = parser.parse_args()
    try:
        pipeline = EmotionPipeline(args.model, feature_type=args.feature_type)
        with args.input.open("rb") as stream:
            from src.preprocessing.face_crop import MAX_UPLOAD_BYTES

            content = stream.read(MAX_UPLOAD_BYTES + 1)
        print(json.dumps(pipeline.predict(content, cropped_faces=args.cropped_faces)))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
