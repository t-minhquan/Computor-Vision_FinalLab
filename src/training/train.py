import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from src.emotion_recognition.svm_fer_classifier import (
    BASELINE_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    EMOTIONS,
    FEATURE_SPECS,
    save_classifier,
)
from src.pipeline import extract_features
from src.preprocessing.face_crop import (
    MAX_UPLOAD_BYTES,
    FaceDetector,
    decode_image,
    prepare_cropped_face,
)


def train_model(
    data_dir: Path, output: Path, *, cropped_faces: bool = False, feature_type: str = "lbp"
) -> dict:
    if feature_type not in FEATURE_SPECS:
        raise ValueError(f"Unknown feature type: {feature_type}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite model: {output}")
    if not data_dir.is_dir():
        raise ValueError(f"Training directory does not exist: {data_dir}")
    folders = {path.name for path in data_dir.iterdir() if path.is_dir()}
    aliases = {"angry": "anger", "happy": "happiness", "sad": "sadness"}
    class_folders = {aliases.get(name, name): data_dir / name for name in folders}
    if len(folders) != len(EMOTIONS) or set(class_folders) != set(EMOTIONS):
        raise ValueError(
            "Training directory must contain exactly these folders: "
            + ", ".join(EMOTIONS)
            + ". FER2013 aliases angry, happy, sad are also accepted."
        )
    detector = None if cropped_faces else FaceDetector()
    features, labels = [], []
    for emotion in EMOTIONS:
        paths = sorted(
            p
            for p in class_folders[emotion].iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(paths) < 2:
            raise ValueError(f"Class {emotion} needs at least two labeled images.")
        print(f"Extracting {emotion}: {len(paths)} images", file=sys.stderr, flush=True)
        for path in paths:
            try:
                with path.open("rb") as stream:
                    image = decode_image(stream.read(MAX_UPLOAD_BYTES + 1))
                if cropped_faces:
                    face = prepare_cropped_face(image)
                else:
                    face, _ = detector.crop(image)
                features.append(extract_features(face, feature_type=feature_type))
                labels.append(emotion)
                if len(labels) % 500 == 0:
                    print(
                        f"Extracted {len(labels)} images ({feature_type})",
                        file=sys.stderr,
                        flush=True,
                    )
            except (ValueError, OSError) as exc:
                raise ValueError(f"Cannot use training image {path}: {exc}") from exc
    classifier = SVC(
        kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", verbose=feature_type == "hog"
    )
    print(f"Training SVM on {len(labels)} images...", file=sys.stderr, flush=True)
    classifier.fit(np.stack(features), labels)
    save_classifier(classifier, output, feature_type=feature_type)
    return {
        "model": str(output),
        "feature_type": feature_type,
        "features_per_image": FEATURE_SPECS[feature_type][0],
        "samples": len(labels),
        "class_counts": dict(Counter(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FER from seven labeled image folders.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feature-type", choices=FEATURE_SPECS, default="lbp")
    parser.add_argument(
        "--cropped-faces",
        action="store_true",
        help="Use already cropped faces (e.g. FER2013); skip Haar detection during training.",
    )
    args = parser.parse_args()
    output = args.output or (
        BASELINE_MODEL_PATH if args.feature_type == "hog" else DEFAULT_MODEL_PATH
    )
    try:
        print(
            json.dumps(
                train_model(
                    args.data_dir,
                    output,
                    cropped_faces=args.cropped_faces,
                    feature_type=args.feature_type,
                ),
                indent=2,
            )
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
