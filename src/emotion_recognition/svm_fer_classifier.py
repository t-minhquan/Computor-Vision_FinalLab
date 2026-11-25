from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from src.feature_extraction.hog_descriptor import HOG_FEATURE_SIZE, HOG_FEATURE_VERSION
from src.feature_extraction.lbp_descriptor import FEATURE_SIZE, FEATURE_VERSION

EMOTIONS = ("neutral", "happiness", "sadness", "surprise", "anger", "disgust", "fear")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "fer.joblib"
BASELINE_MODEL_PATH = DEFAULT_MODEL_PATH.with_name("hog_baseline.joblib")
FEATURE_SPECS = {
    "lbp": (FEATURE_SIZE, FEATURE_VERSION),
    "hog": (HOG_FEATURE_SIZE, HOG_FEATURE_VERSION),
}


def validate_classifier(classifier: SVC, *, feature_type: str = "lbp") -> None:
    if feature_type not in FEATURE_SPECS:
        raise ValueError(f"Unknown feature type: {feature_type}")
    if not isinstance(classifier, SVC):
        raise ValueError("Model must be an sklearn SVC.")
    check_is_fitted(classifier)
    feature_size, _ = FEATURE_SPECS[feature_type]
    if classifier.n_features_in_ != feature_size or set(classifier.classes_) != set(EMOTIONS):
        raise ValueError(
            f"Model must use {feature_size} {feature_type} features and all seven emotion labels."
        )


def save_classifier(classifier: SVC, path: Path, *, feature_type: str = "lbp") -> None:
    validate_classifier(classifier, feature_type=feature_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally destroying a previous trained model.
    with path.open("xb") as stream:
        joblib.dump(
            {"feature_version": FEATURE_SPECS[feature_type][1], "classifier": classifier}, stream
        )


def load_classifier(path: Path, *, feature_type: str = "lbp") -> SVC:
    if feature_type not in FEATURE_SPECS:
        raise ValueError(f"Unknown feature type: {feature_type}")
    # joblib can execute code: only load a trusted, locally trained artifact.
    artifact = joblib.load(path)
    if (
        not isinstance(artifact, dict)
        or artifact.get("feature_version") != FEATURE_SPECS[feature_type][1]
    ):
        raise ValueError("Model feature version does not match this pipeline.")
    classifier = artifact.get("classifier")
    validate_classifier(classifier, feature_type=feature_type)
    return classifier


def predict_emotion(classifier: SVC, features: np.ndarray) -> str:
    return str(classifier.predict(features.reshape(1, -1))[0])
