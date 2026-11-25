import cv2
import numpy as np
import pytest
from sklearn.svm import SVC

from src.emotion_recognition.svm_fer_classifier import EMOTIONS, save_classifier
from src.feature_extraction.lbp_descriptor import FEATURE_SIZE


@pytest.fixture
def image_bytes():
    image = np.random.default_rng(42).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


@pytest.fixture
def trained_model(tmp_path):
    # Synthetic features exercise serialization, not recognition accuracy.
    rng = np.random.default_rng(42)
    features = rng.random((21, FEATURE_SIZE), dtype=np.float32)
    classifier = SVC(kernel="rbf", C=10, class_weight="balanced").fit(features, list(EMOTIONS) * 3)
    path = tmp_path / "fer.joblib"
    save_classifier(classifier, path)
    return path
