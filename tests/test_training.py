import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.emotion_recognition.svm_fer_classifier import EMOTIONS, load_classifier
from src.preprocessing.face_crop import FaceDetector
from src.training.train import train_model


def test_training_saves_compatible_model(tmp_path, image_bytes, monkeypatch):
    data = tmp_path / "train"
    for emotion in EMOTIONS:
        folder = data / emotion
        folder.mkdir(parents=True)
        for i in range(2):
            (folder / f"{i}.png").write_bytes(image_bytes)
    monkeypatch.setattr(
        FaceDetector,
        "crop",
        lambda self, image: (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0, 128, 128)),
    )
    output = tmp_path / "fer.joblib"
    result = train_model(data, output)
    assert result["samples"] == 14
    assert set(load_classifier(output).classes_) == set(EMOTIONS)
    with TestClient(create_app(output)) as client:
        response = client.post("/api/v1/predict", files={"file": ("face.png", image_bytes)})
        assert response.status_code == 200
        assert response.json()["emotion"] in EMOTIONS
    with pytest.raises(FileExistsError):
        train_model(data, output)


def test_training_rejects_incomplete_classes(tmp_path):
    with pytest.raises(ValueError, match="exactly these folders"):
        train_model(tmp_path, tmp_path / "fer.joblib")


def test_training_rejects_bad_images(tmp_path):
    data = tmp_path / "train"
    for emotion in EMOTIONS:
        folder = data / emotion
        folder.mkdir(parents=True)
        for i in range(2):
            (folder / f"{i}.png").write_bytes(b"bad image")
    output = tmp_path / "fer.joblib"
    with pytest.raises(ValueError, match="Cannot use training image"):
        train_model(data, output)
    assert not output.exists()


def test_fer2013_labels_and_cropped_faces(tmp_path, monkeypatch):
    aliases = {"anger": "angry", "happiness": "happy", "sadness": "sad"}
    rng = np.random.default_rng(42)
    for emotion in EMOTIONS:
        folder = tmp_path / "train" / aliases.get(emotion, emotion)
        folder.mkdir(parents=True)
        for i in range(2):
            image = rng.integers(0, 256, (48, 48), dtype=np.uint8)
            assert cv2.imwrite(str(folder / f"{i}.png"), image)

    def no_detector(self):
        raise AssertionError("Already cropped faces must not instantiate Haar.")

    monkeypatch.setattr(FaceDetector, "__init__", no_detector)
    output = tmp_path / "fer.joblib"
    report = train_model(tmp_path / "train", output, cropped_faces=True)
    assert report["samples"] == 14
    assert report["class_counts"] == dict.fromkeys(EMOTIONS, 2)
    assert set(load_classifier(output).classes_) == set(EMOTIONS)


def test_training_rejects_duplicate_alias_folders(tmp_path):
    for name in (*EMOTIONS, "angry"):
        (tmp_path / name).mkdir()
    with pytest.raises(ValueError, match="exactly these folders"):
        train_model(tmp_path, tmp_path / "fer.joblib", cropped_faces=True)
