from io import BytesIO

import cv2
import joblib
import numpy as np
import pytest
from PIL import Image

from src.emotion_recognition.svm_fer_classifier import (
    EMOTIONS,
    load_classifier,
    save_classifier,
)
from src.feature_extraction.lbp_descriptor import FEATURE_SIZE, UNIFORM_LOOKUP, extract_lbp
from src.pipeline import EmotionPipeline, extract_features
from src.preprocessing.contrast_equalization import normalize_contrast
from src.preprocessing.dog_filter import filter_dog
from src.preprocessing.face_crop import FaceDetector, decode_image
from src.preprocessing.gamma_correction import correct_gamma


def test_normalization_handles_black_and_flat_faces():
    for value in (0, 128, 255):
        face = np.full((128, 128), value, dtype=np.uint8)
        features = extract_features(face)
        assert features.shape == (FEATURE_SIZE,)
        assert np.isfinite(features).all()
        assert features.sum() == pytest.approx(1)
        np.testing.assert_array_equal(normalize_contrast(np.zeros_like(face)), 127)


def test_gamma_and_dog_preserve_expected_behavior():
    image = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    np.testing.assert_allclose(correct_gamma(image, 1), image, atol=1e-5)
    assert correct_gamma(image)[0, 10] > image[0, 10]
    dog = filter_dog(image)
    assert dog.dtype == np.float32
    assert dog.min() < 0 < dog.max()
    for gamma in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            correct_gamma(image, gamma)
    with pytest.raises(ValueError):
        filter_dog(image, 4, 1)


def test_uniform_mapping_and_spatial_bins():
    assert set(UNIFORM_LOOKUP) == set(range(59))
    assert np.count_nonzero(UNIFORM_LOOKUP == 58) == 198
    image = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    features = extract_lbp(image).reshape(64, 59)
    # Interior blocks see the same horizontal ramp, with five neighbors >= center.
    block = features[3 * 8 + 3]
    assert np.count_nonzero(block) == 1
    assert block[UNIFORM_LOOKUP[199]] == pytest.approx(1 / 64)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 8)),
        np.full((128, 128), np.nan),
        np.full((128, 128), -1),
        np.full((128, 128), 256),
    ],
)
def test_feature_input_validation(image):
    with pytest.raises(ValueError):
        extract_features(image)


def test_decode_rejects_invalid_unsupported_and_large_images(image_bytes):
    assert decode_image(image_bytes).shape == (128, 128, 3)
    for content in (b"", b"not an image", image_bytes[:50]):
        with pytest.raises(ValueError):
            decode_image(content)
    stream = BytesIO()
    Image.new("RGB", (20, 20)).save(stream, format="GIF")
    with pytest.raises(ValueError, match="JPEG and PNG"):
        decode_image(stream.getvalue())
    stream = BytesIO()
    Image.new("L", (4000, 3001)).save(stream, format="PNG")
    with pytest.raises(ValueError, match="megapixel"):
        decode_image(stream.getvalue())


@pytest.mark.parametrize("boxes", [[], [(0, 0, 30, 30), (30, 30, 30, 30)]])
def test_face_count_is_enforced(boxes):
    class Cascade:
        def detectMultiScale(self, *args, **kwargs):
            return boxes

    detector = FaceDetector()
    detector.cascade = Cascade()
    with pytest.raises(ValueError, match="exactly one face"):
        detector.crop(np.zeros((128, 128, 3), dtype=np.uint8))


def test_face_crop_scales_bounding_box():
    class Cascade:
        def detectMultiScale(self, image, **kwargs):
            assert max(image.shape) == 960
            return [(100, 50, 100, 120)]

    detector = FaceDetector()
    detector.cascade = Cascade()
    face, box = detector.crop(np.zeros((1200, 1920, 3), dtype=np.uint8))
    assert face.shape == (128, 128)
    assert box == (200, 100, 200, 240)


def test_model_roundtrip_and_prediction(trained_model, image_bytes, monkeypatch):
    pipeline = EmotionPipeline(trained_model)
    monkeypatch.setattr(
        pipeline.detector,
        "crop",
        lambda image: (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0, 128, 128)),
    )
    result = pipeline.predict(image_bytes)
    assert result["emotion"] in EMOTIONS
    assert result["latency_ms"] >= 0
    with pytest.raises(FileExistsError):
        save_classifier(load_classifier(trained_model), trained_model)
    assert load_classifier(trained_model).n_features_in_ == FEATURE_SIZE


def test_bad_model_version_is_rejected(tmp_path):
    path = tmp_path / "wrong.joblib"
    joblib.dump({"feature_version": "wrong"}, path)
    with pytest.raises(ValueError, match="version"):
        load_classifier(path)


def test_cropped_prediction_matches_training_input(trained_model, monkeypatch):
    pipeline = EmotionPipeline(trained_model)
    image = np.random.default_rng(42).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    expected = extract_features(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (128, 128)))

    def no_detector(image):
        raise AssertionError("Already cropped faces must not be detected or cropped again.")

    def capture_features(classifier, features):
        np.testing.assert_array_equal(features, expected)
        return "neutral"

    monkeypatch.setattr(pipeline.detector, "crop", no_detector)
    monkeypatch.setattr("src.pipeline.predict_emotion", capture_features)
    result = pipeline.predict(encoded.tobytes(), cropped_faces=True)
    assert result["emotion"] == "neutral"
    assert result["face_box"] == {"x": 0, "y": 0, "width": 48, "height": 48}
