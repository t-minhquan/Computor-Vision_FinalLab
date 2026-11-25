import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.emotion_recognition.svm_fer_classifier import EMOTIONS, load_classifier
from src.evaluation.evaluate import evaluate_model
from src.feature_extraction.hog_descriptor import HOG_FEATURE_SIZE, extract_hog
from src.pipeline import EmotionPipeline, extract_features
from src.training.train import train_model


def test_hog_has_expected_size_and_detects_gradient_direction():
    blank = np.zeros((128, 128), dtype=np.uint8)
    np.testing.assert_array_equal(extract_hog(blank), np.zeros(HOG_FEATURE_SIZE))
    horizontal = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    features = extract_hog(horizontal)
    assert features.shape == (1764,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert not np.allclose(features, extract_hog(horizontal.T.copy()))


def test_hog_baseline_skips_the_proposed_lighting_pipeline(monkeypatch):
    image = np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8)

    def forbidden(*args, **kwargs):
        raise AssertionError("HOG baseline must not apply proposed illumination normalization.")

    for name in ("correct_gamma", "filter_dog", "normalize_contrast"):
        monkeypatch.setattr(f"src.pipeline.{name}", forbidden)
    np.testing.assert_array_equal(extract_features(image, feature_type="hog"), extract_hog(image))
    with pytest.raises(ValueError):
        extract_features(image, feature_type="unknown")
    with pytest.raises(ValueError):
        extract_hog(np.zeros((48, 48), dtype=np.uint8))


def test_hog_train_save_load_and_evaluate(tmp_path):
    data = tmp_path / "data"
    aliases = {"anger": "angry", "happiness": "happy", "sadness": "sad"}
    rng = np.random.default_rng(42)
    for label in EMOTIONS:
        folder = data / aliases.get(label, label)
        folder.mkdir(parents=True)
        for index in range(2):
            image = rng.integers(0, 256, (48, 48), dtype=np.uint8)
            assert cv2.imwrite(str(folder / f"{index}.png"), image)
    model_path = tmp_path / "hog_baseline.joblib"
    train = train_model(data, model_path, cropped_faces=True, feature_type="hog")
    assert train["samples"] == 14
    assert train["features_per_image"] == 1764
    model = load_classifier(model_path, feature_type="hog")
    assert model.get_params()["kernel"] == "rbf"
    assert model.C == 10 and model.gamma == "scale" and model.class_weight == "balanced"
    assert set(model.classes_) == set(EMOTIONS)
    with pytest.raises(ValueError, match="version"):
        load_classifier(model_path)
    with pytest.raises(FileExistsError):
        train_model(data, model_path, cropped_faces=True, feature_type="hog")
    pipeline = EmotionPipeline(model_path, feature_type="hog")
    assert (
        pipeline.predict((data / "happy/0.png").read_bytes(), cropped_faces=True)["emotion"]
        in EMOTIONS
    )
    evaluation = evaluate_model(
        data, model_path, tmp_path / "results", cropped_faces=True, feature_type="hog"
    )
    assert evaluation["feature_type"] == "hog"
    assert evaluation["total_images"] == 14
    assert evaluation["failed_images"] == 0
    assert evaluation["features_per_image"] == 1764
    assert "raw grayscale HOG" in (tmp_path / "results/REPORT.md").read_text()


def test_lbp_artifact_cannot_be_loaded_as_hog(trained_model):
    with pytest.raises(ValueError, match="version"):
        load_classifier(trained_model, feature_type="hog")


def test_hog_cli_default_does_not_target_existing_lbp_model(tmp_path, monkeypatch):
    import src.training.train as training

    baseline_path = tmp_path / "hog_baseline.joblib"
    calls = []

    def fake_train(data, output, **kwargs):
        calls.append((output, kwargs))
        return {"model": str(output)}

    monkeypatch.setattr(training, "BASELINE_MODEL_PATH", baseline_path)
    monkeypatch.setattr(training, "train_model", fake_train)
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--data-dir", str(tmp_path), "--feature-type", "hog", "--cropped-faces"],
    )
    training.main()
    assert calls == [(baseline_path, {"cropped_faces": True, "feature_type": "hog"})]


def test_hog_colab_notebook_uses_new_bundle_and_matching_model_loader():
    path = Path(__file__).resolve().parents[1] / "notebooks/02_train_hog_baseline_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "hog_colab_cell", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            sources.append(source)
    combined = "\n".join(sources)
    assert "fer2013_hog_colab.zip" in combined
    assert 'CONTENT_DIR / "fer_hog_baseline"' in combined
    assert '"--feature-type", "hog"' in combined
    assert "feature_type='hog'" in combined
    assert 'MODEL_PATH = RUN_DIR / "hog_baseline.joblib"' in combined
    assert ".fit(" not in combined
