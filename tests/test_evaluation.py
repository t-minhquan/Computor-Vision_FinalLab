import csv
import json

import pytest

from src.emotion_recognition.svm_fer_classifier import EMOTIONS
from src.evaluation.evaluate import evaluate_model


def make_dataset(root, image_bytes):
    aliases = {"anger": "angry", "happiness": "happy", "sadness": "sad"}
    for label in EMOTIONS:
        folder = root / aliases.get(label, label)
        folder.mkdir(parents=True)
        (folder / "sample.png").write_bytes(image_bytes)


def test_evaluation_exports_all_classes_and_predictions(tmp_path, image_bytes, trained_model):
    data = tmp_path / "test"
    make_dataset(data, image_bytes)
    output = tmp_path / "results"
    summary = evaluate_model(data, trained_model, output, cropped_faces=True)
    assert summary["total_images"] == summary["processed_images"] == 7
    assert summary["failed_images"] == 0
    # Identical images across all seven labels yield one correct prediction.
    assert summary["accuracy"] == pytest.approx(1 / 7)
    assert set(summary["per_class"]) == set(EMOTIONS)
    assert sum(sum(row) for row in summary["confusion_matrix"]) == 7
    assert json.loads((output / "summary.json").read_text())["input_mode"] == "cropped_faces"
    with (output / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    assert sum(row["correct"] == "True" for row in rows) == 1
    assert (output / "classification_report.csv").is_file()
    assert (output / "confusion_matrix.csv").is_file()
    assert "5W1H" in (output / "REPORT.md").read_text()
    with pytest.raises(FileExistsError):
        evaluate_model(data, trained_model, output, cropped_faces=True)


def test_evaluation_keeps_processing_failures_in_denominator(tmp_path, image_bytes, trained_model):
    data = tmp_path / "test"
    make_dataset(data, image_bytes)
    (data / "happy/broken.png").write_bytes(b"invalid image")
    output = tmp_path / "results"
    summary = evaluate_model(data, trained_model, output, cropped_faces=True)
    assert summary["total_images"] == 8
    assert summary["processed_images"] == 7
    assert summary["failed_images"] == 1
    assert summary["accuracy"] == pytest.approx(1 / 8)
    assert summary["confusion_columns"][-1] == "processing_error"
    assert sum(sum(row) for row in summary["confusion_matrix"]) == 8
    with (output / "predictions.csv").open(newline="") as stream:
        failures = [row for row in csv.DictReader(stream) if row["error"]]
    assert len(failures) == 1
    assert failures[0]["file"] == "happy/broken.png"
    assert failures[0]["correct"] == "False"


def test_evaluation_rejects_incomplete_label_folders(tmp_path, trained_model):
    with pytest.raises(ValueError, match="seven emotion folders"):
        evaluate_model(tmp_path, trained_model, tmp_path / "results", cropped_faces=True)
