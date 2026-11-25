"""Evaluate a trained FER model on every labeled image in a directory."""

import argparse
import csv
import json
import os
import platform
import statistics
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.emotion_recognition.svm_fer_classifier import DEFAULT_MODEL_PATH, EMOTIONS, FEATURE_SPECS
from src.pipeline import EmotionPipeline
from src.preprocessing.face_crop import MAX_UPLOAD_BYTES

FAILED = "processing_error"


def evaluate_model(
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    *,
    cropped_faces: bool = False,
    feature_type: str = "lbp",
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation results: {output_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Evaluation directory does not exist: {data_dir}")
    aliases = {"angry": "anger", "happy": "happiness", "sad": "sadness"}
    folders = [p for p in data_dir.iterdir() if p.is_dir()]
    class_folders = {aliases.get(p.name, p.name): p for p in folders}
    if len(folders) != len(EMOTIONS) or set(class_folders) != set(EMOTIONS):
        raise ValueError(
            "Evaluation needs exactly seven emotion folders; FER2013 aliases are accepted."
        )
    samples = []
    for label in EMOTIONS:
        paths = sorted(
            p
            for p in class_folders[label].iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not paths:
            raise ValueError(f"No evaluation images for class {label}.")
        samples.extend((p, label) for p in paths)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = perf_counter()
    pipeline = EmotionPipeline(model_path, feature_type=feature_type)
    if not pipeline.ready:
        raise ValueError(f"Trained model not found: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    truth, predicted, latencies = [], [], []
    failures = 0
    with (output_dir / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["file", "true_label", "predicted_label", "correct", "latency_ms", "error"],
        )
        writer.writeheader()
        for index, (path, label) in enumerate(samples, 1):
            row = {
                "file": path.relative_to(data_dir).as_posix(),
                "true_label": label,
                "predicted_label": FAILED,
                "correct": False,
                "latency_ms": "",
                "error": "",
            }
            try:
                with path.open("rb") as image:
                    content = image.read(MAX_UPLOAD_BYTES + 1)
                result = pipeline.predict(content, cropped_faces=cropped_faces)
                row.update(
                    predicted_label=result["emotion"],
                    correct=result["emotion"] == label,
                    latency_ms=result["latency_ms"],
                )
                latencies.append(result["latency_ms"])
            except (ValueError, OSError) as exc:
                failures += 1
                row["error"] = str(exc)
            truth.append(label)
            predicted.append(row["predicted_label"])
            writer.writerow(row)
            if index % 50 == 0 or index == len(samples):
                stream.flush()
                elapsed = perf_counter() - started
                print(
                    f"Evaluated {index}/{len(samples)} | errors={failures} | elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

    labels = list(EMOTIONS)
    report = classification_report(
        truth, predicted, labels=labels, output_dict=True, zero_division=0
    )
    matrix_labels = labels + ([FAILED] if failures else [])
    matrix = confusion_matrix(truth, predicted, labels=matrix_labels)[: len(labels)]
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "input_mode": "cropped_faces" if cropped_faces else "face_detection",
        "feature_type": feature_type,
        "feature_version": FEATURE_SPECS[feature_type][1],
        "features_per_image": FEATURE_SPECS[feature_type][0],
        "svm_parameters": {
            key: pipeline.classifier.get_params()[key]
            for key in ("C", "kernel", "gamma", "class_weight")
        },
        "total_images": len(samples),
        "processed_images": len(samples) - failures,
        "failed_images": failures,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "latency_std_ms": statistics.pstdev(latencies) if latencies else None,
        "elapsed_seconds": perf_counter() - started,
        "labels": labels,
        "confusion_columns": matrix_labels,
        "confusion_matrix": matrix.tolist(),
        "per_class": {label: report[label] for label in labels},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "scikit-learn", "scipy", "joblib", "opencv-python-headless")
            },
        },
        "protocol": "All labeled images; no fitting or parameter tuning. Processing errors count as incorrect and false negatives. Latency covers successful pipeline calls, excluding file reads and model loading. Folder names supply ground truth; split provenance is the caller's responsibility.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "classification_report.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label in labels:
            writer.writerow(
                [
                    label,
                    *(report[label][key] for key in ("precision", "recall", "f1-score", "support")),
                ]
            )
    with (output_dir / "confusion_matrix.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true / predicted", *matrix_labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])

    model_link = Path(os.path.relpath(model_path.resolve(), output_dir.resolve())).as_posix()
    data_link = Path(os.path.relpath(data_dir.resolve(), output_dir.resolve())).as_posix()
    lines = [
        "# FER evaluation report",
        "",
        "- **Motivation/Background**: Measure the trained FER pipeline on labeled evaluation images.",
        "- **Purpose**: Report overall and per-class performance, including processing failures.",
        "- **Overview Pipeline**: Decode, prepare face, "
        + (
            "raw grayscale HOG, SVM."
            if feature_type == "hog"
            else "normalize lighting, Spatial-LBP, SVM."
        ),
        "- **Detailed Plan**: Protocol, overall metrics, class metrics and confusion matrix.",
        f"- **References**: [Model]({model_link}), [data]({data_link}), [summary](summary.json), [predictions](predictions.csv), [class report](classification_report.csv), [confusion matrix](confusion_matrix.csv).",
        f"- **Created**: {summary['finished_at']}",
        f"- **Last Updated**: {summary['finished_at']}",
        "",
        "---",
        "",
        "> **5W1H**",
        "> - **What**: Complete evaluation of the supplied labeled directory.",
        "> - **Why**: Measure classification quality and processing coverage.",
        f"> - **When**: {started_at} to {summary['finished_at']}.",
        f"> - **Where**: Local CPU, {platform.system()}; artifacts beside this report.",
        "> - **Who**: FER project evaluation CLI, for project review.",
        f"> - **How**: Features `{feature_type}`, mode `{summary['input_mode']}`. {summary['protocol']}",
        "",
        "| Metric | Value | Meaning |",
        "| :--- | ---: | :--- |",
        f"| Images | {len(samples)} | Every labeled image discovered |",
        f"| Processing errors | {failures} | Counted as incorrect, not silently excluded |",
        f"| Accuracy | {summary['accuracy']:.4f} | Correct predictions / all images |",
        f"| Macro-F1 | {summary['macro_f1']:.4f} | Equal-weight mean F1 over seven classes |",
        f"| Weighted-F1 | {summary['weighted_f1']:.4f} | F1 weighted by class support |",
    ]
    if latencies:
        lines.append(
            f"| Mean latency | {summary['mean_latency_ms']:.2f} ms | Successful pipeline calls, excluding file reads and model load |"
        )
    lines.extend(
        [
            "",
            "## Per-class metrics",
            "",
            "| Label | Precision | Recall | F1 | Support |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in labels:
        row = report[label]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows are reference labels; columns are predictions. A processing-error column is added only if needed.",
            "",
            "| True / predicted | " + " | ".join(matrix_labels) + " |",
            "| :--- | " + " | ".join("---:" for _ in matrix_labels) + " |",
        ]
    )
    for label, row in zip(labels, matrix):
        lines.append("| " + label + " | " + " | ".join(map(str, row.tolist())) + " |")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--feature-type", choices=FEATURE_SPECS, default="lbp")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cropped-faces",
        action="store_true",
        help="Evaluate already cropped faces such as FER2013.",
    )
    args = parser.parse_args()
    output = args.output_dir or Path("experiments/results") / datetime.now().strftime(
        "%Y%m%d_%H%M%S_" + args.feature_type + "_eval"
    )
    print(
        f"Model: {args.model}\nData: {args.data_dir}\nResults: {output}\nCropped faces: {args.cropped_faces}",
        file=sys.stderr,
        flush=True,
    )
    try:
        summary = evaluate_model(
            args.data_dir,
            args.model,
            output,
            cropped_faces=args.cropped_faces,
            feature_type=args.feature_type,
        )
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "output_dir",
                        "total_images",
                        "failed_images",
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "mean_latency_ms",
                        "elapsed_seconds",
                    )
                },
                indent=2,
            )
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
