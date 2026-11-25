# FER evaluation report

- **Motivation/Background**: Measure the trained FER pipeline on labeled evaluation images.
- **Purpose**: Report overall and per-class performance, including processing failures.
- **Overview Pipeline**: Decode, prepare face, normalize lighting, Spatial-LBP, SVM.
- **Detailed Plan**: Protocol, overall metrics, class metrics and confusion matrix.
- **References**: [Model](../../../models/fer.joblib), [data](../../../data/raw/test), [summary](summary.json), [predictions](predictions.csv), [class report](classification_report.csv), [confusion matrix](confusion_matrix.csv).
- **Created**: 2026-11-25T17:05:26+07:00
- **Last Updated**: 2026-11-25T17:05:26+07:00

---

> **5W1H**
> - **What**: Complete evaluation of the supplied labeled directory.
> - **Why**: Measure classification quality and processing coverage.
> - **When**: 2026-11-25T16:54:37+07:00 to 2026-11-25T17:05:26+07:00.
> - **Where**: Local CPU, Windows; artifacts beside this report.
> - **Who**: FER project evaluation CLI, for project review.
> - **How**: Mode `cropped_faces`. All labeled images; no fitting or parameter tuning. Processing errors count as incorrect and false negatives. Latency covers successful pipeline calls, excluding file reads and model loading. Folder names supply ground truth; split provenance is the caller's responsibility.

| Metric | Value | Meaning |
| :--- | ---: | :--- |
| Images | 7178 | Every labeled image discovered |
| Processing errors | 0 | Counted as incorrect, not silently excluded |
| Accuracy | 0.5784 | Correct predictions / all images |
| Macro-F1 | 0.5696 | Equal-weight mean F1 over seven classes |
| Weighted-F1 | 0.5772 | F1 weighted by class support |
| Mean latency | 79.32 ms | Successful pipeline calls, excluding file reads and model load |

## Per-class metrics

| Label | Precision | Recall | F1 | Support |
| :--- | ---: | ---: | ---: | ---: |
| neutral | 0.5116 | 0.5385 | 0.5247 | 1233 |
| happiness | 0.7291 | 0.7858 | 0.7564 | 1774 |
| sadness | 0.4773 | 0.4555 | 0.4661 | 1247 |
| surprise | 0.7710 | 0.6968 | 0.7320 | 831 |
| anger | 0.4548 | 0.4781 | 0.4662 | 958 |
| disgust | 0.8387 | 0.4685 | 0.6012 | 111 |
| fear | 0.4562 | 0.4268 | 0.4410 | 1024 |

## Confusion matrix

Rows are reference labels; columns are predictions. A processing-error column is added only if needed.

| True / predicted | neutral | happiness | sadness | surprise | anger | disgust | fear |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| neutral | 664 | 150 | 176 | 26 | 116 | 1 | 100 |
| happiness | 114 | 1394 | 99 | 27 | 81 | 0 | 59 |
| sadness | 203 | 140 | 568 | 22 | 153 | 3 | 158 |
| surprise | 53 | 48 | 28 | 579 | 35 | 0 | 88 |
| anger | 135 | 103 | 134 | 17 | 458 | 4 | 107 |
| disgust | 5 | 8 | 10 | 2 | 25 | 52 | 9 |
| fear | 124 | 69 | 175 | 78 | 139 | 2 | 437 |
