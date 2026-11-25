# FER evaluation report

- **Motivation/Background**: Measure the trained FER pipeline on labeled evaluation images.
- **Purpose**: Report overall and per-class performance, including processing failures.
- **Overview Pipeline**: Decode, prepare face, raw grayscale HOG, SVM.
- **Detailed Plan**: Protocol, overall metrics, class metrics and confusion matrix.
- **References**: [Model](../../../models/hog_baseline.joblib), [data](../../../data/raw/test), [summary](summary.json), [predictions](predictions.csv), [class report](classification_report.csv), [confusion matrix](confusion_matrix.csv).
- **Created**: 2026-11-25T18:38:43+07:00
- **Last Updated**: 2026-11-25T18:38:43+07:00

---

> **5W1H**
> - **What**: Complete evaluation of the supplied labeled directory.
> - **Why**: Measure classification quality and processing coverage.
> - **When**: 2026-11-25T18:30:02+07:00 to 2026-11-25T18:38:43+07:00.
> - **Where**: Local CPU, Windows; artifacts beside this report.
> - **Who**: FER project evaluation CLI, for project review.
> - **How**: Features `hog`, mode `cropped_faces`. All labeled images; no fitting or parameter tuning. Processing errors count as incorrect and false negatives. Latency covers successful pipeline calls, excluding file reads and model loading. Folder names supply ground truth; split provenance is the caller's responsibility.

| Metric | Value | Meaning |
| :--- | ---: | :--- |
| Images | 7178 | Every labeled image discovered |
| Processing errors | 0 | Counted as incorrect, not silently excluded |
| Accuracy | 0.5977 | Correct predictions / all images |
| Macro-F1 | 0.5985 | Equal-weight mean F1 over seven classes |
| Weighted-F1 | 0.5959 | F1 weighted by class support |
| Mean latency | 56.80 ms | Successful pipeline calls, excluding file reads and model load |

## Per-class metrics

| Label | Precision | Recall | F1 | Support |
| :--- | ---: | ---: | ---: | ---: |
| neutral | 0.5395 | 0.5434 | 0.5414 | 1233 |
| happiness | 0.7326 | 0.8016 | 0.7655 | 1774 |
| sadness | 0.4819 | 0.5004 | 0.4910 | 1247 |
| surprise | 0.7837 | 0.7196 | 0.7503 | 831 |
| anger | 0.4937 | 0.4885 | 0.4911 | 958 |
| disgust | 0.8649 | 0.5766 | 0.6919 | 111 |
| fear | 0.4852 | 0.4336 | 0.4580 | 1024 |

## Confusion matrix

Rows are reference labels; columns are predictions. A processing-error column is added only if needed.

| True / predicted | neutral | happiness | sadness | surprise | anger | disgust | fear |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| neutral | 670 | 150 | 177 | 26 | 99 | 0 | 111 |
| happiness | 91 | 1422 | 103 | 35 | 71 | 0 | 52 |
| sadness | 202 | 131 | 624 | 18 | 134 | 3 | 135 |
| surprise | 41 | 50 | 33 | 598 | 35 | 1 | 73 |
| anger | 118 | 110 | 146 | 18 | 468 | 4 | 94 |
| disgust | 5 | 5 | 6 | 1 | 24 | 64 | 6 |
| fear | 115 | 73 | 206 | 67 | 117 | 2 | 444 |
