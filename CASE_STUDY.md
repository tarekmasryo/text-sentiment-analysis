# Case Study — IMDB Sentiment Classification

## Problem

Build a binary sentiment classifier for English movie reviews while keeping the evaluation clean and reproducible.

The goal is not to create the most complex NLP model. The goal is to show a reliable workflow that avoids common mistakes such as preprocessing leakage, threshold selection on the test set, or publishing metrics without error analysis.

## Data

Dataset: IMDB Dataset of 50K Movie Reviews from Kaggle.

Expected columns:

- `review`: raw movie review text
- `sentiment`: `positive` or `negative`

The notebook checks missing values, exact duplicates, label conflicts, and duplicate cleaned reviews before the split.

## Approach

The final notebook uses classical NLP models because they are fast, explainable, and strong on this dataset.

Models compared:

- Multinomial Naive Bayes
- Logistic Regression
- Calibrated Linear SVM

All text preprocessing is part of the scikit-learn pipeline:

```text
TextCleaner → TF-IDF → Model
```

This keeps preprocessing consistent during cross-validation, final fitting, and inference.

## Evaluation

The workflow uses:

- Train / validation / test split
- Cross-validation on the training split
- Validation-based model and threshold selection
- Final test evaluation only at the end

Metrics include:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Brier score

The notebook also includes confusion matrix, calibration, feature signals, error analysis, and behavior checks by review length and prediction confidence.

## Artifacts

A full run saves:

```text
artifacts/sentiment_pipeline.joblib
artifacts/threshold.json
artifacts/metrics.json
artifacts/training_config.json
artifacts/model_card.md
```

A reload smoke test verifies that the saved pipeline can be loaded and used for inference.

## Main takeaway

Classical TF-IDF models remain strong for this task when the evaluation workflow is handled carefully.

The model should be treated as validated for English movie reviews only. Other domains or languages need their own evaluation before use.
