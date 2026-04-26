# 🎬 IMDB Sentiment Classification

A practical notebook for **binary sentiment analysis** on the **IMDB 50K Movie Reviews** dataset.  
The focus is not just getting a high score, but building a clean and reliable workflow: data checks → leakage-safe pipelines → validation-based model and threshold selection → final test evaluation → reusable artifacts.

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue)](#)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](#)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](#)
[![CI](https://img.shields.io/badge/CI-safe--ci-success)](#)

---

## 🚀 Why this notebook?

- **Clean data checks**: missing values, duplicates, label conflicts, and cleaned-text duplicates.
- **Leakage-safe workflow**: text preprocessing and TF-IDF stay inside the model pipeline.
- **Reliable splitting**: train / validation / test split with the test set used only at the end.
- **Strong classical baselines**:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Calibrated Linear SVM
- **Validation-based selection**: model and threshold are selected on validation data, not the test split.
- **Final evaluation**: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix, ROC/PR curves, and calibration curve.
- **Readable analysis**: distinctive terms, feature signals, error analysis, and model behavior checks.
- **Reusable artifacts**: saved pipeline, threshold, metrics, training config, model card, and reload smoke test.

---

## 📂 Dataset

- **Dataset**: IMDB Dataset of 50K Movie Reviews from Kaggle
- **Source file**: `IMDB Dataset.csv`
- **Rows**: 50,000 before cleaning
- **Columns**:
  - `review` — raw movie review text
  - `sentiment` — `positive` / `negative`

> The dataset file is **not included** in this repository.

For Kaggle, add the dataset as an input before running the notebook.

For local runs, place the file here:

```text
data/raw/IMDB Dataset.csv
```

Then set this near the top of the notebook:

```python
DATA_PATH_OVERRIDE = "data/raw/IMDB Dataset.csv"
```

---

## 📁 Repository layout

```text
.
├── imdb_sentiment_classification.ipynb
├── src/
│   └── sentiment_lab/
│       ├── __init__.py
│       └── preprocessing.py
├── data/
│   └── raw/               # local dataset location; CSV is not tracked
├── artifacts/             # generated outputs after running the notebook
├── CASE_STUDY.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .github/
    └── workflows/
        └── safe-ci.yml
```

---

## 🧱 Notebook outline

1. **Setup & imports**
2. **Dataset loading**
3. **Data quality checks**
4. **Text cleaning preview**
5. **Exploratory analysis**
6. **Train / validation / test split**
7. **Leakage-safe model pipelines**
8. **Cross-validation on the training split**
9. **Validation-based model and threshold selection**
10. **Final test evaluation**
11. **ROC, PR, calibration, and confusion matrix**
12. **Feature signal inspection**
13. **Error analysis**
14. **Model behavior checks**
15. **Artifact saving**
16. **Reload smoke test**
17. **Inference demo**
18. **Final summary**

---

## 🛠️ Environment

- **Python**: 3.10–3.12
- **Core libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `joblib`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## ⚡ Quick start

```bash
git clone https://github.com/tarekmasryo/text-sentiment-analysis.git
cd text-sentiment-analysis

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook imdb_sentiment_classification.ipynb
```

For local runs, place the dataset at:

```text
data/raw/IMDB Dataset.csv
```

and set:

```python
DATA_PATH_OVERRIDE = "data/raw/IMDB Dataset.csv"
```

---

## 📈 Outputs & artifacts

A full notebook run saves generated files under `artifacts/`:

```text
sentiment_pipeline.joblib
threshold.json
metrics.json
training_config.json
model_card.md
reload_smoke_test.py
```

Typical held-out test results are around:

```text
F1:      ~0.91
ROC-AUC: ~0.97
PR-AUC:  ~0.97
```

Exact results can vary slightly by environment and library versions.

---

## 🔍 Methodology notes

- The data is validated and deduplicated before splitting.
- The EDA is descriptive only and is not used for model selection or threshold tuning.
- Text preprocessing stays inside the `sklearn` pipeline.
- Cross-validation is run on the training split only.
- The model and threshold are selected using validation data.
- The test split is used once for the final estimate.
- The saved pipeline is reloaded and checked before the notebook ends.

---

## ⚠️ Limitations

The model works best when the input is close to the data used here: **English movie reviews**.

For another domain, such as product reviews, social media comments, support tickets, or Arabic text, the model should be evaluated again on representative data before its predictions are used.

---

## ✅ Development checks

These checks are lightweight and do not require the dataset:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m compileall src
```

The CI workflow runs these checks without executing the notebook.

---

## 🧾 Case study

See [`CASE_STUDY.md`](CASE_STUDY.md) for the project story, decisions, and takeaways.

---

## 🙌 Credits

Dataset: **IMDB Dataset of 50K Movie Reviews** from Kaggle.  
Author: **Tarek Masryo**
