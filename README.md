# 🎬 IMDB Reviews — EDA + Classical Models + BiLSTM Baseline

A practical notebook for **binary sentiment analysis** on the classic **IMDB 50K Reviews** dataset.  
Clean EDA → strong classical baselines (NB / LogReg / Linear SVM + calibration) → **F1-based threshold tuning** → **explainability** → optional **BiLSTM** baseline.

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue)](#)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](#)
[![CI](https://img.shields.io/badge/CI-safe--ci-success)](#)

---

## 🚀 Why this notebook?
- **Kaggle-friendly**: path-flexible loading, deterministic seeds, artifacts saved.  
- **Clear EDA**: class distribution, text lengths, top n-grams.  
- **Strong baselines**: TF-IDF + MultinomialNB / Logistic Regression / Linear SVM (calibrated).  
- **Robust evaluation**: stratified CV, ROC/PR curves, F1-optimized threshold, calibration plot, Brier score.  
- **Explainability**: top weighted terms from Logistic Regression (no leakage).  
- **Error analysis**: quick FP/FN peek.  
- **Deep learning (optional)**: compact **BiLSTM** baseline with tokenization, embedding, learning curves, confusion matrix.  

---

## 📂 Dataset
- **Source file**: `IMDB Dataset.csv`  
- **Rows**: 50,000  
- **Columns**:
  - `review` — raw movie review text  
  - `sentiment` — `positive` / `negative`  

> The dataset file is **not included** in this repo.  
> For local runs, place it under: `data/raw/IMDB Dataset.csv`

Data loading supports **local** `data/raw/` and **Kaggle** `/kaggle/input/` via `repo_utils/pathing.py`.

---

## 📁 Repo layout

```text
.
├── text-sentiment-classification.ipynb
├── data/
│   └── raw/               # put IMDB Dataset.csv here (local runs)
├── artifacts/             # saved models / vectorizer / tables
├── repo_utils/
│   └── pathing.py         # local + Kaggle path helpers
├── CASE_STUDY.md
├── requirements.txt
├── requirements-dev.txt
└── .gitignore
```

---

## 🧱 Notebook Outline
1. **Setup & Imports**  
2. **Load & Peek**  
3. **Light Cleaning** (HTML strip, lowercasing, punctuation/digits removal; optional stopwords & lemmatization)  
4. **EDA** (distributions, text lengths, n-grams)  
5. **Vectorization** (TF-IDF)  
6. **Classical Models** (NB / LogReg / LinearSVM with calibration, stratified CV)  
7. **Holdout Evaluation** (metrics, ROC/PR curves, confusion matrix)  
8. **Calibration & Brier score**  
9. **Threshold tuning (F1)**  
10. **Explainability (LogReg coefficients)**  
11. **Error analysis (FP/FN)**  
12. **BiLSTM Baseline** (2 epochs)  
13. **Artifacts saved** (vectorizer, best model, summary CSV)  

---

## 🛠️ Environment
- **Python**: 3.10–3.12  
- **Core**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`  
- **NLP (optional)**: `contractions`, `nltk`  
- **DL (optional)**: `tensorflow>=2.15`  

```bash
pip install -r requirements.txt
```

Notes:
- For **classical models only**, `requirements.txt` is enough.
- To run the **BiLSTM** section, install TensorFlow separately:
  ```bash
  pip install "tensorflow>=2.15"
  ```

---

## ⚡ Quick Start
```bash
git clone https://github.com/tarekmasryo/text-sentiment-analysis.git
cd text-sentiment-analysis

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook text-sentiment-classification.ipynb
```

- Place `IMDB Dataset.csv` under `data/raw/` if not running on Kaggle.
- Alternatively, set a full path with `DATA_PATH`:
  - Windows (PowerShell): `$env:DATA_PATH="C:\path\IMDB Dataset.csv"`
  - macOS/Linux: `export DATA_PATH="/path/IMDB Dataset.csv"`

---

## ✅ Quality checks (optional)

These checks are lightweight and **do not** run the notebook (no data required):

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
```

Notes:
- Ruff is configured to **exclude `.ipynb`** files (CI stays stable).
- Auto-fix import order and simple issues:
  ```bash
  ruff check . --fix
  ```

---

## 📈 Outputs & Artifacts
- **CV table**: mean ± std for Accuracy / F1 / ROC-AUC across folds.  
- **Curves**: ROC, Precision-Recall, Calibration.  
- **Confusion matrices**: default 0.5 and F1-optimized threshold.  
- **Explainability**: top +/− terms from Logistic Regression.  
- **Saved artifacts** (in `artifacts/`): vectorizer, best model, metrics tables.

**Example results (from one run):**  
- LinearSVM (calibrated): Acc ≈ 0.90 · F1 ≈ 0.90  
- BiLSTM (2 epochs): Acc ≈ 0.85  

---

## 🔍 Notes on Methodology
- **No leakage**: vectorizer fit on train only.  
- **Calibration**: SVM calibrated; Brier score reported.  
- **Thresholding**: F1-optimal threshold from PR curve.  
- **Reproducibility**: `SEED=42`, stratified splits.  

---

## 🧾 Case Study
See **CASE_STUDY.md** for the project story, decisions, and takeaways (without repeated run steps).

## 🙌 Credits
Dataset: **IMDB 50K Reviews** (Kaggle).  
Author: **Tarek Masryo** · GitHub / Kaggle / HuggingFace
