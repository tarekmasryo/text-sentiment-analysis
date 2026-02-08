# 🎬 IMDB Reviews — EDA + Classical Models + BiLSTM Baseline

A practical, opinionated notebook for **binary sentiment analysis** on the classic **IMDB 50K Reviews** dataset.  
Clean EDA → strong classical baselines (NB / LogReg / Linear SVM + calibration) → **threshold tuning by F1** → **explainability** → **BiLSTM** baseline.

---

## 🚀 Why this notebook?
- **Kaggle-ready**: path-flexible loading, deterministic seeds, artifacts saved.  
- **Clear EDA**: class distribution, text lengths, top n-grams.  
- **Strong baselines**: TF-IDF + MultinomialNB / Logistic Regression / Linear SVM (calibrated).  
- **Robust evaluation**: stratified 5-fold CV, ROC/PR curves, F1-optimized threshold, calibration plot, Brier score.  
- **Explainability**: top weighted terms from Logistic Regression (no leakage).  
- **Error analysis**: quick FP/FN peek.  
- **Deep learning**: compact **BiLSTM** baseline with tokenization, embedding, accuracy curves, confusion matrix.  

---

## 📂 Dataset
- **Source file**: `IMDB Dataset.csv`  
- **Rows**: 50,000  
- **Columns**:
  - `review` — raw movie review text  
  - `sentiment` — `positive` / `negative`  

> Data loading supports **local** `data/raw/` and **Kaggle** `/kaggle/input/` via `repo_utils/pathing.py`.

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
13. **Artifacts saved** (TF-IDF, best model, summary CSV)  

---

## 🛠️ Environment
- **Python**: 3.10–3.12  
- **Core**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`  
- **NLP (optional)**: `contractions`, `nltk`  
- **DL**: `tensorflow>=2.14`  

```bash
pip install -r requirements.txt
```

Notes:
- For **classical models only**, `requirements.txt` is enough.
- To run the **BiLSTM** section, install TensorFlow separately: `pip install tensorflow>=2.15`.


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

---

## 📈 Outputs & Artifacts
- **CV table** (`model_summary.csv`): mean ± std for Accuracy / F1 / ROC-AUC across folds.  
- **Curves**: ROC, Precision-Recall, Calibration.  
- **Confusion matrices**: default 0.5 and F1-optimized threshold.  
- **Explainability**: top +/− terms from Logistic Regression.  
- **Artifacts**:
  - `tfidf_vectorizer.joblib`
  - `<best_model>_model.joblib`
  - `model_summary.csv`

**Sample Results (holdout set):**  
- LinearSVM (calibrated): Acc = 0.908 · F1 = 0.908 · ROC-AUC = 0.967 · AP = 0.966  
- BiLSTM (2 epochs): Acc ≈ 0.855  

---

## 🔍 Notes on Methodology
- **No leakage**: vectorizer fit on train only.  
- **Calibration**: SVM calibrated with sigmoid; Brier score reported.  
- **Thresholding**: F1-optimal threshold from PR curve.  
- **Reproducibility**: `SEED=42`, stratified splits.  

---


---

## 🧾 Case Study
See **CASE_STUDY.md** for the project story, decisions, and takeaways (without repeated run steps).

## 🙌 Credits
Dataset: **IMDB 50K Reviews** (Kaggle).  
Author: **Tarek Masryo** · [GitHub](https://github.com/tarekmasryo) · [Kaggle](https://www.kaggle.com/tarekmasryo) · [HuggingFace](https://huggingface.co/TarekMasryo)
