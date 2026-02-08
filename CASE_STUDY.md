# 🧠 Case Study — IMDB Sentiment (50K Reviews)

## Problem
Turn raw review text into a reliable **binary sentiment** signal (`positive` vs `negative`) with baselines that are:
- easy to reproduce
- strong enough to be a real reference point
- calibrated / threshold-aware when scores are used for decisions

---

## Data
- **Dataset:** IMDB 50K Reviews
- **Grain:** one row per review
- **Fields:** `review` (text), `sentiment` (label)

Practical notes:
- Text contains **HTML** and informal writing (negations, contractions, typos).
- Keep preprocessing **light** to avoid erasing signal (especially negation).

---

## Approach
### EDA
- Label balance, review length distribution
- Common tokens / n-grams (sanity checks, not “pretty charts”)

### Baselines
- **TF‑IDF** vectorization (fit on train only)
- Classical models:
  - MultinomialNB
  - Logistic Regression
  - Linear SVM (with optional calibration)

### Evaluation
- Holdout split + stratified CV for stability checks
- Metrics: Accuracy, F1, ROC‑AUC, PR‑AUC
- For score-based decisions: tune a **threshold** by F1 on the PR curve
- Calibration: reliability curve + Brier score (when probabilistic scores exist)

### Deep Learning (baseline, optional)
- Compact **BiLSTM** to compare against strong classical TF‑IDF baselines
- Kept intentionally minimal (baseline, not SOTA)

---

## Results (what to look for)
- Strong TF‑IDF + linear models typically dominate as a baseline.
- Calibration improves interpretability when using probabilities for downstream decisions.
- BiLSTM is included as a reference point, not as the default recommendation.

Artifacts saved:
- `artifacts/model_summary.csv`
- `artifacts/tfidf_vectorizer.joblib`
- `artifacts/<best_model>_model.joblib`

---

## Decisions & Takeaways
- **Default recommendation:** calibrated linear model on TF‑IDF (fast, strong, explainable).
- **Thresholding matters:** “0.5” is rarely optimal when optimizing F1 or operating constraints.
- **Explainability:** logistic regression coefficients are a quick sanity-check tool.

---

## Next Steps
- Add a small **error taxonomy** (negation, sarcasm, domain terms) from FP/FN samples.
- Try a compact transformer baseline (DistilBERT) for a modern reference, with the same evaluation discipline.
