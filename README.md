# Chinese Weibo 7-Class Emotion Classifier

ML mini-project: classify Chinese microblog (Weibo) posts into seven emotion classes using classical NLP
(jieba / character n-grams, TF-IDF, linear SVM / logistic regression / Naive Bayes), with a leakage-free evaluation
protocol and an interactive web demo.

> **Scope, stated honestly.** The supplied `data/labeledData.csv` is a *mixed-topic* Chinese Weibo corpus
> (≈5% of posts mention COVID-related words). The labels are the numeric IDs 0–6. This project is a 7-class emotion
> experiment on that corpus; it is **not** a validated COVID-vaccination sentiment model.
> Class names shown in the demo (fear, disgust, happiness, surprise, like/gratitude, sadness, anger) are
> **tentative** – inferred from each class's most characteristic words (`results/top_features_per_class.csv`), not from
> official dataset documentation.

## Quick start (demo works immediately – trained model and results are included)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py                                   # web demo  (http://localhost:8501)
python -m src.predict_demo --text "感谢医护人员，向你们致敬！"   # CLI demo
python -m src.predict_demo                             # interactive CLI
```

> If loading the bundled model ever fails (different scikit-learn version), just run `python -m src.train` once (~1 min) to rebuild it.

Reproduce everything from scratch (~3 min on a laptop CPU): `python run_all.py`  ·  run tests: `pytest -q`

## Headline results (held-out test set, 2,113 unique posts)

| Metric | Value |
|---|---|
| Accuracy | **68.7 %** (95 % bootstrap CI 66.8 – 70.6) |
| Macro-F1 | **0.672** (CI 0.650 – 0.693) |
| 5-fold CV macro-F1 (train+val) | 0.683 ± 0.016 |
| Majority-class baseline accuracy | 28.3 % |
| Best classes (F1) | like/gratitude 0.86 · happiness 0.80 |
| Hardest classes (F1) | surprise 0.51 · fear 0.58 |
| Most common confusions | disgust ↔ anger, sadness → happiness |

Selected model: **char 1–3-gram TF-IDF + LinearSVC (C = 0.3, balanced class weights) with sigmoid-calibrated probabilities**,
chosen on *validation* macro-F1 among 19 candidates; the test set was scored once.

### The big finding: the original evaluation was inflated by duplicate posts
48 % of the rows in the raw file are repeated texts (e.g. one "致敬！…" post appears 43 times), so a random split puts the
same post in both train and test. Re-running the original-style approach under both protocols:

| Model | Original split (with duplicates) | Clean split (de-duplicated) |
|---|---|---|
| Original-style (word TF-IDF + LR, emoticons deleted) | 78.5 % acc. (57 % of test rows also in train) | **58.9 %** |
| **Improved (this repo)** | 84.9 % acc. | **68.7 %** |

On the original split the old model scored 92 % on leaked rows but only 61 % on unseen rows. The improved model is
~10 accuracy points better under honest evaluation. (`results/leakage_audit.csv`, `python -m src.leakage_audit`)

## What changed vs. the first version → see [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md)

## Project layout
```
app.py                    Streamlit demo (single post, CSV batch, results dashboard)
run_all.py                reproduce audit → training → leakage audit
src/config.py             paths, seed, tentative label names, COVID keyword list
src/preprocess.py         cleaning (keeps [emoticons]), jieba tokenizer, dedup key
src/data.py               loading, de-duplication, conflict removal, stratified split
src/models.py             19 candidate pipelines (NB / LogReg / LinearSVC × word / char / word+char)
src/train.py              model selection, single test evaluation, CV, ablation, error analysis, saves model
src/evaluate.py           metrics, bootstrap CI, plots, top features
src/leakage_audit.py      2×2 experiment quantifying the duplicate-leakage inflation
src/predict.py            inference API (top-k probabilities)
src/predict_demo.py       CLI demo
src/inspect_dataset.py    dataset audit
tests/                    11 pytest tests (cleaning, de-dup, leakage-free splits, inference)
results/                  metrics, figures, error lists, top features per class
models/best_model.joblib  trained model (4.5 MB)
docs/                     IMPROVEMENTS.md · VIVA_NOTES.md · PAPER_NOTES.md
```

## Outputs in `results/`
`summary.json` · `model_comparison.csv/.png` · `test_classification_report.csv` · `confusion_matrix.png` ·
`per_class_f1.png` · `class_distribution.png` · `leakage_audit.csv/.json` · `ablation_emoticons.csv` ·
`top_features_per_class.csv` · `confident_errors.csv` · `covid_keyword_sample.csv` · `dataset_audit.json`

## Limitations
1. Label names are tentative; verify against the dataset's original documentation before publishing them.
2. Mixed-topic data: results do not transfer automatically to COVID/vaccination posts. A keyword-screened subset
   (127 test posts) scores 79.5 % accuracy, but it is small and skewed to class 4, so treat it as indicative only.
3. The file has no date column, so a time-trend analysis (as in the reference paper) is not possible here.
4. ~69 % accuracy is near the ceiling for bag-of-n-gram models on short, noisy, ambiguous emotion text. Next steps:
   a Chinese pre-trained transformer (e.g. `hfl/chinese-roberta-wwm-ext`) – not used here because it needs a GPU /
   model download.
5. Short posts (<5 characters) and sarcasm are frequent error sources (`results/confident_errors.csv`).
