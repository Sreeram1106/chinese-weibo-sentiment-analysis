"""Quantify how much the ORIGINAL evaluation protocol was inflated by duplicate posts.

2x2 experiment:   {original-style model, improved model}  x  {original split, clean split}
The original split = random 60/20/20 on the raw file (48% duplicate texts -> leakage).
"""
from __future__ import annotations

import json
import re
import warnings

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import RESULTS_DIR, SEED
from .data import clean_dataset, load_raw, split_data
from .models import char_tfidf
from .preprocess import dedup_key

warnings.filterwarnings("ignore")


class OriginalTokenizer:
    """Re-implementation of the original project's cleaning (emoticons deleted) + jieba."""
    def __call__(self, text):
        t = re.sub(r"https?://\S+", " ", str(text))
        t = re.sub(r"@[\w\-\u4e00-\u9fff]+", " ", t)
        t = re.sub(r"\[[^\[\]]{1,8}\]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return [w.strip() for w in jieba.lcut(t) if w.strip()]


def original_style_model():
    return Pipeline([("f", TfidfVectorizer(tokenizer=OriginalTokenizer(), token_pattern=None, max_features=150000,
                                           ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def improved_model():
    return Pipeline([("feat", char_tfidf()), ("clf", LinearSVC(C=0.3, class_weight="balanced", random_state=SEED))])


def run_cell(make_model, train, val, test):
    m = make_model().fit(train["text"], train["label"])
    pred = m.predict(test["text"])
    keys = set(train["text"].map(dedup_key))
    leaked = test["text"].map(dedup_key).isin(keys).values
    out = {"test_accuracy": accuracy_score(test["label"], pred),
           "test_macro_f1": f1_score(test["label"], pred, average="macro"),
           "test_rows_also_in_train": float(leaked.mean())}
    if leaked.any() and (~leaked).any():
        out["acc_on_leaked_rows"] = accuracy_score(test["label"][leaked], pred[leaked])
        out["acc_on_unseen_rows"] = accuracy_score(test["label"][~leaked], pred[~leaked])
    return {k: round(float(v), 4) for k, v in out.items()}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    raw = load_raw()
    # original protocol: split the raw file directly
    tr, tmp = train_test_split(raw, test_size=0.4, random_state=SEED, stratify=raw["label"])
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=tmp["label"])
    leaky = (tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True))
    clean, _ = clean_dataset(raw)
    proper = split_data(clean)

    res = {}
    for mname, mk in (("original-style (word TF-IDF + LR, emoticons deleted)", original_style_model),
                      ("improved (char TF-IDF + LinearSVC, emoticons kept)", improved_model)):
        for pname, splits in (("original split (with duplicates)", leaky), ("clean split (de-duplicated)", proper)):
            print(f"{mname} | {pname}")
            res[f"{mname} | {pname}"] = run_cell(mk, *splits)
            print("  ", res[f"{mname} | {pname}"])
    (RESULTS_DIR / "leakage_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"model": k.split(" | ")[0], "protocol": k.split(" | ")[1], **v} for k, v in res.items()]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "leakage_audit.csv", index=False)


if __name__ == "__main__":
    main()
