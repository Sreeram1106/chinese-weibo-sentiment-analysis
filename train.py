"""Train, select, evaluate and save the Chinese 7-class emotion classifier.

Protocol (no test-set peeking):
  1. clean + de-duplicate  ->  stratified 60/20/20 split
  2. fit every candidate on TRAIN, rank by VALIDATION macro-F1
  3. fit the winner (with calibrated probabilities) on TRAIN, score ONCE on TEST
  4. report bootstrap CIs, 5-fold CV on train+val, ablations, error analysis
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import evaluate as ev
from .config import (COVID_TERMS, LABEL_NAMES, LABEL_NAMES_ARE_VERIFIED, MODEL_PATH, MODELS_DIR,
                     RESULTS_DIR, SEED)
from .data import load_splits
from .models import build_candidates, char_tfidf, with_probabilities

warnings.filterwarnings("ignore")


def main():
    RESULTS_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)
    raw, clean, (train, val, test), audit = load_splits()
    classes = sorted(clean["label"].unique())
    print(f"Raw rows {audit['raw_rows']:,} -> {audit['after_dedup_rows']:,} unique "
          f"({audit['duplicate_share_removed']:.1%} removed). Split {len(train)}/{len(val)}/{len(test)}")

    ev.plot_class_distribution(raw["label"].value_counts(), clean["label"].value_counts(),
                               RESULTS_DIR / "class_distribution.png")

    # ---- 1. model search on validation ---------------------------------------------------
    rows, fitted = [], {}
    for name, pipe in build_candidates().items():
        pipe.fit(train["text"], train["label"])
        pred = pipe.predict(val["text"])
        m = ev.acc_f1(val["label"], pred)
        rows.append({"model": name, "validation_accuracy": m["accuracy"], "validation_macro_f1": m["macro_f1"]})
        fitted[name] = pipe
        print(f"  {name:32s} val acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    dpred = dummy.predict(val["text"])
    rows.append({"model": "Majority-class baseline", "validation_accuracy": accuracy_score(val["label"], dpred),
                 "validation_macro_f1": f1_score(val["label"], dpred, average="macro")})
    comp = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    comp.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    ev.plot_model_comparison(comp, RESULTS_DIR / "model_comparison.png")
    best_name = comp.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"\nSelected (validation macro-F1): {best_name}")

    # ---- 2. interpretable weights ---------------------------------------------------------
    ev.top_features(best_pipe, classes).to_csv(RESULTS_DIR / "top_features_per_class.csv", index=False)

    # ---- 3. final model (probabilities) -> single test evaluation -------------------------
    final = with_probabilities(clone(best_pipe)).fit(train["text"], train["label"])
    test_pred = final.predict(test["text"])
    test_m = ev.acc_f1(test["label"], test_pred)
    ci = ev.bootstrap_ci(test["label"], test_pred)
    report = ev.per_class_report(test["label"], test_pred)
    report.to_csv(RESULTS_DIR / "test_classification_report.csv")
    ev.plot_confusion(test["label"], test_pred, classes, RESULTS_DIR / "confusion_matrix.png",
                      "Held-out test confusion matrix (de-duplicated)")
    ev.plot_per_class_f1(report, RESULTS_DIR / "per_class_f1.png")
    print(f"TEST accuracy={test_m['accuracy']:.4f} {ci['accuracy_ci95']}  macro-F1={test_m['macro_f1']:.4f} {ci['macro_f1_ci95']}")

    # ---- 4. stability: 5-fold CV on train+val -------------------------------------------
    trval = pd.concat([train, val], ignore_index=True)
    cv = cross_val_score(clone(best_pipe), trval["text"], trval["label"], scoring="f1_macro",
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED))

    # ---- 5. ablation: does keeping Weibo emoticons help? ---------------------------------
    clf_params = best_pipe.named_steps["clf"].get_params()
    abl = []
    for label, keep in (("emoticons kept (ours)", True), ("emoticons deleted (original cleaning)", False)):
        feat = char_tfidf(); feat.set_params(preprocessor=__import__("src.preprocess", fromlist=["x"]).TextCleaner(keep))
        p = Pipeline([("feat", feat), ("clf", clone(best_pipe.named_steps["clf"]))]).fit(train["text"], train["label"])
        abl.append({"variant": label, **{f"val_{k}": v for k, v in ev.acc_f1(val["label"], p.predict(val["text"])).items()}})
    pd.DataFrame(abl).to_csv(RESULTS_DIR / "ablation_emoticons.csv", index=False)

    # ---- 6. COVID-keyword subset robustness check ----------------------------------------
    pattern = "|".join(COVID_TERMS)
    mask = test["text"].str.contains(pattern, case=False, regex=True).values
    covid = {"n_covid_keyword_test_rows": int(mask.sum()), "n_other_test_rows": int((~mask).sum())}
    if mask.sum() >= 30:
        covid["covid_keyword_subset"] = ev.acc_f1(test["label"][mask], test_pred[mask])
        covid["class_counts_covid_subset"] = {int(k): int(v) for k, v in test["label"][mask].value_counts().sort_index().items()}
    covid["other_subset"] = ev.acc_f1(test["label"][~mask], test_pred[~mask])
    covid["caveat"] = "Keyword screen only; small subset; not a validated COVID/vaccination test set."

    # ---- 7. error analysis -----------------------------------------------------------------
    proba = final.predict_proba(test["text"])
    err = test.assign(pred=test_pred, confidence=proba.max(axis=1))
    err = err[err["label"] != err["pred"]].sort_values("confidence", ascending=False)
    err["true_name"] = err["label"].map(LABEL_NAMES); err["pred_name"] = err["pred"].map(LABEL_NAMES)
    err.head(100).to_csv(RESULTS_DIR / "confident_errors.csv", index=False, encoding="utf-8-sig")
    pairs = err.groupby(["label", "pred"]).size().sort_values(ascending=False).head(8)
    top_conf = [{"true": ev.tag(int(a)), "predicted": ev.tag(int(b)), "count": int(n)} for (a, b), n in pairs.items()]

    # ---- 8. persist ---------------------------------------------------------------------------
    joblib.dump({"model": final, "classes": [int(c) for c in classes], "name": best_name}, MODEL_PATH, compress=3)
    summary = {
        "selected_model": best_name,
        "data_audit": audit,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "validation": comp.iloc[0][["validation_accuracy", "validation_macro_f1"]].to_dict(),
        "test": {**test_m, **ci},
        "cv5_macro_f1_train_plus_val": {"mean": float(cv.mean()), "std": float(cv.std())},
        "majority_baseline_val_accuracy": float(comp[comp.model == "Majority-class baseline"].validation_accuracy.iloc[0]),
        "covid_keyword_check": covid,
        "most_common_confusions": top_conf,
        "label_names_verified": LABEL_NAMES_ARE_VERIFIED,
        "note": "Metrics are on a de-duplicated, mixed-topic Chinese Weibo corpus; they are NOT COVID-vaccination-specific.",
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CV macro-F1 {cv.mean():.4f} +/- {cv.std():.4f}\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
