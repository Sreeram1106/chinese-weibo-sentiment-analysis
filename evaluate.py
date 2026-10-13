"""Evaluation helpers: metrics, bootstrap CIs, plots, error analysis, feature inspection."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)

from .config import LABEL_NAMES


def tag(c: int) -> str:
    return f"{c}: {LABEL_NAMES.get(c, '?')}"


def acc_f1(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 42) -> dict:
    """95% percentile bootstrap CI for accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {"accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T
    df.index = [tag(int(i)) if str(i).isdigit() else i for i in df.index]
    return df


def plot_confusion(y_true, y_pred, classes, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [tag(c) for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), labels)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return cm


def plot_model_comparison(df: pd.DataFrame, path):
    d = df.sort_values("validation_macro_f1")
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(d) + 1.5))
    ax.barh(d["model"], d["validation_macro_f1"], color="#4C78A8")
    ax.set_xlim(max(0, d["validation_macro_f1"].min() - 0.05), d["validation_macro_f1"].max() + 0.02)
    ax.set_xlabel("Validation macro-F1"); ax.set_title("Model search (de-duplicated data)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_per_class_f1(report: pd.DataFrame, path):
    d = report.loc[[i for i in report.index if i[0].isdigit()]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(d.index, d["f1-score"], color="#59A14F")
    ax.set_ylim(0, 1); ax.set_ylabel("F1 (test)"); ax.set_title("Per-class F1 on held-out test set")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_class_distribution(raw_counts, clean_counts, path):
    cls = sorted(raw_counts.index)
    x = np.arange(len(cls)); w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [raw_counts[c] for c in cls], w, label="raw file", color="#BAB0AC")
    ax.bar(x + w / 2, [clean_counts[c] for c in cls], w, label="after de-duplication", color="#E15759")
    ax.set_xticks(x, [tag(c) for c in cls], rotation=35, ha="right")
    ax.set_ylabel("posts"); ax.set_title("Class distribution before / after cleaning"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def top_features(pipe, classes, k: int = 12) -> pd.DataFrame:
    """Most positively weighted n-grams per class for a fitted linear pipeline."""
    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["feat"].get_feature_names_out()
    names = np.array([n.split("__", 1)[-1] for n in names])
    rows = []
    for ci, c in enumerate(clf.classes_):
        for rank, j in enumerate(np.argsort(clf.coef_[ci])[::-1][:k], 1):
            rows.append({"class": tag(int(c)), "rank": rank, "feature": names[j], "weight": float(clf.coef_[ci][j])})
    return pd.DataFrame(rows)
