"""Dataset loading, de-duplication and leakage-free splitting."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, SEED
from .preprocess import clean_text, dedup_key


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if not {"review", "label"} <= set(df.columns):
        raise ValueError("Expected columns: review,label")
    df = df.rename(columns={"review": "text"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove empties, (near-)duplicates and texts with contradictory labels.

    Returns the cleaned frame plus an audit dict describing what was removed.
    """
    audit = {"raw_rows": int(len(df))}
    df = df.copy()
    df["key"] = df["text"].map(dedup_key)
    df = df[df["key"].str.len() > 0]
    audit["after_empty_removal"] = int(len(df))

    n_labels = df.groupby("key")["label"].nunique()
    conflicting = n_labels[n_labels > 1].index
    audit["conflicting_label_texts_dropped"] = int(len(conflicting))
    df = df[~df["key"].isin(conflicting)]

    df = df.drop_duplicates(subset="key", keep="first")
    audit["after_dedup_rows"] = int(len(df))
    audit["duplicate_share_removed"] = round(1 - audit["after_dedup_rows"] / audit["raw_rows"], 4)
    df = df.drop(columns="key").reset_index(drop=True)
    df["text"] = df["text"].map(lambda t: t.strip())
    return df, audit


def split_data(df: pd.DataFrame, seed: int = SEED):
    """Stratified 60/20/20 split.  Because ``df`` is de-duplicated, no text can
    appear in more than one split."""
    train, tmp = train_test_split(df, test_size=0.4, random_state=seed, stratify=df["label"])
    val, test = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_splits(seed: int = SEED):
    raw = load_raw()
    clean, audit = clean_dataset(raw)
    train, val, test = split_data(clean, seed)
    return raw, clean, (train, val, test), audit
