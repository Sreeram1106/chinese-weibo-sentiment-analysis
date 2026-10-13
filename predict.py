"""Inference API used by the CLI, the web demo and the tests."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np

from .config import LABEL_NAMES, MODEL_PATH


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    if not MODEL_PATH.exists() and path == MODEL_PATH:
        raise FileNotFoundError("Model not found. Run:  python -m src.train")
    try:
        return joblib.load(path)
    except Exception as e:  # typically a scikit-learn version mismatch
        raise RuntimeError("Could not load models/best_model.joblib (likely a scikit-learn version mismatch). "
                           "Regenerate it on this machine with:  python -m src.train   (~1 min)") from e


def predict_proba(texts: list[str], bundle=None) -> np.ndarray:
    bundle = bundle or load_model()
    return bundle["model"].predict_proba(texts)


def predict(text: str, top_k: int = 3, bundle=None) -> dict:
    """Classify one post.  Returns the winner plus top-k (id, name, probability)."""
    bundle = bundle or load_model()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a non-empty text.")
    proba = bundle["model"].predict_proba([text])[0]
    classes = bundle["model"].classes_
    order = np.argsort(proba)[::-1]
    ranked = [{"label": int(classes[i]), "name": LABEL_NAMES.get(int(classes[i]), "?"),
               "probability": float(proba[i])} for i in order]
    return {"text": text, "label": ranked[0]["label"], "name": ranked[0]["name"],
            "confidence": ranked[0]["probability"], "top": ranked[:top_k], "all": ranked}
