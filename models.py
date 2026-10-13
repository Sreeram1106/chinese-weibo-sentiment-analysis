"""Model factory: every candidate is a single sklearn Pipeline that accepts RAW text."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import SEED
from .preprocess import ChineseTokenizer, TextCleaner


def word_tfidf(ngram=(1, 2)):
    return TfidfVectorizer(tokenizer=ChineseTokenizer(), token_pattern=None, lowercase=False,
                           ngram_range=ngram, min_df=2, sublinear_tf=True, max_features=200_000)


def char_tfidf(ngram=(1, 3)):
    return TfidfVectorizer(analyzer="char", preprocessor=TextCleaner(), ngram_range=ngram,
                           min_df=2, sublinear_tf=True, max_features=300_000)


def features(kind: str):
    if kind == "word":
        return word_tfidf()
    if kind == "char":
        return char_tfidf()
    if kind == "word+char":
        return FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())])
    raise ValueError(kind)


def build_candidates() -> dict[str, Pipeline]:
    """Return {name: untrained pipeline}.  Names encode the hyper-parameters."""
    cands: dict[str, Pipeline] = {}
    cands["ComplementNB | word"] = Pipeline([("feat", features("word")), ("clf", ComplementNB(alpha=0.3))])
    for fk in ("word", "char", "word+char"):
        for C in (1, 3, 10):
            cands[f"LogReg C={C} | {fk}"] = Pipeline([
                ("feat", features(fk)),
                ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED))])
        for C in (0.1, 0.3, 1):
            cands[f"LinearSVC C={C} | {fk}"] = Pipeline([
                ("feat", features(fk)),
                ("clf", LinearSVC(C=C, class_weight="balanced", random_state=SEED))])
    return cands


def with_probabilities(pipe: Pipeline) -> Pipeline:
    """Give margin-only models (LinearSVC) a predict_proba via calibration."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe
    return Pipeline([("feat", pipe.named_steps["feat"]),
                     ("clf", CalibratedClassifierCV(clf, cv=3, method="sigmoid"))])
