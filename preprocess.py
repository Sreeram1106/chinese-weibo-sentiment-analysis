"""Chinese text cleaning, normalisation and tokenisation.

Design notes
------------
* Weibo emoticons such as ``[good]`` / ``[委屈]`` are *kept* as a single token
  (``EMO_good``).  They are among the strongest emotion cues in the data, and
  the original project deleted them.
* URLs and @mentions carry no emotion and are removed.
* Tokenisation is done with jieba; results are memoised so repeated fits
  (grid search / CV) do not re-segment the same sentence.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import jieba

jieba.setLogLevel(60)  # silence "Building prefix dict" noise

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@[\w\-\u4e00-\u9fff]+")
_EMOTICON = re.compile(r"\[([^\[\]\s]{1,8})\]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_REPEAT_PUNCT = re.compile(r"([!?！？。.~，,])\1{2,}")
_REPEAT_CHAR = re.compile(r"(.)\1{3,}")
_SPACES = re.compile(r"\s+")
_KEEP_FOR_KEY = re.compile(r"[^\w\u4e00-\u9fff]+")


def clean_text(text: object, keep_emoticons: bool = True) -> str:
    """Normalise a raw post.  Returns "" for empty / non-text input.

    ``keep_emoticons=False`` reproduces the original project's behaviour
    (emoticons deleted) and is only used for the ablation study."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _ZERO_WIDTH.sub("", text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    if keep_emoticons:
        text = _EMOTICON.sub(lambda m: f" EMO_{m.group(1)} ", text)
    else:
        text = _EMOTICON.sub(" ", text)
    text = _REPEAT_PUNCT.sub(r"\1\1", text)      # "!!!!!" -> "!!"
    text = _REPEAT_CHAR.sub(r"\1\1\1", text)      # "哈哈哈哈哈哈" -> "哈哈哈"
    return _SPACES.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Aggressive key used to find (near-)duplicate posts."""
    return _KEEP_FOR_KEY.sub("", clean_text(text)).lower()


@lru_cache(maxsize=200_000)
def _segment(cleaned: str) -> tuple[str, ...]:
    return tuple(t for t in (w.strip() for w in jieba.lcut(cleaned)) if t)


class ChineseTokenizer:
    """Picklable callable: raw text -> list of jieba tokens (after cleaning)."""

    def __call__(self, text: str) -> list[str]:
        return list(_segment(clean_text(text)))


class TextCleaner:
    """Picklable callable used as ``preprocessor`` for char n-gram features."""

    def __init__(self, keep_emoticons: bool = True):
        self.keep_emoticons = keep_emoticons

    def __call__(self, text: str) -> str:
        return clean_text(text, keep_emoticons=self.keep_emoticons)
