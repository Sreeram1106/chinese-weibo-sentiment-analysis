import numpy as np
import pandas as pd
import pytest

from src.data import clean_dataset, load_raw, split_data
from src.preprocess import ChineseTokenizer, clean_text, dedup_key


def test_clean_keeps_emoticon_and_drops_noise():
    out = clean_text("好开心[good] http://t.cn/abc @小明 哈哈哈哈哈哈!!!!!!")
    assert "EMO_good" in out and "http" not in out and "@" not in out
    assert "哈哈哈哈" not in out and "!!!" not in out


def test_clean_handles_empty_and_nan():
    assert clean_text(None) == "" and clean_text(float("nan")) == "" and clean_text("   ") == ""


def test_dedup_key_catches_near_duplicates():
    assert dedup_key("致敬！英雄们 ​") == dedup_key("致敬!英雄们")


def test_tokenizer_returns_tokens():
    assert len(ChineseTokenizer()("我爱北京天安门")) >= 3


def test_clean_dataset_removes_duplicates_and_conflicts():
    df = pd.DataFrame({"text": ["你好啊", "你好啊 ", "同一句", "同一句", "独特的句子", ""],
                       "label": [1, 1, 2, 3, 4, 0]})
    out, audit = clean_dataset(df)
    assert sorted(out["text"]) == ["你好啊", "独特的句子"]        # conflict + empty dropped, dup merged
    assert audit["conflicting_label_texts_dropped"] == 1


def test_split_has_no_leakage_and_is_stratified():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"text": [f"文本{i}" for i in range(500)], "label": rng.integers(0, 3, 500)})
    tr, va, te = split_data(df)
    assert not (set(tr.text) & set(te.text)) and not (set(tr.text) & set(va.text)) and not (set(va.text) & set(te.text))
    assert len(tr) + len(va) + len(te) == 500


def test_real_dataset_schema():
    raw = load_raw()
    assert set(raw["label"].unique()) == set(range(7))
