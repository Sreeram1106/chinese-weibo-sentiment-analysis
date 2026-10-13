import pytest

from src.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")


def test_predict_structure_and_probabilities():
    from src.predict import predict
    r = predict("感谢每一位医护人员的无私奉献，向你们致敬！")
    assert r["label"] in range(7)
    assert len(r["all"]) == 7
    assert abs(sum(x["probability"] for x in r["all"]) - 1) < 1e-6
    assert r["confidence"] == r["top"][0]["probability"]


def test_sanity_examples():
    from src.predict import predict
    assert predict("气死我了！！太过分了")["name"] in {"anger", "disgust"}
    assert predict("太开心了哈哈哈哈")["name"] == "happiness"


def test_empty_input_rejected():
    from src.predict import predict
    with pytest.raises(ValueError):
        predict("   ")


def test_summary_is_consistent_with_split():
    import json
    from src.config import RESULTS_DIR
    s = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    assert s["split_sizes"]["train"] + s["split_sizes"]["validation"] + s["split_sizes"]["test"] == s["data_audit"]["after_dedup_rows"]
    assert s["test"]["accuracy"] > s["majority_baseline_val_accuracy"] + 0.2
