"""Streamlit demo.   Run:  streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src.config import LABEL_NAMES, MODEL_PATH, RESULTS_DIR
from src.predict import load_model, predict

st.set_page_config(page_title="Weibo Emotion Classifier", page_icon="💬", layout="wide")

EXAMPLES = {
    "Happiness": "今天终于拿到录取通知书了，太开心了！[哈哈]",
    "Gratitude": "感谢每一位医护人员的无私奉献，向你们致敬！",
    "Anger": "什么破服务，气死我了，再也不来了！！",
    "Surprise": "居然是真的？？没想到会这样，太震惊了",
    "Sadness": "想起奶奶，眼泪忍不住流下来，真的好想她",
    "Fear": "半夜听到门外有声音，吓得我不敢动，好害怕",
    "Disgust": "这家店的东西太恶心了，真的很无聊又烦人",
}

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()
bundle = load_model()

st.title("💬 Chinese Weibo Emotion Classifier")
st.caption("7-class text classification · char n-gram TF-IDF + calibrated linear SVM · trained on a de-duplicated Weibo corpus")
st.warning("Class **names are tentative** (inferred from each class's most characteristic words); the dataset only provides IDs 0–6. "
           "The corpus is mixed-topic, so this is **not** a validated COVID-vaccination sentiment model.", icon="⚠️")

tab_demo, tab_batch, tab_results = st.tabs(["Try it", "Batch (CSV)", "Honest results"])

with tab_demo:
    st.write("Pick an example or type your own Chinese post:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, txt) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state["text"] = txt
    text = st.text_area("Post", key="text", height=110, placeholder="在这里输入微博内容…")
    if st.button("Classify", type="primary") or text:
        if not text.strip():
            st.info("Enter some text to classify.")
        else:
            res = predict(text, bundle=bundle)
            left, right = st.columns([1, 2])
            left.metric("Prediction", f"{res['label']} · {res['name']}")
            left.metric("Confidence", f"{res['confidence']:.1%}")
            if res["confidence"] < 0.4:
                left.info("Low confidence – the post is ambiguous or short.")
            df = pd.DataFrame(res["all"]).assign(class_=lambda d: d["label"].astype(str) + " " + d["name"])
            right.bar_chart(df.set_index("class_")["probability"], horizontal=True)

with tab_batch:
    st.write("Upload a CSV with a **text** (or `review`) column. Optional `label` column adds accuracy.")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        d = pd.read_csv(up, encoding="utf-8-sig")
        col = "text" if "text" in d.columns else "review" if "review" in d.columns else None
        if col is None:
            st.error("CSV needs a `text` or `review` column.")
        else:
            d = d.dropna(subset=[col]).head(5000)
            proba = bundle["model"].predict_proba(d[col].astype(str).tolist())
            classes = bundle["model"].classes_
            d["pred_label"] = classes[proba.argmax(1)]
            d["pred_name"] = d["pred_label"].map(LABEL_NAMES)
            d["confidence"] = proba.max(1).round(3)
            if "label" in d.columns:
                st.metric("Accuracy on your file", f"{(d['label'] == d['pred_label']).mean():.1%}")
            st.dataframe(d, use_container_width=True)
            st.download_button("Download predictions", d.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                               "predictions.csv", "text/csv")

with tab_results:
    sp = RESULTS_DIR / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        t = s["test"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{t['accuracy']:.1%}", f"95% CI {t['accuracy_ci95'][0]:.1%}–{t['accuracy_ci95'][1]:.1%}", delta_color="off")
        c2.metric("Test macro-F1", f"{t['macro_f1']:.3f}")
        c3.metric("5-fold CV macro-F1", f"{s['cv5_macro_f1_train_plus_val']['mean']:.3f}")
        c4.metric("Majority baseline acc.", f"{s['majority_baseline_val_accuracy']:.1%}")
    la = RESULTS_DIR / "leakage_audit.csv"
    if la.exists():
        st.subheader("Why the original numbers were inflated")
        st.write("48% of the raw file consists of repeated posts, so a random split puts the same post in train *and* test. "
                 "De-duplicating gives the honest picture:")
        st.dataframe(pd.read_csv(la)[["model", "protocol", "test_accuracy", "test_macro_f1", "test_rows_also_in_train"]], hide_index=True)
    for img, cap in (("confusion_matrix.png", "Confusion matrix"), ("per_class_f1.png", "Per-class F1"),
                     ("class_distribution.png", "Class distribution")):
        if (RESULTS_DIR / img).exists():
            st.image(str(RESULTS_DIR / img), caption=cap, width=640)
