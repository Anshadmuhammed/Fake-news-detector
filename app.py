import streamlit as st
import pickle
import re
import os

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("model.pkl not found. Please run  `python train.py`  first.")
        st.stop()
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict(text):
    cleaned = clean_text(text)
    label = model.predict([cleaned])[0]
    # Check if model supports probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        classes = model.classes_
        conf = dict(zip(classes, proba))
    else:
        score = model.decision_function([cleaned])[0]
        # Convert decision score to pseudo-confidence
        import numpy as np
        prob = 1 / (1 + np.exp(-abs(score)))
        conf = {label: prob, ("REAL" if label == "FAKE" else "FAKE"): 1 - prob}
    return label, conf

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📰 Fake News Detector")
st.markdown("Paste a news headline or article text below to check if it is **Real** or **Fake**.")
st.markdown("---")

input_text = st.text_area(
    "Enter news headline or article text:",
    height=160,
    placeholder="e.g. Scientists discover new vaccine effective against multiple virus strains..."
)

col1, col2 = st.columns([1, 1])
with col1:
    analyse = st.button("🔍 Analyse", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()

if analyse:
    if not input_text.strip():
        st.warning("Please enter some text to analyse.")
    else:
        with st.spinner("Analysing..."):
            label, conf = predict(input_text)

        real_conf = conf.get("REAL", 0) * 100
        fake_conf = conf.get("FAKE", 0) * 100

        st.markdown("---")
        st.markdown("### 🔎 Result")

        if label == "REAL":
            st.success(f"## ✅ REAL NEWS  —  {real_conf:.1f}% confidence")
        else:
            st.error(f"## ❌ FAKE NEWS  —  {fake_conf:.1f}% confidence")

        st.markdown("#### Confidence Breakdown")
        st.write(f"✅ Real: **{real_conf:.1f}%**")
        st.progress(int(real_conf))
        st.write(f"❌ Fake: **{fake_conf:.1f}%**")
        st.progress(int(fake_conf))

        st.markdown("---")
        st.info(
            "⚠️ **Disclaimer:** This tool is for educational purposes only. "
            "Always verify news through trusted, authoritative sources."
        )

st.caption("Built with Python · Scikit-learn · NLP · Streamlit")
