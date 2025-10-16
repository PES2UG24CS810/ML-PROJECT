import streamlit as st
import joblib
import pandas as pd
import re
import os
import csv
from rapidfuzz import fuzz, process

# 🌈 PAGE CONFIGURATION

st.set_page_config(
    page_title="Quora Insincere Question Detector",
    page_icon="💡",
    layout="centered",
)

# 🎨 CUSTOM CSS — Synthwave Sunset Background

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Poppins:wght@400;600&display=swap');

    .stApp {
        position: relative;
        overflow: hidden;
        font-family: 'Poppins', sans-serif;
        color: #fff;
        background: linear-gradient(to top, #120458, #240046, #3c096c, #5a189a);
        height: 100vh;
        background-attachment: fixed;
    }

    /* Animated Horizon Glow */
    .stApp::before {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 45%;
        background: linear-gradient(to top, #ff007f, transparent);
        opacity: 0.8;
        animation: glowShift 8s ease-in-out infinite alternate;
        z-index: 0;
    }

    /* Moving Grid Floor */
    .stApp::after {
        content: "";
        position: absolute;
        bottom: 0;
        width: 100%;
        height: 40%;
        background-image:
            linear-gradient(90deg, rgba(255, 0, 255, 0.2) 1px, transparent 1px),
            linear-gradient(rgba(255, 0, 255, 0.2) 1px, transparent 1px);
        background-size: 40px 40px;
        transform: perspective(300px) rotateX(70deg);
        animation: moveGrid 12s linear infinite;
        z-index: 0;
    }

    @keyframes moveGrid {
        from { background-position: 0 0, 0 0; }
        to { background-position: 0 40px, 40px 0; }
    }

    @keyframes glowShift {
        0% { opacity: 0.5; }
        50% { opacity: 0.9; }
        100% { opacity: 0.5; }
    }

    .title {
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        letter-spacing: 2px;
        font-weight: 800;
        color: #ff66c4;
        text-shadow: 0 0 20px #ff00ff, 0 0 40px #ff00ff;
        z-index: 2;
        position: relative;
        margin-top: 20px;
    }

    .subtitle {
        text-align: center;
        color: #ffb3ec;
        font-size: 18px;
        margin-bottom: 35px;
        position: relative;
        z-index: 2;
    }

    .main-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 0 25px rgba(255, 0, 255, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        z-index: 2;
    }

    textarea {
        background-color: rgba(0,0,0,0.5) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid #ff66c4 !important;
        font-size: 16px !important;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff006e, #ff8fa3, #ffbe0b);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        padding: 10px 26px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.4);
        z-index: 2;
        position: relative;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #ffbe0b, #ff8fa3, #ff006e);
        transform: scale(1.1);
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.6);
    }

    .result {
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        border-radius: 15px;
        padding: 15px;
        color: white;
        margin-top: 25px;
        animation: pulse 2s infinite;
        z-index: 2;
        position: relative;
    }

    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.9; }
        50% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(1); opacity: 0.9; }
    }

    .sincere {
        background: linear-gradient(90deg, #00ffcc, #00bfa5);
        box-shadow: 0 0 25px #00ffcc;
    }

    .insincere {
        background: linear-gradient(90deg, #ff1744, #c51162);
        box-shadow: 0 0 25px #ff0044;
    }
    </style>
""", unsafe_allow_html=True)

# ⚙️ LOAD MODEL & VECTORIZER

try:
    model = joblib.load("best_quora_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("❌ Model or vectorizer not found. Please ensure both .pkl files are available.")
    st.stop()


# LOAD DATASET

df_train = None
if os.path.exists("train.csv"):
    try:
        df_train = pd.read_csv("train.csv", on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL)
        df_train['question_text'] = df_train['question_text'].astype(str).str.strip().str.lower()
    except Exception as e:
        st.warning(f"⚠ Could not read train.csv: {e}")

# TEXT CLEANING

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 💬 UI SECTION

st.markdown("<h1 class='title'>💬 Quora Insincere Question Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Detect insincere vs sincere Quora questions using Machine Learning ⚡</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    user_input = st.text_area("✍ Enter your question:", height=120,
                              placeholder="Example: Why is Prayag Raj called Allahabad?")
    analyze_btn = st.button("🚀 Analyze Question")

    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠ Please enter a valid question.")
        else:
            question_clean = clean_text(user_input)
            found_in_dataset = False

            # 🔍 Check dataset with fuzzy match
            if df_train is not None:
                questions_list = df_train['question_text'].tolist()
                match_result = process.extractOne(question_clean, questions_list, scorer=fuzz.token_sort_ratio)

                if match_result and match_result[1] >= 90:
                    found_in_dataset = True
                    best_match = match_result[0]
                    actual_label = df_train[df_train['question_text'] == best_match].iloc[0]['target']
                    label_text = "🚨 Insincere" if actual_label == 1 else "✅ Sincere"
                    color_class = "insincere" if actual_label == 1 else "sincere"
                    st.markdown(f"<div class='result {color_class}'>{label_text} (From Dataset)</div>",
                                unsafe_allow_html=True)

            # Predict if not found
            if not found_in_dataset:
                vec = vectorizer.transform([question_clean])
                pred = model.predict(vec)[0]
                label_text = "🚨 Insincere" if pred == 1 else "✅ Sincere"
                color_class = "insincere" if pred == 1 else "sincere"
                st.markdown(f"<div class='result {color_class}'>{label_text} (Predicted)</div>",
                            unsafe_allow_html=True)

                # 🎯 Confidence bar
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(vec)[0][pred]
                    st.progress(float(prob))
                    st.caption(f"Model confidence: **{prob:.2%}**")
