# Streamlit/main.py

import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path

# ─── 1. PATH SETUP ────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
ROOT    = HERE.parent
ENC_DIR = ROOT / "Results" / "Encoders"
MOD_DIR = ROOT / "Results" / "Models"

MODEL_FILE   = MOD_DIR / "best_logistic_model.pkl"
ORDINAL_FILE = ENC_DIR / "ordinal_encoder.pkl"

# ─── 2. FEATURES & ORDER ──────────────────────────────────────────────────────
ordinal_cols = ["Sleep Duration", "Dietary Habits", "Degree"]
label_features = [
    "Gender",
    "City",
    "Working Professional or Student",
    "Profession",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]
feature_order = [
    "Gender", "Age", "City", "Working Professional or Student", "Profession",
    "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction",
    "Job Satisfaction", "Sleep Duration", "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?", "Work/Study Hours",
    "Financial Stress", "Family History of Mental Illness",
]

# ─── 3. LOAD ENCODERS & MODEL ────────────────────────────────────────────────
# 3a. Ordinal encoder
ordinal_encoder = joblib.load(ORDINAL_FILE)

# 3b. Label encoders via glob
label_encoders = {}
for feat in label_features:
    # turn "Have you ever had suicidal thoughts ?" → "Have_you_ever_had_suicidal_thoughts"
    key = re.sub(r"[^\w\s]", "", feat).strip().replace(" ", "_")
    pattern = f"{key}*label_encoder.pkl"
    matches = list(ENC_DIR.glob(pattern))
    
    if not matches:
        st.error(f"❌ No encoder file found for feature: '{feat}' (tried '{pattern}')")
        st.stop()
    if len(matches) > 1:
        st.warning(f"⚠️ Multiple matches for '{feat}': {matches}. Using the first.")
    
    enc_path = matches[0]
    label_encoders[feat] = joblib.load(enc_path)

# 3c. Trained model
model = joblib.load(MODEL_FILE)

# ─── 4. STREAMLIT UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Depression Prediction", layout="centered")
st.title("🧠 Depression Analysis & Prediction")
st.sidebar.header("Enter your information:")

def user_input_features() -> pd.DataFrame:
    data = {}
    # ordinal inputs
    for i, col in enumerate(ordinal_cols):
        opts = list(ordinal_encoder.categories_[i])
        data[col] = st.sidebar.selectbox(col, opts)
    # label inputs
    for feat, le in label_encoders.items():
        data[feat] = st.sidebar.selectbox(feat, list(le.classes_))
    # numeric inputs
    data["Age"]               = st.sidebar.number_input("Age",               0, 100, 30)
    data["Academic Pressure"] = st.sidebar.number_input("Academic Pressure", 0.0, 10.0, 0.0, 0.1)
    data["Work Pressure"]     = st.sidebar.number_input("Work Pressure",     0.0, 10.0, 0.0, 0.1)
    data["CGPA"]              = st.sidebar.number_input("CGPA",              0.0, 10.0, 0.0, 0.1)
    data["Study Satisfaction"]= st.sidebar.number_input("Study Satisfaction",0.0, 10.0, 0.0, 0.1)
    data["Job Satisfaction"]  = st.sidebar.number_input("Job Satisfaction",  0.0, 10.0, 0.0, 0.1)
    data["Work/Study Hours"]  = st.sidebar.number_input("Work/Study Hours",  0,   24,  8)
    data["Financial Stress"]  = st.sidebar.number_input("Financial Stress",  0,   10,  3)
    return pd.DataFrame(data, index=[0])

# collect user input
input_df = user_input_features()

# ─── 5. PREPROCESS ───────────────────────────────────────────────────────────
# 5a. Ordinal → numeric
input_df[ordinal_cols] = ordinal_encoder.transform(input_df[ordinal_cols])
# 5b. Label → numeric
for feat, le in label_encoders.items():
    input_df[feat] = le.transform(input_df[feat])
# 5c. Reorder to match training
input_df = input_df[feature_order]

# ─── 6. PREDICT & DISPLAY ────────────────────────────────────────────────────
pred  = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0, 1]

st.subheader("Your Input Summary")
st.write(input_df)

st.subheader("Prediction")
if pred == 1:
    st.error("🔴 Depression Likely")
else:
    st.success("🟢 No Depression")

st.subheader("Confidence")
st.write(f"{proba:.2%}")
