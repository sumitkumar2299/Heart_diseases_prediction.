

import streamlit as st
import pandas as pd
import pickle
# import joblib

# ------------------- Load Model -------------------

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# with open(os.path.join(BASE_DIR, "KNN_heart.pkl"), "rb") as f:
#     model = pickle.load(f)
@st.cache_resource
def load_model():
    try:
        # model = joblib.load("KNN_heart.pkl")
        with open("KNN_heart.pkl", "rb") as f:
            model = pickle.load(f)
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("columns.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError:
        st.error("❌ Model files not found! Please make sure 'KNN_heart.pkl', 'scaler.pkl', and 'columns.pkl' are in the same folder.")
        st.stop()

model, scaler, expected_columns = load_model()

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ------------------- Sidebar -------------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This app predicts the **risk of heart disease** using a machine learning model trained on medical data.

- ⚙️ Built with **Streamlit**
- 🤖 ML Model: **KNN Classifier**
- 👨‍💻 Developer: *Sumit Kumar*

---
**Disclaimer:** This app is for educational purposes only and not a substitute for professional medical advice.
""")

# ------------------- Main Header -------------------
st.markdown("<h1 style='text-align: center; color: #0056b3;'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Provide your health details below to check your risk.</p>", unsafe_allow_html=True)
st.write("---")

# ------------------- Input Form -------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ['M', 'F'])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

    with col2:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dL", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ['Normal', "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submit_btn = st.form_submit_button("🔍 Predict Risk")

# ------------------- Prediction Logic -------------------
if submit_btn:
    # Create input dict
    raw_input = {
        'Age': age,
        'RestingBp': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBs': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # ------------------- Show Result -------------------
    st.write("---")
    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease**\n\nPlease consult a doctor for a full check-up.")
    else:
        st.success("✅ **Low Risk of Heart Disease**\n\nLooks good! But remember, this is not medical advice.")
