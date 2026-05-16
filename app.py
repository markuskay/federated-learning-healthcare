# ============================================================
# STREAMLIT GUI — MULTI-HOSPITAL OUTCOME PREDICTION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# ------------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Outcome Prediction System",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Hospital Outcome Prediction System")
st.markdown(
    "This application predicts patient outcomes using hospital-specific trained models."
)

# ------------------------------------------------------------
# MODEL DIRECTORY
# ------------------------------------------------------------
MODEL_DIR = r"C:\Users\HP\Desktop\fedmodel\models"

hospital_models = {
    "Toronto General Hospital": "Toronto_General",
    "Cleveland Clinic": "Cleveland_Clinic",
    "Johns Hopkins Hospital": "Johns_Hopkins",
    "Massachusetts General Hospital": "Massachusetts_General",
    "Mayo Clinic": "Mayo_Clinic"
}

# ------------------------------------------------------------
# SIDEBAR — HOSPITAL SELECTION
# ------------------------------------------------------------
st.sidebar.header("Hospital Selection")
selected_hospital = st.sidebar.selectbox(
    "Select Hospital",
    list(hospital_models.keys())
)

hospital_key = hospital_models[selected_hospital]

model_path = os.path.join(MODEL_DIR, f"{hospital_key}_model.h5")
preprocessor_path = os.path.join(MODEL_DIR, f"{hospital_key}_preprocessor.pkl")

# ------------------------------------------------------------
# LOAD MODEL + PREPROCESSOR
# ------------------------------------------------------------
model = tf.keras.models.load_model(model_path)
preprocessor = joblib.load(preprocessor_path)

# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------
st.subheader("Patient Information")

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=60, max_value=220, value=120)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)

    with col2:
        blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=400, value=110)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=400, value=180)

    gender = st.selectbox("Gender", ["Male", "Female"])
    diagnosis = st.selectbox(
        "Diagnosis",
        ["Diabetes", "Heart Disease", "Cancer", "Hypertension", "Asthma", "COVID-19", "Healthy"]
    )

    submit = st.form_submit_button("Predict Outcome")

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
if submit:

    input_df = pd.DataFrame([{
        "Age": age,
        "Blood_Pressure": blood_pressure,
        "Heart_Rate": heart_rate,
        "Blood_Sugar": blood_sugar,
        "Cholesterol": cholesterol,
        "Gender": gender,
        "Diagnosis": diagnosis
    }])

    processed_input = preprocessor.transform(input_df)

    prediction_prob = model.predict(processed_input)[0][0]
    prediction = int(prediction_prob >= 0.5)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"Predicted Outcome: **Higher Risk**")
    else:
        st.success(f"Predicted Outcome: **Lower Risk**")

    st.metric(
        label="Model Confidence",
        value=f"{prediction_prob:.2%}" if prediction == 1 else f"{1 - prediction_prob:.2%}"
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "Outcome predictions are generated using hospital-specific machine learning models."
)
