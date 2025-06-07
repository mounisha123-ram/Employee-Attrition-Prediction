# performance_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Employee Performance Rating Prediction", layout="centered")
st.title("🎯 Employee Performance Rating Prediction")

# Load model, scaler, and feature names
with open("prediction2_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("prediction2_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("prediction2_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Sidebar Navigation
section = st.sidebar.radio("Select Section", ["Prediction"])

if section == "Prediction":
    st.subheader("Enter Employee Details")

    # Input fields for all 6 features used in training
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5])  # 1: Below College, 2: College, ..., 5: Doctor
    job_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=10000)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)

    # Create a DataFrame using the input
    input_data = pd.DataFrame([[
        education,
        job_involvement,
        job_level,
        monthly_income,
        years_at_company,
        years_in_current_role
    ]], columns=feature_names)

    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Predict on button click
    if st.button("🚀 Predict Performance Rating"):
        prediction = model.predict(scaled_input)[0]
        st.success(f"📊 Predicted Performance Rating: {prediction}")
