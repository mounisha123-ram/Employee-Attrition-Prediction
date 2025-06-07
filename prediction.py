import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("🔍 Employee Attrition Prediction")

# Load model, scaler, and feature names
with open("prediction1_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("prediction1_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("prediction1_feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Sidebar Navigation
section = st.sidebar.radio("Select Section", ["Prediction"])

if section == "Prediction":
    st.subheader("Enter Employee Details")

    # Input fields
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=20000)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    over_time = st.selectbox("Over Time", ["Yes", "No"])

    # Manual encoding (same order as training)
    department_map = {"Sales": 2, "Research & Development": 1, "Human Resources": 0}
    marital_map = {"Single": 2, "Married": 1, "Divorced": 0}
    overtime_map = {"Yes": 1, "No": 0}

    encoded_input = pd.DataFrame([[
        age,
        department_map[department],
        monthly_income,
        job_satisfaction,
        years_at_company,
        marital_map[marital_status],
        overtime_map[over_time]
    ]], columns=feature_names)

    # Scale input
    scaled_input = scaler.transform(encoded_input)

    # Predict button
    if st.button("🚀 Predict Attrition"):
        prediction = model.predict(scaled_input)[0]
        result = "Yes - At Risk of Attrition" if prediction == 1 else "No - Likely to Stay"

        st.success(f"📢 Prediction: {result}")
