import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved model, scaler, and encoders
with open('attrition_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('attrition_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('attrition_label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('attrition_input_columns.pkl', 'rb') as f:
    features = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("🔍 Attrition Prediction App")

st.write("Enter employee details to predict the attrition risk.")

# Input fields
age = st.slider("Age", 18, 60, 30)
department = st.selectbox("Department", label_encoders['department'].classes_)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=100)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
years_at_company = st.slider("Years at Company", 0, 40, 5)
marital_status = st.selectbox("Marital Status", label_encoders['marital_status'].classes_)
over_time = st.selectbox("Over Time", label_encoders['over_time'].classes_)

# Encode categorical inputs
input_data = {
    'age': age,
    'department': label_encoders['department'].transform([department])[0],
    'monthly_income': monthly_income,
    'job_satisfaction': job_satisfaction,
    'years_at_company': years_at_company,
    'marital_status': label_encoders['marital_status'].transform([marital_status])[0],
    'over_time': label_encoders['over_time'].transform([over_time])[0]
}

input_df = pd.DataFrame([input_data])

# Scale inputs
input_scaled = scaler.transform(input_df)


# After prediction button
if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of attrition ({probability:.2%} probability)")
    else:
        st.success(f"✅ Low risk of attrition ({probability:.2%} probability)")

    # 🔍 Show probability chart
    st.subheader("📊 Prediction Probability")
    probs = model.predict_proba(input_scaled)[0]
    prob_df = pd.DataFrame({
        'Attrition': ['No', 'Yes'],
        'Probability': probs
    })

    # Bar chart
    fig, ax = plt.subplots()
    sns.barplot(x='Attrition', y='Probability', data=prob_df, palette='Blues_d', ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Optionally: show values as metrics
    st.metric("Probability - No", f"{probs[0]*100:.2f}%")
    st.metric("Probability - Yes", f"{probs[1]*100:.2f}%")

    # Optional: Radar Chart or Feature Importance?

