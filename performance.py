# performance.py (for Streamlit app)

import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and features
with open('performance_rating_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_performance.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('features.pkl', 'rb') as f:
    expected_features = pickle.load(f)

# Page setup
st.set_page_config(page_title="Performance Rating Predictor", layout="centered")
st.title("🔍 Employee Performance Rating Prediction")
st.write("Provide employee details to predict the **Performance Rating**.")

# Input form
education = st.selectbox("Education Level (1 = Below College, 5 = Doctor)", [1, 2, 3, 4, 5])
job_involvement = st.slider("Job Involvement (1 = Low, 4 = Very High)", 1, 4, 3)
job_level = st.selectbox("Job Level (1 = Entry Level, 5 = Executive)", [1, 2, 3, 4, 5])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=500)
years_at_company = st.slider("Years at Company", 0, 40, 5)
years_in_current_role = st.slider("Years in Current Role", 0, 18, 3)

# Prepare input
input_data = pd.DataFrame([{
    'education': education,
    'job_involvement': job_involvement,
    'job_level': job_level,
    'monthly_income': monthly_income,
    'years_at_company': years_at_company,
    'years_in_current_role': years_in_current_role
}])

# Ensure feature order
input_data = input_data[expected_features]

# Scale
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Performance Rating"):
    prediction = model.predict(input_scaled)[0]
    st.subheader("🎯 Predicted Performance Rating:")
    st.success(f"{prediction} (on scale 1-4)")

    # Show probability chart
    probs = model.predict_proba(input_scaled)[0]
    prob_df = pd.DataFrame({
        'Rating': model.classes_,
        'Probability': probs
    })
    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(prob_df.set_index('Rating'))
