import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("🧑‍💼 Employee Attrition Prediction App")

# Load model, scaler, and input columns
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('input_columns.pkl', 'rb') as f:
    input_columns = pickle.load(f)

# Mapping dictionaries
business_travel_map = {"Non-Travel": 0, "Travel Frequently": 1, "Travel Rarely": 2}
department_map = {"Sales": 0, "Research & Development": 1, "Human Resources": 2}
education_field_map = {
    "Life Sciences": 0, "Medical": 1, "Marketing": 2,
    "Technical Degree": 3, "Human Resources": 4, "Other": 5
}
gender_map = {"Male": 0, "Female": 1}
marital_status_map = {"Single": 0, "Married": 1, "Divorced": 2}
over_time_map = {"No": 0, "Yes": 1}
job_role_map = {
    "Sales Executive": 0, "Research Scientist": 1, "Laboratory Technician": 2,
    "Manufacturing Director": 3, "Healthcare Representative": 4, "Manager": 5,
    "Sales Representative": 6, "Research Director": 7, "Human Resources": 8
}

# Form input
with st.form("prediction_form"):
    st.subheader("🔍 Enter Employee Information:")

    age = st.slider("Age", 18, 65)
    business_travel = business_travel_map[st.selectbox("Business Travel", list(business_travel_map.keys()))]
    daily_rate = st.number_input("Daily Rate", min_value=0)
    department = department_map[st.selectbox("Department", list(department_map.keys()))]
    distance_from_home = st.slider("Distance From Home", 0, 50)
    education = st.selectbox("Education Level (1–5)", [1, 2, 3, 4, 5])
    education_field = education_field_map[st.selectbox("Education Field", list(education_field_map.keys()))]
    environment_satisfaction = st.selectbox("Environment Satisfaction (1–4)", [1, 2, 3, 4])
    gender = gender_map[st.selectbox("Gender", list(gender_map.keys()))]
    hourly_rate = st.slider("Hourly Rate", 0, 100)
    job_involvement = st.selectbox("Job Involvement (1–4)", [1, 2, 3, 4])
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    job_role = job_role_map[st.selectbox("Job Role", list(job_role_map.keys()))]
    job_satisfaction = st.selectbox("Job Satisfaction (1–4)", [1, 2, 3, 4])
    marital_status = marital_status_map[st.selectbox("Marital Status", list(marital_status_map.keys()))]
    monthly_income = st.number_input("Monthly Income", min_value=0)
    monthly_rate = st.number_input("Monthly Rate", min_value=0)
    num_companies_worked = st.slider("Number of Companies Worked", 0, 10)
    over_time = over_time_map[st.selectbox("Over Time", list(over_time_map.keys()))]
    percent_salary_hike = st.slider("Percent Salary Hike", 0, 100)
    performance_rating = st.selectbox("Performance Rating (1–4)", [1, 2, 3, 4])
    relationship_satisfaction = st.selectbox("Relationship Satisfaction (1–4)", [1, 2, 3, 4])
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    total_working_years = st.slider("Total Working Years", 0, 40)
    training_times_last_year = st.slider("Training Times Last Year", 0, 10)
    work_life_balance = st.selectbox("Work Life Balance (1–4)", [1, 2, 3, 4])
    years_at_company = st.slider("Years at Company", 0, 40)
    years_in_current_role = st.slider("Years in Current Role", 0, 20)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 20)
    years_with_curr_manager = st.slider("Years With Current Manager", 0, 20)

    submit = st.form_submit_button("Predict Attrition")

# Run prediction
if submit:
    tenure_category = pd.cut([years_at_company], bins=[0, 2, 5, 10, 100],
                             labels=['<2 yrs', '2-5 yrs', '5-10 yrs', '10+ yrs'],
                             right=False).codes[0]

    engagement_score = (0.4 * job_satisfaction +
                        0.3 * job_involvement +
                        0.2 * work_life_balance +
                        0.1 * relationship_satisfaction)

    performance_tenure = performance_rating * years_at_company

    input_dict = {
        'age': age, 'business_travel': business_travel, 'daily_rate': daily_rate,
        'department': department, 'distance_from_home': distance_from_home, 'education': education,
        'education_field': education_field, 'environment_satisfaction': environment_satisfaction,
        'gender': gender, 'hourly_rate': hourly_rate, 'job_involvement': job_involvement,
        'job_level': job_level, 'job_role': job_role, 'job_satisfaction': job_satisfaction,
        'marital_status': marital_status, 'monthly_income': monthly_income, 'monthly_rate': monthly_rate,
        'num_companies_worked': num_companies_worked, 'over_time': over_time,
        'percent_salary_hike': percent_salary_hike, 'performance_rating': performance_rating,
        'relationship_satisfaction': relationship_satisfaction, 'stock_option_level': stock_option_level,
        'total_working_years': total_working_years, 'training_times_last_year': training_times_last_year,
        'work_life_balance': work_life_balance, 'years_at_company': years_at_company,
        'years_in_current_role': years_in_current_role, 'years_since_last_promotion': years_since_last_promotion,
        'years_with_curr_manager': years_with_curr_manager, 'tenure_category': tenure_category,
        'engagement_score': engagement_score, 'performance_tenure': performance_tenure
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=input_columns)

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    pred_proba = model.predict_proba(scaled_input)[0][prediction]

    st.markdown("### 🎯 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ The model predicts this employee is **likely to leave**. (Confidence: {pred_proba:.2%})")
    else:
        st.success(f"✅ The model predicts this employee will **likely stay**. (Confidence: {pred_proba:.2%})")

    st.markdown("### 🔧 Feature Engineering Insights:")

    tenure_labels = ['<2 yrs', '2-5 yrs', '5-10 yrs', '10+ yrs']
    st.metric(label="Tenure Category", value=tenure_labels[tenure_category])
    st.metric(label="Engagement Score", value=f"{engagement_score:.2f}")

    if engagement_score < 2.0:
        engagement_level = "🔴 Low Engagement"
        engagement_color = "red"
    elif 2.0 <= engagement_score < 3.0:
        engagement_level = "🟡 Moderate Engagement"
        engagement_color = "orange"
    else:
        engagement_level = "🟢 High Engagement"
        engagement_color = "green"

    st.markdown(f"**Engagement Level:** <span style='color:{engagement_color}; font-size:16px'>{engagement_level}</span>", unsafe_allow_html=True)
    st.metric(label="Performance x Tenure", value=f"{performance_tenure:.2f}")

    st.markdown("#### Engagement Score Breakdown")
components = {
    "Job Satisfaction": job_satisfaction * 0.4,
    "Job Involvement": job_involvement * 0.3,
    "Work Life Balance": work_life_balance * 0.2,
    "Relationship Satisfaction": relationship_satisfaction * 0.1
}

fig, ax = plt.subplots(figsize=(3, 2))  # smaller size
sns.barplot(x=list(components.keys()), y=list(components.values()), color="#1f77b4", ax=ax)  # blue
ax.set_ylabel("Weighted Score", fontsize=8)
ax.set_ylim(0, 1.6)
ax.tick_params(axis='x', labelrotation=20, labelsize=7)
ax.tick_params(axis='y', labelsize=7)
ax.set_title("Component Contributions", fontsize=9)
fig.tight_layout()
st.pyplot(fig)


st.markdown("#### Performance x Tenure Components")
perf_tenure_df = pd.DataFrame({
    "Metric": ["Performance Rating", "Years at Company"],
    "Value": [performance_rating, years_at_company]
})

fig2, ax2 = plt.subplots(figsize=(2.8, 2))  # smaller size
sns.barplot(x="Metric", y="Value", data=perf_tenure_df, color="#1f77b4", ax=ax2)  # blue
ax2.set_ylim(0, max(perf_tenure_df["Value"]) + 1)
ax2.set_ylabel("Value", fontsize=8)
ax2.set_title("Performance Factors", fontsize=9)
ax2.tick_params(axis='x', labelsize=7)
ax2.tick_params(axis='y', labelsize=7)
fig2.tight_layout()
st.pyplot(fig2)
