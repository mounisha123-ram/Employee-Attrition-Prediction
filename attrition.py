# attrition_dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("👥 Employee Attrition Prediction Dashboard")

# Sidebar navigation
section = st.sidebar.radio("Select Section", ["Prediction", "EDA", "Model Evaluation Results"])

# Load model and scaler
with open("attrition_logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("attrition_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load cleaned data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_employee_attrition_data.csv")

df = load_data()

# ---------------------- Prediction Section ---------------------- #
# Feature engineering inside the app
if section == "Prediction":
    st.header("🔍 Employee Attrition Prediction")

    # Base inputs
    age = st.number_input("Age", min_value=18, max_value=60)
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    daily_rate = st.number_input("Daily Rate", min_value=100)
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    distance_from_home = st.number_input("Distance From Home", min_value=1)
    education = st.slider("Education (1=Below College, 5=Doctor)", 1, 5)
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hourly_rate = st.number_input("Hourly Rate", min_value=20)
    job_involvement = st.slider("Job Involvement (1-4)", 1, 4)
    job_level = st.slider("Job Level", 1, 5)
    job_role = st.selectbox("Job Role", df["job_role"].unique())
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    monthly_income = st.number_input("Monthly Income", min_value=1000)
    monthly_rate = st.number_input("Monthly Rate", min_value=1000)
    num_companies_worked = st.number_input("Num Companies Worked", min_value=0)
    over_time = st.selectbox("Over Time", ["Yes", "No"])
    percent_salary_hike = st.slider("Percent Salary Hike", 10, 25)
    performance_rating = st.slider("Performance Rating", 1, 4)
    relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4)
    stock_option_level = st.slider("Stock Option Level", 0, 3)
    total_working_years = st.number_input("Total Working Years", min_value=0)
    training_times_last_year = st.slider("Training Times Last Year", 0, 6)
    work_life_balance = st.slider("Work Life Balance", 1, 4)
    years_at_company = st.number_input("Years at Company", min_value=0)
    years_in_current_role = st.number_input("Years in Current Role", min_value=0)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0)
    years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0)

    # 🎯 Feature Engineering
    # tenure_category from years_at_company
    if years_at_company <= 2:
        tenure_category = "0-2"
    elif years_at_company <= 5:
        tenure_category = "3-5"
    elif years_at_company <= 10:
        tenure_category = "6-10"
    else:
        tenure_category = "10+"

    # engagement_score from satisfaction ratings
    engagement_score = round((job_satisfaction + work_life_balance + environment_satisfaction) / 3, 2)

    # performance_tenure from performance_rating * years_at_company
    performance_tenure = performance_rating * years_at_company

    # Input dictionary with engineered features
    input_dict = {
        'age': age,
        'business_travel': business_travel,
        'daily_rate': daily_rate,
        'department': department,
        'distance_from_home': distance_from_home,
        'education': education,
        'education_field': education_field,
        'environment_satisfaction': environment_satisfaction,
        'gender': gender,
        'hourly_rate': hourly_rate,
        'job_involvement': job_involvement,
        'job_level': job_level,
        'job_role': job_role,
        'job_satisfaction': job_satisfaction,
        'marital_status': marital_status,
        'monthly_income': monthly_income,
        'monthly_rate': monthly_rate,
        'num_companies_worked': num_companies_worked,
        'over_time': over_time,
        'percent_salary_hike': percent_salary_hike,
        'performance_rating': performance_rating,
        'relationship_satisfaction': relationship_satisfaction,
        'stock_option_level': stock_option_level,
        'total_working_years': total_working_years,
        'training_times_last_year': training_times_last_year,
        'work_life_balance': work_life_balance,
        'years_at_company': years_at_company,
        'years_in_current_role': years_in_current_role,
        'years_since_last_promotion': years_since_last_promotion,
        'years_with_curr_manager': years_with_curr_manager,
        'tenure_category': tenure_category,
        'engagement_score': engagement_score,
        'performance_tenure': performance_tenure
    }

    input_df = pd.DataFrame([input_dict])
    full_df = pd.concat([df.drop(columns=['attrition']), input_df], axis=0)
    full_df_encoded = pd.get_dummies(full_df)

    # Align with trained feature set
    with open("attrition_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    model_input = full_df_encoded.tail(1).reindex(columns=feature_names, fill_value=0)

    input_scaled = scaler.transform(model_input)

    if st.button("Predict Attrition"):
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Likely to leave the company\nPrediction Confidence: {prob:.2f}")
        else:
            st.success(f"✅ Not likely to leave the company\nPrediction Confidence: {1 - prob:.2f}")
            
        # Display feature engineering insights
        st.markdown("### 🔧 Feature Engineering Insights:")

        # Tenure Category Label Mapping
        tenure_labels = {'0-2': '<2 yrs', '3-5': '2-5 yrs', '6-10': '5-10 yrs', '10+': '10+ yrs'}
        st.metric(label="Tenure Category", value=tenure_labels[tenure_category])

        # Engagement Score
        st.metric(label="Engagement Score", value=f"{engagement_score:.2f}")

        # Engagement Level Visualization
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

        # Performance x Tenure
        st.metric(label="Performance x Tenure", value=f"{performance_tenure:.2f}")

        # 🎯 Engagement Score Breakdown
        st.markdown("#### Engagement Score Breakdown")
        components = {
            "Job Satisfaction": job_satisfaction * 0.4,
            "Job Involvement": job_involvement * 0.3,
            "Work Life Balance": work_life_balance * 0.2,
            "Relationship Satisfaction": relationship_satisfaction * 0.1
        }

        fig, ax = plt.subplots(figsize=(3, 2))
        sns.barplot(x=list(components.keys()), y=list(components.values()), color="#1f77b4", ax=ax)
        ax.set_ylabel("Weighted Score", fontsize=8)
        ax.set_ylim(0, 1.6)
        ax.tick_params(axis='x', labelrotation=20, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title("Component Contributions", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)

        # 🎯 Performance Tenure Breakdown
        st.markdown("#### Performance x Tenure Components")
        perf_tenure_df = pd.DataFrame({
            "Metric": ["Performance Rating", "Years at Company"],
            "Value": [performance_rating, years_at_company]
        })

        fig2, ax2 = plt.subplots(figsize=(2.8, 2))
        sns.barplot(x="Metric", y="Value", data=perf_tenure_df, color="#1f77b4", ax=ax2)
        ax2.set_ylim(0, max(perf_tenure_df["Value"]) + 1)
        ax2.set_ylabel("Value", fontsize=8)
        ax2.set_title("Performance Factors", fontsize=9)
        ax2.tick_params(axis='x', labelsize=7)
        ax2.tick_params(axis='y', labelsize=7)
        fig2.tight_layout()
        st.pyplot(fig2)


# ---------------------- EDA Section ---------------------- #
elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    eda_option = st.selectbox("Select EDA Plot", [
        "Attrition Count", "Correlation Matrix", "Box Plots", "Histograms",
        "Categorical Plots", "Education vs Attrition"
    ])

    if eda_option == "Attrition Count":
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(y='attrition', data=df, palette='pastel', ax=ax)
        for p in ax.patches:
            ax.text(p.get_width() + 1, p.get_y() + p.get_height()/2, int(p.get_width()), va='center')
        ax.set_title("Attrition Count")
        st.pyplot(fig)

    elif eda_option == "Correlation Matrix":
        corr = df.select_dtypes(include=['float64', 'int64']).corr()
        filtered = corr[(corr.abs() > 0.5) & (corr.abs() < 1.0)]
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(filtered, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    elif eda_option == "Box Plots":
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            fig = px.box(df, y=col, title=f"Box Plot of {col}", points="all")
            st.plotly_chart(fig)

    elif eda_option == "Histograms":
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            fig = px.histogram(df, x=col, title=f"Histogram of {col}", nbins=30, marginal="rug")
            st.plotly_chart(fig)

    elif eda_option == "Categorical Plots":
        cat_cols = ['job_role', 'gender', 'job_satisfaction', 'department', 'over_time']
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.countplot(x=col, hue='attrition', data=df, palette='Blues', ax=ax)
            ax.set_title(f"{col.replace('_', ' ').title()} vs Attrition")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    elif eda_option == "Education vs Attrition":
        edu_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=df['education'].map(edu_map), hue='attrition', data=df, palette='Blues', ax=ax)
        ax.set_title("Education vs Attrition")
        st.pyplot(fig)

# ---------------------- Model Results Section ---------------------- #
elif section == "Model Evaluation Results":
    st.header("Model Used --> LogisticRegression")

    X_test = pd.read_csv("X_test_attrition_scaled.csv").values
    y_test = pd.read_csv("y_attrition_test.csv")["attrition"].values

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("🔢 Accuracy", f"{acc*100:.2f}%")

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)