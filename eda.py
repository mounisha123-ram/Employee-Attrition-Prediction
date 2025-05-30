import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set up page config
st.set_page_config(page_title="Employee Attrition EDA", layout="wide")

# Load dataset
df = pd.read_csv("Employee-Attrition - Employee-Attrition.csv")

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r'([a-z])([A-Z])', r'\1_\2', regex=True)
    .str.replace(' ', '_')
    .str.lower()
)

# Drop unwanted columns
df.drop(['employee_count', 'standard_hours', 'over18', 'employee_number'], axis=1, inplace=True)

# Title
st.title("📊 Employee Attrition - Exploratory Data Analysis")

# Sidebar filters (optional)
with st.sidebar:
    st.header("🔎 Filter Options")
    attrition_filter = st.selectbox("Filter by Attrition", options=["All", "Yes", "No"])
    if attrition_filter != "All":
        df = df[df['attrition'] == attrition_filter]

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Correlation & Distribution", "Attrition Analysis", "Custom Plots"])

# ----------------------------- Tab 1: Overview -----------------------------
with tab1:
    st.header("📌 Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Attrition Count")
    fig = px.histogram(df, x='attrition', text_auto=True)
    fig.update_traces(textfont_size=14, textangle=0, textposition="outside")
    fig.update_layout(title_text='Attrition Count')
    st.plotly_chart(fig)

# ---------------- Tab 2: Correlation & Distribution ----------------
with tab2:
    st.header("🔗 Correlation Heatmap")
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    numerical_features = [col for col in numerical_features if col != 'attrition']

    corr = df[numerical_features].corr()
    fig2, ax2 = plt.subplots(figsize=(16,10))
    sns.heatmap(corr, annot=True, cmap='BuPu', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📦 Boxplots for Outliers")
    for col in numerical_features:
        fig = px.box(df, y=col, title=f'Box Plot of {col}')
        st.plotly_chart(fig)

    st.subheader("📊 Histograms")
    for col in numerical_features:
        fig = px.histogram(df, x=col, title=f'Histogram of {col}', marginal="rug", nbins=30)
        st.plotly_chart(fig)

# ---------------- Tab 3: Attrition Analysis ----------------
with tab3:
    st.header(" Categorical Variable Analysis by Attrition")

    st.subheader("Monthly Income by Attrition")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='attrition', y='monthly_income', data=df, palette='Set2', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Job Satisfaction vs. Attrition")
    fig4, ax4 = plt.subplots()
    sns.countplot(x='job_satisfaction', hue='attrition', data=df, palette='mako', ax=ax4)
    st.pyplot(fig4)

    st.subheader("Department vs. Attrition")
    fig5, ax5 = plt.subplots()
    sns.countplot(x='department', hue='attrition', data=df, palette='coolwarm', ax=ax5)
    plt.xticks(rotation=30)
    st.pyplot(fig5)

    st.subheader("Business Travel vs. Attrition")
    fig6, ax6 = plt.subplots()
    sns.countplot(x='business_travel', hue='attrition', data=df, palette='viridis', ax=ax6)
    plt.xticks(rotation=30)
    st.pyplot(fig6)

    st.subheader("Education Field vs. Attrition")
    fig7, ax7 = plt.subplots()
    sns.countplot(x='education_field', hue='attrition', data=df, palette='hot', ax=ax7)
    plt.xticks(rotation=45)
    st.pyplot(fig7)

    st.subheader("Distance From Home vs. Attrition")
    fig8, ax8 = plt.subplots()
    sns.histplot(data=df, x='distance_from_home', hue='attrition', multiple='stack', bins=10, ax=ax8)
    st.pyplot(fig8)

    st.subheader("OverTime vs. Attrition")
    fig9, ax9 = plt.subplots()
    sns.countplot(x='over_time', hue='attrition', data=df, palette='Set3', ax=ax9)
    st.pyplot(fig9)

    st.subheader("Job Role vs. Attrition")
    fig10, ax10 = plt.subplots(figsize=(10, 5))
    sns.countplot(x='job_role', hue='attrition', data=df, palette='hot', ax=ax10)
    plt.xticks(rotation=45)
    st.pyplot(fig10)

    st.subheader("Gender vs. Attrition")
    fig11, ax11 = plt.subplots()
    sns.countplot(x='gender', hue='attrition', data=df, palette='hot', ax=ax11)
    st.pyplot(fig11)

    st.subheader("Age Distribution")
    fig12, ax12 = plt.subplots()
    sns.kdeplot(df['age'], fill=True, ax=ax12)
    st.pyplot(fig12)

# ---------------- Tab 4: Custom Analysis ----------------
with tab4:
    # Mapping education for readability
    edu_map = {1 :'Below College', 2: 'College', 3 :'Bachelor', 4 :'Master', 5: 'Doctor'}
    df['education_level'] = df['education'].map(edu_map)

    # Define ordinal features
    ordinal_features = {
        'education': 'Education Level',
        'environment_satisfaction': 'Environment Satisfaction',
        'job_involvement': 'Job Involvement',
        'job_satisfaction': 'Job Satisfaction',
        'performance_rating': 'Performance Rating',
        'relationship_satisfaction': 'Relationship Satisfaction',
        'work_life_balance': 'Work Life Balance'
    }

    # Dropdown to choose a feature
    selected_feature = st.selectbox("Select an ordinal feature to visualize vs. Attrition", list(ordinal_features.keys()))

    # Plot the selected feature vs Attrition with count labels
    st.subheader(f"📊 {ordinal_features[selected_feature]} vs. Attrition")
    fig, ax = plt.subplots(figsize=(8,5))
    plot = sns.countplot(x=selected_feature, hue='attrition', data=df, palette='coolwarm', ax=ax)

    # Add count labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3)

    st.pyplot(fig)

    # Show preview of all ordinal features
    st.subheader("📋 Preview of All Ordinal Features")
    st.dataframe(df[list(ordinal_features.keys())].head())


