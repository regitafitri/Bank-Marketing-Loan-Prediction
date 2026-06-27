import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================
# LOAD DATA
# =========================================
@st.cache_data
def load_data():
    return pd.read_csv("bank (data final project).csv")

df = load_data()

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():
    return joblib.load("bank_marketing_pipeline_xgb.pkl")

pipeline = load_model()

# =========================================
# TITLE
# =========================================
st.title("🏦 Bank Deposit Prediction Dashboard")

st.markdown("""
Predict whether a customer will subscribe to a term deposit.
""")

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("Customer Information")

age = st.sidebar.number_input("Age", 18, 100, 40)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.","blue-collar","entrepreneur",
        "housemaid","management","retired",
        "self-employed","services","student",
        "technician","unemployed","unknown"
    ]
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["married","single","divorced"]
)

education = st.sidebar.selectbox(
    "Education",
    ["primary","secondary","tertiary","unknown"]
)

default = st.sidebar.selectbox(
    "Credit Default",
    ["no","yes"]
)

balance = st.sidebar.number_input(
    "Balance",
    value=1000
)

housing = st.sidebar.selectbox(
    "Housing Loan",
    ["no","yes"]
)

loan = st.sidebar.selectbox(
    "Personal Loan",
    ["no","yes"]
)

contact = st.sidebar.selectbox(
    "Contact Type",
    ["cellular","telephone","unknown"]
)

day = st.sidebar.slider(
    "Last Contact Day",
    1,31,15
)

month = st.sidebar.selectbox(
    "Month",
    [
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"
    ]
)

duration = st.sidebar.number_input(
    "Call Duration",
    min_value=0,
    value=300
)

campaign = st.sidebar.number_input(
    "Campaign Contacts",
    min_value=1,
    value=1
)

pdays = st.sidebar.number_input(
    "Pdays",
    value=-1
)

previous = st.sidebar.number_input(
    "Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.sidebar.selectbox(
    "Previous Outcome",
    ["unknown","failure","success","other"]
)

# =========================================
# PREDICT BUTTON
# =========================================
if st.sidebar.button("Predict"):

    input_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    prob = pipeline.predict_proba(input_df)[0][1]

    score = int(prob * 100)

    st.subheader("Prediction Result")

    st.metric(
        "Probability",
        f"{prob:.2%}"
    )

    st.metric(
        "Score",
        f"{score}/100"
    )

    if score >= 80:
        st.success("Very Likely to Subscribe")
    elif score >= 60:
        st.warning("Likely to Subscribe")
    else:
        st.error("Unlikely to Subscribe")

    st.progress(score / 100)

# =========================================
# DATA OVERVIEW
# =========================================
st.subheader("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write(df.head())

with col2:
    fig, ax = plt.subplots()

    df["deposit"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)
