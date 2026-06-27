import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Bank Deposit Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("bank (data final project).csv")

df = load_data()

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_pipeline():
    return joblib.load("bank_marketing_pipeline_xgb.pkl")

pipeline = joblib.load("bank_marketing_pipeline_xgb.pkl")

# ==============================
# SESSION STATE
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# TITLE
# ==============================
st.title("🏦 Bank Deposit Prediction Dashboard")

st.markdown("Predict whether a customer will subscribe to a term deposit.")

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.header("Customer Input")

age = st.sidebar.number_input("Age", 18, 100, 40)

job = st.sidebar.selectbox("Job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed", "services",
    "student", "technician", "unemployed", "unknown"
])

marital = st.sidebar.selectbox("Marital", ["married", "single", "divorced"])

education = st.sidebar.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])

default = st.sidebar.selectbox("Credit Default", ["no", "yes"])

balance = st.sidebar.number_input("Balance", value=1000)

housing = st.sidebar.selectbox("Housing Loan", ["no", "yes"])

loan = st.sidebar.selectbox("Personal Loan", ["no", "yes"])

contact = st.sidebar.selectbox("Contact", ["cellular", "telephone", "unknown"])

day = st.sidebar.slider("Day", 1, 31, 15)

month = st.sidebar.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])

duration = st.sidebar.number_input("Duration", 0, 5000, 300)

campaign = st.sidebar.number_input("Campaign", 1, 50, 1)

pdays = st.sidebar.number_input("Pdays", value=-1)

previous = st.sidebar.number_input("Previous", value=0)

poutcome = st.sidebar.selectbox("Poutcome", ["unknown", "failure", "success", "other"])

predict_button = st.sidebar.button("Predict")

# ==============================
# PREDICTION
# ==============================
if predict_button:

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

    if score >= 80:
        category = "Very Likely"
    elif score >= 60:
        category = "Likely"
    else:
        category = "Unlikely"

    st.session_state.last_prediction = {
        "prob": prob,
        "score": score,
        "category": category,
        "input_df": input_df
    }

    st.session_state.history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Probability": f"{prob:.1%}",
        "Score": score,
        "Category": category
    })

# ==============================
# OUTPUT
# ==============================
if "last_prediction" in st.session_state:

    pred = st.session_state.last_prediction

    st.subheader("Result")

    st.metric("Probability", f"{pred['prob']:.1%}")
    st.metric("Score", f"{pred['score']}/100")
    st.metric("Category", pred["category"])

    st.progress(pred["score"]/100)

    st.write(pred["input_df"])
else:
    st.info("Input data lalu klik Predict")
    
# ==============================
# HISTORY
# ==============================
if st.session_state.history:
    st.subheader("History")
    st.dataframe(pd.DataFrame(st.session_state.history))
