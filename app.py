import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model and preprocessor
model = joblib.load('models/best_rf_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

st.set_page_config(page_title="Credit Scoring Model", layout="centered")
st.title("💳 Credit Scoring Model")
st.markdown("Predict whether a loan applicant is **good** or **bad** credit risk.")

st.sidebar.header("Applicant Details")

#  Numerical inputs (sliders/numbers) 
duration = st.sidebar.slider("Loan duration (months)", 4, 72, 18)
credit_amount = st.sidebar.number_input("Credit amount", 0, 20000, 3000, step=500)
installment_rate = st.sidebar.slider("Installment rate (% of disposable income)", 1, 4, 2)
residence_since = st.sidebar.slider("Years at current residence", 1, 4, 2)
age = st.sidebar.slider("Age", 18, 100, 35)
existing_credits = st.sidebar.slider("Number of existing credits", 1, 4, 1)
num_dependents = st.sidebar.slider("Number of dependents", 1, 2, 1)

#Categorical inputs (dropdowns) 
checking = st.sidebar.selectbox(
    "Status of checking account",
    ["A11", "A12", "A13", "A14"],
    format_func=lambda x: {
        "A11": "< 0 DM", "A12": "0 to < 200 DM",
        "A13": ">= 200 DM / salary", "A14": "no checking account"
    }[x]
)
credit_hist = st.sidebar.selectbox(
    "Credit history",
    ["A30", "A31", "A32", "A33", "A34"],
    format_func=lambda x: {
        "A30": "no credits taken / all paid back",
        "A31": "all paid back duly",
        "A32": "existing paid back duly",
        "A33": "delay in past",
        "A34": "critical / other credits"
    }[x]
)
purpose = st.sidebar.selectbox(
    "Loan purpose",
    ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"]
)
savings = st.sidebar.selectbox(
    "Savings account / bonds",
    ["A61", "A62", "A63", "A64", "A65"],
    format_func=lambda x: {
        "A61": "< 100 DM", "A62": "100 to < 500 DM",
        "A63": "500 to < 1000 DM", "A64": ">= 1000 DM",
        "A65": "unknown / no savings"
    }[x]
)
employment = st.sidebar.selectbox(
    "Present employment since",
    ["A71", "A72", "A73", "A74", "A75"],
    format_func=lambda x: {
        "A71": "unemployed", "A72": "< 1 year",
        "A73": "1–4 years", "A74": "4–7 years", "A75": ">= 7 years"
    }[x]
)
personal_status = st.sidebar.selectbox(
    "Personal status & sex",
    ["A91", "A92", "A93", "A94"]
)
other_debtors = st.sidebar.selectbox(
    "Other debtors / guarantors",
    ["A101", "A102", "A103"]
)
property_type = st.sidebar.selectbox(
    "Property",
    ["A121", "A122", "A123", "A124"]
)
other_install = st.sidebar.selectbox(
    "Other installment plans",
    ["A141", "A142", "A143"]
)
housing = st.sidebar.selectbox(
    "Housing",
    ["A151", "A152", "A153"],
    format_func=lambda x: {"A151": "rent", "A152": "own", "A153": "for free"}[x]
)
job = st.sidebar.selectbox(
    "Job",
    ["A171", "A172", "A173", "A174"]
)
telephone = st.sidebar.selectbox(
    "Telephone",
    ["A191", "A192"],
    format_func=lambda x: {"A191": "none", "A192": "yes, registered"}[x]
)
foreign = st.sidebar.selectbox(
    "Foreign worker",
    ["A201", "A202"],
    format_func=lambda x: {"A201": "yes", "A202": "no"}[x]
)

# Build the input dictionary
input_data = {
    'duration': duration,
    'credit_amount': credit_amount,
    'installment_rate': installment_rate,
    'residence_since': residence_since,
    'age': age,
    'existing_credits': existing_credits,
    'num_dependents': num_dependents,
    'checking_account': checking,
    'credit_history': credit_hist,
    'purpose': purpose,
    'savings_account': savings,
    'employment': employment,
    'personal_status_sex': personal_status,
    'other_debtors': other_debtors,
    'property': property_type,
    'other_installment_plans': other_install,
    'housing': housing,
    'job': job,
    'telephone': telephone,
    'foreign_worker': foreign
}

input_df = pd.DataFrame([input_data])

# Predict
if st.sidebar.button("🔮 Predict Creditworthiness"):
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)[0]
    proba = model.predict_proba(input_processed)[0, 1]  # probability of class 1 (good)
    
    if prediction == 1:
        st.success(f"✅ **Good Credit Risk** (probability: {proba:.2%})")
        st.markdown("The applicant is likely to repay the loan.")
    else:
        st.error(f"❌ **Bad Credit Risk** (probability of good: {proba:.2%})")
        st.markdown("The applicant may default — further review recommended.")
    
    # Display input summary
    st.subheader("Input Summary")
    st.json(input_data)