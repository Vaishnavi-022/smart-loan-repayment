import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Smart Loan Repayment Prediction",
    page_icon="💰",
    layout="wide"
)

# Load trained model
model = joblib.load("models/xgb_loan_repayment_model.pkl")


def get_risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


st.title("💰 Smart Loan Repayment Prediction")

st.write(
    "Enter borrower and loan details to estimate repayment status, "
    "default probability, and risk category."
)

st.divider()

st.subheader("Borrower & Loan Information")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0
    )

    term = st.selectbox(
        "Loan Term",
        [" 36 months", " 60 months"]
    )

    int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=12.5
    )

    installment = st.number_input(
        "Monthly Installment",
        min_value=0.0,
        value=333.0
    )

    grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    emp_length = st.selectbox(
        "Employment Length",
        [
            "< 1 year",
            "1 year",
            "2 years",
            "3 years",
            "4 years",
            "5 years",
            "6 years",
            "7 years",
            "8 years",
            "9 years",
            "10+ years"
        ]
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    annual_inc = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=60000.0
    )

    verification_status = st.selectbox(
        "Verification Status",
        ["Verified", "Source Verified", "Not Verified"]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "major_purchase",
            "small_business",
            "car",
            "medical",
            "moving",
            "vacation",
            "house",
            "wedding",
            "renewable_energy",
            "educational",
            "other"
        ]
    )


with col2:

    dti = st.number_input(
        "Debt-to-Income Ratio (DTI)",
        min_value=0.0,
        value=18.5
    )

    delinq_2yrs = st.number_input(
        "Delinquencies in Last 2 Years",
        min_value=0,
        value=0,
        step=1
    )

    fico_range_low = st.number_input(
        "FICO Range Low",
        min_value=300,
        max_value=850,
        value=700
    )

    fico_range_high = st.number_input(
        "FICO Range High",
        min_value=300,
        max_value=850,
        value=704
    )

    inq_last_6mths = st.number_input(
        "Credit Inquiries (Last 6 Months)",
        min_value=0,
        value=1,
        step=1
    )

    open_acc = st.number_input(
        "Open Credit Accounts",
        min_value=0,
        value=8,
        step=1
    )

    pub_rec = st.number_input(
        "Public Records",
        min_value=0,
        value=0,
        step=1
    )

    revol_bal = st.number_input(
        "Revolving Balance",
        min_value=0.0,
        value=5000.0
    )

    revol_util = st.number_input(
        "Revolving Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=35.0
    )

    total_acc = st.number_input(
        "Total Credit Accounts",
        min_value=0,
        value=15,
        step=1
    )


st.divider()


if st.button("🔍 Predict Loan Risk", use_container_width=True):

    loan_data = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "fico_range_low": fico_range_low,
        "fico_range_high": fico_range_high,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc
    }

    input_df = pd.DataFrame([loan_data])

    fully_paid_probability = model.predict_proba(input_df)[0][1]

    default_probability = 1 - fully_paid_probability

    predicted_status = (
        "Fully Paid"
        if fully_paid_probability >= 0.5
        else "Charged Off"
    )

    risk_category = get_risk_category(default_probability)

    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Predicted Status",
            predicted_status
        )

    with result_col2:
        st.metric(
            "Default Probability",
            f"{default_probability * 100:.2f}%"
        )

    with result_col3:
        st.metric(
            "Risk Category",
            risk_category
        )

    st.progress(
        min(float(default_probability), 1.0),
        text=f"Default Probability: {default_probability * 100:.2f}%"
    )

    if risk_category == "Low Risk":
        st.success(
            "This loan falls into the Low Risk category."
        )

    elif risk_category == "Medium Risk":
        st.warning(
            "This loan falls into the Medium Risk category."
        )

    else:
        st.error(
            "This loan falls into the High Risk category."
        )