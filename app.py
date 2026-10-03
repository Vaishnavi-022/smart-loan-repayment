import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Loan Repayment Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1220;
    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #111c32, #0e2438);
    border: 1px solid #243b53;
    border-radius: 20px;
    padding: 35px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #9fb3c8;
    font-size: 17px;
    line-height: 1.6;
}

/* Cards */
.card {
    background: #111c32;
    border: 1px solid #243b53;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

.card-title {
    font-size: 21px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}

.card-text {
    color: #9fb3c8;
    line-height: 1.6;
}

/* Stats */
.stat {
    background: #111c32;
    border: 1px solid #243b53;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.stat-number {
    font-size: 28px;
    font-weight: 800;
    color: #22d3ee;
}

.stat-label {
    color: #94a3b8;
    font-size: 14px;
}

/* Result */
.result-card {
    background: linear-gradient(135deg, #102a43, #12344d);
    border: 1px solid #1e88a8;
    border-radius: 20px;
    padding: 30px;
    margin-top: 25px;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    color: white;
}

.result-value {
    font-size: 24px;
    font-weight: 700;
    color: #22d3ee;
}

/* Risk */
.risk-low {
    background: #123524;
    border: 1px solid #287d4c;
    border-radius: 12px;
    padding: 15px;
    color: #86efac;
}

.risk-medium {
    background: #3a2b0b;
    border: 1px solid #a16207;
    border-radius: 12px;
    padding: 15px;
    color: #fde68a;
}

.risk-high {
    background: #3b1111;
    border: 1px solid #991b1b;
    border-radius: 12px;
    padding: 15px;
    color: #fca5a5;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 700;
    height: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d1729;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/xgb_loan_repayment_model.pkl")


model = load_model()


# ============================================================
# RISK FUNCTION
# ============================================================

def get_risk_category(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💳 Smart Loan")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        "An ML-powered system that predicts loan repayment "
        "outcomes and estimates default risk."
    )

    st.markdown("---")

    st.markdown("### Risk Thresholds")

    st.write("🟢 **Low Risk:** 0% – 30%")

    st.write("🟡 **Medium Risk:** 30% – 60%")

    st.write("🔴 **High Risk:** 60% – 100%")

    st.markdown("---")

    st.caption("Built using XGBoost + Streamlit")


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
💳 Smart Loan Repayment Prediction
</div>

<div class="hero-subtitle">
An AI-powered loan risk assessment system that predicts repayment
status and estimates the probability of default using borrower and
loan information.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# QUICK STATS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat">
        <div class="stat-number">1.34M+</div>
        <div class="stat-label">Training Records</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat">
        <div class="stat-number">20</div>
        <div class="stat-label">Input Features</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat">
        <div class="stat-number">0.719</div>
        <div class="stat-label">XGBoost ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat">
        <div class="stat-number">3</div>
        <div class="stat-label">Risk Categories</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("")


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("""
<div class="card">

<div class="card-title">
⚙️ How It Works
</div>

<div class="card-text">

<strong>1. Enter Borrower Data</strong><br>
Provide financial, employment and credit information.

<br><br>

<strong>2. ML Prediction</strong><br>
The trained XGBoost model analyzes the provided information.

<br><br>

<strong>3. Risk Assessment</strong><br>
The system calculates default probability and assigns a risk category.

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("## 🧾 Loan & Borrower Information")

st.caption(
    "Enter the details below to generate a repayment prediction."
)


# ============================================================
# BORROWER PROFILE
# ============================================================

st.markdown("### 👤 Borrower Profile")

col1, col2 = st.columns(2)

with col1:

    annual_inc = st.number_input(
        "Annual Income ($)",
        min_value=0.0,
        value=60000.0,
        step=1000.0
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
        ],
        index=5
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "MORTGAGE",
            "OWN",
            "OTHER",
            "NONE",
            "ANY"
        ]
    )

with col2:

    verification_status = st.selectbox(
        "Verification Status",
        [
            "Verified",
            "Source Verified",
            "Not Verified"
        ]
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
            "educational",
            "renewable_energy",
            "other"
        ]
    )

    dti = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=100.0,
        value=18.5,
        step=0.1
    )


# ============================================================
# CREDIT PROFILE
# ============================================================

st.markdown("### 📊 Credit Profile")

col1, col2, col3 = st.columns(3)

with col1:

    fico_range_low = st.number_input(
        "FICO Score Low",
        min_value=300,
        max_value=850,
        value=700
    )

    fico_range_high = st.number_input(
        "FICO Score High",
        min_value=300,
        max_value=850,
        value=704
    )

    delinq_2yrs = st.number_input(
        "Delinquencies (2 Years)",
        min_value=0,
        value=0,
        step=1
    )

with col2:

    inq_last_6mths = st.number_input(
        "Credit Inquiries (6 Months)",
        min_value=0,
        value=1,
        step=1
    )

    open_acc = st.number_input(
        "Open Accounts",
        min_value=0,
        value=8,
        step=1
    )

    total_acc = st.number_input(
        "Total Accounts",
        min_value=0,
        value=15,
        step=1
    )

with col3:

    pub_rec = st.number_input(
        "Public Records",
        min_value=0,
        value=0,
        step=1
    )

    revol_bal = st.number_input(
        "Revolving Balance ($)",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

    revol_util = st.number_input(
        "Revolving Utilization (%)",
        min_value=0.0,
        max_value=200.0,
        value=35.0,
        step=1.0
    )


# ============================================================
# LOAN DETAILS
# ============================================================

st.markdown("### 💰 Loan Details")

col1, col2, col3 = st.columns(3)

with col1:

    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    term = st.selectbox(
        "Loan Term",
        [
            " 36 months",
            " 60 months"
        ]
    )

with col2:

    int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=12.5,
        step=0.1
    )

    installment = st.number_input(
        "Monthly Installment ($)",
        min_value=0.0,
        value=333.0,
        step=10.0
    )

with col3:

    grade = st.selectbox(
        "Loan Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ],
        index=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("")

predict_button = st.button(
    "🚀 Predict Loan Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

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

    loan_df = pd.DataFrame([loan_data])

    fully_paid_probability = model.predict_proba(loan_df)[0][1]

    default_probability = 1 - fully_paid_probability

    predicted_status = (
        "Fully Paid"
        if fully_paid_probability >= 0.5
        else "Charged Off"
    )

    risk_category = get_risk_category(default_probability)


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("""
    <div class="result-card">
        <div class="result-title">
            🎯 Prediction Result
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted Status",
            predicted_status
        )

    with col2:

        st.metric(
            "Default Probability",
            f"{default_probability * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Risk Category",
            risk_category
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.markdown("### 📈 Default Probability")

    st.progress(
        min(float(default_probability), 1.0)
    )

    st.write(
        f"Estimated probability of default: "
        f"**{default_probability * 100:.2f}%**"
    )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    if risk_category == "Low Risk":

        st.markdown(
            f"""
            <div class="risk-low">
            🟢 <strong>Low Risk</strong><br>
            The predicted default probability is below 30%.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif risk_category == "Medium Risk":

        st.markdown(
            f"""
            <div class="risk-medium">
            🟡 <strong>Medium Risk</strong><br>
            The predicted default probability is between 30% and 60%.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="risk-high">
            🔴 <strong>High Risk</strong><br>
            The predicted default probability is 60% or higher.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# RISK GUIDE
# ============================================================

st.markdown("")

st.markdown("""
<div class="card">

<div class="card-title">
📌 Risk Classification Guide
</div>

</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="risk-low">
    🟢 <strong>Low Risk</strong><br>
    Default probability below 30%
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="risk-medium">
    🟡 <strong>Medium Risk</strong><br>
    Default probability 30% – 60%
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="risk-high">
    🔴 <strong>High Risk</strong><br>
    Default probability 60% or higher
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("")
st.markdown("## 🤖 Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "ROC-AUC": [
        0.7094,
        0.7128,
        0.7190
    ],
    "Precision": [
        0.8852,
        0.8840,
        0.8101
    ],
    "Recall": [
        0.6306,
        0.6483,
        0.9866
    ],
    "F1 Score": [
        0.7365,
        0.7480,
        0.8897
    ]
})

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.markdown("")

st.markdown("""
<div class="card">

<div class="card-title">
📚 Project Overview
</div>

<div class="card-text">

<strong>Dataset:</strong> Lending Club accepted loan dataset<br><br>

<strong>Target:</strong> Fully Paid vs Charged Off<br><br>

<strong>Machine Learning:</strong> Logistic Regression, Random Forest and XGBoost<br><br>

<strong>Final Model:</strong> XGBoost<br><br>

<strong>Application:</strong> Streamlit interactive prediction dashboard

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("")

st.markdown("""
<div style="
text-align:center;
padding:25px;
color:#64748b;
border-top:1px solid #243b53;
margin-top:30px;
">

Smart Loan Repayment Prediction<br>

Built with Python • Scikit-learn • XGBoost • Streamlit

</div>
""", unsafe_allow_html=True)