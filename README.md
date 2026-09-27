# 💰 Smart Loan Repayment Prediction

An end-to-end Machine Learning project that predicts loan repayment status, estimates default probability, and classifies borrowers into risk categories.

## 🚀 Live Demo

https://smart-loan-repayment.streamlit.app/

## 📌 Project Overview

Loan repayment risk assessment is an important task in financial decision-making. This project uses borrower and loan information available at the time of loan issuance to predict whether a loan is likely to be fully paid or charged off.

The project includes:

- Data preprocessing
- Feature engineering and selection
- Logistic Regression
- Random Forest
- XGBoost
- Model evaluation using ROC-AUC, Precision, Recall and F1-score
- Default probability prediction
- Risk categorization
- Interactive Streamlit web application
- Cloud deployment

## 🎯 Prediction Outputs

For a given loan application, the system provides:

- **Predicted Status:** Fully Paid / Charged Off
- **Default Probability:** Probability that the loan will default
- **Risk Category:** Low Risk / Medium Risk / High Risk

### Risk Categories

| Default Probability | Risk Category |
|---|---|
| 0% – <30% | Low Risk |
| 30% – <60% | Medium Risk |
| 60% – 100% | High Risk |

> Risk thresholds are project-defined thresholds used for demonstration.

## 📊 Dataset

The project uses the **Lending Club accepted loans dataset** from Kaggle.

Dataset:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

The model uses only features that are available before or at the time of loan issuance to avoid target leakage.

## 🧠 Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Performance

| Model | ROC-AUC | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7094 | 0.8852 | 0.6306 | 0.7365 |
| Random Forest | 0.7128 | 0.8840 | 0.6483 | 0.7480 |
| XGBoost | 0.7190 | 0.8101 | 0.9866 | 0.8897 |

XGBoost was used as the final model for the deployed application.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

## 📁 Project Structure

```text
Smart-Loan-Repayment/
│
├── data/
│   └── accepted_2007_to_2018Q4.csv.gz
│
├── models/
│   └── xgb_loan_repayment_model.pkl
│
├── loan_prediction.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore