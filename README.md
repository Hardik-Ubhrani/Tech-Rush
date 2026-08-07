# AI-Powered Financial Fraud Management System

An enterprise banking financial fraud detection platform built with **Streamlit**, **LightGBM**, and **TreeSHAP Explainable AI**. The system reuses preprocessing, feature engineering, and trained model artifacts from `Financial_Fraud_Management_System.ipynb` to deliver real-time sub-12ms transaction risk scoring, explainable decision attribution, and automated SAR investigation reporting.

---

## 🌟 Key Features

- **Real-Time Fraud Inference**: Millisecond risk scoring using trained LightGBM model with a calibrated decision threshold of `0.93`.
- **Domain Feature Engineering**: Reuses 8 domain features (`errorBalanceOrig`, `errorBalanceDest`, `is_account_drain`, `hour_of_day`, `Dest_Total_Received_Volume`, `is_Mule_Suspect`, `is_high_risk_type`, `amount_bracket`).
- **Explainable AI (SHAP)**: Waterfall plots and feature attribution for regulatory compliance and audit trails.
- **Model Comparison Benchmarks**: Direct performance evaluation across **Random Forest**, **XGBoost**, and **LightGBM**.
- **Automated Case Investigation & PDF Export**: Instant case report generation with downloadable PDF export.

---

## 📁 Project Structure

```
project/
├── Financial_Fraud_Management_System.ipynb
├── app.py
├── pages/
│   ├── 1_dashboard.py
│   ├── 2_prediction.py
│   ├── 3_model_comparison.py
│   ├── 4_investigation_report.py
│   └── 5_about.py
├── backend/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── predictor.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── utils.py
├── models/
│   ├── lightgbm_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── feature_columns.pkl
│   ├── threshold.pkl
│   └── shap_explainer.pkl
├── assets/
│   └── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

Install the required Python dependencies:

```bash
pip install -r assets/requirements.txt
```

### 2. Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will automatically check for trained artifacts in `models/`. If any file is missing, `backend/model_loader.py` will extract the dataset, fit the LightGBM model and SHAP explainer, and export all `.pkl` files automatically.
