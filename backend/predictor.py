import time
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from backend.preprocessing import preprocess_single_transaction
from backend.feature_engineering import add_engineered_features
from backend.model_loader import load_all_artifacts

def predict_transaction(
    transaction_type: str,
    amount: float,
    old_sender_balance: float,
    new_sender_balance: float,
    old_receiver_balance: float,
    new_receiver_balance: float,
    transaction_hour: int = 14,
    transaction_day: int = 15,
) -> dict:
    """
    Executes real-time single transaction prediction pipeline using loaded LightGBM model
    and SHAP TreeExplainer. Reuses notebook feature order, scaling, threshold, and XAI logic.
    """
    start_time = time.perf_counter()

    # 1. Load serialized model objects
    artifacts = load_all_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_columns = artifacts["feature_columns"]
    threshold = artifacts["threshold"] # 0.93
    explainer = artifacts["shap_explainer"]

    # 2. Raw input dict
    raw_tx = {
        "type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(old_sender_balance),
        "newbalanceOrig": float(new_sender_balance),
        "oldbalanceDest": float(old_receiver_balance),
        "newbalanceDest": float(new_receiver_balance),
        "hour": int(transaction_hour),
        "day": int(transaction_day),
        "nameOrig": "C_USER_INPUT",
        "nameDest": "M_DEST_INPUT",
    }

    # 3. Preprocessing
    df = preprocess_single_transaction(raw_tx)

    # 4. Feature Engineering
    df_feat = add_engineered_features(df, is_single=True)

    # 5. Drop high-cardinality ID columns
    df_model = df_feat.drop(columns=["nameOrig", "nameDest"], errors="ignore")

    # 6. Reorder & ensure all required feature columns exist
    for col in feature_columns:
        if col not in df_model.columns:
            df_model[col] = 0.0

    df_model = df_model[feature_columns].copy()

    # 7. Apply RobustScaler to numeric columns
    numeric_cols = [c for c in feature_columns if c != "type"]
    df_scaled = df_model.copy()
    df_scaled[numeric_cols] = scaler.transform(df_model[numeric_cols])

    # Failsafe numeric casting
    for col in df_scaled.columns:
        if col != "type":
            df_scaled[col] = pd.to_numeric(df_scaled[col], errors="coerce").fillna(0).astype(float)

    if "type" in df_scaled.columns:
        df_scaled["type"] = df_scaled["type"].astype("category")

    # 8. Model inference probability
    prob_array = model.predict_proba(df_scaled)
    fraud_prob = float(prob_array[0, 1])

    # 9. Apply Notebook Threshold (0.93)
    is_fraud = int(fraud_prob >= threshold)

    # 10. Risk Score (0 to 100 scale)
    risk_score = round(fraud_prob * 100.0, 2)

    # 11. Risk Level & Decision Rules
    if fraud_prob >= threshold:
        risk_level = "CRITICAL HIGH RISK"
        decision = "DECLINE & FREEZE ACCOUNT"
        rec_action = (
            "AUTOMATED ACTION: Transaction blocked immediately. Origin account flagged for suspicious activity. "
            "Dispatch SAR (Suspicious Activity Report) to Banking Compliance Division and issue customer SMS verification."
        )
    elif fraud_prob >= 0.50:
        risk_level = "MEDIUM HIGH RISK"
        decision = "FLAG FOR MANUAL REVIEW"
        rec_action = (
            "AUTOMATED ACTION: Transaction placed on temporary 15-minute authorization hold. "
            "Forwarded to Tier-2 Fraud Analyst queue for mandatory balance ledger verification."
        )
    else:
        risk_level = "LOW RISK"
        decision = "APPROVE TRANSACTION"
        rec_action = "AUTOMATED ACTION: Transaction authorized automatically with real-time risk verification pass."

    # Confidence score calculation
    confidence = round(max(fraud_prob, 1.0 - fraud_prob) * 100.0, 2)

    # 12. SHAP Values Generation
    shap_vals = explainer(df_scaled)
    single_shap = shap_vals[0]

    # Extract top features by absolute SHAP contribution
    feature_names = df_scaled.columns.tolist()
    shap_contributions = single_shap.values
    if len(shap_contributions.shape) > 1 and shap_contributions.shape[1] == 2:
        # Binary classification SHAP output for class 1
        shap_values_class1 = shap_contributions[:, 1]
    else:
        shap_values_class1 = shap_contributions

    top_feature_list = []
    for f_name, val, raw_val in zip(feature_names, shap_values_class1, df_model.iloc[0].values):
        top_feature_list.append({
            "feature": f_name,
            "shap_value": float(val),
            "feature_value": str(raw_val),
            "abs_impact": abs(float(val))
        })

    top_feature_list = sorted(top_feature_list, key=lambda x: x["abs_impact"], reverse=True)
    top_feature_importance = top_feature_list[:6]

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return {
        "prediction": is_fraud,
        "fraud_probability": fraud_prob,
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "recommended_action": rec_action,
        "top_features": top_feature_importance,
        "shap_values": single_shap,
        "shap_values_raw": shap_values_class1,
        "feature_names": feature_names,
        "raw_input": raw_tx,
        "latency_ms": latency_ms,
        "threshold": threshold,
    }
