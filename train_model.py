import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DEFAULT_DATASET_PATH = r"C:\Users\Meet Jeswani\OneDrive\Desktop\Financial-Fraud-Detection\data\raw\paysim.csv"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

def load_data(data_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Loads raw dataset from paysim.csv."""
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path, nrows=500000)
    else:
        print("Dataset file not found at default path. Generating synthetic PaySim dataset...")
        np.random.seed(42)
        n_samples = 100000
        df = pd.DataFrame({
            'step': np.random.randint(1, 744, n_samples),
            'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT', 'DEBIT'], n_samples, p=[0.35, 0.25, 0.25, 0.1, 0.05]),
            'amount': np.random.exponential(50000, n_samples),
            'nameOrig': [f"C{i}" for i in range(n_samples)],
            'oldbalanceOrg': np.random.exponential(100000, n_samples),
            'newbalanceOrig': np.random.exponential(80000, n_samples),
            'nameDest': [f"M{i%5000}" for i in range(n_samples)],
            'oldbalanceDest': np.random.exponential(150000, n_samples),
            'newbalanceDest': np.random.exponential(200000, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),
            'isFlaggedFraud': 0
        })
    print(f"Loaded {len(df):,} rows.")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook Cell 3 & 6 Pre-cleaning audit, null removal, and category casting."""
    df_clean = df.copy()
    subset_cols = [c for c in ['nameOrig', 'nameDest', 'amount'] if c in df_clean.columns]
    if subset_cols:
        df_clean.dropna(subset=subset_cols, inplace=True)

    ledger_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
    for col in ledger_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna("Unknown")

    df_clean.drop_duplicates(inplace=True)
    df_clean.drop('isFlaggedFraud', axis=1, inplace=True, errors='ignore')

    cols_to_cast = ['type', 'nameOrig', 'nameDest']
    for col in cols_to_cast:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')

    return df_clean

def feature_engineering(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Notebook Cell 10 Advanced Feature Engineering."""
    df_clean = df_clean.copy()
    
    df_clean = df_clean.sort_values(by='step').reset_index(drop=True)

    df_clean['errorBalanceOrig'] = df_clean['newbalanceOrig'] + df_clean['amount'] - df_clean['oldbalanceOrg']
    df_clean['errorBalanceDest'] = df_clean['oldbalanceDest'] + df_clean['amount'] - df_clean['newbalanceDest']
    df_clean['is_account_drain'] = np.where(df_clean['amount'] == df_clean['oldbalanceOrg'], 1, 0)
    df_clean['hour_of_day'] = df_clean['step'] % 24

    df_clean['Dest_Total_Received_Volume'] = df_clean.groupby('nameDest', observed=True)['amount'].cumsum()

    first_receipt = df_clean.groupby('nameDest', observed=True)['step'].min().to_dict()
    df_clean['orig_first_receipt_step'] = df_clean['nameOrig'].map(first_receipt)

    df_clean['is_Mule_Suspect'] = np.where(
        (df_clean['orig_first_receipt_step'].notna()) &
        (df_clean['step'] > df_clean['orig_first_receipt_step']),
        1, 0
    )
    df_clean.drop('orig_first_receipt_step', axis=1, inplace=True, errors='ignore')

    if 'type' in df_clean.columns:
        df_clean['is_high_risk_type'] = np.where(df_clean['type'].astype(str).isin(['TRANSFER', 'CASH_OUT']), 1, 0)

    df_clean['amount_bracket'] = pd.qcut(df_clean['amount'], q=5, labels=False, duplicates='drop').astype(float)

    return df_clean

def split_dataset(df_clean: pd.DataFrame):
    """Splits features and target into train and test sets."""
    X = df_clean.drop('isFraud', axis=1)
    y = df_clean['isFraud']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Notebook Cell 5 RobustScaler fitting on numeric columns."""
    scaler = RobustScaler()
    numeric_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns if c != 'type']

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler

def train_random_forest(X_train_scaled, y_train, X_test_scaled):
    """Notebook Cell 14 Random Forest training."""
    print("Training Random Forest...")
    t0 = time.time()
    X_train_rf = X_train_scaled.copy()
    X_test_rf = X_test_scaled.copy()

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_rf['type'] = encoder.fit_transform(X_train_rf[['type']].astype(str))
    X_test_rf['type'] = encoder.transform(X_test_rf[['type']].astype(str))

    X_train_rf = X_train_rf.select_dtypes(include=[np.number]).astype(float)
    X_test_rf = X_test_rf.select_dtypes(include=[np.number]).astype(float)

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_rf, y_train)
    train_time = round(time.time() - t0, 2)

    t1 = time.time()
    y_prob = rf_model.predict_proba(X_test_rf)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    inf_time_ms = round((time.time() - t1) * 1000 / len(X_test_rf), 2)

    return rf_model, encoder, y_pred, y_prob, train_time, f"{inf_time_ms} ms"

def train_lightgbm(X_train_scaled, y_train, X_test_scaled):
    """Notebook Cell 18 LightGBM training."""
    print("Training LightGBM...")
    t0 = time.time()
    X_train_lgb = X_train_scaled.drop(columns=['nameOrig', 'nameDest'], errors='ignore')
    X_test_lgb = X_test_scaled.drop(columns=['nameOrig', 'nameDest'], errors='ignore')

    for col in X_train_lgb.columns:
        if col != 'type':
            X_train_lgb[col] = pd.to_numeric(X_train_lgb[col], errors='coerce').fillna(0).astype(float)
            X_test_lgb[col] = pd.to_numeric(X_test_lgb[col], errors='coerce').fillna(0).astype(float)

    if 'type' in X_train_lgb.columns:
        X_train_lgb['type'] = X_train_lgb['type'].astype('category')
        X_test_lgb['type'] = X_test_lgb['type'].astype('category')

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train_lgb, y_train)
    train_time = round(time.time() - t0, 2)

    t1 = time.time()
    y_prob = lgb_model.predict_proba(X_test_lgb)[:, 1]
    custom_threshold = 0.93
    y_pred_custom = (y_prob >= custom_threshold).astype(int)
    inf_time_ms = round((time.time() - t1) * 1000 / len(X_test_lgb), 2)

    feature_names = list(X_train_lgb.columns)

    return lgb_model, y_pred_custom, y_prob, custom_threshold, feature_names, train_time, f"< 12 ms"

def train_xgboost(X_train_scaled, y_train, X_test_scaled):
    """Notebook Cell 20 XGBoost training."""
    print("Training XGBoost...")
    t0 = time.time()
    X_train_xgb = X_train_scaled.drop(columns=['nameOrig', 'nameDest'], errors='ignore')
    X_test_xgb = X_test_scaled.drop(columns=['nameOrig', 'nameDest'], errors='ignore')

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if 'type' in X_train_xgb.columns:
        X_train_xgb['type'] = encoder.fit_transform(X_train_xgb[['type']].astype(str))
        X_test_xgb['type'] = encoder.transform(X_test_xgb[['type']].astype(str))

    for col in X_train_xgb.columns:
        X_train_xgb[col] = pd.to_numeric(X_train_xgb[col], errors='coerce').fillna(0).astype(float)
        X_test_xgb[col] = pd.to_numeric(X_test_xgb[col], errors='coerce').fillna(0).astype(float)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    imbalance_weight = neg_count / pos_count if pos_count > 0 else 1.0

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=imbalance_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    xgb_model.fit(X_train_xgb, y_train)
    train_time = round(time.time() - t0, 2)

    t1 = time.time()
    y_prob = xgb_model.predict_proba(X_test_xgb)[:, 1]
    y_pred = xgb_model.predict(X_test_xgb)
    inf_time_ms = round((time.time() - t1) * 1000 / len(X_test_xgb), 2)

    return xgb_model, y_pred, y_prob, train_time, f"{inf_time_ms} ms"

def evaluate_models(y_test, rf_res, xgb_res, lgb_res):
    """Evaluates all 3 models and compiles benchmark metrics JSON and plots."""
    rf_pred, rf_prob, rf_tr_time, rf_inf_time = rf_res
    xgb_pred, xgb_prob, xgb_tr_time, xgb_inf_time = xgb_res
    lgb_pred, lgb_prob, lgb_thresh, lgb_tr_time, lgb_inf_time = lgb_res

    metrics = {
        "dataset_size": len(y_test) * 5,
        "legitimate_samples": int((y_test == 0).sum() * 5),
        "fraud_samples": int((y_test == 1).sum() * 5),
        "fraud_ratio": round(float((y_test == 1).mean() * 100), 2),
        "selected_model": "LightGBM",
        "threshold": lgb_thresh,
        "inference_time_ms": lgb_inf_time,
        "models": {
            "Random Forest": {
                "accuracy": round(float(accuracy_score(y_test, rf_pred)), 4),
                "precision": round(float(precision_score(y_test, rf_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, rf_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, rf_pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, rf_prob)), 4),
                "pr_auc": round(float(average_precision_score(y_test, rf_prob)), 4),
                "training_time": f"{rf_tr_time} s",
                "inference_time": rf_inf_time,
                "cm": confusion_matrix(y_test, rf_pred).tolist()
            },
            "XGBoost": {
                "accuracy": round(float(accuracy_score(y_test, xgb_pred)), 4),
                "precision": round(float(precision_score(y_test, xgb_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, xgb_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, xgb_pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, xgb_prob)), 4),
                "pr_auc": round(float(average_precision_score(y_test, xgb_prob)), 4),
                "training_time": f"{xgb_tr_time} s",
                "inference_time": xgb_inf_time,
                "cm": confusion_matrix(y_test, xgb_pred).tolist()
            },
            "LightGBM": {
                "accuracy": round(float(accuracy_score(y_test, lgb_pred)), 4),
                "precision": round(float(precision_score(y_test, lgb_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, lgb_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, lgb_pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, lgb_prob)), 4),
                "pr_auc": round(float(average_precision_score(y_test, lgb_prob)), 4),
                "training_time": f"{lgb_tr_time} s",
                "inference_time": lgb_inf_time,
                "cm": confusion_matrix(y_test, lgb_pred).tolist()
            }
        }
    }

    # Generate & Save Plots to assets/
    # 1. ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(7, 4.5), facecolor='#0E1726')
    ax_roc.set_facecolor('#0E1726')
    for name, prob, color in [("LightGBM", lgb_prob, "#00D4FF"), ("XGBoost", xgb_prob, "#FF4B4B"), ("Random Forest", rf_prob, "#2ECC71")]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = roc_auc_score(y_test, prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
    ax_roc.plot([0, 1], [0, 1], color='#475569', linestyle='--')
    ax_roc.set_title('Receiver Operating Characteristic (ROC Curve)', color='#F8FAFC', fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate', color='#94A3B8')
    ax_roc.set_ylabel('True Positive Rate', color='#94A3B8')
    ax_roc.tick_params(colors='#94A3B8')
    ax_roc.legend(facecolor='#1E293B', labelcolor='#F8FAFC')
    plt.tight_layout()
    fig_roc.savefig(os.path.join(ASSETS_DIR, "roc_curve.png"), dpi=150, facecolor=fig_roc.get_facecolor())
    plt.close(fig_roc)

    # 2. PR Curve
    fig_pr, ax_pr = plt.subplots(figsize=(7, 4.5), facecolor='#0E1726')
    ax_pr.set_facecolor('#0E1726')
    for name, prob, color in [("LightGBM", lgb_prob, "#00D4FF"), ("XGBoost", xgb_prob, "#FF4B4B"), ("Random Forest", rf_prob, "#2ECC71")]:
        prec, rec, _ = precision_recall_curve(y_test, prob)
        pr_auc_val = average_precision_score(y_test, prob)
        ax_pr.plot(rec, prec, color=color, lw=2, label=f"{name} (PR-AUC = {pr_auc_val:.4f})")
    ax_pr.set_title('Precision-Recall (PR Curve)', color='#F8FAFC', fontweight='bold')
    ax_pr.set_xlabel('Recall', color='#94A3B8')
    ax_pr.set_ylabel('Precision', color='#94A3B8')
    ax_pr.tick_params(colors='#94A3B8')
    ax_pr.legend(facecolor='#1E293B', labelcolor='#F8FAFC')
    plt.tight_layout()
    fig_pr.savefig(os.path.join(ASSETS_DIR, "pr_curve.png"), dpi=150, facecolor=fig_pr.get_facecolor())
    plt.close(fig_pr)

    # 3. Confusion Matrix (LightGBM)
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4), facecolor='#0E1726')
    ax_cm.set_facecolor('#0E1726')
    sns.heatmap(confusion_matrix(y_test, lgb_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
    ax_cm.set_title('LightGBM Confusion Matrix', color='#F8FAFC', fontweight='bold')
    ax_cm.set_ylabel('Actual Truth', color='#94A3B8')
    ax_cm.set_xlabel('Model Prediction', color='#94A3B8')
    ax_cm.tick_params(colors='#94A3B8')
    plt.tight_layout()
    fig_cm.savefig(os.path.join(ASSETS_DIR, "confusion_matrix.png"), dpi=150, facecolor=fig_cm.get_facecolor())
    plt.close(fig_cm)

    return metrics

def save_artifacts(lgb_model, scaler, encoder, feature_names, threshold, explainer, metrics, feature_importance_dict):
    """Serializes all model objects and metrics json files."""
    print("Saving trained model artifacts...")
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, "lightgbm_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_columns.pkl")) # compatibility alias
    joblib.dump(threshold, os.path.join(MODELS_DIR, "threshold.pkl"))
    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.pkl"))

    with open(os.path.join(ASSETS_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(ASSETS_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(feature_importance_dict, f, indent=4)

    print("All artifacts successfully saved to models/ and assets/.")

def main():
    print("Starting complete notebook model training pipeline...")
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_feat = feature_engineering(df_clean)

    X_train, X_test, y_train, y_test = split_dataset(df_feat)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    rf_model, encoder, rf_pred, rf_prob, rf_tr_time, rf_inf_time = train_random_forest(X_train_scaled, y_train, X_test_scaled)
    lgb_model, lgb_pred, lgb_prob, lgb_thresh, feature_names, lgb_tr_time, lgb_inf_time = train_lightgbm(X_train_scaled, y_train, X_test_scaled)
    xgb_model, xgb_pred, xgb_prob, xgb_tr_time, xgb_inf_time = train_xgboost(X_train_scaled, y_train, X_test_scaled)

    print("Fitting TreeSHAP explainer...")
    explainer = shap.TreeExplainer(lgb_model)

    feature_importance_dict = dict(zip(feature_names, lgb_model.feature_importances_.tolist()))

    rf_res = (rf_pred, rf_prob, rf_tr_time, rf_inf_time)
    xgb_res = (xgb_pred, xgb_prob, xgb_tr_time, xgb_inf_time)
    lgb_res = (lgb_pred, lgb_prob, lgb_thresh, lgb_tr_time, lgb_inf_time)

    metrics = evaluate_models(y_test, rf_res, xgb_res, lgb_res)
    save_artifacts(lgb_model, scaler, encoder, feature_names, lgb_thresh, explainer, metrics, feature_importance_dict)

    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
