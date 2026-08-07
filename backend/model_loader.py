import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

REQUIRED_FILES = [
    "lightgbm_model.pkl",
    "scaler.pkl",
    "feature_names.pkl",
    "threshold.pkl",
    "shap_explainer.pkl",
]

def get_artifact_paths():
    fn_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    if not os.path.exists(fn_path):
        fn_path = os.path.join(MODELS_DIR, "feature_columns.pkl")

    enc_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    if not os.path.exists(enc_path):
        enc_path = os.path.join(MODELS_DIR, "encoder.pkl")

    return {
        "model": os.path.join(MODELS_DIR, "lightgbm_model.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
        "encoder": enc_path,
        "feature_names": fn_path,
        "threshold": os.path.join(MODELS_DIR, "threshold.pkl"),
        "shap_explainer": os.path.join(MODELS_DIR, "shap_explainer.pkl"),
    }

def load_all_artifacts():
    """
    Loads pre-trained model artifacts from models/ directory.
    Frontend and backend predictor use this directly without retraining.
    """
    paths = get_artifact_paths()

    return {
        "model": joblib.load(paths["model"]),
        "scaler": joblib.load(paths["scaler"]),
        "encoder": joblib.load(paths["encoder"]) if os.path.exists(paths["encoder"]) else None,
        "feature_columns": joblib.load(paths["feature_names"]),
        "feature_names": joblib.load(paths["feature_names"]),
        "threshold": joblib.load(paths["threshold"]),
        "shap_explainer": joblib.load(paths["shap_explainer"]),
    }
