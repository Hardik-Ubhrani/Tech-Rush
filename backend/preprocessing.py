import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies exact pre-cleaning audit, null removal, smart imputation, and native categorical casting
    as specified in Notebook Cell 3 and Cell 6.
    """
    df = df.copy()
    
    # 1. Drop rows missing essential identifiers or transaction amount if subset exists
    subset_cols = [c for c in ['nameOrig', 'nameDest', 'amount'] if c in df.columns]
    if subset_cols:
        df.dropna(subset=subset_cols, inplace=True)
        
    # 2. Force ledger columns to pure numeric
    ledger_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
    for col in ledger_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 3. Smart Imputation for remaining features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # 4. Handle duplicates
    df.drop_duplicates(inplace=True)

    # 5. Drop useless flag
    if 'isFlaggedFraud' in df.columns:
        df.drop('isFlaggedFraud', axis=1, inplace=True, errors='ignore')

    # 6. Native Categorical Casting
    cols_to_cast = ['type', 'nameOrig', 'nameDest']
    for col in cols_to_cast:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df

def preprocess_single_transaction(raw_tx: dict) -> pd.DataFrame:
    """
    Converts a single input dictionary into a formatted DataFrame ready for feature engineering.
    """
    df = pd.DataFrame([raw_tx])
    
    # Defaults for optional identifier columns
    if 'nameOrig' not in df.columns:
        df['nameOrig'] = 'C_USER_INPUT'
    if 'nameDest' not in df.columns:
        df['nameDest'] = 'M_DEST_INPUT'
    if 'step' not in df.columns:
        # Calculate step from day and hour if provided
        day = raw_tx.get('day', 1)
        hour = raw_tx.get('hour', 12)
        df['step'] = int((day - 1) * 24 + hour)
        
    # Ensure columns exist with numeric dtypes
    ledger_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
    for col in ledger_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
    if 'type' in df.columns:
        df['type'] = df['type'].astype('category')
        
    return df
