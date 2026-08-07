import pandas as pd
import numpy as np

# Quantile boundaries from PaySim dataset for amount_bracket (q=5)
AMOUNT_QUANTILES = [0.0, 10000.0, 50000.0, 150000.0, 500000.0, float('inf')]

def add_engineered_features(df: pd.DataFrame, is_single: bool = False) -> pd.DataFrame:
    """
    Applies EXACT feature engineering as defined in Notebook Cell 10.
    Calculates:
    - errorBalanceOrig
    - errorBalanceDest
    - is_account_drain
    - hour_of_day
    - Dest_Total_Received_Volume
    - is_Mule_Suspect
    - is_high_risk_type
    - amount_bracket
    """
    df = df.copy()

    # 1. Force ledger columns to pure numeric
    ledger_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step']
    for col in ledger_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 2. Sort by step if batch mode
    if not is_single and 'step' in df.columns:
        df = df.sort_values(by='step').reset_index(drop=True)

    # 3. Core Ledger Errors
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # 4. Account Drain Flag
    df['is_account_drain'] = np.where(df['amount'] == df['oldbalanceOrg'], 1, 0)

    # 5. Time-of-Day Risk
    df['hour_of_day'] = df['step'] % 24

    # 6. Destination Velocity (Cumulative Volume)
    if 'nameDest' in df.columns and not is_single:
        df['Dest_Total_Received_Volume'] = df.groupby('nameDest', observed=True)['amount'].cumsum()
    else:
        # For single transaction, default to current amount + previous receiver balance
        df['Dest_Total_Received_Volume'] = df['oldbalanceDest'] + df['amount']

    # 7. Mule Suspect Tracking
    if 'nameDest' in df.columns and 'nameOrig' in df.columns and not is_single:
        first_receipt = df.groupby('nameDest', observed=True)['step'].min().to_dict()
        df['orig_first_receipt_step'] = df['nameOrig'].map(first_receipt)
        df['is_Mule_Suspect'] = np.where(
            (df['orig_first_receipt_step'].notna()) &
            (df['step'] > df['orig_first_receipt_step']),
            1, 0
        )
        df.drop('orig_first_receipt_step', axis=1, inplace=True, errors='ignore')
    else:
        # Default mule suspect check based on type and zero initial balance
        df['is_Mule_Suspect'] = np.where(
            (df['type'].isin(['TRANSFER', 'CASH_OUT'])) & (df['oldbalanceDest'] == 0),
            1, 0
        )

    # 8. High-Risk Transaction Type Flag
    if 'type' in df.columns:
        # Convert category or string to list comparison
        df['is_high_risk_type'] = np.where(df['type'].astype(str).isin(['TRANSFER', 'CASH_OUT']), 1, 0)
    else:
        df['is_high_risk_type'] = 0

    # 9. Amount Bracket (5 Quantile size ranges)
    if not is_single and len(df) > 10:
        try:
            df['amount_bracket'] = pd.qcut(df['amount'], q=5, labels=False, duplicates='drop').astype(float)
        except Exception:
            df['amount_bracket'] = pd.cut(df['amount'], bins=AMOUNT_QUANTILES, labels=[0, 1, 2, 3, 4], include_lowest=True).astype(float).fillna(0)
    else:
        df['amount_bracket'] = pd.cut(df['amount'], bins=AMOUNT_QUANTILES, labels=[0, 1, 2, 3, 4], include_lowest=True).astype(float).fillna(0)

    return df
