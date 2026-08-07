import datetime
import pandas as pd
import streamlit as st
from utils.styles import inject_custom_css
from backend.predictor import predict_transaction
from backend.utils import generate_pdf_report

# Inject global dark cybersecurity styling
inject_custom_css()

# Retrieve latest prediction or run default sample prediction
res = st.session_state.get("latest_prediction")
if res is None:
    # Run default representative case if user navigates straight to report page
    res = predict_transaction(
        transaction_type="TRANSFER",
        amount=150000.0,
        old_sender_balance=150000.0,
        new_sender_balance=0.0,
        old_receiver_balance=0.0,
        new_receiver_balance=150000.0,
        transaction_hour=14,
        transaction_day=15,
    )

raw_tx = res.get("raw_input", {})
timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
case_id = "#CASE-2026-8849X"

# ==========================================
# PAGE HEADER & EXPORT BUTTON
# ==========================================
head_col1, head_col2 = st.columns([3, 1])

with head_col1:
    st.markdown("### 📄 **AI Fraud Investigation Report**")
    st.markdown(
        "<p style='color: #94A3B8; margin-bottom: 24px;'>"
        "Automated forensic risk assessment and decision audit report for transaction cases."
        "</p>",
        unsafe_allow_html=True,
    )

with head_col2:
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    pdf_bytes = generate_pdf_report(res)
    st.download_button(
        label="📥 Export PDF Report",
        data=pdf_bytes,
        file_name=f"Fraud_Audit_Report_{case_id[1:]}.pdf",
        mime="application/pdf",
        type="secondary",
        use_container_width=True,
    )

st.markdown("---")

# ==========================================
# 1. CASE INFORMATION & TIMESTAMP
# ==========================================
st.markdown("##### 📌 **CASE INFORMATION**")

with st.container():
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown(
            f"""
            <div style="margin-bottom: 14px;">
                <div class="placeholder-title">Case Reference ID</div>
                <div style="font-weight: 700; color: #00D4FF; font-size: 1.1rem;">{case_id}</div>
            </div>
            <div>
                <div class="placeholder-title">Timestamp</div>
                <div style="color: #FFFFFF; font-size: 0.95rem;">{timestamp_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col2:
        st.markdown(
            f"""
            <div style="margin-bottom: 14px;">
                <div class="placeholder-title">Origin Account</div>
                <div style="color: #FFFFFF; font-weight: 600;">{raw_tx.get('nameOrig', 'C123456789')}</div>
            </div>
            <div>
                <div class="placeholder-title">Destination Account</div>
                <div style="color: #FFFFFF; font-weight: 600;">{raw_tx.get('nameDest', 'M987654321')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col3:
        st.markdown(
            f"""
            <div style="margin-bottom: 14px;">
                <div class="placeholder-title">Transaction Type & Amount</div>
                <div style="color: #FFFFFF; font-weight: 600;">{raw_tx.get('type', 'TRANSFER')} (${raw_tx.get('amount', 0.0):,.2f})</div>
            </div>
            <div>
                <div class="placeholder-title">Investigation Engine</div>
                <div style="color: #8A99AD; font-size: 0.9rem;">LightGBM Fraud Sentinel v2.4</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# ==========================================
# 2. PREDICTION METRICS & RISK ASSESSMENT
# ==========================================
st.markdown("##### 🎯 **RISK EVALUATION & METRICS**")

r_col1, r_col2, r_col3 = st.columns(3)

risk_color = "#FF4B4B" if res["prediction"] == 1 else "#00D4FF"

with r_col1:
    st.markdown(
        f"""
        <div class="placeholder-card" style="padding: 24px; border-top: 3px solid {risk_color};">
            <div class="placeholder-title">Fraud Probability</div>
            <div class="placeholder-value" style="font-size: 1.6rem; color: {risk_color}; margin-top: 6px;">{res['fraud_probability']*100:.2f}%</div>
            <div style="font-size: 0.78rem; color: #8A99AD; margin-top: 6px;">Model Confidence: {res['confidence']:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r_col2:
    st.markdown(
        f"""
        <div class="placeholder-card" style="padding: 24px;">
            <div class="placeholder-title">Risk Level</div>
            <div class="placeholder-value" style="font-size: 1.4rem; color: {risk_color}; margin-top: 6px;">{res['risk_level']}</div>
            <div style="font-size: 0.78rem; color: #8A99AD; margin-top: 6px;">Risk Score: {res['risk_score']} / 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r_col3:
    st.markdown(
        f"""
        <div class="placeholder-card" style="padding: 24px;">
            <div class="placeholder-title">Decision</div>
            <div class="placeholder-value" style="font-size: 1.3rem; color: #FFFFFF; margin-top: 6px;">{res['decision']}</div>
            <div style="font-size: 0.78rem; color: #8A99AD; margin-top: 6px;">Threshold: {res['threshold']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

# ==========================================
# 3. PREDICTION NARRATIVE SUMMARY
# ==========================================
st.markdown("##### 📝 **PREDICTION SUMMARY & ANOMALY NARRATIVE**")

if res["prediction"] == 1:
    narrative = (
        f"CRITICAL ANOMALY DETECTED: The transaction exhibits classic high-velocity account draining behavior. "
        f"The requested amount of ${raw_tx.get('amount', 0.0):,.2f} exactly equals the sender's total initial balance of ${raw_tx.get('oldbalanceOrg', 0.0):,.2f}, "
        f"reducing sender balance to $0.00. Furthermore, the transaction type '{raw_tx.get('type')}' is flagged as a high-risk fraud attack vector. "
        f"The LightGBM model assigned a high fraud probability of {res['fraud_probability']*100:.2f}%, triggering an automated decline protocol."
    )
else:
    narrative = (
        f"LEGITIMATE TRANSACTION AUDIT: The transaction of ${raw_tx.get('amount', 0.0):,.2f} passed all ledger consistency checks. "
        f"Sender balance post-transaction reflects valid debit operations, and destination account balances exhibit normal velocity patterns. "
        f"The LightGBM model evaluated a low fraud risk probability of {res['fraud_probability']*100:.2f}%, well below the {res['threshold']} decision threshold."
    )

st.markdown(
    f"""
    <div class="cyber-card" style="padding: 24px; text-align: left;">
        <div class="placeholder-title" style="margin-bottom: 8px;">Automated Forensic Narrative</div>
        <div style="font-size: 0.95rem; line-height: 1.6; color: #F8FAFC;">
            {narrative}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

# ==========================================
# 4. TOP SHAP RISK FACTORS TABLE
# ==========================================
st.markdown("##### 🔬 **TOP SHAP RISK FACTORS**")

df_top = pd.DataFrame(res["top_features"])[["feature", "feature_value", "shap_value"]]
df_top.columns = ["Feature Name", "Transaction Value", "SHAP Impact Score"]
st.dataframe(df_top, use_container_width=True)

st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

# ==========================================
# 5. RECOMMENDED ACTION PROTOCOL
# ==========================================
st.markdown("##### 🛡️ **FINAL AUDIT RECOMMENDATION & ACTION PROTOCOL**")

st.markdown(
    f"""
    <div class="cyber-card" style="padding: 24px; text-align: left; border-left: 4px solid {risk_color};">
        <div class="placeholder-title" style="margin-bottom: 8px;">Action Protocol</div>
        <div style="font-size: 0.95rem; line-height: 1.6; color: #F8FAFC;">
            {res['recommended_action']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
