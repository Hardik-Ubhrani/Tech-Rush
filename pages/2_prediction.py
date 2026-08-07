import streamlit as st
import pandas as pd
from utils.styles import inject_custom_css
from backend.predictor import predict_transaction
from backend.utils import create_shap_waterfall_fig

# Inject global dark cybersecurity styling
inject_custom_css()

# ==========================================
# PAGE HEADER
# ==========================================
st.markdown("### 🔍 **Single Transaction Fraud Prediction**")
st.markdown(
    "<p style='color: #94A3B8; margin-bottom: 28px;'>"
    "Input transaction details below to run real-time inference and SHAP explainability analysis."
    "</p>",
    unsafe_allow_html=True,
)

# ==========================================
# ELEGANT TRANSACTION FORM
# ==========================================
st.markdown("##### 📝 **TRANSACTION INPUT PARAMETERS**")

with st.container():
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "DEPOSIT", "DEBIT"],
            index=0,
            help="Select the financial transaction category",
        )

        amount = st.number_input(
            "Amount ($)",
            min_value=0.0,
            value=150000.00,
            step=1000.00,
            format="%.2f",
            help="Total monetary value of the transaction",
        )

        old_sender_balance = st.number_input(
            "Old Sender Balance ($)",
            min_value=0.0,
            value=150000.00,
            step=1000.00,
            format="%.2f",
            help="Sender balance prior to transaction execution",
        )

        new_sender_balance = st.number_input(
            "New Sender Balance ($)",
            min_value=0.0,
            value=0.00,
            step=1000.00,
            format="%.2f",
            help="Sender balance post-transaction execution",
        )

    with col2:
        old_receiver_balance = st.number_input(
            "Old Receiver Balance ($)",
            min_value=0.0,
            value=0.00,
            step=1000.00,
            format="%.2f",
            help="Receiver balance prior to transaction execution",
        )

        new_receiver_balance = st.number_input(
            "New Receiver Balance ($)",
            min_value=0.0,
            value=150000.00,
            step=1000.00,
            format="%.2f",
            help="Receiver balance post-transaction execution",
        )

        transaction_hour = st.slider(
            "Transaction Hour (0 - 23)",
            min_value=0,
            max_value=23,
            value=14,
            help="Hour of the day when transaction occurred",
        )

        transaction_day = st.slider(
            "Transaction Day (1 - 31)",
            min_value=1,
            max_value=31,
            value=15,
            help="Day of the month when transaction occurred",
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# Large Button: Analyze Transaction
analyze_btn = st.button("Analyze Transaction", type="primary", use_container_width=True)

st.markdown("<div style='margin-bottom: 36px;'></div>", unsafe_allow_html=True)

# Handle prediction execution
if analyze_btn:
    with st.spinner("Executing real-time LightGBM inference & SHAP explanation..."):
        res = predict_transaction(
            transaction_type=transaction_type,
            amount=amount,
            old_sender_balance=old_sender_balance,
            new_sender_balance=new_sender_balance,
            old_receiver_balance=old_receiver_balance,
            new_receiver_balance=new_receiver_balance,
            transaction_hour=transaction_hour,
            transaction_day=transaction_day,
        )
        st.session_state["latest_prediction"] = res
        st.success(f"Inference completed in {res['latency_ms']} ms")

# Retrieve latest prediction if available
res = st.session_state.get("latest_prediction")

# ==========================================
# MODEL INFERENCE & EXPLAINABILITY OUTPUT
# ==========================================
st.markdown("##### 📊 **MODEL INFERENCE & EXPLAINABILITY OUTPUT**")

if res is not None:
    # 1. Prediction Metrics
    m_col1, m_col2, m_col3 = st.columns(3)

    risk_color = "#FF4B4B" if res["prediction"] == 1 else "#00D4FF"

    with m_col1:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 4px solid {risk_color};">
                <div class="kpi-title">Risk Level</div>
                <div class="kpi-value" style="color: {risk_color}; font-size: 1.3rem;">{res['risk_level']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Confidence</div>
                <div class="kpi-value">{res['confidence']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Fraud Probability</div>
                <div class="kpi-value" style="color: {risk_color};">{res['fraud_probability']*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Decision</div>
                <div class="kpi-value" style="font-size: 1.1rem;">{res['decision']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Risk Score</div>
                <div class="kpi-value">{res['risk_score']} / 100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Inference Latency</div>
                <div class="kpi-value">{res['latency_ms']} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # Action Protocol
    st.markdown("##### 🛡️ **RECOMMENDED ACTION PROTOCOL**")
    st.info(res["recommended_action"])

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 2. SHAP Waterfall Plot
    st.markdown("##### 🔬 **SHAP FEATURE EXPLANATION WATERFALL**")
    fig_shap = create_shap_waterfall_fig(res["top_features"])
    st.pyplot(fig_shap)

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 3. Top Features Table
    st.markdown("##### 📋 **TOP CONTRIBUTING RISK FEATURES**")
    df_top = pd.DataFrame(res["top_features"])[["feature", "feature_value", "shap_value"]]
    df_top.columns = ["Feature Name", "Transaction Value", "SHAP Impact Score (+ = Fraud Risk)"]
    st.dataframe(df_top, use_container_width=True)

else:
    # Placeholder state when no transaction has been analyzed yet
    st.caption("Click 'Analyze Transaction' above to execute live LightGBM model prediction.")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown('<div class="placeholder-card"><div class="placeholder-title">Risk Level</div><div class="placeholder-value">--</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder-card"><div class="placeholder-title">Confidence</div><div class="placeholder-value">--</div></div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="placeholder-card"><div class="placeholder-title">Fraud Probability</div><div class="placeholder-value">--</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder-card"><div class="placeholder-title">Decision</div><div class="placeholder-value">--</div></div>', unsafe_allow_html=True)
    with m_col3:
        st.markdown('<div class="placeholder-card"><div class="placeholder-title">Risk Score</div><div class="placeholder-value">--</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder-card"><div class="placeholder-title">Recommended Action</div><div class="placeholder-value">--</div></div>', unsafe_allow_html=True)
