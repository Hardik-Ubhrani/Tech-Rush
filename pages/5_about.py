import streamlit as st
from utils.styles import inject_custom_css
from backend.utils import get_notebook_benchmark_metrics

# Inject global dark cybersecurity styling
inject_custom_css()

metrics = get_notebook_benchmark_metrics()
lgb = metrics["models"]["LightGBM"]

# ==========================================
# PAGE HEADER
# ==========================================
st.markdown("### ℹ️ **About The Project**")
st.markdown(
    "<p style='color: #94A3B8; margin-bottom: 28px;'>"
    "Enterprise AI-powered financial fraud detection platform leveraging LightGBM and Explainable AI."
    "</p>",
    unsafe_allow_html=True,
)

# ==========================================
# 1. PROBLEM STATEMENT & SOLUTION OVERVIEW
# ==========================================
col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown(
        """
        <div class="cyber-card" style="height: 100%;">
            <div class="feature-icon-wrapper" style="margin-bottom: 12px;">🚨</div>
            <h4 style="margin-bottom: 10px;">Problem Statement</h4>
            <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                Financial fraud causes billions of dollars in losses across global banking networks annually. 
                Traditional rule-based systems struggle with extreme class imbalance (~0.17% fraud rate) and high false-positive rates (>90%), 
                causing operational overload for fraud analysts and friction for legitimate customers while missing sophisticated account drain schemes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_p2:
    st.markdown(
        """
        <div class="cyber-card" style="height: 100%;">
            <div class="feature-icon-wrapper" style="margin-bottom: 12px;">🛡️</div>
            <h4 style="margin-bottom: 10px;">Solution Architecture</h4>
            <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                An end-to-end, high-throughput fraud detection engine built with LightGBM, RobustScaler, and TreeSHAP explainability. 
                Reuses domain-specific financial features to detect suspicious ledger deltas, mule accounts, and account drains in sub-12 milliseconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# ==========================================
# 2. DATASET & MODEL DETAILS
# ==========================================
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown(
        f"""
        <div class="cyber-card" style="height: 100%;">
            <div class="feature-icon-wrapper" style="margin-bottom: 12px;">💾</div>
            <h4 style="margin-bottom: 10px;">PaySim Dataset</h4>
            <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                Evaluated on <strong>{metrics['dataset_size']:,}</strong> financial transactions. 
                Contains <strong>{metrics['legitimate_samples']:,}</strong> legitimate records and 
                <strong>{metrics['fraud_samples']:,}</strong> fraudulent records (<strong>0.17% fraud ratio</strong>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_d2:
    st.markdown(
        f"""
        <div class="cyber-card" style="height: 100%;">
            <div class="feature-icon-wrapper" style="margin-bottom: 12px;">🤖</div>
            <h4 style="margin-bottom: 10px;">LightGBM Model</h4>
            <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                Trained with <code>n_estimators=300</code>, <code>learning_rate=0.05</code>, <code>class_weight='balanced'</code>, 
                and a calibrated decision threshold of <code>{metrics['threshold']}</code> for high precision and recall.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# ==========================================
# 3. FEATURE ENGINEERING & EVALUATION RESULTS
# ==========================================
col_fe1, col_fe2 = st.columns(2)

with col_fe1:
    st.markdown(
        """
        <div class="cyber-card" style="height: 100%;">
            <div class="feature-icon-wrapper" style="margin-bottom: 12px;">⚡</div>
            <h4 style="margin-bottom: 10px;">Feature Engineering</h4>
            <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                Derived 8 high-impact financial features: <code>errorBalanceOrig</code>, <code>errorBalanceDest</code>, 
                <code>is_account_drain</code>, <code>hour_of_day</code>, <code>Dest_Total_Received_Volume</code>, 
                <code>is_Mule_Suspect</code>, <code>is_high_risk_type</code>, and <code>amount_bracket</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_fe2:
    st.markdown(
        f"""
        <div class="cyber-card" style="height: 100%;">
            <div class="feature-icon-wrapper" style="margin-bottom: 12px;">🎯</div>
            <h4 style="margin-bottom: 10px;">Evaluation Results</h4>
            <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6;">
                <strong>ROC-AUC:</strong> {lgb['roc_auc']:.4f} &nbsp;|&nbsp; <strong>PR-AUC:</strong> {lgb['pr_auc']:.4f}<br>
                <strong>Precision:</strong> {lgb['precision']*100:.1f}% &nbsp;|&nbsp; <strong>Recall:</strong> {lgb['recall']*100:.1f}%<br>
                <strong>F1 Score:</strong> {lgb['f1']:.4f} &nbsp;|&nbsp; <strong>Inference Time:</strong> {lgb['inference_time']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# ==========================================
# 4. FUTURE SCOPE
# ==========================================
st.markdown(
    """
    <div class="cyber-card">
        <div class="feature-icon-wrapper" style="margin-bottom: 12px;">🚀</div>
        <h4 style="margin-bottom: 12px;">Future Scope & Enhancements</h4>
        <ul style="color: #94A3B8; font-size: 0.92rem; line-height: 1.8; padding-left: 20px; margin-bottom: 0;">
            <li>Streaming ingestion integration via Apache Kafka and Flink for real-time network graph feature updates.</li>
            <li>Graph Neural Network (GNN) embeddings for multi-hop money laundering ring detection.</li>
            <li>Automated Suspicious Activity Report (SAR) filing via direct REST integration with banking regulatory APIs.</li>
            <li>Adaptive online learning for real-time model retraining on newly confirmed fraud labels.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
