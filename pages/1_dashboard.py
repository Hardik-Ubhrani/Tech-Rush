import streamlit as st
from utils.styles import inject_custom_css

# Inject modern dark cybersecurity design theme
inject_custom_css()

# ==========================================
# HERO SECTION
# ==========================================
st.markdown(
    """
    <div class="hero-container animate-fade-in">
        <div class="hero-badge">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #00D4FF; box-shadow: 0 0 8px #00D4FF;"></span>
            Enterprise Banking Security Platform
        </div>
        <div class="hero-title">
            AI Powered Financial Fraud Detection System
        </div>
        <div class="hero-subtitle">
            Real-time fraud detection using LightGBM, Explainable AI and advanced feature engineering.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero Action Buttons
btn_col1, btn_col2, _ = st.columns([1.3, 1.8, 4])
with btn_col1:
    if st.button("Start Prediction", type="primary", use_container_width=True):
        st.switch_page("pages/2_prediction.py")

with btn_col2:
    if st.button("View Model Comparison", type="secondary", use_container_width=True):
        st.switch_page("pages/3_model_comparison.py")

st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)

from backend.utils import get_notebook_benchmark_metrics

# Fetch exact notebook metrics
metrics = get_notebook_benchmark_metrics()
lgb_metrics = metrics["models"]["LightGBM"]

# ==========================================
# KPI CARDS
# ==========================================
st.markdown("##### 📈 **MODEL PERFORMANCE BENCHMARKS (NOTEBOOK SOURCE OF TRUTH)**")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">ROC-AUC</div>
            <div class="kpi-value">{lgb_metrics['roc_auc']:.4f}</div>
            <div class="kpi-sub">↑ Top-tier Discrimination</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">F1 Score</div>
            <div class="kpi-value">{lgb_metrics['f1']:.2f}</div>
            <div class="kpi-sub">↑ Calibrated Precision/Recall</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Precision</div>
            <div class="kpi-value">{lgb_metrics['precision']*100:.1f}%</div>
            <div class="kpi-sub">↓ Low False Positive Rate</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Inference Time</div>
            <div class="kpi-value">{lgb_metrics['inference_time']}</div>
            <div class="kpi-sub">⚡ Real-time Latency</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# Dataset Summary Metrics Row
d1, d2, d3, d4, d5 = st.columns(5)
with d1:
    st.metric("Total Dataset Rows", f"{metrics['dataset_size']:,}")
with d2:
    st.metric("Legitimate Samples", f"{metrics['legitimate_samples']:,}")
with d3:
    st.metric("Fraud Samples", f"{metrics['fraud_samples']:,}")
with d4:
    st.metric("Fraud Ratio", f"{metrics['fraud_ratio']:.2f}%")
with d5:
    st.metric("Custom Threshold", f"{metrics['threshold']}")

st.markdown("<div style='margin-bottom: 36px;'></div>", unsafe_allow_html=True)

# ==========================================
# WORKFLOW PIPELINE (HORIZONTAL)
# ==========================================
st.markdown("##### 🔄 **END-TO-END FRAUD INFERENCE WORKFLOW**")

st.markdown(
    """
    <div class="workflow-container">
        <div class="workflow-node">
            <div class="workflow-icon">💳</div>
            <div class="workflow-label">Transaction</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-node">
            <div class="workflow-icon">⚡</div>
            <div class="workflow-label">Feature Engineering</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-node">
            <div class="workflow-icon">🤖</div>
            <div class="workflow-label">LightGBM</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-node">
            <div class="workflow-icon">🎯</div>
            <div class="workflow-label">Risk Score</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-node">
            <div class="workflow-icon">🔬</div>
            <div class="workflow-label">SHAP Explainability</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-node">
            <div class="workflow-icon">🛡️</div>
            <div class="workflow-label">Final Decision</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 36px;'></div>", unsafe_allow_html=True)

# ==========================================
# FEATURE HIGHLIGHTS CARDS
# ==========================================
st.markdown("##### ✨ **SYSTEM CAPABILITIES & CORE FEATURES**")

feat1, feat2, feat3, feat4 = st.columns(4)

with feat1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">⚡</div>
            <div class="feature-title">Real-time Prediction</div>
            <div class="feature-desc">
                Instantaneous transaction risk scoring engine evaluating high-dimensional feature vectors to detect fraud before completion.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feat2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">🔬</div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">
                Transparent model decisions powered by SHAP feature contribution waterfall plots for regulatory audit compliance and trust.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feat3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">📊</div>
            <div class="feature-title">Feature Engineering</div>
            <div class="feature-desc">
                Enterprise domain-specific financial features including balance delta ratios, account velocities, and merchant risk profiles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with feat4:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">📄</div>
            <div class="feature-title">Investigation Report</div>
            <div class="feature-desc">
                Comprehensive automated case reports providing evidence breakdown, risk drivers, and SAR-ready export documentation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
