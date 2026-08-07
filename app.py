import streamlit as st
from utils.styles import inject_custom_css

st.set_page_config(
    page_title="Enterprise Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark cyber-security theme
inject_custom_css()

# Page definition with required sidebar icons
dashboard_page = st.Page(
    "pages/1_dashboard.py",
    title="Dashboard",
    icon="🏠",
    default=True,
)

prediction_page = st.Page(
    "pages/2_prediction.py",
    title="Prediction",
    icon="🔍",
)

model_comparison_page = st.Page(
    "pages/3_model_comparison.py",
    title="Model Comparison",
    icon="📊",
)

investigation_report_page = st.Page(
    "pages/4_investigation_report.py",
    title="Investigation Report",
    icon="📄",
)

about_page = st.Page(
    "pages/5_about.py",
    title="About",
    icon="ℹ️",
)

pg = st.navigation(
    {
        "Navigation": [
            dashboard_page,
            prediction_page,
            model_comparison_page,
            investigation_report_page,
            about_page,
        ]
    }
)

pg.run()
