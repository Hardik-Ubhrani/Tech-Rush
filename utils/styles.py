import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Main app background */
        .stApp {
            background-color: #0B1220;
            color: #FFFFFF;
        }

        /* Hide standard Streamlit header clutter if preferred */
        header[data-testid="stHeader"] {
            background: transparent;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0B1220;
            border-right: 1px solid rgba(0, 212, 255, 0.12);
        }

        section[data-testid="stSidebar"] .stSelectbox, 
        section[data-testid="stSidebar"] .stMultiSelect {
            background-color: #131C2F;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 15px rgba(0, 212, 255, 0.15); }
            50% { box-shadow: 0 0 25px rgba(0, 212, 255, 0.3); }
            100% { box-shadow: 0 0 15px rgba(0, 212, 255, 0.15); }
        }

        .animate-fade-in {
            animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        /* Hero Container */
        .hero-container {
            background: linear-gradient(135deg, #131C2F 0%, #0F172A 100%);
            border: 1px solid rgba(0, 212, 255, 0.25);
            border-radius: 16px;
            padding: 40px 36px;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.5s ease-out;
        }

        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 450px;
            height: 450px;
            background: radial-gradient(circle, rgba(0, 212, 255, 0.12) 0%, rgba(0, 212, 255, 0) 70%);
            pointer-events: none;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            color: #00D4FF;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
            text-transform: uppercase;
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #FFFFFF !important;
            margin-bottom: 12px;
            line-height: 1.2;
            letter-spacing: -0.5px;
        }

        .hero-subtitle {
            font-size: 1.125rem;
            color: #94A3B8;
            max-width: 720px;
            margin-bottom: 28px;
            line-height: 1.6;
        }

        /* Card Container */
        .cyber-card {
            background-color: #131C2F;
            border: 1px solid rgba(0, 212, 255, 0.15);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
        }

        .cyber-card:hover {
            border-color: rgba(0, 212, 255, 0.45);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 212, 255, 0.12);
        }

        /* KPI Card Specifics */
        .kpi-card {
            background: linear-gradient(180deg, #131C2F 0%, #0E1626 100%);
            border: 1px solid rgba(0, 212, 255, 0.18);
            border-radius: 14px;
            padding: 22px;
            transition: all 0.3s ease;
            height: 100%;
        }

        .kpi-card:hover {
            border-color: #00D4FF;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.18);
            transform: translateY(-3px);
        }

        .kpi-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 8px;
        }

        .kpi-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #00D4FF;
            line-height: 1.1;
            letter-spacing: -0.5px;
        }

        .kpi-sub {
            font-size: 0.8rem;
            color: #00E676;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Workflow Step Cards */
        .workflow-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 24px 16px;
            background: #131C2F;
            border: 1px solid rgba(0, 212, 255, 0.15);
            border-radius: 16px;
            margin: 20px 0 32px 0;
            overflow-x: auto;
        }

        .workflow-node {
            flex: 1;
            min-width: 130px;
            background: #0B1220;
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 12px;
            padding: 16px 12px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .workflow-node:hover {
            border-color: #00D4FF;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.25);
            transform: scale(1.03);
        }

        .workflow-icon {
            font-size: 1.5rem;
            margin-bottom: 8px;
        }

        .workflow-label {
            font-size: 0.88rem;
            font-weight: 700;
            color: #FFFFFF;
        }

        .workflow-arrow {
            color: #00D4FF;
            font-size: 1.4rem;
            font-weight: bold;
            opacity: 0.7;
            user-select: none;
        }

        /* Feature Highlight Cards */
        .feature-card {
            background-color: #131C2F;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 24px;
            height: 100%;
            transition: all 0.3s ease;
        }

        .feature-card:hover {
            border-color: rgba(0, 212, 255, 0.4);
            box-shadow: 0 6px 24px rgba(0, 212, 255, 0.12);
            transform: translateY(-2px);
        }

        .feature-icon-wrapper {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            background: rgba(0, 212, 255, 0.12);
            border: 1px solid rgba(0, 212, 255, 0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            margin-bottom: 16px;
        }

        .feature-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 8px;
        }

        .feature-desc {
            font-size: 0.88rem;
            color: #94A3B8;
            line-height: 1.55;
        }

        /* Placeholder Containers */
        .placeholder-card {
            background-color: #131C2F;
            border: 1px dashed rgba(0, 212, 255, 0.25);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .placeholder-card:hover {
            border-color: rgba(0, 212, 255, 0.5);
            background-color: rgba(19, 28, 47, 0.8);
        }

        .placeholder-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #8A99AD;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 6px;
        }

        .placeholder-value {
            font-size: 1.1rem;
            color: #64748B;
            font-weight: 500;
        }

        /* Table Styling */
        .cyber-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(0, 212, 255, 0.2);
            margin-bottom: 24px;
        }
        .cyber-table th {
            background-color: #0B1220;
            color: #00D4FF;
            padding: 14px 16px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: left;
            border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        }
        .cyber-table td {
            background-color: #131C2F;
            color: #FFFFFF;
            padding: 16px;
            font-size: 0.92rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .cyber-table tr:last-child td {
            border-bottom: none;
        }
        .cyber-table tr.highlight-row td {
            background-color: rgba(0, 212, 255, 0.08);
            font-weight: 600;
        }

        /* Utility color classes */
        .text-primary { color: #00D4FF !important; }
        .text-success { color: #00E676 !important; }
        .text-warning { color: #FFC107 !important; }
        .text-danger { color: #FF5252 !important; }
        .text-muted { color: #8A99AD !important; }

        /* Custom Streamlit Buttons */
        div.stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.25s ease !important;
            padding: 10px 24px !important;
            height: auto !important;
        }

        div.stButton > button[kind="primary"] {
            background-color: #00D4FF !important;
            color: #0B1220 !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: #33E0FF !important;
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.45) !important;
            transform: translateY(-1px) !important;
        }

        div.stButton > button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(0, 212, 255, 0.3) !important;
        }

        div.stButton > button[kind="secondary"]:hover {
            background-color: rgba(0, 212, 255, 0.12) !important;
            border-color: #00D4FF !important;
            color: #00D4FF !important;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0B1220;
        }
        ::-webkit-scrollbar-thumb {
            background: #131C2F;
            border-radius: 4px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #00D4FF;
        }
        </style>
    """, unsafe_allow_html=True)
