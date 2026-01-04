import streamlit as st
import base64

def load_css():
    """Charge le CSS personnalisé pour le thème sombre/néon."""
    st.markdown("""
    <style>
        /* General Theme */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #1E232F;
            border: 1px solid #2B3342;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #4F8BF9;
            box-shadow: 0 6px 12px rgba(79, 139, 249, 0.2);
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #4F8BF9 0%, #9B5DE5 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            opacity: 0.9;
            box-shadow: 0 0 15px rgba(79, 139, 249, 0.5);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #1E232F;
            border-radius: 5px;
            color: #FAFAFA;
            padding: 10px 20px;
            border: 1px solid #2B3342;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2B3342 !important;
            border-color: #4F8BF9 !important;
            color: #4F8BF9 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def safe_query_params():
    """Récupère les paramètres d'URL de manière sécurisée."""
    try:
        return st.query_params
    except:
        return {}

def render_header():
    """Affiche l'en-tête de l'application."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="background: linear-gradient(90deg, #4F8BF9, #9B5DE5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; margin-bottom: 0.5rem;">
            TBC Analysis Dashboard
        </h1>
        <p style="color: #A0AEC0; font-size: 1.2rem;">
            Évaluation Thermomécanique des Aubes de Turbine Multicouches
        </p>
    </div>
    """, unsafe_allow_html=True)
