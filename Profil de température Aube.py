import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.constants import CONSTANTS
from core import styles

# Import des onglets (Modules)
from tabs import mechanical, optimization, dashboard_home
from tabs import analysis_detailed, study_parametric, mapping_3d, theory_interactive
from core.reporting import generate_html_report

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(
    page_title="TBC Analysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du style CSS Premium
st.markdown(styles.load_css(), unsafe_allow_html=True)

# Constantes calculées
T_secu = CONSTANTS['T_crit'] * CONSTANTS['Securite_pct']

# ==========================================
# 2. INTERFACE SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.markdown("---")
    
    st.subheader("1. Paramètres Globaux")
    
    alpha_in = st.slider(
        "Épaisseur Céramique (α)", 
        min_value=0.0, max_value=3.0, value=0.20, step=0.05,
        key="alpha_input",
        help="Définit l'épaisseur relative de la couche TBC ($h_3 = \\alpha \\cdot h_1$)"
    )
    
    beta_in = st.slider(
        "Anisotropie Céramique (β)", 
        min_value=0.0, max_value=2.0, value=0.8, step=0.1,
        key="beta_input",
        help="Ratio k33 / k_eta. Si < 1, la conduction latérale est favorisée."
    )
    
    lw_in = st.number_input(
        "Longueur d'Onde $L_w$ (m)", 
        min_value=0.01, max_value=5.0, value=0.1, step=0.01,
        key="lw_input",
        help="Modélise la taille d'une variation de température latérale (un 'point chaud')."
    )
    
    st.markdown("---")
    
    st.subheader("2. Conditions aux Limites")
    
    # Valeurs par défaut extraites des constantes
    t_bottom_default = CONSTANTS['T_bottom']
    t_top_default = CONSTANTS['T_top']

    # La session_state est automatiquement gérée par les clés
    t_bottom_in = st.number_input("Température Base (°C)", key="T_bottom", value=t_bottom_default, step=10)
    t_top_in = st.number_input("Température Surface (°C)", key="T_top", value=t_top_default, step=10)

    def reset_temperatures():
        """Callback pour réinitialiser les températures."""
        st.session_state.T_bottom = t_bottom_default
        st.session_state.T_top = t_top_default

    st.button("Réinitialiser T°", on_click=reset_temperatures, help="Restaure les valeurs par défaut.")
    
    st.markdown("---")

    st.subheader("3. Scénario Catastrophe")
    t_bottom_catastrophe_in = st.number_input(
        "Température Base Catastrophe (°C)",
        value=t_bottom_default, step=10, key="t_bottom_cata"
    )
    t_top_catastrophe_in = st.number_input(
        "Température Surface Catastrophe (°C)",
        value=t_top_default + 100, step=10, key="t_top_cata"
    )

    st.markdown("---")
    st.caption(f"**Limites de Température**\n\n- T Critique: {CONSTANTS['T_crit']}°C\n- T Sécurité: {T_secu:.0f}°C")


# ==========================================
# 3. APPLICATION PRINCIPALE
# ==========================================

st.title("🛡️ Analyse Thermique de Revêtement (TBC)")

# Onglets principaux
tab_dashboard, tab_single, tab_multi, tab_3d, tab_mech, tab_opt, tab_theory = st.tabs([
    "🏠 Dashboard",
    "🔎 Analyse Détaillée & Impacts", 
    "📚 Étude Paramétrique (2D)",
    "🧊 Cartographie 3D (Alpha/Beta)",
    "⚙️ Calcul Mécanique",
    "📊 Sensibilité & Optimisation",
    "📖 Démarche & Théorie"
])

# --- 0. Dashboard Principal ---
with tab_dashboard:
    dashboard_home.render()

# --- 1. Analyse Détaillée ---
with tab_single:
    analysis_detailed.render(
        alpha_in, beta_in, lw_in, 
        t_bottom_in, t_top_in, 
        t_bottom_catastrophe_in, t_top_catastrophe_in
    )

# --- 2. Étude Paramétrique ---
with tab_multi:
    study_parametric.render(
        beta_in, lw_in, t_bottom_in, t_top_in
    )

# --- 3. Cartographie 3D ---
with tab_3d:
    mapping_3d.render(
        lw_in, t_bottom_in, t_top_in
    )

# --- 4. Calcul Mécanique ---
with tab_mech:
    mechanical.render()

# --- 5. Sensibilité & Optimisation ---
with tab_opt:
    optimization.render()

# --- 6. Théorie ---
with tab_theory:
    theory_interactive.render()
