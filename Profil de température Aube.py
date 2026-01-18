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
        min_value=0.0, max_value=2.0, value=0.20, step=0.01,
        key="alpha_input",
        help="""**Rapport d'épaisseur TBC/Substrat** (h₃ = α × h₁)
        
🎯 **Plages recommandées:**
- α < 0.1 : Protection minimale (50 µm)
- α = 0.2-0.5 : Applications standard turbines (100-250 µm)
- α = 0.5-1.0 : Thick TBC (250-500 µm)
- α > 1.0 : Cas extrêmes (>500 µm)

📐 Avec h₁ = 500 µm, α = 0.2 → h₃ = 100 µm de céramique YSZ.""")
    
    beta_in = st.slider(
        "Anisotropie Céramique (β)", 
        min_value=0.1, max_value=1.5, value=0.8, step=0.05,
        key="beta_input",
        help="""**Ratio de conductivité** (β = k₃₃ / k_η)
        
🔬 **Physique:**
- β = 0.5-0.8 : YSZ colonnaire EBPVD (anisotropie marquée)
- β = 0.8-1.0 : YSZ APS standard
- β = 1.0 : Matériau isotrope
- β > 1.0 : Conduction normale favorisée (rare)

📊 **Valeurs typiques YSZ:** β ≈ 0.7-1.0""")
    
    lw_in = st.number_input(
        "Longueur d'Onde Lw (m)", 
        min_value=0.005, max_value=1.0, value=0.1, step=0.005, format="%.3f",
        key="lw_input",
        help="""**Période spatiale des variations thermiques**

🌡️ Modélise la taille caractéristique d'un gradient thermique.

📏 **Échelles physiques:**
- Lw = 5-20 mm : Échelle microfissures, hot spots locaux
- Lw = 50-150 mm : Variations inter-aubes typiques
- Lw = 200-500 mm : Gradients macro turbine

⚙️ Nombres d'onde: δ = π/Lw (plus Lw petit → gradient intense)""")
    
    st.markdown("---")
    
    st.subheader("2. Conditions aux Limites")
    
    # Valeurs par défaut extraites des constantes
    t_bottom_default = CONSTANTS['T_bottom']
    t_top_default = CONSTANTS['T_top']

    # La session_state est automatiquement gérée par les clés
    t_bottom_in = st.number_input(
        "Température Base (°C)", 
        key="T_bottom", 
        value=t_bottom_default, 
        step=10,
        help="Température du substrat côté refroidissement (canal interne). Typique: 400-600°C."
    )
    t_top_in = st.number_input(
        "Température Surface (°C)", 
        key="T_top", 
        value=t_top_default, 
        step=10,
        help="Température de surface exposée aux gaz chauds. Typique: 1200-1400°C (turbines haute pression)."
    )

    def reset_temperatures():
        """Callback pour réinitialiser les températures."""
        st.session_state.T_bottom = t_bottom_default
        st.session_state.T_top = t_top_default

    st.button("🔄 Réinitialiser T°", on_click=reset_temperatures, help="Restaure les valeurs par défaut (500°C / 1400°C).")
    
    st.markdown("---")

    st.subheader("3. Scénario Catastrophe")
    st.caption("⚠️ Simule une perte de refroidissement ou surchauffe")
    
    t_bottom_catastrophe_in = st.number_input(
        "Température Base Catastrophe (°C)",
        value=t_bottom_default, step=10, key="t_bottom_cata",
        help="Température élevée si perte de refroidissement interne."
    )
    t_top_catastrophe_in = st.number_input(
        "Température Surface Catastrophe (°C)",
        value=t_top_default + 100, step=10, key="t_top_cata",
        help="Surchauffe gaz chauds (ex: défaillance injection carburant)."
    )

    st.markdown("---")
    
    # Informations de référence
    st.markdown("""
    <div style="background: rgba(59,130,246,0.1); padding: 0.8rem; border-radius: 8px; border-left: 3px solid #3b82f6;">
        <div style="color: #60a5fa; font-weight: 600; font-size: 0.8rem;">📚 Références</div>
        <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.3rem;">
            • ProjectEstaca.pdf (8 étapes)<br>
            • ONERA/Safran (Inconel 718)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"**Limites:** T_crit = {CONSTANTS['T_crit']}°C | T_sécu = {T_secu:.0f}°C")


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
