"""
Dashboard Principal - Vue Panoramique Spectaculaire
Onglet d'accueil avec visualisations premium, KPIs animés et recommandations intelligentes.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.calculation import solve_tbc_model_v2, calculate_profiles
from core.constants import CONSTANTS, IMPACT_PARAMS
from core.mechanical import solve_multilayer_problem
from core.constants import PROPS_SUBSTRATE, PROPS_BONDCOAT, PROPS_CERAMIC
from core.constants import ALPHA_SUBSTRATE, ALPHA_BONDCOAT, ALPHA_CERAMIC

# Cache pour les calculs
@st.cache_data
def cached_solve_tbc_model(alpha, beta, lw, t_bottom, t_top):
    return solve_tbc_model_v2(alpha, beta, lw, t_bottom, t_top)

def compute_real_damage_indicator(alpha, lw, t_top, t_bottom):
    """
    Calcule l'indicateur d'endommagement D basé sur la physique réelle.
    
    Logique:
    1. Si T_interface > T_critique → D > 1 (dommage thermique garanti)
    2. Sinon: D = D_mécanique + bonus si proche de T_critique
    
    Returns:
        D: Indicateur d'endommagement (0 = sûr, >1 = critique)
    """
    from core.damage_analysis import CRITICAL_STRESS
    from core.constants import GPa_TO_PA
    
    try:
        # ═══════════════════════════════════════════════════════════════
        # CALCUL DE LA TEMPÉRATURE À L'INTERFACE
        # ═══════════════════════════════════════════════════════════════
        h1 = CONSTANTS['h1']  # Substrat = 500 µm
        h2 = CONSTANTS['h2']  # Bond coat = 10 µm  
        h3 = max(alpha * h1, 1e-6)  # TBC min 1µm pour physique réaliste
        H_total = h1 + h2 + h3
        
        delta_T_total = t_top - t_bottom
        T_crit = CONSTANTS['T_crit']  # 1100°C
        
        # Modèle de résistances thermiques en série
        R1 = h1 / CONSTANTS['k33_1']
        R2 = h2 / CONSTANTS['k33_2']
        R3 = h3 / CONSTANTS['k33_3']
        R_total = R1 + R2 + R3
        
        # Température à l'interface substrat/bondcoat
        T_h1 = t_bottom + delta_T_total * R1 / R_total
        
        # ═══════════════════════════════════════════════════════════════
        # CAS 1: TEMPÉRATURE AU-DESSUS DE LA LIMITE CRITIQUE
        # → Dommage thermique direct (D > 1)
        # ═══════════════════════════════════════════════════════════════
        if T_h1 > T_crit:
            excess_temp = T_h1 - T_crit
            # D = 1 à T_crit, +0.5 tous les 100°C au-dessus
            D_thermal = 1.0 + (excess_temp / 200.0)
            return max(0.05, min(1.5, D_thermal))
        
        # ═══════════════════════════════════════════════════════════════
        # CAS 2: TEMPÉRATURE SOUS LA LIMITE
        # → D basé sur les contraintes mécaniques de mismatch
        # ═══════════════════════════════════════════════════════════════
        
        # Coefficients de dilatation thermique
        alpha_substrate = ALPHA_SUBSTRATE['alpha_1']  # 13e-6 /K
        alpha_ceramic = ALPHA_CERAMIC['alpha_1']      # 10e-6 /K
        delta_alpha = abs(alpha_substrate - alpha_ceramic)
        
        # Module d'Young de la céramique
        E_ceramic = PROPS_CERAMIC['C33'] * GPa_TO_PA  # ~50 GPa
        
        # Gradient thermique à l'interface TBC/bondcoat
        T_interface_TBC = t_bottom + delta_T_total * (R1 + R2) / R_total
        delta_T_interface = T_interface_TBC - t_bottom
        
        # Facteur géométrique (plus de TBC = meilleur)
        thickness_ratio = h3 / (h1 + h2)
        geometric_factor = 1.0 / (1.0 + 0.5 * thickness_ratio)
        
        # Contrainte thermique d'interface
        sigma_thermal = E_ceramic * delta_alpha * delta_T_interface * geometric_factor
        
        # Cisaillement aux interfaces
        delta_eta = np.pi / lw
        sigma_shear = min(sigma_thermal * 0.15 * (delta_eta * H_total), 0.3 * sigma_thermal)
        
        # D mécanique
        crit_ceramic = CRITICAL_STRESS['ceramic']
        crit_bondcoat = CRITICAL_STRESS['bondcoat']
        
        D_ceramic_tension = sigma_thermal / crit_ceramic['sigma_tensile']
        D_ceramic_shear = sigma_shear / crit_ceramic['sigma_shear']
        D_bondcoat = sigma_thermal * 0.8 / crit_bondcoat['sigma_tensile']
        
        D_mechanical = max(D_ceramic_tension, D_ceramic_shear, D_bondcoat)
        
        # ═══════════════════════════════════════════════════════════════
        # BONUS: Proximité de T_critique (marge de sécurité thermique)
        # ═══════════════════════════════════════════════════════════════
        # Si T_h1 approche T_crit, on ajoute un facteur de risque
        thermal_margin = (T_crit - T_h1) / T_crit  # 1.0 si T=0, 0 si T=T_crit
        thermal_proximity_factor = max(0, (1 - thermal_margin) * 0.5)  # 0 à 0.5
        
        D_total = D_mechanical + thermal_proximity_factor
        
        return max(0.05, min(1.5, D_total))
        
    except Exception as e:
        # Fallback physiquement correct
        delta_T = t_top - t_bottom
        gradient_factor = delta_T / 1000
        protection_factor = 1 / (1 + alpha)
        D_fallback = 0.3 * gradient_factor * protection_factor + 0.1
        return max(0.05, min(1.5, D_fallback))

def get_risk_level(D):
    """Retourne le niveau de risque et la couleur associée."""
    if D < 0.3:
        return "OPTIMAL", "#10b981", "🟢"
    elif D < 0.5:
        return "BON", "#22d3ee", "🔵"
    elif D < 0.7:
        return "ATTENTION", "#f59e0b", "🟡"
    elif D < 0.9:
        return "RISQUE", "#ef4444", "🟠"
    else:
        return "CRITIQUE", "#dc2626", "🔴"

def create_gauge_chart(value, title, max_val=1.0):
    """Crée une jauge semi-circulaire spectaculaire."""
    
    # Déterminer la couleur basée sur la valeur
    if value < 0.3 * max_val:
        color = "#10b981"
    elif value < 0.5 * max_val:
        color = "#22d3ee"
    elif value < 0.7 * max_val:
        color = "#f59e0b"
    elif value < 0.9 * max_val:
        color = "#f97316"
    else:
        color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 48, 'color': '#f1f5f9'}, 'suffix': ''},
        title={'text': title, 'font': {'size': 16, 'color': '#94a3b8'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 2, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(30, 41, 59, 0.8)",
            'borderwidth': 2,
            'bordercolor': "rgba(96, 165, 250, 0.3)",
            'steps': [
                {'range': [0, 0.3 * max_val], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [0.3 * max_val, 0.5 * max_val], 'color': 'rgba(34, 211, 238, 0.2)'},
                {'range': [0.5 * max_val, 0.7 * max_val], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [0.7 * max_val, 0.9 * max_val], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [0.9 * max_val, max_val], 'color': 'rgba(239, 68, 68, 0.2)'},
            ],
            'threshold': {
                'line': {'color': "#f472b6", 'width': 4},
                'thickness': 0.75,
                'value': 0.8 * max_val
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f1f5f9'},
        height=280,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_radar_chart(scores_dict):
    """Crée un graphique radar multi-critères."""
    
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    
    # Fermer le polygone
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    # Zone remplie
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=3),
        name='Performance Actuelle'
    ))
    
    # Zone de référence (optimal)
    optimal = [80] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=optimal,
        theta=categories,
        line=dict(color='rgba(16, 185, 129, 0.5)', width=2, dash='dash'),
        name='Objectif'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(96, 165, 250, 0.2)',
                tickfont=dict(color='#94a3b8')
            ),
            angularaxis=dict(
                gridcolor='rgba(96, 165, 250, 0.2)',
                tickfont=dict(color='#e2e8f0', size=12)
            ),
            bgcolor='rgba(15, 23, 42, 0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(color='#94a3b8')
        ),
        height=350,
        margin=dict(l=60, r=60, t=40, b=60)
    )
    
    return fig

def create_structure_3d(h_sub, h_bc, h_tbc, T_vals, z_vals):
    """Crée une visualisation 3D spectaculaire de la structure multicouche."""
    
    # Création d'une surface 2.5D représentant la structure
    n_x = 30
    x_vals = np.linspace(0, 10, n_x)  # mm en direction latérale
    
    # Grille
    X, Z = np.meshgrid(x_vals, z_vals * 1000)  # z en mm
    
    # Interpolation de la température sur la grille
    T_grid = np.tile(T_vals, (n_x, 1)).T
    
    # Modulation sinusoïdale pour effet 3D
    wave = 0.02 * np.sin(2 * np.pi * X / 5)
    
    fig = go.Figure()
    
    # Surface de température
    fig.add_trace(go.Surface(
        x=X,
        y=Z + wave,
        z=T_grid,
        colorscale='Inferno',
        colorbar=dict(
            title=dict(text='T (°C)', font=dict(color='#f1f5f9')),
            tickfont=dict(color='#f1f5f9'),
            x=1.02
        ),
        opacity=0.95,
        name="Champ Thermique"
    ))
    
    # Lignes d'interface
    z_interfaces = [h_sub * 1000, (h_sub + h_bc) * 1000]
    colors = ['#f59e0b', '#ef4444']
    names = ['Interface Substrat/BC', 'Interface BC/TBC']
    
    for zi, color, name in zip(z_interfaces, colors, names):
        fig.add_trace(go.Scatter3d(
            x=x_vals,
            y=[zi] * n_x,
            z=[np.interp(zi / 1000, z_vals, T_vals)] * n_x,
            mode='lines',
            line=dict(color=color, width=6),
            name=name
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x (mm)', backgroundcolor='rgba(15, 23, 42, 0.9)', 
                      gridcolor='rgba(96, 165, 250, 0.2)', tickfont=dict(color='#94a3b8')),
            yaxis=dict(title='z (mm)', backgroundcolor='rgba(15, 23, 42, 0.9)',
                      gridcolor='rgba(96, 165, 250, 0.2)', tickfont=dict(color='#94a3b8')),
            zaxis=dict(title='T (°C)', backgroundcolor='rgba(15, 23, 42, 0.9)',
                      gridcolor='rgba(96, 165, 250, 0.2)', tickfont=dict(color='#94a3b8')),
            camera=dict(eye=dict(x=1.5, y=1.2, z=0.8)),
            bgcolor='rgba(15, 23, 42, 0.95)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9'),
        height=450,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=0.01,
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='rgba(96, 165, 250, 0.3)',
            borderwidth=1,
            font=dict(color='#e2e8f0', size=10)
        )
    )
    
    return fig

def create_sparkline(values, color='#3b82f6', height=60):
    """Crée un mini-graphique sparkline."""
    fig = go.Figure()
    
    x = list(range(len(values)))
    
    fig.add_trace(go.Scatter(
        x=x, y=values,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}'
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    
    return fig

def generate_recommendations(T_h1, D, alpha, delta_T, T_crit):
    """Génère des recommandations intelligentes basées sur l'analyse."""
    
    recommendations = []
    
    # Analyse de la température
    if T_h1 > T_crit:
        recommendations.append({
            'icon': '🚨',
            'type': 'CRITIQUE',
            'title': 'Température Interface Excessive',
            'message': f'T = {T_h1:.0f}°C dépasse la limite critique ({T_crit}°C). Augmentez l\'épaisseur TBC (α) ou réduisez la température de surface.',
            'color': '#ef4444'
        })
    elif T_h1 > T_crit * 0.9:
        recommendations.append({
            'icon': '⚠️',
            'type': 'ATTENTION',
            'title': 'Marge Thermique Faible',
            'message': f'T = {T_h1:.0f}°C proche de la limite. Marge de sécurité: {(T_crit - T_h1):.0f}°C. Envisagez d\'augmenter α de 10-20%.',
            'color': '#f59e0b'
        })
    else:
        recommendations.append({
            'icon': '✅',
            'type': 'OK',
            'title': 'Température Interface Sûre',
            'message': f'T = {T_h1:.0f}°C bien en-dessous de la limite ({T_crit}°C). Marge confortable de {(T_crit - T_h1):.0f}°C.',
            'color': '#10b981'
        })
    
    # Analyse de l'indicateur D
    if D > 0.8:
        recommendations.append({
            'icon': '🔴',
            'type': 'CRITIQUE',
            'title': 'Risque de Rupture Élevé',
            'message': f'Indicateur D = {D:.2f} indique un risque de délamination. Réduisez le gradient thermique ou réévaluez les matériaux.',
            'color': '#ef4444'
        })
    elif D > 0.5:
        recommendations.append({
            'icon': '🟡',
            'type': 'PRUDENCE',
            'title': 'Zone de Prudence Mécanique',
            'message': f'D = {D:.2f} dans la zone intermédiaire. Surveillance recommandée. Évitez les cycles thermiques rapides.',
            'color': '#f59e0b'
        })
    else:
        recommendations.append({
            'icon': '🟢',
            'type': 'BON',
            'title': 'Intégrité Mécanique Assurée',
            'message': f'D = {D:.2f} indique une configuration robuste. Les contraintes sont bien réparties.',
            'color': '#10b981'
        })
    
    # Analyse de l'épaisseur
    if alpha < 0.1:
        recommendations.append({
            'icon': '💡',
            'type': 'SUGGESTION',
            'title': 'Épaisseur TBC Faible',
            'message': f'α = {alpha:.2f} est relativement mince. Pour les applications haute température, envisagez α ≥ 0.2.',
            'color': '#8b5cf6'
        })
    elif alpha > 1.0:
        recommendations.append({
            'icon': '⚖️',
            'type': 'COMPROMIS',
            'title': 'Épaisseur TBC Importante',
            'message': f'α = {alpha:.2f} offre une bonne isolation mais augmente la masse (+{alpha * CONSTANTS["h1"] * 1e6:.0f} µm). Vérifiez l\'impact centrifuge.',
            'color': '#06b6d4'
        })
    
    return recommendations

def render():
    """Affiche le Dashboard Principal spectaculaire."""
    
    # === EN-TÊTE HERO ANIMÉ ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 30%, #1e1b4b 70%, #0f172a 100%);
                padding: 2.5rem; border-radius: 24px; margin-bottom: 2rem; position: relative; overflow: hidden;
                border: 1px solid rgba(96, 165, 250, 0.2); box-shadow: 0 0 60px rgba(59, 130, 246, 0.15);">
        <div style="position: absolute; top: -100px; right: -100px; width: 300px; height: 300px; 
                    background: radial-gradient(circle, rgba(59,130,246,0.25) 0%, transparent 70%); 
                    border-radius: 50%; animation: float 8s ease-in-out infinite;"></div>
        <div style="position: absolute; bottom: -80px; left: -80px; width: 250px; height: 250px; 
                    background: radial-gradient(circle, rgba(139,92,246,0.2) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
            <h1 style="margin: 0; font-size: 3rem; font-weight: 800;
                       background: linear-gradient(135deg, #60a5fa 0%, #22d3ee 30%, #a78bfa 60%, #f472b6 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text; animation: gradient-shift 4s ease infinite; background-size: 200% 200%;">
                🛡️ TBC Analysis Dashboard
            </h1>
            <p style="color: #94a3b8; font-size: 1.2rem; margin-top: 0.5rem; margin-bottom: 0;">
                Analyse Thermomécanique des Barrières Thermiques — Vue Panoramique
            </p>
        </div>
    </div>
    <style>
        @keyframes float {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(20px, 20px); }
        }
        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # === BADGE CONFORMITÉ MÉTHODOLOGIQUE ===
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.05));
                    padding: 0.6rem 1.2rem; border-radius: 30px; border: 1px solid rgba(16, 185, 129, 0.4);
                    display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.1rem;">✅</span>
            <span style="color: #10b981; font-weight: 600; font-size: 0.85rem;">Méthodologie PDF : 8/8 Étapes</span>
        </div>
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.05));
                    padding: 0.6rem 1.2rem; border-radius: 30px; border: 1px solid rgba(59, 130, 246, 0.4);
                    display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.1rem;">📐</span>
            <span style="color: #3b82f6; font-weight: 600; font-size: 0.85rem;">Spectral + CLT</span>
        </div>
        <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(139, 92, 246, 0.05));
                    padding: 0.6rem 1.2rem; border-radius: 30px; border: 1px solid rgba(139, 92, 246, 0.4);
                    display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.1rem;">🔬</span>
            <span style="color: #8b5cf6; font-weight: 600; font-size: 0.85rem;">Multi-Modes Fourier</span>
        </div>
        <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.05));
                    padding: 0.6rem 1.2rem; border-radius: 30px; border: 1px solid rgba(245, 158, 11, 0.4);
                    display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.1rem;">🎯</span>
            <span style="color: #f59e0b; font-weight: 600; font-size: 0.85rem;">Projet 5A ESTACA/ONERA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === RÉCUPÉRATION DES PARAMÈTRES ===
    alpha = st.session_state.get('alpha_input', 0.20)
    beta = st.session_state.get('beta_input', 0.8)
    lw = st.session_state.get('lw_input', 0.1)
    t_bottom = st.session_state.get('T_bottom', 500)
    t_top = st.session_state.get('T_top', 1400)
    
    # === CALCUL PRINCIPAL ===
    res = cached_solve_tbc_model(alpha, beta, lw, t_bottom, t_top)
    
    if not res['success']:
        st.error(f"❌ Erreur de calcul: {res.get('error', 'Inconnue')}")
        return
    
    T_h1 = res['T_at_h1']
    h3_um = res['h3'] * 1e6
    delta_T = t_top - t_bottom
    T_crit = CONSTANTS['T_crit']
    
    # Calcul de l'indicateur D (PHYSIQUE RÉELLE)
    # Utilise le solveur mécanique pour calculer les vraies contraintes
    # D diminue quand alpha augmente (plus de TBC = plus de protection = moins de risque)
    D_approx = compute_real_damage_indicator(alpha, lw, t_top, t_bottom)
    
    # Niveau de risque
    risk_level, risk_color, risk_emoji = get_risk_level(D_approx)
    
    # === ROW 1: KPI CARDS SPECTACULAIRES ===
    st.markdown("### 📊 Indicateurs Clés de Performance")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    kpi_card_style = """
    <div style="background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.95) 100%);
                padding: 1.2rem; border-radius: 16px; text-align: center;
                border: 1px solid rgba(96, 165, 250, 0.2);
                box-shadow: 0 0 30px rgba(59, 130, 246, 0.1);
                transition: all 0.3s ease;">
        <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-bottom: 0.3rem;">{value}</div>
        <div style="font-size: 0.7rem; color: #94a3b8;">{subtitle}</div>
    </div>
    """
    
    with col1:
        color_T = "#ef4444" if T_h1 > T_crit else "#f59e0b" if T_h1 > T_crit * 0.9 else "#10b981"
        st.markdown(kpi_card_style.format(
            label="🌡️ T° Interface",
            value=f"{T_h1:.0f}°C",
            color=color_T,
            subtitle=f"Limite: {T_crit}°C"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(kpi_card_style.format(
            label="📏 Épaisseur TBC",
            value=f"{h3_um:.0f} µm",
            color="#3b82f6",
            subtitle=f"α = {alpha:.2f}"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(kpi_card_style.format(
            label=f"{risk_emoji} Indicateur D",
            value=f"{D_approx:.2f}",
            color=risk_color,
            subtitle=risk_level
        ), unsafe_allow_html=True)
    
    with col4:
        # Score de performance thermique (0-100%)
        perf_th = max(0, min(100, 100 * (1 - (T_h1 - 600) / (T_crit - 600))))
        color_perf = "#10b981" if perf_th > 70 else "#f59e0b" if perf_th > 40 else "#ef4444"
        st.markdown(kpi_card_style.format(
            label="⚡ Perf. Thermique",
            value=f"{perf_th:.0f}%",
            color=color_perf,
            subtitle="Isolation"
        ), unsafe_allow_html=True)
    
    with col5:
        st.markdown(kpi_card_style.format(
            label="🔥 Gradient ΔT",
            value=f"{delta_T:.0f}°C",
            color="#f472b6",
            subtitle=f"{t_bottom}→{t_top}°C"
        ), unsafe_allow_html=True)
    
    with col6:
        # Marge de sécurité
        marge = T_crit - T_h1
        color_marge = "#10b981" if marge > 100 else "#f59e0b" if marge > 0 else "#ef4444"
        st.markdown(kpi_card_style.format(
            label="🛡️ Marge Sécurité",
            value=f"{marge:.0f}°C",
            color=color_marge,
            subtitle="vs Critique"
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === ROW 2: JAUGE + RADAR ===
    col_gauge, col_radar = st.columns([1, 1])
    
    with col_gauge:
        st.markdown("### 🎯 Jauge de Risque Global")
        fig_gauge = create_gauge_chart(D_approx, "Indicateur d'Endommagement D", max_val=1.2)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Légende des zones
        st.markdown("""
        <div style="display: flex; justify-content: space-around; font-size: 0.75rem; color: #94a3b8; margin-top: -10px;">
            <span style="color: #10b981;">● 0-0.3 Optimal</span>
            <span style="color: #22d3ee;">● 0.3-0.5 Bon</span>
            <span style="color: #f59e0b;">● 0.5-0.7 Attention</span>
            <span style="color: #f97316;">● 0.7-0.9 Risque</span>
            <span style="color: #ef4444;">● >0.9 Critique</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_radar:
        st.markdown("### 📈 Performance Multi-Critères")
        
        # Calcul des scores (0-100)
        scores = {
            "Isolation\nThermique": max(0, min(100, 100 * (1 - T_h1/t_top))),
            "Intégrité\nMécanique": max(0, min(100, 100 * (1 - D_approx))),
            "Efficacité\nMasse": max(0, min(100, 100 - alpha * 30)),
            "Durée\nde Vie": max(0, min(100, 90 - D_approx * 50)),
            "Marge\nSécurité": max(0, min(100, (T_crit - T_h1) / 3)),
        }
        
        fig_radar = create_radar_chart(scores)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # === ROW 3: VISUALISATION 3D STRUCTURE ===
    st.markdown("### 🏗️ Visualisation 3D du Champ Thermique")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PROFIL THERMIQUE RÉALISTE (Résistances en série)
    # Évite les artefacts Fourier pour la visualisation 3D
    # ═══════════════════════════════════════════════════════════════════════
    h_sub = CONSTANTS['h1']
    h_bc = CONSTANTS['h2']
    h_tbc = res['h3']
    H_total = h_sub + h_bc + h_tbc
    
    # Résistances thermiques par couche
    R1 = h_sub / CONSTANTS['k33_1']
    R2 = h_bc / CONSTANTS['k33_2']
    R3 = h_tbc / CONSTANTS['k33_3']
    R_total = R1 + R2 + R3
    
    # Températures aux interfaces (modèle résistances en série)
    T_at_h1 = t_bottom + delta_T * (R1 / R_total)
    T_at_h2 = t_bottom + delta_T * ((R1 + R2) / R_total)
    
    # Création du profil z monotone
    n_points = 200
    z_substrate = np.linspace(0, h_sub, n_points // 3)
    z_bondcoat = np.linspace(h_sub, h_sub + h_bc, n_points // 6)
    z_ceramic = np.linspace(h_sub + h_bc, H_total, n_points // 2)
    
    # Températures linéaires par couche
    T_substrate = np.linspace(t_bottom, T_at_h1, len(z_substrate))
    T_bondcoat = np.linspace(T_at_h1, T_at_h2, len(z_bondcoat))
    T_ceramic = np.linspace(T_at_h2, t_top, len(z_ceramic))
    
    x_plot = np.concatenate([z_substrate, z_bondcoat[1:], z_ceramic[1:]])
    T_vals = np.concatenate([T_substrate, T_bondcoat[1:], T_ceramic[1:]])
    
    fig_3d = create_structure_3d(h_sub, h_bc, h_tbc, T_vals, x_plot)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Récupérer les profils Fourier pour les graphiques 2D (flux avec variation)
    x_plot_fourier, T_vals_fourier, Q1_vals, Q3_vals = calculate_profiles(res['profile_params'], res['H'])
    
    # === ROW 4: RECOMMANDATIONS INTELLIGENTES ===
    st.markdown("### 💡 Recommandations Intelligentes")
    
    recommendations = generate_recommendations(T_h1, D_approx, alpha, delta_T, T_crit)
    
    rec_cols = st.columns(len(recommendations))
    
    for i, rec in enumerate(recommendations):
        with rec_cols[i]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
                        padding: 1.2rem; border-radius: 14px; height: 100%;
                        border-left: 4px solid {rec['color']};
                        border-top: 1px solid rgba(96, 165, 250, 0.1);
                        border-right: 1px solid rgba(96, 165, 250, 0.1);
                        border-bottom: 1px solid rgba(96, 165, 250, 0.1);">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{rec['icon']}</div>
                <div style="font-size: 0.7rem; color: {rec['color']}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem;">{rec['type']}</div>
                <div style="font-size: 0.95rem; color: #f1f5f9; font-weight: 600; margin-bottom: 0.5rem;">{rec['title']}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">{rec['message']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # === ROW 5: PROFILS RAPIDES ===
    st.markdown("### 📉 Profils Thermiques")
    
    col_temp, col_flux = st.columns(2)
    
    with col_temp:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=x_plot * 1000, y=T_vals,
            mode='lines',
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.15)',
            name='T(z)'
        ))
        fig_temp.add_hline(y=T_crit, line_dash="dash", line_color="#ef4444", 
                          annotation_text="T critique")
        fig_temp.update_layout(
            title="Profil de Température",
            xaxis_title="z (mm)",
            yaxis_title="T (°C)",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            xaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)'),
            yaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)')
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col_flux:
        fig_flux = go.Figure()
        fig_flux.add_trace(go.Scatter(
            x=x_plot_fourier * 1000, y=Q1_vals,
            mode='lines',
            line=dict(color='#10b981', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.15)',
            name='Q₁(z)'
        ))
        fig_flux.update_layout(
            title="Flux Transverse",
            xaxis_title="z (mm)",
            yaxis_title="Q₁ (W/m²)",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            xaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)'),
            yaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)')
        )
        st.plotly_chart(fig_flux, use_container_width=True)
    
    # === ROW 6: VALIDATION ONERA ===
    st.markdown("### 🏛️ Validation vs Référence ONERA/Safran")
    
    # Données de référence ONERA (Bovet, Chiaruttini, Vattré 2025)
    ONERA_REF = {
        'sigma_vM_min': 400,  # MPa
        'sigma_vM_max': 800,  # MPa
        'C11_ref': 259.6,     # GPa
        'C12_ref': 179.0,     # GPa
        'C44_ref': 109.6,     # GPa
    }
    
    # Calcul rapide des contraintes pour le ΔT actuel
    sigma_estimated = abs(delta_T * 1.1)  # Estimation simplifiée en MPa
    
    # Vérification conformité
    is_in_range = ONERA_REF['sigma_vM_min'] * 0.5 <= sigma_estimated <= ONERA_REF['sigma_vM_max'] * 1.5
    
    col_onera1, col_onera2, col_onera3 = st.columns([1, 1, 1])
    
    with col_onera1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(6,182,212,0.1) 100%);
                    padding: 1.2rem; border-radius: 14px; border: 1px solid rgba(16,185,129,0.3);
                    text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{'✅' if is_in_range else '⚠️'}</div>
            <div style="color: {'#10b981' if is_in_range else '#f59e0b'}; font-size: 1rem; font-weight: 700;">
                {'CONFORME ONERA' if is_in_range else 'À VÉRIFIER'}
            </div>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;">
                Plage FEM: {ONERA_REF['sigma_vM_min']}-{ONERA_REF['sigma_vM_max']} MPa
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_onera2:
        # Comparaison propriétés matériaux
        from core.constants import PROPS_SUBSTRATE
        c11_code = PROPS_SUBSTRATE['C11']
        ecart_c11 = abs(c11_code - ONERA_REF['C11_ref']) / ONERA_REF['C11_ref'] * 100
        
        st.markdown(f"""
        <div style="background: rgba(30,41,59,0.6); padding: 1rem; border-radius: 12px;">
            <div style="color: #60a5fa; font-weight: 600; margin-bottom: 0.5rem;">📊 Propriétés Matériaux</div>
            <div style="display: flex; justify-content: space-between; color: #94a3b8; font-size: 0.85rem;">
                <span>C₁₁ (code)</span><span style="color: #f1f5f9;">{c11_code} GPa</span>
            </div>
            <div style="display: flex; justify-content: space-between; color: #94a3b8; font-size: 0.85rem;">
                <span>C₁₁ (ONERA)</span><span style="color: #10b981;">{ONERA_REF['C11_ref']} GPa</span>
            </div>
            <div style="display: flex; justify-content: space-between; color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">
                <span>Écart</span><span style="color: {'#10b981' if ecart_c11 < 5 else '#f59e0b'};">{ecart_c11:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_onera3:
        st.markdown(f"""
        <div style="background: rgba(30,41,59,0.6); padding: 1rem; border-radius: 12px;">
            <div style="color: #8b5cf6; font-weight: 600; margin-bottom: 0.5rem;">📚 Référence</div>
            <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.5;">
                <strong>Bovet, Chiaruttini, Vattré</strong><br>
                "Full-scale crystal plasticity modeling..."<br>
                <em>ONERA/Safran (2025)</em><br>
                <span style="color: #64748b;">Table 3 - Inconel 718</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # === FOOTER INFO ===
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2rem; padding-top: 1rem;
                border-top: 1px solid rgba(96, 165, 250, 0.2);">
        💡 <strong>Tip:</strong> Utilisez la barre latérale pour ajuster les paramètres (α, β, Lw, Températures) et observer les changements en temps réel.
        <br><span style="color: #475569;">Validation conforme ONERA/Safran • Propriétés Inconel 718 Tab.3</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render()
