import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.calculation import solve_tbc_model, calculate_profiles
from core.constants import CONSTANTS, IMPACT_PARAMS

# --- Fonctions de calcul décorées pour la mise en cache ---
@st.cache_data
def cached_solve_tbc_model(alpha, beta, lw, t_bottom, t_top):
    """Wrapper pour mettre en cache les résultats de solve_tbc_model."""
    return solve_tbc_model(alpha, beta, lw, t_bottom, t_top)

@st.cache_data
def find_alpha_for_temp(target_temp, beta, lw, t_bottom, t_top, alpha_min=0.01, alpha_max=8.0, tol=1e-3, max_iter=30):
    """
    Trouve la valeur d'alpha qui résulte en une température cible donnée à l'interface h1,
    en utilisant la méthode de la bissection.
    """
    
    def get_temp_at_alpha(alpha):
        """Fonction objective : retourne la température pour un alpha donné."""
        res = cached_solve_tbc_model(alpha, beta, lw, t_bottom, t_top)
        if res['success']:
            return res['T_at_h1']
        return 1e9 # Retourne un nombre élevé en cas d'échec du calcul

    # La température diminue lorsque alpha augmente, donc f(alpha) est décroissante.
    # f(alpha) = T(alpha) - target_temp
    # Pour la bissection, nous avons besoin de f(alpha_min) > 0 et f(alpha_max) < 0.
    
    f_min = get_temp_at_alpha(alpha_min) - target_temp
    f_max = get_temp_at_alpha(alpha_max) - target_temp

    if np.sign(f_min) == np.sign(f_max):
        return {'success': False, 'message': "La T° cible est hors de l'intervalle de recherche. Essayez d'autres paramètres."}

    # Inversion si l'utilisateur a entré une plage incorrecte
    if f_min < f_max:
        alpha_min, alpha_max = alpha_max, alpha_min
        f_min, f_max = f_max, f_min

    for i in range(max_iter):
        alpha_mid = (alpha_min + alpha_max) / 2
        if alpha_mid == alpha_min or alpha_mid == alpha_max: # Précision atteinte
             return {'success': True, 'alpha': alpha_mid}

        f_mid = get_temp_at_alpha(alpha_mid) - target_temp

        if abs(f_mid) < tol:
            return {'success': True, 'alpha': alpha_mid}

        if np.sign(f_mid) == np.sign(f_min):
            alpha_min = alpha_mid
            f_min = f_mid
        else:
            alpha_max = alpha_mid
    
    return {'success': False, 'message': "Le solveur n'a pas convergé. La solution est peut-être hors de la plage de recherche."}


# ==========================================
# 1. CONFIGURATION & STYLE (CSS "Premium")
# ==========================================
# --- Palette de couleurs pour la cohérence ---
PALETTE = {
    'temp': '#3b82f6',       # Blue-500
    'flux_norm': '#ef4444',  # Red-500
    'flux_trans': '#10b981', # Emerald-500
    'accent': '#f59e0b',     # Amber-500
    'grid': '#e2e8f0',       # Slate-200
    'text': '#334155',       # Slate-700
    'bg': '#ffffff'          # White
}

st.set_page_config(
    page_title="TBC Analysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS pour raffiner l'interface
# Style CSS pour raffiner l'interface
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* H1 Title Styling */
    h1 {
        color: #1e3a8a; /* Dark Blue */
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* H2/H3 Headers */
    h2, h3 {
        color: #334155;
        font-weight: 600;
    }
    
    h3 {
        border-bottom: 2px solid #3b82f6; /* Blue-500 */
        padding-bottom: 8px;
        margin-top: 20px;
    }

    /* Metric Cards - Premium Look */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #3b82f6;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #0f172a;
        font-weight: 700;
    }

    /* Warning/Info Boxes */
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 16px;
        color: #92400e;
        font-weight: 500;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #64748b;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #3b82f6;
        border-top: 2px solid #3b82f6;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Custom Divider */
    hr {
        margin: 2em 0;
        border: 0;
        border-top: 1px solid #e2e8f0;
    }

    .block-container { padding-top: 2rem; padding-bottom: 4rem; }
</style>
""", unsafe_allow_html=True)

# Recalculate T_secu from imported constants
T_secu = CONSTANTS['T_crit'] * CONSTANTS['Securite_pct']

# ==========================================
# 2. INTERFACE SIDEBAR (MODIFIÉE)
# ==========================================
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.markdown("---")
    
    st.subheader("1. Paramètres Globaux")
    
    # --- MODIFICATION : Alpha est maintenant ici ---
    alpha_in = st.slider(
        "Épaisseur Céramique (α)", 
        min_value=0.0, max_value=3.0, value=0.20, step=0.05,
        help="Définit l'épaisseur relative de la couche TBC ($h_3 = \alpha \cdot h_1$)"
    )
    
    beta_in = st.slider(
        "Anisotropie Céramique (β)", 
        min_value=0.0, max_value=2.0, value=0.8, step=0.1,
        help="Ratio k33 / k_eta. Si < 1, la conduction latérale est favorisée."
    )
    
    lw_in = st.number_input(
        "Longueur d'Onde $L_w$ (m)", 
        min_value=0.01, max_value=5.0, value=0.1, step=0.01,
        help="Modélise la taille d'une variation de température latérale (un 'point chaud'). Un petit Lw simule un défaut intense et localisé, tandis qu'un grand Lw représente une variation thermique graduelle."
    )
    
    st.markdown("---")
    
    st.subheader("2. Conditions aux Limites")
    
    # Valeurs par défaut extraites des constantes
    t_bottom_default = CONSTANTS['T_bottom']
    t_top_default = CONSTANTS['T_top']

    # La session_state est automatiquement créée par les clés des widgets
    t_bottom_in = st.number_input("Température Base (°C)", key="T_bottom", value=t_bottom_default, step=10)
    t_top_in = st.number_input("Température Surface (°C)", key="T_top", value=t_top_default, step=10)

    def reset_temperatures():
        """Callback pour réinitialiser les températures."""
        st.session_state.T_bottom = t_bottom_default
        st.session_state.T_top = t_top_default

    st.button("Réinitialiser T°", on_click=reset_temperatures, help="Restaure les températures de base et de surface par défaut.")
    
    st.markdown("---")

    st.subheader("3. Scénario Catastrophe")
    t_bottom_catastrophe_in = st.number_input(
        "Température Base Catastrophe (°C)",
        value=t_bottom_default,
        step=10,
        key="t_bottom_cata",
        help="Définit le T_bottom pour le scénario catastrophe."
    )
    t_top_catastrophe_in = st.number_input(
        "Température Surface Catastrophe (°C)",
        value=t_top_default + 100,
        step=10,
        key="t_top_cata",
        help="Définit un T_top élevé pour lequel l'épaisseur 'alpha' sera calculée afin de maintenir l'interface à T_crit."
    )

    st.markdown("---")
    st.caption(f"**Limites de Température**\n\n- T Critique: {CONSTANTS['T_crit']}°C\n- T Sécurité: {T_secu:.0f}°C")

def display_detailed_analysis_tab(alpha_in, beta_in, lw_in, t_bottom, t_top):
    """Affiche l'onglet d'analyse détaillée pour un cas unique."""
    res = cached_solve_tbc_model(alpha_in, beta_in, lw_in, t_bottom, t_top)
    
    if not res['success']:
        st.error(f"Erreur lors du calcul : {res.get('error', 'Erreur inconnue')}")
        return

    # Conversion dimensions
    h1_mic = CONSTANTS['h1'] * 1e6
    h3_mic = res['h3'] * 1e6
    
    # --- A. VISUALISATION COUPE TRANSVERSALE & KPI ---
    col_visu, col_kpi_val = st.columns([1, 3])
    
    with col_visu:
            # Petit graph de la coupe
        fig_geo = go.Figure()
        fig_geo.add_trace(go.Bar(y=[''], x=[h1_mic], orientation='h', name='Alliage', marker=dict(color='#95a5a6')))
        fig_geo.add_trace(go.Bar(y=[''], x=[CONSTANTS['h2'] * 1e6], orientation='h', name='Liaison', marker=dict(color='#d35400')))
        fig_geo.add_trace(go.Bar(y=[''], x=[h3_mic], orientation='h', name='TBC', marker=dict(color='#d6eaf8')))
        fig_geo.update_layout(barmode='stack', height=100, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False))
        st.plotly_chart(fig_geo, use_container_width=True)
        st.caption("Coupe (Échelle réelle)")

    with col_kpi_val:
        # KPI et STATUT
        T_h1 = res['T_at_h1']
        delta_T = T_h1 - CONSTANTS['T_crit']

        if T_h1 > CONSTANTS['T_crit']: status_icon = "🚨"
        elif T_h1 <= T_secu: status_icon = "✅"
        else: status_icon = "⚠️"

        c1, c2, c3 = st.columns(3)
        c1.metric("📏 Épaisseur TBC ($h_3$)", f"{h3_mic:.0f} µm")
        c2.metric("⚡ Conductivité Trans.", f"{res['k_eta_3']:.2f} W/mK")
        c3.metric(f"{status_icon} T° Interface Alliage", f"{T_h1:.2f} °C", delta=f"{-delta_T:.2f} vs Limite")

    st.divider()


    # --- TÂCHE 3 : NOTE DE SYNTHÈSE / WARNING ---
    st.markdown("""
    <div class="warning-box">
        ⚠️ NOTE DE SYNTHÈSE :<br>
        Attention, l'optimisation thermique (baisse de T°) implique souvent une augmentation de l'épaisseur (Alpha).
        Cela induit des contraintes mécaniques (masse/stress centrifuge) non calculées ici.
    </div>
    """, unsafe_allow_html=True)

    # --- TÂCHE 1 : TABLEAU DE QUANTIFICATION (Full Width) ---
    st.markdown("#### 📊 Impact Global (Approche Inverse)")
    st.caption("Comparaison des épaisseurs requises pour maintenir la T° critique (1100°C) à l'interface.")

    # La température cible est toujours T_crit.
    target_temp = CONSTANTS['T_crit']

    # 1. Calcul Alpha Nominal (Optimisé)
    nom_res = find_alpha_for_temp(
        target_temp=target_temp,
        beta=beta_in,
        lw=lw_in,
        t_bottom=t_bottom,
        t_top=t_top
    )

    # 2. Calcul Alpha Catastrophe (Optimisé)
    cata_res = find_alpha_for_temp(
        target_temp=target_temp,
        beta=beta_in,
        lw=lw_in,
        t_bottom=t_bottom_catastrophe_in,
        t_top=t_top_catastrophe_in
    )

    if nom_res['success'] and cata_res['success']:
        alpha_nom = nom_res['alpha']
        alpha_cata = cata_res['alpha']
        
        # Calcul des impacts
        def get_metrics(alpha_val):
            h3_val = alpha_val * CONSTANTS['h1']
            blade_surface = 2 * IMPACT_PARAMS['blade_height'] * IMPACT_PARAMS['blade_chord']
            vol = h3_val * blade_surface
            mass = vol * IMPACT_PARAMS['rho_ceram']
            cost = vol * IMPACT_PARAMS['cost_per_vol']
            co2 = mass * IMPACT_PARAMS['co2_per_kg']
            return h3_val, blade_surface, vol, mass, cost, co2

        # 1. Nominal Optimisé
        h3_n, s_n, v_n, m_n, c_n, co_n = get_metrics(alpha_nom)
        # 2. Catastrophe Optimisé
        h3_c, s_c, v_c, m_c, c_c, co_c = get_metrics(alpha_cata)
        # 3. Simulé (Manuel)
        h3_s, s_s, v_s, m_s, c_s, co_s = get_metrics(alpha_in)
        
        df_imp = pd.DataFrame({
            "Critère": ["Alpha (α)", "Épaisseur (µm)", "Surcharge (kg/aube)", "Coût (€/aube)", "Carbone (kgCO2/aube)"],
            "Nominal (Calculé)": [f"{alpha_nom:.2f}", f"{h3_n*1e6:.0f}", f"{m_n:.3f}", f"{c_n:.2f}", f"{co_n:.2f}"],
            "Catastrophe (Calculé)": [f"{alpha_cata:.2f}", f"{h3_c*1e6:.0f}", f"{m_c:.3f}", f"{c_c:.2f}", f"{co_c:.2f}"],
            "Simulé (Manuel)": [f"{alpha_in:.2f}", f"{h3_s*1e6:.0f}", f"{m_s:.3f}", f"{c_s:.2f}", f"{co_s:.2f}"]
        })
        st.dataframe(df_imp, hide_index=True, use_container_width=True)
        
        st.info(f"""
        **Analyse :**
        - **Nominal (Calculé)** : Épaisseur min. pour T_top={t_top}°C.
        - **Catastrophe (Calculé)** : Épaisseur min. pour T_top={t_top_catastrophe_in}°C.
        - **Simulé (Manuel)** : Valeurs actuelles avec votre Alpha={alpha_in:.2f}.
        """)

    else:
        st.warning("Calcul impossible pour l'un des scénarios (hors limites).")

    st.divider()

    # --- C. GRAPHIQUES DÉTAILLÉS (Full Width) ---
    x_plot, T_vals, Q1_vals, Q3_vals = calculate_profiles(res['profile_params'], res['H'])
    x_mm = x_plot * 1000
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("🌡️ Profil de Température", "⬇️ Flux Normal (Q3)", "↔️ Flux Transverse (Q1)"))
    
    # --- AMÉLIORATION : Zones Matériaux et Lignes de Température ---
    h1_mm = CONSTANTS['h1'] * 1000
    h2_mm = CONSTANTS['h2'] * 1000
    h3_mm = res['h3'] * 1000
    
    # 1. Lignes de température critiques (Traces explicites pour visibilité et légende)
    fig.add_trace(go.Scatter(
        x=[x_mm[0], x_mm[-1]], y=[CONSTANTS['T_crit'], CONSTANTS['T_crit']],
        mode='lines', name='T° Critique',
        line=dict(color='#ef4444', width=2, dash='dash'),
        hoverinfo='name+y',
        legendgroup="limits", legendgrouptitle_text="Limites"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[x_mm[0], x_mm[-1]], y=[T_secu, T_secu],
        mode='lines', name='T° Sécurité',
        line=dict(color='#f97316', width=2, dash='dash'),
        hoverinfo='name+y',
        legendgroup="limits"
    ), row=1, col=1)

    # 2. Zones matériaux en fond
    zones = [
        {'x0': 0, 'x1': h1_mm, 'color': "#cbd5e1", 'label': "Alliage"}, # Slate-300
        {'x0': h1_mm, 'x1': h1_mm + h2_mm, 'color': "#fdba74", 'label': "Liaison"}, # Orange-300
        {'x0': h1_mm + h2_mm, 'x1': h1_mm + h2_mm + h3_mm, 'color': "#bae6fd", 'label': "Céramique"} # Sky-200
    ]
    
    # Ajout de traces invisibles pour créer une légende pour les zones
    for i, zone in enumerate(zones):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=zone['color'], symbol='square'),
            showlegend=True, name=zone['label'],
            legendgroup=f"zones",
            legendgrouptitle_text="Couches" if i == 0 else ""
        ), row=1, col=1)

    # Création des rectangles de couleur pour les zones
    for r in [1, 2, 3]:
        for zone in zones:
            fig.add_vrect(
                x0=zone['x0'], x1=zone['x1'], 
                fillcolor=zone['color'], opacity=0.3,
                layer="below", line_width=0, 
                row=r, col=1
            )
    
    # Courbes avec style premium
    fig.add_trace(go.Scatter(x=x_mm, y=T_vals, name="Température", line=dict(color=PALETTE['temp'], width=3), showlegend=True, legendgroup="curves", legendgrouptitle_text="Profils"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_mm, y=Q3_vals, name="Flux Normal", line=dict(color=PALETTE['flux_norm'], width=2), showlegend=True, legendgroup="curves"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_mm, y=Q1_vals, name="Flux Transverse", line=dict(color=PALETTE['flux_trans'], width=2), fill='tozeroy', showlegend=True, legendgroup="curves"), row=3, col=1)
    
    # Mettre à jour la plage de l'axe Y pour inclure les températures critiques
    if len(T_vals) > 0:
        min_y_range = min(T_vals.min(), T_secu) * 0.98
        max_y_range = max(T_vals.max(), CONSTANTS['T_crit']) * 1.02
        fig.update_yaxes(range=[min_y_range, max_y_range], row=1, col=1)

    # Layout Premium
    fig.update_layout(
        height=700, 
        showlegend=True, 
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12, color=PALETTE['text'])
    )
    
    # Grilles
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'], zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'], zeroline=False)

    st.plotly_chart(fig, use_container_width=True)
        


def display_parametric_study_tab(beta_in, lw_in, t_bottom, t_top):
    """Affiche l'onglet d'étude paramétrique pour alpha."""
    st.markdown("### 🔢 Sélection des Valeurs d'Alpha")

    mode_input = st.radio("Mode :", ["🎯 Liste Manuelle", "📏 Intervalle (Range)"], horizontal=True)
    alphas_to_test = []

    if mode_input == "🎯 Liste Manuelle":
        col_sel1, col_sel2 = st.columns([3, 1])
        with col_sel1:
            options_base = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0, 1.5, 2.0]
            alphas_selected = st.multiselect("Valeurs :", options=options_base, default=[0.04, 0.10, 0.25])
        alphas_to_test = sorted(alphas_selected)
    else: 
        c_start, c_end, c_step = st.columns(3)
        with c_start: a_start = st.number_input("Début", min_value=0.0, max_value=3.0, value=0.05, format="%.2f")
        with c_end: a_end = st.number_input("Fin", min_value=0.0, max_value=3.0, value=0.50, format="%.2f")
        with c_step: a_step = st.number_input("Pas", min_value=0.01, value=0.05, format="%.2f")
        if a_start < a_end: alphas_to_test = np.arange(a_start, a_end + a_step/100, a_step)

    if st.button(f"🚀 Lancer Simulation ({len(alphas_to_test)} cas)", type="primary"):
        results_list = []
        for a in alphas_to_test:
            # On utilise Beta de la sidebar, mais Alpha de la boucle
            r = cached_solve_tbc_model(a, beta_in, lw_in, t_bottom, t_top)
            if r['success']:
                r['alpha'] = a
                results_list.append(r)
        
        if results_list:
            # --- TÂCHE 2 : Données étendues pour tableau et graphiques ---
            processed_data = []
            for r in results_list:
                h3 = r['h3']
                blade_surface = 2 * IMPACT_PARAMS['blade_height'] * IMPACT_PARAMS['blade_chord']
                vol = h3 * blade_surface
                mass = vol * IMPACT_PARAMS['rho_ceram']
                cost = vol * IMPACT_PARAMS['cost_per_vol']
                co2 = mass * IMPACT_PARAMS['co2_per_kg']
                
                processed_data.append({
                    'alpha': r['alpha'],
                    'T_h1': r['T_at_h1'],
                    'dQ1_h1': r['dQ1_h1'],
                    'h3_um': h3 * 1e6,
                    'surface_m2': blade_surface,
                    'volume_m3': vol,
                    'mass_kg': mass,
                    'cost_eur': cost,
                    'co2_kg': co2,
                })
    
            df_results = pd.DataFrame(processed_data)
    
            # --- GRAPHIQUES ---
            col_t, col_q = st.columns(2)
            with col_t:
                fig_trend = px.line(df_results, x='alpha', y='T_h1', markers=True, 
                                    title="Température vs Alpha", labels={'alpha': 'Alpha', 'T_h1': 'T (°C)'})
                fig_trend.update_traces(line_color=PALETTE['temp'], line_width=3, marker=dict(size=8, color=PALETTE['temp'], line=dict(width=1, color='white')))
                fig_trend.add_hline(y=CONSTANTS['T_crit'], line_color='#ef4444', line_dash='dash', line_width=2,
                                    annotation_text="Limite Critique", annotation_position="top right",
                                    annotation_font_size=12, annotation_font_color="#ef4444")
                fig_trend.add_hline(y=T_secu, line_color='#f97316', line_dash='dash', line_width=2,
                                    annotation_text="Limite Sécurité", annotation_position="top right",
                                    annotation_font_size=12, annotation_font_color="#f97316")
                
                fig_trend.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(family="Inter, sans-serif", size=12, color=PALETTE['text']),
                    hovermode="x unified"
                )
                fig_trend.update_xaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'])
                fig_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'])
                
                st.plotly_chart(fig_trend, use_container_width=True)
    
            with col_q:
                fig_flux = px.line(df_results, x='alpha', y='dQ1_h1', markers=True,
                                   title="Saut Flux Transverse vs Alpha", labels={'alpha': 'Alpha', 'dQ1_h1': 'ΔQ1 (W/m²)'})
                fig_flux.update_traces(line_color=PALETTE['accent'], line_width=3, marker=dict(size=8, color=PALETTE['accent'], line=dict(width=1, color='white')))
                
                fig_flux.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(family="Inter, sans-serif", size=12, color=PALETTE['text']),
                    hovermode="x unified"
                )
                fig_flux.update_xaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'])
                fig_flux.update_yaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'])
                
                st.plotly_chart(fig_flux, use_container_width=True)
    
            # --- TABLEAU DE RÉSULTATS ---
            st.markdown("### 📊 Tableau de synthèse")
            
            df_display = df_results.rename(columns={
                'alpha': 'Alpha (α)',
                'T_h1': 'T° Interface (°C)',
                'dQ1_h1': 'Saut de Flux (W/m²)',
                'h3_um': 'Épaisseur TBC (µm)',
                'surface_m2': 'Surface (m²)',
                'volume_m3': 'Volume (m³)',
                'mass_kg': 'Surcharge (kg/aube)',
                'cost_eur': 'Coût (€/aube)',
                'co2_kg': 'Carbone (kgCO2/aube)'
            })[[
                'Alpha (α)', 'Épaisseur TBC (µm)', 'T° Interface (°C)', 
                'Surface (m²)', 'Volume (m³)', 'Surcharge (kg/aube)',
                'Coût (€/aube)', 'Carbone (kgCO2/aube)', 'Saut de Flux (W/m²)'
            ]]
    
            st.dataframe(df_display.style.format({
                "Alpha (α)": "{:.2f}",
                "Épaisseur TBC (µm)": "{:.0f}",
                "T° Interface (°C)": "{:.1f}",
                "Surface (m²)": "{:.4f}",
                "Volume (m³)": "{:.6f}",
                "Surcharge (kg/aube)": "{:.3f}",
                "Coût (€/aube)": "{:.2f}",
                "Carbone (kgCO2/aube)": "{:.2f}",
                "Saut de Flux (W/m²)": "{:.2e}",
            }), use_container_width=True)



def display_3d_mapping_tab(lw_in, t_bottom, t_top):



    """Affiche l'onglet de cartographie 3D."""



    st.header("🧊 Cartographie 3D : Preuve d'Hétérogénéité")



    st.markdown("""



    Cette visualisation permet de comparer la réponse **continue** (Température) et **discrète/hétérogène** (Saut de Flux).



    Le saut de flux démontre que la matière n'est pas un milieu continu classique.



    """, unsafe_allow_html=True)



    



    col_3d_params, col_3d_viz = st.columns([1, 3])



    



    with col_3d_params:
        st.subheader("Paramètres 3D")
        
        # --- AMÉLIORATION : Plages Configurables ---
        with st.expander("🛠️ Plages de Simulation", expanded=True):
            c_a1, c_a2 = st.columns(2)
            with c_a1: a_min = st.number_input("Alpha Min", 0.05, 5.0, 0.1, 0.05)
            with c_a2: a_max = st.number_input("Alpha Max", 0.05, 5.0, 2.0, 0.05)
            
            c_b1, c_b2 = st.columns(2)
            with c_b1: b_min = st.number_input("Beta Min", 0.05, 5.0, 0.1, 0.05)
            with c_b2: b_max = st.number_input("Beta Max", 0.05, 5.0, 2.0, 0.05)

        res_grid = st.slider("Résolution (points/axe)", 5, 20, 10)
        
        plot_type = st.radio(
            "Variable Physique (Axe Z) :",
            ["Température T(h1)", "Saut de Flux ΔQ1(h1)"],
            help="Sélectionnez 'Saut de Flux' pour visualiser la réponse discrète du matériau."
        )
        
        if st.button("🔄 Générer Surface 3D"):
            # Utilisation des plages configurées
            alpha_vals = np.linspace(a_min, a_max, res_grid)
            beta_vals = np.linspace(b_min, b_max, res_grid)
            z_data = []
            
            # Boucle de calcul 2D (Range Alpha x Range Beta)
            progress_bar = st.progress(0)
            for i, b in enumerate(beta_vals):
                z_row = []
                for a in alpha_vals:
                    r = cached_solve_tbc_model(a, b, lw_in, t_bottom, t_top)
                    if r['success']:
                        # Choix de la variable selon la sélection
                        if plot_type == "Température T(h1)":
                            val = r['T_at_h1']
                        else:
                            val = r['dQ1_h1'] # Le fameux saut discret
                    else:
                        val = np.nan
                    z_row.append(val)
                z_data.append(z_row)
                progress_bar.progress((i + 1) / res_grid)
            
            # Stockage des résultats
            st.session_state['z_3d'] = z_data
            st.session_state['x_3d'] = alpha_vals
            st.session_state['y_3d'] = beta_vals
            st.session_state['plot_type'] = plot_type
            progress_bar.empty()



    with col_3d_viz:

        if 'z_3d' in st.session_state:

            current_type = st.session_state.get('plot_type', "Température")

            if "Flux" in current_type:

                z_title = "Saut ΔQ1 (W/m²)"

                colors = "Plasma" 

                main_title = "Surface 3D : Discontinuité du Flux (Preuve Hétérogène)"

            else:

                z_title = "Température (°C)"

                colors = "RdBu_r"

                main_title = "Surface 3D : Température Interface"



            fig_3d = go.Figure(data=[go.Surface(

                z=st.session_state['z_3d'], 

                x=st.session_state['x_3d'], 

                y=st.session_state['y_3d'],

                colorscale=colors, 

                colorbar=dict(title=z_title)

            )])

            

            fig_3d.update_layout(
                title=dict(text=main_title, font=dict(size=20, color=PALETTE['text'])),
                scene=dict(
                    xaxis=dict(title='Alpha (Épaisseur)', backgroundcolor='white', gridcolor=PALETTE['grid'], showbackground=True),
                    yaxis=dict(title='Beta (Anisotropie)', backgroundcolor='white', gridcolor=PALETTE['grid'], showbackground=True),
                    zaxis=dict(title=z_title, backgroundcolor='white', gridcolor=PALETTE['grid'], showbackground=True),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                height=700, 
                margin=dict(l=0, r=0, b=0, t=50),
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif", color=PALETTE['text'])
            )

            st.plotly_chart(fig_3d, use_container_width=True)

            

            if "Flux" in current_type:
                st.info("ℹ️ **Note :** Les variations brusques sur cette surface illustrent la réponse discrète du matériau aux changements de géométrie et d'anisotropie.")
            else:
                st.info("""
                ℹ️ **Note Physique : Invariance selon Beta**
                Vous remarquerez que la température ne varie pas selon l'axe Beta (Anisotropie).
                C'est normal : le modèle 1D résout l'équation de la chaleur selon l'axe normal (x3). 
                La température dépend uniquement de la conductivité normale ($K_{33}$), alors que Beta modifie la conductivité transverse ($K_{11}$).
                """)

        else:

            st.info("👈 Sélectionnez la variable et cliquez sur le bouton pour générer.")





# ==========================================

# 3. APPLICATION PRINCIPALE

# ==========================================

st.title("🛡️ Analyse Thermique de Revêtement (TBC)")



# --- MISE À JOUR STRUCTURE : 3 ONGLETS ---

tab_single, tab_multi, tab_3d = st.tabs([

    "🔎 Analyse Détaillée & Impacts", 

    "📚 Étude Paramétrique (2D)",

    "🧊 Cartographie 3D (Alpha/Beta)"

])



with tab_single:



    display_detailed_analysis_tab(alpha_in, beta_in, lw_in, t_bottom_in, t_top_in)







with tab_multi:



    display_parametric_study_tab(beta_in, lw_in, t_bottom_in, t_top_in)







with tab_3d:



    display_3d_mapping_tab(lw_in, t_bottom_in, t_top_in)
