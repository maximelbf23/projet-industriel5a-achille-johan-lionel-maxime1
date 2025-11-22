import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. CONFIGURATION & STYLE (CSS "Premium")
# ==========================================
st.set_page_config(
    page_title="TBC Analysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS pour raffiner l'interface
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h3 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    /* Style pour la Tâche 3 : Warning Box */
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        color: #856404;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DÉFINITION DES PARAMÈTRES PHYSIQUES (FIXES)
# ==========================================
CONSTANTS = {
    'h1': 0.0005,      # Superalliage (m)
    'h2': 0.00001,     # Liaison (m)
    'k33_1': 20.0,     # Superalliage (W/mK)
    'k33_2': 8.0,      # Liaison (W/mK)
    'k33_3': 1.5,      # Céramique (W/mK) - Base
    'T_bottom': 500,   # T(x3=0)
    'T_top': 1400,     # T(x3=H)
    'T_crit': 1100,    # Température critique
    'Securite_pct': 0.8
}

# --- TÂCHE 1 : CONSTANTES POUR L'IMPACT ---
IMPACT_PARAMS = {
    'rho_ceram': 6050,      # Masse volumique (kg/m^3)
    'cost_per_vol': 25000,  # Coût estimé par volume (€/m^3)
    'co2_per_kg': 15.5      # Empreinte carbone (kgCO2/kg matière)
}

T_secu = CONSTANTS['T_crit'] * CONSTANTS['Securite_pct']

# ==========================================
# 3. MOTEUR DE CALCUL (CŒUR DU MODÈLE)
# ==========================================
def solve_tbc_model(alpha, beta_ceramique, lw_val):
    """
    Résout le système d'équations thermiques pour une configuration donnée.
    """
    try:
        # 1. Géométrie
        h1, h2 = CONSTANTS['h1'], CONSTANTS['h2']
        x_i1 = h1
        x_i2 = h1 + h2
        h3 = alpha * h1
        H = h1 + h2 + h3
        
        # 2. Propriétés Thermiques
        k_eta_1 = CONSTANTS['k33_1'] # Isotrope
        k_eta_2 = CONSTANTS['k33_2'] # Isotrope
        
        # Sécurité div par zéro
        beta_safe = max(beta_ceramique, 1e-6)
        k_eta_3 = CONSTANTS['k33_3'] / beta_safe # Anisotropie
        
        # Mode de Fourier
        delta_eta = np.pi / lw_val
        
        # Valeurs propres (Lambdas)
        lambdas = []
        pairs = [(k_eta_1, CONSTANTS['k33_1']), (k_eta_2, CONSTANTS['k33_2']), (k_eta_3, CONSTANTS['k33_3'])]
        
        for k_e, k_n in pairs:
            lambdas.append(delta_eta * np.sqrt(k_e / k_n))
        
        l1, l2, l3 = lambdas
        
        # Coefficients C (Flux)
        C1 = CONSTANTS['k33_1'] * l1
        C2 = CONSTANTS['k33_2'] * l2
        C3 = CONSTANTS['k33_3'] * l3

        # 3. Assemblage Matrice Système (6x6)
        M = np.zeros((6, 6))
        F = np.zeros(6)

        # T(0) = T_bottom
        M[0,0]=1; M[0,1]=1; F[0]=CONSTANTS['T_bottom']
        
        # Interface 1
        M[1,0]=np.exp(l1*x_i1); M[1,1]=np.exp(-l1*x_i1); M[1,2]=-np.exp(l2*x_i1); M[1,3]=-np.exp(-l2*x_i1)
        M[2,0]=C1*np.exp(l1*x_i1); M[2,1]=-C1*np.exp(-l1*x_i1); M[2,2]=-C2*np.exp(l2*x_i1); M[2,3]=C2*np.exp(-l2*x_i1)
        
        # Interface 2
        M[3,2]=np.exp(l2*x_i2); M[3,3]=np.exp(-l2*x_i2); M[3,4]=-np.exp(l3*x_i2); M[3,5]=-np.exp(-l3*x_i2)
        M[4,2]=C2*np.exp(l2*x_i2); M[4,3]=-C2*np.exp(-l2*x_i2); M[4,4]=-C3*np.exp(l3*x_i2); M[4,5]=C3*np.exp(-l3*x_i2)
        
        # T(H) = T_top
        M[5,4]=np.exp(l3*H); M[5,5]=np.exp(-l3*H); F[5]=CONSTANTS['T_top']

        # Résolution
        coeffs = np.linalg.solve(M, F)
        A1, B1, A2, B2, A3, B3 = coeffs

        # 4. Fonctions de Calcul
        def get_profiles(x_arr):
            condlist = [x_arr <= x_i1, (x_arr > x_i1) & (x_arr <= x_i2), x_arr > x_i2]
            
            T_funcs = [
                lambda x: A1*np.exp(l1*x) + B1*np.exp(-l1*x),
                lambda x: A2*np.exp(l2*x) + B2*np.exp(-l2*x),
                lambda x: A3*np.exp(l3*x) + B3*np.exp(-l3*x)
            ]
            T = np.piecewise(x_arr, condlist, T_funcs)
            
            Q1_funcs = [
                lambda x: -k_eta_1 * delta_eta * T_funcs[0](x),
                lambda x: -k_eta_2 * delta_eta * T_funcs[1](x),
                lambda x: -k_eta_3 * delta_eta * T_funcs[2](x)
            ]
            Q1 = np.piecewise(x_arr, condlist, Q1_funcs)
            
            Q3_funcs = [
                lambda x: -(C1*A1*np.exp(l1*x) - C1*B1*np.exp(-l1*x)),
                lambda x: -(C2*A2*np.exp(l2*x) - C2*B2*np.exp(-l2*x)),
                lambda x: -(C3*A3*np.exp(l3*x) - C3*B3*np.exp(-l3*x))
            ]
            Q3 = np.piecewise(x_arr, condlist, Q3_funcs)
            
            return T, Q1, Q3

        T_h1 = A1*np.exp(l1*x_i1) + B1*np.exp(-l1*x_i1)
        T_h2 = A2*np.exp(l2*x_i2) + B2*np.exp(-l2*x_i2)
        
        # Calcul Sauts Flux
        Q1_h1_minus = -k_eta_1 * delta_eta * T_h1
        Q1_h1_plus  = -k_eta_2 * delta_eta * T_h1
        dQ1_h1 = Q1_h1_plus - Q1_h1_minus
        
        Q1_h2_minus = -k_eta_2 * delta_eta * T_h2
        Q1_h2_plus  = -k_eta_3 * delta_eta * T_h2
        dQ1_h2 = Q1_h2_plus - Q1_h2_minus

        return {
            'success': True,
            'H': H, 'h3': h3, 'k_eta_3': k_eta_3,
            'T_at_h1': T_h1, 'T_at_h2': T_h2,
            'dQ1_h1': dQ1_h1, 'dQ1_h2': dQ1_h2,
            'get_profiles': get_profiles,
            'params': {'alpha': alpha, 'beta': beta_ceramique, 'lw': lw_val}
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

# ==========================================
# 4. INTERFACE SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚙️ Paramètres")
    st.markdown("---")
    
    st.subheader("1. Paramètres Globaux")
    beta_in = st.slider(
        "Anisotropie Céramique (β)", 
        min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="Ratio k33 / k_eta. Si < 1, la conduction latérale est favorisée."
    )
    
    lw_in = st.number_input(
        "Longueur d'Onde $L_w$ (m)", 
        min_value=0.01, max_value=5.0, value=0.1, step=0.01,
        help="Taille caractéristique du défaut."
    )
    
    st.markdown("---")
    st.caption(f"**Limites de Température**\n\n- T Critique: {CONSTANTS['T_crit']}°C\n- T Sécurité: {T_secu:.0f}°C")

# ==========================================
# 5. APPLICATION PRINCIPALE
# ==========================================
st.title("🛡️ Analyse Thermique de Revêtement (TBC)")

# --- MISE À JOUR STRUCTURE : 3 ONGLETS ---
tab_single, tab_multi, tab_3d = st.tabs([
    "🔎 Analyse Détaillée & Impacts", 
    "📚 Étude Paramétrique (2D)",
    "🧊 Cartographie 3D (Alpha/Beta)"
])

# ------------------------------------------
# ONGLET 1 : CAS UNIQUE + TÂCHE 1 (IMPACT) + TÂCHE 3 (WARNING)
# ------------------------------------------
with tab_single:
    col_input, col_kpi = st.columns([1, 3])
    
    with col_input:
        st.markdown("#### Configuration")
        alpha_single = st.slider("Épaisseur Céramique (α)", 0.05, 2.0, 0.20, 0.05, key="a_single")
        res = solve_tbc_model(alpha_single, beta_in, lw_in)
    
    if res['success']:
        # Conversion dimensions
        h1_mic = CONSTANTS['h1'] * 1e6
        h2_mic = CONSTANTS['h2'] * 1e6
        h3_mic = res['h3'] * 1e6
        
        with col_kpi:
            # --- A. VISUALISATION COUPE TRANSVERSALE ---
            fig_geo = go.Figure()
            fig_geo.add_trace(go.Bar(
                y=[''], x=[h1_mic], orientation='h', name='Superalliage',
                marker=dict(color='#95a5a6', line=dict(width=1)),
                text=f"<b>Alliage</b><br>{h1_mic:.0f} µm", textposition='auto', hoverinfo='x+name'
            ))
            fig_geo.add_trace(go.Bar(
                y=[''], x=[h2_mic], orientation='h', name='Liaison',
                marker=dict(color='#d35400', line=dict(width=1)),
                hoverinfo='x+name'
            ))
            fig_geo.add_trace(go.Bar(
                y=[''], x=[h3_mic], orientation='h', name='Céramique',
                marker=dict(color='#d6eaf8', line=dict(width=1)),
                text=f"<b>Céramique (TBC)</b><br>{h3_mic:.0f} µm", textposition='auto', hoverinfo='x+name'
            ))
            fig_geo.update_layout(barmode='stack', height=80, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False))
            st.plotly_chart(fig_geo, use_container_width=True)
        
        # --- B. KPI et STATUT ---
        T_h1 = res['T_at_h1']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Épaisseur TBC ($h_3$)", f"{h3_mic:.0f} µm")
        c2.metric("Conductivité Trans.", f"{res['k_eta_3']:.2f} W/mK")
        delta_T = T_h1 - CONSTANTS['T_crit']
        c3.metric("T° Interface Alliage", f"{T_h1:.2f} °C", delta=f"{-delta_T:.2f} vs Limite")
        
        with c4:
            if T_h1 > CONSTANTS['T_crit']: st.error(f"🚨 CRITIQUE")
            elif T_h1 <= T_secu: st.success("✅ SÉCURISÉ")
            else: st.warning("⚠️ SURVEILLANCE")

        st.divider()

        # --- TÂCHE 3 : NOTE DE SYNTHÈSE / WARNING ---
        st.markdown("""
        <div class="warning-box">
            ⚠️ NOTE DE SYNTHÈSE :<br>
            Attention, l'optimisation thermique (baisse de T°) implique souvent une augmentation de l'épaisseur (Alpha).
            Cela induit des contraintes mécaniques (masse/stress centrifuge) non calculées ici.
        </div>
        """, unsafe_allow_html=True)

        # --- C. GRAPHIQUES DÉTAILLÉS ---
        col_graphes, col_impact = st.columns([2, 1])
        
        with col_graphes:
            x_plot = np.linspace(0, res['H'], 500)
            T_vals, Q1_vals, Q3_vals = res['get_profiles'](x_plot)
            x_mm = x_plot * 1000
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                subplot_titles=("🌡️ Profil de Température", "⬇️ Flux Normal (Q3)", "↔️ Flux Transverse (Q1)"))
            
            # Zone Critique
            fig.add_hrect(y0=CONSTANTS['T_crit'], y1=max(np.max(T_vals), 1500), fillcolor="red", opacity=0.05, row=1, col=1)
            
            # Courbes
            fig.add_trace(go.Scatter(x=x_mm, y=T_vals, name="Température", line=dict(color='#2980b9', width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_mm, y=Q3_vals, name="Flux Normal", line=dict(color='#c0392b', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_mm, y=Q1_vals, name="Flux Transverse", line=dict(color='#27ae60', width=2), fill='tozeroy'), row=3, col=1)
            
            # Interfaces
            interfaces = [CONSTANTS['h1']*1000, (CONSTANTS['h1']+CONSTANTS['h2'])*1000]
            for xi in interfaces:
                for r in [1,2,3]: fig.add_vline(x=xi, line_dash="dot", line_color="gray", row=r, col=1)

            fig.update_layout(height=600, showlegend=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        # --- TÂCHE 1 : TABLEAU DE QUANTIFICATION ---
        with col_impact:
            st.markdown("#### 📊 Impact Global")
            st.markdown("Comparaison **Nominal** (actuel) vs **Catastrophe** (α=2.0).")
            
            # Calcul des impacts
            alpha_cata = 2.0
            h3_nom = res['h3']
            h3_cata = alpha_cata * CONSTANTS['h1']
            
            def get_metrics(h_val):
                vol = h_val * 1.0 # Base 1m²
                mass = vol * IMPACT_PARAMS['rho_ceram']
                cost = vol * IMPACT_PARAMS['cost_per_vol']
                co2 = mass * IMPACT_PARAMS['co2_per_kg']
                return mass, cost, co2

            m1, c1, co1 = get_metrics(h3_nom)
            m2, c2, co2 = get_metrics(h3_cata)
            
            df_imp = pd.DataFrame({
                "Critère": ["Surcharge (kg/m²)", "Coût (€/m²)", "Carbone (kgCO2)"],
                "Nominal": [f"{m1:.2f}", f"{c1:.0f}", f"{co1:.1f}"],
                "Catastrophe": [f"{m2:.2f}", f"{c2:.0f}", f"{co2:.1f}"],
                "Delta": [f"+{m2-m1:.2f}", f"+{c2-c1:.0f}", f"+{co2-co1:.1f}"]
            })
            st.table(df_imp)

# ------------------------------------------
# ONGLET 2 : ÉTUDE PARAMÉTRIQUE (Liste ou Range)
# ------------------------------------------
with tab_multi:
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
        with c_start: a_start = st.number_input("Début", 0.05, format="%.2f")
        with c_end: a_end = st.number_input("Fin", 0.50, format="%.2f")
        with c_step: a_step = st.number_input("Pas", 0.05, format="%.2f")
        if a_start < a_end: alphas_to_test = np.arange(a_start, a_end + a_step/100, a_step)

    if st.button(f"🚀 Lancer Simulation ({len(alphas_to_test)} cas)", type="primary"):
        results_list = []
        for a in alphas_to_test:
            r = solve_tbc_model(a, beta_in, lw_in)
            if r['success']:
                r['alpha'] = a
                results_list.append(r)
            
        if results_list:
            df_trends = pd.DataFrame([{
                'alpha': r['alpha'], 'T_h1': r['T_at_h1'], 'dQ1_h1': r['dQ1_h1']
            } for r in results_list])

            col_t, col_q = st.columns(2)
            with col_t:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=df_trends['alpha'], y=df_trends['T_h1'], mode='lines+markers', name='T(Alliage)'))
                fig_trend.add_hline(y=CONSTANTS['T_crit'], line_color='red', line_dash='dash')
                fig_trend.update_layout(title="Température vs Alpha", xaxis_title="Alpha", yaxis_title="T (°C)")
                st.plotly_chart(fig_trend, use_container_width=True)
            with col_q:
                fig_flux = go.Figure()
                fig_flux.add_trace(go.Scatter(x=df_trends['alpha'], y=df_trends['dQ1_h1'], mode='lines+markers', line_color='orange', name='Saut Q1'))
                fig_flux.update_layout(title="Saut Flux Transverse vs Alpha", xaxis_title="Alpha", yaxis_title="ΔQ1")
                st.plotly_chart(fig_flux, use_container_width=True)

# ------------------------------------------
# ONGLET 3 : CARTOGRAPHIE 3D (TÂCHE 2 - MISE À JOUR)
# ------------------------------------------
with tab_3d:
    st.header("🧊 Cartographie 3D : Preuve d'Hétérogénéité")
    st.markdown("""
    Cette visualisation permet de comparer la réponse **continue** (Température) et **discrète/hétérogène** (Saut de Flux).
    Le saut de flux démontre que la matière n'est pas un milieu continu classique.
    """)
    
    col_3d_params, col_3d_viz = st.columns([1, 3])
    
    with col_3d_params:
        st.subheader("Paramètres 3D")
        res_grid = st.slider("Résolution (points/axe)", 5, 20, 10)
        
        # --- NOUEAU SELECTEUR POUR LE SUPERVISEUR ---
        plot_type = st.radio(
            "Variable Physique (Axe Z) :",
            ["Température T(h1)", "Saut de Flux ΔQ1(h1)"],
            help="Sélectionnez 'Saut de Flux' pour visualiser la réponse discrète du matériau."
        )
        
        if st.button("🔄 Générer Surface 3D"):
            alpha_vals = np.linspace(0.1, 2.0, res_grid)
            beta_vals = np.linspace(0.1, 2.0, res_grid)
            z_data = []
            
            # Boucle de calcul
            for b in beta_vals:
                z_row = []
                for a in alpha_vals:
                    r = solve_tbc_model(a, b, lw_in)
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
            
            # Stockage des résultats
            st.session_state['z_3d'] = z_data
            st.session_state['x_3d'] = alpha_vals
            st.session_state['y_3d'] = beta_vals
            st.session_state['plot_type'] = plot_type # On retient ce qu'on a tracé

    with col_3d_viz:
        if 'z_3d' in st.session_state:
            # Adaptation Titres et Couleurs selon la variable
            current_type = st.session_state.get('plot_type', "Température")
            if "Flux" in current_type:
                z_title = "Saut ΔQ1 (W/m²)"
                colors = "Plasma" # Couleur différente pour le flux
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
                title=main_title,
                scene=dict(
                    xaxis_title='Alpha (Épaisseur)', 
                    yaxis_title='Beta (Anisotropie)', 
                    zaxis_title=z_title
                ),
                height=650, margin=dict(l=0, r=0, b=0, t=30)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
            if "Flux" in current_type:
                st.info("ℹ️ **Note :** Les variations brusques de cette surface illustrent la réponse discrète du matériau aux changements de géométrie et d'anisotropie.")
        else:
            st.info("👈 Sélectionnez la variable et cliquez sur le bouton pour générer.")