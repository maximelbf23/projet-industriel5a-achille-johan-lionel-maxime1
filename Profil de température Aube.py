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

# Style CSS pour raffiner l'interface (Cartes, Tableaux)
st.markdown("""
<style>
    /* Style des métriques KPI */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Titres de section */
    h3 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    /* Ajustement du padding haut */
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
T_secu = CONSTANTS['T_crit'] * CONSTANTS['Securite_pct']

# ==========================================
# 3. MOTEUR DE CALCUL (CŒUR DU MODÈLE)
# ==========================================
def solve_tbc_model(alpha, beta_ceramique, lw_val):
    """
    Résout le système d'équations thermiques pour une configuration donnée.
    """
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
    # k_eta et k33 correspondants pour les 3 couches
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

    try:
        # T(0) = T_bottom
        M[0,0]=1; M[0,1]=1; F[0]=CONSTANTS['T_bottom']
        
        # Interface 1 (Continuité T et Flux Normal)
        M[1,0]=np.exp(l1*x_i1); M[1,1]=np.exp(-l1*x_i1); M[1,2]=-np.exp(l2*x_i1); M[1,3]=-np.exp(-l2*x_i1)
        M[2,0]=C1*np.exp(l1*x_i1); M[2,1]=-C1*np.exp(-l1*x_i1); M[2,2]=-C2*np.exp(l2*x_i1); M[2,3]=C2*np.exp(-l2*x_i1)
        
        # Interface 2 (Continuité T et Flux Normal)
        M[3,2]=np.exp(l2*x_i2); M[3,3]=np.exp(-l2*x_i2); M[3,4]=-np.exp(l3*x_i2); M[3,5]=-np.exp(-l3*x_i2)
        M[4,2]=C2*np.exp(l2*x_i2); M[4,3]=-C2*np.exp(-l2*x_i2); M[4,4]=-C3*np.exp(l3*x_i2); M[4,5]=C3*np.exp(-l3*x_i2)
        
        # T(H) = T_top
        M[5,4]=np.exp(l3*H); M[5,5]=np.exp(-l3*H); F[5]=CONSTANTS['T_top']

        # Résolution
        coeffs = np.linalg.solve(M, F)
        A1, B1, A2, B2, A3, B3 = coeffs

        # 4. Fonctions de Calcul des Profils (Vectorisées pour rapidité)
        def get_profiles(x_arr):
            # Masques pour les zones
            condlist = [x_arr <= x_i1, (x_arr > x_i1) & (x_arr <= x_i2), x_arr > x_i2]
            
            # Température T(x)
            T_funcs = [
                lambda x: A1*np.exp(l1*x) + B1*np.exp(-l1*x),
                lambda x: A2*np.exp(l2*x) + B2*np.exp(-l2*x),
                lambda x: A3*np.exp(l3*x) + B3*np.exp(-l3*x)
            ]
            T = np.piecewise(x_arr, condlist, T_funcs)
            
            # Flux Transverse Q1(x) = -k_eta * delta_eta * T(x)
            # Note: Q1 dépend de T(x) calculé localement
            Q1_funcs = [
                lambda x: -k_eta_1 * delta_eta * T_funcs[0](x),
                lambda x: -k_eta_2 * delta_eta * T_funcs[1](x),
                lambda x: -k_eta_3 * delta_eta * T_funcs[2](x)
            ]
            Q1 = np.piecewise(x_arr, condlist, Q1_funcs)
            
            # Flux Normal Q3(x)
            Q3_funcs = [
                lambda x: -(C1*A1*np.exp(l1*x) - C1*B1*np.exp(-l1*x)),
                lambda x: -(C2*A2*np.exp(l2*x) - C2*B2*np.exp(-l2*x)),
                lambda x: -(C3*A3*np.exp(l3*x) - C3*B3*np.exp(-l3*x))
            ]
            Q3 = np.piecewise(x_arr, condlist, Q3_funcs)
            
            return T, Q1, Q3

        # 5. Calcul des Valeurs Clés (Scalaires)
        T_h1 = A1*np.exp(l1*x_i1) + B1*np.exp(-l1*x_i1)
        T_h2 = A2*np.exp(l2*x_i2) + B2*np.exp(-l2*x_i2)
        
        # Sauts de Flux Transverse (Discontinuités)
        # Q1 à gauche et à droite de h1
        Q1_h1_minus = -k_eta_1 * delta_eta * T_h1
        Q1_h1_plus  = -k_eta_2 * delta_eta * T_h1
        dQ1_h1 = Q1_h1_plus - Q1_h1_minus
        
        # Q1 à gauche et à droite de h2
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
    st.info("Ces paramètres affectent les deux modes de simulation.")
    
    beta_in = st.slider(
        "Anisotropie Céramique (β)", 
        min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="Ratio k33 / k_eta. Si < 1, la conduction latérale est favorisée."
    )
    
    lw_in = st.number_input(
        "Longueur d'Onde $L_w$ (m)", 
        min_value=0.01, max_value=5.0, value=0.1, step=0.01,
        help="Taille caractéristique du défaut ou de la perturbation."
    )
    
    st.markdown("---")
    st.caption(f"**Limites de Température**\n\n- T Critique: {CONSTANTS['T_crit']}°C\n- T Sécurité: {T_secu:.0f}°C")

# ==========================================
# 5. APPLICATION PRINCIPALE
# ==========================================
st.title("🛡️ Analyse Thermique de Revêtement (TBC)")

# Onglets pour séparer les analyses
tab_single, tab_multi = st.tabs(["🔎 Analyse Détaillée (Cas Unique)", "📚 Étude Paramétrique (Comparaison)"])

# ------------------------------------------
# ONGLET 1 : CAS UNIQUE
# ------------------------------------------
with tab_single:
    col_input, col_kpi = st.columns([1, 3])
    
    with col_input:
        st.markdown("#### Configuration")
        alpha_single = st.slider("Épaisseur Céramique (α)", 0.05, 2.0, 0.20, 0.05, key="a_single")
        
        # Exécution simulation
        res = solve_tbc_model(alpha_single, beta_in, lw_in)
    
    if res['success']:
        # Conversion dimensions pour affichage
        h1_mic = CONSTANTS['h1'] * 1e6
        h2_mic = CONSTANTS['h2'] * 1e6
        h3_mic = res['h3'] * 1e6
        
        with col_kpi:
            # --- A. VISUALISATION COUPE TRANSVERSALE ---
            fig_geo = go.Figure()
            # Alliage
            fig_geo.add_trace(go.Bar(
                y=[''], x=[h1_mic], orientation='h', name='Superalliage',
                marker=dict(color='#95a5a6', line=dict(width=1)),
                text=f"<b>Alliage</b><br>{h1_mic:.0f} µm", textposition='auto', hoverinfo='x+name'
            ))
            # Liaison
            fig_geo.add_trace(go.Bar(
                y=[''], x=[h2_mic], orientation='h', name='Liaison',
                marker=dict(color='#d35400', line=dict(width=1)),
                hoverinfo='x+name' # Trop petit pour texte auto
            ))
            # Céramique
            fig_geo.add_trace(go.Bar(
                y=[''], x=[h3_mic], orientation='h', name='Céramique',
                marker=dict(color='#d6eaf8', line=dict(width=1)),
                text=f"<b>Céramique (TBC)</b><br>{h3_mic:.0f} µm", textposition='auto', hoverinfo='x+name'
            ))
            
            fig_geo.update_layout(
                barmode='stack', height=90, margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False, xaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.markdown("##### Coupe Transversale (Échelle réelle des épaisseurs)")
            st.plotly_chart(fig_geo, use_container_width=True)
        
        # --- B. KPI et STATUT ---
        T_h1 = res['T_at_h1']
        k3_eta = res['k_eta_3']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Épaisseur TBC ($h_3$)", f"{h3_mic:.0f} µm")
        c2.metric("Conductivité Trans. ($k_{\eta,3}$)", f"{k3_eta:.2f} W/mK")
        
        delta_T = T_h1 - CONSTANTS['T_crit']
        c3.metric("T° Interface Alliage", f"{T_h1:.2f} °C", delta=f"{-delta_T:.2f} vs Limite", delta_color="normal")
        
        with c4:
            if T_h1 > CONSTANTS['T_crit']:
                st.error(f"🚨 CRITIQUE (> {CONSTANTS['T_crit']}°C)")
            elif T_h1 <= T_secu:
                st.success("✅ SÉCURISÉ")
            else:
                st.warning("⚠️ SURVEILLANCE")

        st.divider()

        # --- C. GRAPHIQUES DÉTAILLÉS ---
        col_graphes, col_data = st.columns([2, 1])
        
        with col_graphes:
            x_plot = np.linspace(0, res['H'], 500)
            T_vals, Q1_vals, Q3_vals = res['get_profiles'](x_plot)
            x_mm = x_plot * 1000
            
            # Sous-graphiques empilés
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                subplot_titles=("🌡️ Profil de Température", "⬇️ Flux Normal (Q3)", "↔️ Flux Transverse (Q1)")
            )
            
            # Zone Critique
            fig.add_hrect(y0=CONSTANTS['T_crit'], y1=max(np.max(T_vals), 1500), fillcolor="red", opacity=0.05, row=1, col=1)
            
            # Courbes
            fig.add_trace(go.Scatter(x=x_mm, y=T_vals, name="Température", line=dict(color='#2980b9', width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_mm, y=Q3_vals, name="Flux Normal", line=dict(color='#c0392b', width=2)), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_mm, y=Q1_vals, name="Flux Transverse", line=dict(color='#27ae60', width=2), fill='tozeroy'), row=3, col=1)
            
            # Lignes Interfaces
            interfaces = [CONSTANTS['h1']*1000, (CONSTANTS['h1']+CONSTANTS['h2'])*1000]
            for xi in interfaces:
                for r in [1,2,3]: fig.add_vline(x=xi, line_dash="dot", line_color="gray", row=r, col=1)

            fig.update_layout(height=700, showlegend=False, hovermode="x unified")
            fig.update_xaxes(title_text="Position (mm)", row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_data:
            st.markdown("#### 📊 Résultats Numériques")
            st.markdown("Valeurs exactes aux interfaces calculées par le modèle.")
            
            results_df = pd.DataFrame({
                "Paramètre": [
                    "Température T(h1)", "Température T(h2)", 
                    "Saut Flux Q1 (h1)", "Saut Flux Q1 (h2)"
                ],
                "Valeur": [
                    f"{res['T_at_h1']:.2f} °C", f"{res['T_at_h2']:.2f} °C",
                    f"{res['dQ1_h1']:.2e} W/m²", f"{res['dQ1_h2']:.2e} W/m²"
                ],
                "Note": ["Interface Alliage", "Interface Liaison", "Discontinuité", "Discontinuité"]
            })
            st.table(results_df)
            
            st.info("""
            **Interprétation :**
            * **Saut de Flux ($ \Delta Q_1 $)** : Indique l'intensité de la redistribution latérale de la chaleur à cause de la différence de conductivité entre les couches.
            * Une valeur élevée signale un fort gradient local.
            """)

# ------------------------------------------
# ONGLET 2 : ÉTUDE PARAMÉTRIQUE (Complété)
# ------------------------------------------
with tab_multi:
    st.markdown("### 🔢 Sélection des Valeurs d'Alpha")
    st.info("Définissez les scénarios d'épaisseur relative ($h_3/h_1$) à simuler.")
    
    # --- 1. SÉLECTION DU MODE DE SAISIE ---
    mode_input = st.radio(
        "Mode de sélection :",
        ["🎯 Sélection Manuelle (Liste)", "📏 Intervalle Automatique (Range)"],
        horizontal=True
    )
    
    alphas_to_test = []
    
    # --- A. MODE MANUEL ---
    if mode_input == "🎯 Sélection Manuelle (Liste)":
        col_sel1, col_sel2 = st.columns([3, 1])
        with col_sel1:
            # Options de base
            options_base = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0, 1.5, 2.0]
            alphas_selected = st.multiselect(
                "Choisissez dans la liste (ou ajoutez à droite) :", 
                options=options_base, 
                default=[0.04, 0.10, 0.25]
            )
        with col_sel2:
            # Ajout valeur personnalisée
            custom_alpha = st.number_input("Ajouter une valeur spécifique :", min_value=0.00, max_value=10.0, step=0.01, value=0.0)
            if custom_alpha > 0 and custom_alpha not in alphas_selected:
                st.toast(f"Valeur {custom_alpha} ajoutée temporairement.")
                alphas_selected.append(custom_alpha)
        
        alphas_to_test = sorted(list(set(alphas_selected))) # Tri et dédoublonnage
        
    # --- B. MODE INTERVALLE ---
    else: 
        c_start, c_end, c_step = st.columns(3)
        with c_start:
            a_start = st.number_input("Début (Min)", value=0.05, min_value=0.01, format="%.2f")
        with c_end:
            a_end = st.number_input("Fin (Max)", value=0.50, min_value=0.01, format="%.2f")
        with c_step:
            a_step = st.number_input("Pas (Step)", value=0.05, min_value=0.01, format="%.2f")
        
        if a_start < a_end and a_step > 0:
            # Génération avec numpy (incluant la borne fin si possible)
            alphas_to_test = np.arange(a_start, a_end + a_step/100, a_step)
            st.success(f"✅ {len(alphas_to_test)} configurations générées : {np.round(alphas_to_test, 3)}")
        else:
            st.error("Paramètres d'intervalle invalides (Début doit être < Fin).")

    st.divider()
    
    # --- 2. EXÉCUTION DU CALCUL ---
    if st.button(f"🚀 Lancer la simulation ({len(alphas_to_test)} cas)", type="primary", disabled=(len(alphas_to_test)==0)):
        
        results_list = []
        progress_bar = st.progress(0)
        
        # Boucle de calcul
        for i, a in enumerate(alphas_to_test):
            r = solve_tbc_model(a, beta_in, lw_in)
            if r['success']:
                r['alpha'] = a
                results_list.append(r)
            # Mise à jour barre de progression
            progress_bar.progress((i + 1) / len(alphas_to_test))
            
        if results_list:
            # Création DataFrame pour analyse
            df_trends = pd.DataFrame([{
                'alpha': r['alpha'], 
                'T_h1': r['T_at_h1'], 
                'T_h2': r['T_at_h2'],
                'dQ1_h1': r['dQ1_h1'],
                'dQ1_h2': r['dQ1_h2']
            } for r in results_list])

            # --- 3. AFFICHAGE DES TENDANCES (GLOBAL) ---
            st.subheader("1. Tendances Globales")
            
            c_t1, c_t2 = st.columns(2)
            with c_t1:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=df_trends['alpha'], y=df_trends['T_h1'], mode='lines+markers', name='T(Alliage)'))
                fig_trend.add_trace(go.Scatter(x=df_trends['alpha'], y=df_trends['T_h2'], mode='lines+markers', name='T(Liaison)', visible='legendonly'))
                fig_trend.add_hline(y=CONSTANTS['T_crit'], line_color='red', line_dash='dash', annotation_text="Critique")
                fig_trend.add_hline(y=T_secu, line_color='green', line_dash='dot', annotation_text="Sécurité")
                fig_trend.update_layout(title="Température Interface vs Alpha", height=400, xaxis_title="Alpha (α)", yaxis_title="Température (°C)")
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with c_t2:
                fig_flux = go.Figure()
                fig_flux.add_trace(go.Scatter(x=df_trends['alpha'], y=df_trends['dQ1_h1'], mode='lines+markers', name='Saut Q1 (Alliage)', line_color='orange'))
                fig_flux.add_trace(go.Scatter(x=df_trends['alpha'], y=df_trends['dQ1_h2'], mode='lines+markers', name='Saut Q1 (Liaison)', line_color='brown'))
                fig_flux.add_hline(y=0, line_color='black', line_width=1)
                fig_flux.update_layout(title="Saut de Flux Transverse (Discontinuité)", height=400, xaxis_title="Alpha (α)", yaxis_title="ΔQ1 (W/m²)")
                st.plotly_chart(fig_flux, use_container_width=True)
            
            # --- 4. AFFICHAGE DES PROFILS (SUPERPOSITION) ---
            st.subheader("2. Comparaison des Profils Spatiaux")
            st.caption("Visualisation de la distribution interne de la température et des flux.")
            
            fig_multi = make_subplots(rows=1, cols=3, subplot_titles=("Température T(x)", "Flux Normal Q3(x)", "Flux Transverse Q1(x)"))
            
            # Optimisation affichage : Si trop de courbes (>15), on en saute certaines pour la lisibilité
            step_display = max(1, len(results_list) // 15)
            
            for i, res in enumerate(results_list):
                if i % step_display == 0: # Filtre d'affichage
                    # Calcul profil
                    x_p = np.linspace(0, res['H'], 200) * 1000 # mm
                    T_p, Q1_p, Q3_p = res['get_profiles'](x_p/1000)
                    
                    # Couleur dégradée (Rouge -> Bleu selon alpha)
                    ratio = i / len(results_list)
                    color_val = f"rgba({int(255 * (1-ratio))}, {int(50 + 150*ratio)}, {int(255*ratio)}, 0.8)"
                    lbl = f"α={res['alpha']:.2f}"
                    
                    fig_multi.add_trace(go.Scatter(x=x_p, y=T_p, line=dict(color=color_val, width=1.5), name=lbl, legendgroup=lbl), 1, 1)
                    fig_multi.add_trace(go.Scatter(x=x_p, y=Q3_p, line=dict(color=color_val, width=1.5, dash='dot'), showlegend=False, legendgroup=lbl), 1, 2)
                    fig_multi.add_trace(go.Scatter(x=x_p, y=Q1_p, line=dict(color=color_val, width=1.5), showlegend=False, legendgroup=lbl), 1, 3)

            fig_multi.update_xaxes(title_text="Position x (mm)")
            fig_multi.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig_multi, use_container_width=True)
            
            # --- 5. EXPORT DONNÉES ---
            with st.expander("📥 Voir les données brutes"):
                st.dataframe(df_trends, use_container_width=True)