import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from core.calculation import solve_tbc_model_v2
from core.constants import CONSTANTS, IMPACT_PARAMS

# --- Fonctions de calcul décorées pour la mise en cache (Locales au module si besoin, ou importées si partagées) ---
# Note: Idéalement importées d'un module commun 'core/cache.py' pour éviter la duplication.
# Pour l'instant on redéfinit pour l'autonomie.
@st.cache_data
def cached_solve_tbc_model(alpha, beta, lw, t_bottom, t_top):
    """Wrapper pour mettre en cache les résultats de solve_tbc_model."""
    return solve_tbc_model_v2(alpha, beta, lw, t_bottom, t_top)

def render(beta_in, lw_in, t_bottom, t_top):
    """Affiche l'onglet d'étude paramétrique multi-physique."""
    
    # === EN-TÊTE HERO SPECTACULAIRE ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e1b4b 100%);
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; position: relative; overflow: hidden;
                border: 1px solid rgba(139, 92, 246, 0.2); box-shadow: 0 0 50px rgba(139, 92, 246, 0.15);">
        <div style="position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; 
                    background: radial-gradient(circle, rgba(139,92,246,0.25) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; 
                    background: radial-gradient(circle, rgba(6,182,212,0.2) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;
                       background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 50%, #c4b5fd 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;">
                📚 Étude Paramétrique Multi-Physique
            </h2>
            <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 1rem;">
                Variation Alpha/Beta • Analyse de sensibilité • Corrélations
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Palette locale (peut être importée d'un module de style global)
    PALETTE = {
        'temp': '#3b82f6',
        'flux_norm': '#ef4444',
        'flux_trans': '#10b981',
        'accent': '#f59e0b',
        'text': '#334155'
    }

    st.markdown("### 🔢 Configuration de l'Étude Multi-Physique")

    c_conf_1, c_conf_2 = st.columns(2)
    
    with c_conf_1:
        st.markdown("#### 1. Variation de l'Épaisseur (Alpha)")
        mode_input_a = st.radio("Mode Alpha :", ["🎯 Liste Manuelle", "📏 Intervalle"], horizontal=True, key="mode_a")
        alphas_to_test = []
        if mode_input_a == "🎯 Liste Manuelle":
            alphas_selected = st.multiselect("Valeurs Alpha :", options=[0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0, 1.5, 2.0], default=[0.04, 0.10, 0.25, 0.50])
            alphas_to_test = sorted(alphas_selected)
        else: 
            c_start, c_end, c_step = st.columns(3)
            with c_start: a_start = st.number_input("Début A", min_value=0.01, max_value=3.0, value=0.05)
            with c_end: a_end = st.number_input("Fin A", min_value=0.01, max_value=3.0, value=0.50)
            with c_step: a_step = st.number_input("Pas A", min_value=0.01, value=0.05)
            if a_start < a_end: alphas_to_test = np.arange(a_start, a_end + a_step/100, a_step)

    with c_conf_2:
        st.markdown("#### 2. Variation de l'Anisotropie (Beta)")
        mode_input_b = st.radio("Mode Beta :", ["🔒 Fixe (Valeur Sidebar)", "🎯 Liste Manuelle"], horizontal=True, key="mode_b")
        betas_to_test = []
        if mode_input_b == "🔒 Fixe (Valeur Sidebar)":
            betas_to_test = [beta_in]
            st.info(f"Beta fixé à **{beta_in}** (voir Sidebar)")
        else:
            betas_selected = st.multiselect("Valeurs Beta :", options=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], default=[1.0, 5.0, 10.0])
            betas_to_test = sorted(betas_selected)

    # Calcul du nombre total de cas
    total_cases = len(alphas_to_test) * len(betas_to_test)
    
    if st.button(f"🚀 Lancer Simulation 2D ({total_cases} cas)", type="primary"):
        results_list = []
        progress_bar = st.progress(0)
        idx = 0
        
        for a in alphas_to_test:
            for b in betas_to_test:
                # Simulation
                r = cached_solve_tbc_model(a, b, lw_in, t_bottom, t_top)
                if r['success']:
                    r['alpha'] = a
                    r['beta'] = b
                    results_list.append(r)
                idx += 1
                progress_bar.progress(min(idx / max(1, total_cases), 1.0))
        
        progress_bar.empty()
        
        if results_list:
            # Traitement des données
            processed_data = []
            for r in results_list:
                h3 = r['h3']
                blade_surface = 2 * IMPACT_PARAMS['blade_height'] * IMPACT_PARAMS['blade_chord']
                vol = h3 * blade_surface
                mass = vol * IMPACT_PARAMS['rho_ceram']
                cost = vol * IMPACT_PARAMS['cost_per_vol']
                
                processed_data.append({
                    'alpha': r['alpha'],
                    'beta': r['beta'],
                    'beta_str': f"β={r['beta']}", # Pour légende discrète
                    'T_h1': r['T_at_h1'],
                    'dQ1_h1': abs(r['dQ1_h1']), # Flux absolu pour clarté
                    'cost_eur': cost,
                    'mass_kg': mass,
                    'h3_um': h3 * 1e6
                })
    
            df_results = pd.DataFrame(processed_data)
    
            # --- GRAPHIQUES ---
            col_t, col_q = st.columns(2)
            
            with col_t:
                # Température vs Alpha (coloré par Beta)
                fig_trend = px.line(df_results, x='alpha', y='T_h1', color='beta_str', markers=True, 
                                    title="Température Interface vs Alpha (Impact Anisotropie)", 
                                    labels={'alpha': 'Alpha', 'T_h1': 'T (°C)', 'beta_str': 'Anisotropie'})
                
                fig_trend.add_hline(y=CONSTANTS['T_crit'], line_color='#ef4444', line_dash='dash', annotation_text="Critique")
                
                fig_trend.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9'), hovermode="x unified"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
    
            with col_q:
                # Flux vs Alpha (coloré par Beta)
                fig_flux = px.line(df_results, x='alpha', y='dQ1_h1', color='beta_str', markers=True,
                                   title="Flux Transverse vs Alpha (Impact Anisotropie)", 
                                   labels={'alpha': 'Alpha', 'dQ1_h1': 'Flux Trans. (W/m²)', 'beta_str': 'Anisotropie'})
                
                fig_flux.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9'), hovermode="x unified"
                )
                st.plotly_chart(fig_flux, use_container_width=True)
    
            # --- IDENTIFICATION DU POINT OPTIMAL ---
            st.markdown("### 🎯 Configuration Optimale Identifiée")
            
            # Trouver le point optimal (T <= T_crit avec alpha minimal = moins de masse)
            df_safe = df_results[df_results['T_h1'] <= CONSTANTS['T_crit']]
            
            if len(df_safe) > 0:
                optimal = df_safe.loc[df_safe['alpha'].idxmin()]
                cols_opt = st.columns(4)
                
                with cols_opt[0]:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 12px; border: 2px solid #10b981; text-align: center;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">🏆 Alpha Optimal</span>
                        <div style="color: #10b981; font-size: 1.8rem; font-weight: 700;">{optimal['alpha']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_opt[1]:
                    st.markdown(f"""
                    <div style="background: rgba(59, 130, 246, 0.15); padding: 1rem; border-radius: 12px; border: 1px solid #3b82f6; text-align: center;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">📏 Épaisseur</span>
                        <div style="color: #3b82f6; font-size: 1.4rem; font-weight: 700;">{optimal['h3_um']:.0f} µm</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_opt[2]:
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.15); padding: 1rem; border-radius: 12px; border: 1px solid #ef4444; text-align: center;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">🌡️ Température</span>
                        <div style="color: #ef4444; font-size: 1.4rem; font-weight: 700;">{optimal['T_h1']:.1f}°C</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_opt[3]:
                    st.markdown(f"""
                    <div style="background: rgba(245, 158, 11, 0.15); padding: 1rem; border-radius: 12px; border: 1px solid #f59e0b; text-align: center;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">💰 Coût</span>
                        <div style="color: #f59e0b; font-size: 1.4rem; font-weight: 700;">{optimal['cost_eur']:.2f}€</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✅ La configuration optimale utilise **α = {optimal['alpha']:.2f}** (β = {optimal['beta']:.1f}) avec un coût minimal de **{optimal['cost_eur']:.2f}€** par aube.")
            else:
                st.warning("⚠️ Aucune configuration ne respecte la limite de température. Augmentez la plage d'alpha ou réduisez la température de surface.")
            
            st.divider()
            
            # --- TABLEAU AVEC EXPORT ---
            st.markdown("### 📊 Résultats Détaillés")
            
            # Boutons d'export
            col_export1, col_export2, col_spacer = st.columns([1, 1, 3])
            
            with col_export1:
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exporter CSV",
                    data=csv,
                    file_name="etude_parametrique_resultats.csv",
                    mime="text/csv",
                    type="secondary"
                )
            
            with col_export2:
                # Export JSON pour intégration
                json_data = df_results.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Exporter JSON",
                    data=json_data,
                    file_name="etude_parametrique_resultats.json",
                    mime="application/json",
                    type="secondary"
                )
            
            st.dataframe(df_results.style.format({
                "alpha": "{:.2f}", 
                "beta": "{:.1f}", 
                "T_h1": "{:.1f}", 
                "dQ1_h1": "{:.2e}",
                "cost_eur": "{:.2f}",
                "mass_kg": "{:.3f}",
                "h3_um": "{:.0f}"
            }).highlight_min(subset=['T_h1', 'cost_eur'], color='#10b98133'), use_container_width=True, height=250)

            # --- ANALYSE DE SENSIBILITÉ ---
            st.markdown("### 🔬 Analyse de Sensibilité")
            st.caption("Sensibilité aux variations locales (±10%) des paramètres autour des valeurs actuelles.")
            
            # On prend la première valeur d'alpha sélectionnée comme référence
            alpha_ref = alphas_to_test[int(len(alphas_to_test)/2)] if len(alphas_to_test) > 0 else 0.1
            beta_ref = betas_to_test[0] if len(betas_to_test) > 0 else beta_in
            
            # Calcul de sensibilité manuelle
            base_res = cached_solve_tbc_model(alpha_ref, beta_ref, lw_in, t_bottom, t_top)
            
            if base_res['success']:
                sens_data = []
                params = [
                    ('alpha', 'Épaisseur (α)', alpha_ref),
                    ('beta', 'Anisotropie (β)', beta_ref),
                    ('lw', 'Longueur d\'Onde (Lw)', lw_in)
                ]
                
                for param_key, param_label, param_val in params:
                    # +10%
                    if param_key == 'alpha':
                        res_plus = cached_solve_tbc_model(param_val * 1.1, beta_ref, lw_in, t_bottom, t_top)
                        res_minus = cached_solve_tbc_model(param_val * 0.9, beta_ref, lw_in, t_bottom, t_top)
                    elif param_key == 'beta':
                        res_plus = cached_solve_tbc_model(alpha_ref, param_val * 1.1, lw_in, t_bottom, t_top)
                        res_minus = cached_solve_tbc_model(alpha_ref, param_val * 0.9, lw_in, t_bottom, t_top)
                    else:
                        res_plus = cached_solve_tbc_model(alpha_ref, beta_ref, param_val * 1.1, t_bottom, t_top)
                        res_minus = cached_solve_tbc_model(alpha_ref, beta_ref, param_val * 0.9, t_bottom, t_top)
                    
                    if res_plus['success'] and res_minus['success']:
                        delta_T = res_plus['T_at_h1'] - res_minus['T_at_h1']
                        delta_Q = abs(res_plus['dQ1_h1']) - abs(res_minus['dQ1_h1'])
                        sens_data.append({'Paramètre': param_label, 'Impact T° (°C)': delta_T, 'Impact Flux (W/m²)': delta_Q})
                
                if sens_data:
                    df_sens = pd.DataFrame(sens_data)
                    
                    c_s1, c_s2 = st.columns(2)
                    with c_s1:
                        fig_sens_t = px.bar(df_sens, x="Paramètre", y="Impact T° (°C)", color="Paramètre",
                                            title="Impact sur la Température (Delta T)",
                                            color_discrete_sequence=[PALETTE['temp'], PALETTE['accent'], PALETTE['flux_trans']])
                        fig_sens_t.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9'), showlegend=False)
                        st.plotly_chart(fig_sens_t, use_container_width=True)
                        
                        st.markdown("""
                        **Interprétation Physique:**
                        - **Épaisseur (α)** : Domine la chute de température. Plus α est grand, plus T diminue.
                        - **Anisotropie (β)** : Impact faible sur T en 1D (le flux principal est normal).
                        """)
                        
                    with c_s2:
                        fig_sens_q = px.bar(df_sens, x="Paramètre", y="Impact Flux (W/m²)", color="Paramètre",
                                            title="Impact sur le Saut de Flux (Delta Q)",
                                            color_discrete_sequence=[PALETTE['temp'], PALETTE['accent'], PALETTE['flux_trans']])
                        fig_sens_q.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9'), showlegend=False)
                        st.plotly_chart(fig_sens_q, use_container_width=True)
                        
                        st.markdown("""
                        **Interprétation Physique:**
                        - **Lw (Longueur d'Onde)** : Modifie les gradients latéraux, impacte la discontinuité du flux.
                        - **Anisotropie (β)** : Affecte la redistribution du flux entre les couches.
                        """)

            # --- MATRICE DE CORRÉLATION ---
            if len(df_results) > 2:
                st.markdown("### 🧩 Matrice de Corrélation Multi-Paramétrique")
                
                cols_candidates = {
                    'alpha': 'Alpha',
                    'beta': 'Beta',
                    'T_h1': 'Température',
                    'dQ1_h1': 'Flux Transv.',
                    'cost_eur': 'Coût'
                }
                
                std_devs = df_results[list(cols_candidates.keys())].std()
                cols_corr = [col for col in cols_candidates.keys() if std_devs[col] > 1e-9]
                labels_corr = [cols_candidates[col] for col in cols_corr]

                if len(cols_corr) < 2:
                    st.warning("Pas assez de paramètres variables pour calculer des corrélations.")
                else:
                    df_corr = df_results[cols_corr].corr()
                    
                    fig_corr = px.imshow(df_corr, text_auto=".2f", aspect="auto",
                                        x=labels_corr, y=labels_corr,
                                        color_continuous_scale="RdBu_r", origin='lower',
                                        title=f"Heatmap des Corrélations ({', '.join(labels_corr)} variables)")
                    
                    fig_corr.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                        font=dict(color='#f1f5f9')
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Explication dynamique
                    has_beta_var = 'beta' in cols_corr
                    html_content = [
                        '<div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 8px; margin-top: 1rem;">',
                        '<h5 style="color: #60a5fa; margin: 0 0 0.5rem 0;">💡 Analyse des Corrélations :</h5>',
                        '<ul style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0;">',
                        '<li><strong>Alpha vs Masse/Coût (1.00)</strong> : Relation géométrique directe.</li>',
                        '<li><strong>Alpha vs Température</strong> : Forte corrélation négative (l\'épaisseur isole).</li>'
                    ]
                    
                    if has_beta_var:
                        html_content.append('<li><strong>Beta vs Flux Transverse</strong> : L\'anisotropie influence la redistribution latérale du flux.</li>')
                    else:
                        html_content.append('<li><em>Note : Beta est constant dans cette simulation (fixé par la Sidebar).<br>👉 Pour voir son impact dans la matrice, sélectionnez le mode <strong>"Liste Manuelle"</strong> ci-dessus et choisissez plusieurs valeurs.</em></li>')

                    html_content.append('</ul>')
                    html_content.append('</div>')
                    
                    st.markdown("".join(html_content), unsafe_allow_html=True)
