import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from core.mechanical import solve_multilayer_problem
from core.constants import MECHANICAL_PROPS, CONSTANTS

def render():
    # === EN-TÊTE HERO SPECTACULAIRE ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e1b4b 100%);
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; position: relative; overflow: hidden;
                border: 1px solid rgba(16, 185, 129, 0.2); box-shadow: 0 0 50px rgba(16, 185, 129, 0.15);">
        <div style="position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; 
                    background: radial-gradient(circle, rgba(16,185,129,0.25) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; 
                    background: radial-gradient(circle, rgba(245,158,11,0.2) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;
                       background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;">
                📊 Sensibilité & Optimisation Multi-Objectifs
            </h2>
            <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 1rem;">
                Carte de risque 2D • Front de Pareto • Compromis Performance/Intégrité
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_heatmap, tab_pareto = st.tabs(["🔥 Carte de Risque (Heatmap)", "⚖️ Front de Pareto (Compromis)"])
    
    # --- TAB 1: HEATMAP ---
    with tab_heatmap:
        st.subheader("Carte de Risque 2D : Épaisseur vs Température")
        st.caption("Visualisation des zones de sûreté (Vert) et de rupture (Rouge) en fonction des paramètres opérationnels.")
        
        c1, c2 = st.columns([1, 3])
        
        with c1:
            h_range = st.slider("Plage Épaisseur TBC (µm)", 100, 1000, (150, 500), step=50, key="hm_h")
            t_range = st.slider("Plage Amplitude Temp. ΔT (°C)", 500, 1500, (800, 1400), step=100, key="hm_t")
            res_hm = st.select_slider("Résolution Calcul", options=["Basse (Rapid)", "Moyenne", "Haute (Lent)"], value="Moyenne")
            
            n_pts = 8 if res_hm == "Basse (Rapid)" else (15 if res_hm == "Moyenne" else 30)
            
            if st.button("🔄 Générer la Heatmap", type="primary"):
                with st.spinner("Calcul de la cartographie des contraintes..."):
                    # Grid generation
                    h_vals = np.linspace(h_range[0], h_range[1], n_pts)
                    t_vals = np.linspace(t_range[0], t_range[1], n_pts)
                    
                    z_grid = []
                    
                    # Propriétés fixes
                    props_sub = MECHANICAL_PROPS.copy()
                    props_bc = MECHANICAL_PROPS.copy()
                    props_tbc = MECHANICAL_PROPS.copy()
                    alpha_fix = {'alpha_1': 14e-6, 'alpha_2': 14e-6, 'alpha_3': 10e-6} # Differencial expansion
                    
                    for t in t_vals:
                        row = []
                        for h in h_vals:
                            # Simulation rapide CLT
                            layer_configs = [
                                (0.001, props_sub, alpha_fix),
                                (50e-6, props_bc, alpha_fix),
                                (h * 1e-6, props_tbc, alpha_fix)
                            ]
                            try:
                                res = solve_multilayer_problem(layer_configs, 0.1, 10.0, t, method='clt')
                                # Max stress (approx)
                                sig = np.max(np.abs(res['stress_profile']['sigma_33']))
                                row.append(sig / 1e6) # MPa
                            except:
                                row.append(0)
                        z_grid.append(row)
                    
                    st.session_state['hm_data'] = {'x': h_vals, 'y': t_vals, 'z': z_grid}

        with c2:
            if 'hm_data' in st.session_state:
                data = st.session_state['hm_data']
                z_array = np.array(data['z'])
                
                # Trouver la zone optimale (stress minimal)
                min_idx = np.unravel_index(np.argmin(z_array), z_array.shape)
                optimal_h = data['x'][min_idx[1]]
                optimal_t = data['y'][min_idx[0]]
                optimal_stress = z_array[min_idx]
                
                fig_hm = go.Figure(data=go.Heatmap(
                    z=data['z'],
                    x=data['x'],
                    y=data['y'],
                    colorscale='Portland',
                    colorbar=dict(title='Max σ₃₃ (MPa)'),
                    hovertemplate='Épaisseur: %{x:.0f} µm<br>Delta T: %{y:.0f} K<br>Stress: %{z:.1f} MPa<extra></extra>'
                ))
                
                # Ajouter le point optimal
                fig_hm.add_trace(go.Scatter(
                    x=[optimal_h],
                    y=[optimal_t],
                    mode='markers+text',
                    marker=dict(size=20, color='#10b981', symbol='star', line=dict(width=2, color='white')),
                    text=['🏆 OPTIMAL'],
                    textposition='top center',
                    textfont=dict(color='#10b981', size=12),
                    name='Point Optimal',
                    showlegend=False
                ))
                
                fig_hm.update_layout(
                    title="Cartographie des Contraintes Max (|σ₃₃|)",
                    xaxis_title="Épaisseur TBC (µm)",
                    yaxis_title="Amplitude Thermique ΔT (°C)",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9')
                )
                
                st.plotly_chart(fig_hm, use_container_width=True)
                
                # Panneau du point optimal recommandé
                st.markdown("### 🎯 Point Optimal Recommandé")
                cols_opt = st.columns(3)
                
                with cols_opt[0]:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.2); padding: 1.2rem; border-radius: 12px; text-align: center; border: 2px solid #10b981;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">📏 Épaisseur TBC</span>
                        <div style="color: #10b981; font-size: 1.8rem; font-weight: 700;">{optimal_h:.0f} µm</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_opt[1]:
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.2); padding: 1.2rem; border-radius: 12px; text-align: center; border: 2px solid #ef4444;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">🌡️ Delta T</span>
                        <div style="color: #ef4444; font-size: 1.8rem; font-weight: 700;">{optimal_t:.0f}°C</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_opt[2]:
                    st.markdown(f"""
                    <div style="background: rgba(59, 130, 246, 0.2); padding: 1.2rem; border-radius: 12px; text-align: center; border: 2px solid #3b82f6;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">⚡ Contrainte Min</span>
                        <div style="color: #3b82f6; font-size: 1.8rem; font-weight: 700;">{optimal_stress:.1f} MPa</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✅ La configuration optimale est **{optimal_h:.0f} µm** à **ΔT = {optimal_t:.0f}°C** avec une contrainte de **{optimal_stress:.1f} MPa**.")
                
                # Export des données
                col_exp1, col_exp2, col_spacer = st.columns([1, 1, 3])
                with col_exp1:
                    # Créer DataFrame pour export
                    export_data = []
                    for i, t in enumerate(data['y']):
                        for j, h in enumerate(data['x']):
                            export_data.append({'epaisseur_um': h, 'delta_T': t, 'stress_MPa': z_array[i, j]})
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Exporter CSV",
                        data=csv,
                        file_name="heatmap_contraintes.csv",
                        mime="text/csv"
                    )
                
            else:
                st.info("Configurez et lancez la simulation pour voir la carte des risques.")

    # --- TAB 2: PARETO ---
    with tab_pareto:
        st.subheader("⚖️ Analyse de Compromis (Pareto)")
        st.markdown("On cherche à **minimiser le Flux Thermique** (meilleure isolation) tout en **minimisant la Contrainte** (durée de vie).")
        
        cols_p = st.columns([1, 2])
        
        with cols_p[0]:
            n_pareto = st.slider("Nombre de scénarios", 10, 100, 50)
            
            if st.button("🧬 Lancer l'Optimisation", type="secondary"):
                # Simulation Monte Carlo / Grid
                h_tests = np.linspace(100, 800, n_pareto)
                
                pareto_results = []
                
                props_tbc = MECHANICAL_PROPS.copy()
                alpha_fix = {'alpha_1': 14e-6, 'alpha_2': 14e-6, 'alpha_3': 10e-6}
                
                # Const for thermal
                k_tbc = 1.5 # W/mK approx
                
                for h in h_tests:
                    # 1. Thermal Perf (Flux = k * dT / h)
                    # On veut minimiser flux => Maximiser h
                    flux_proxy = k_tbc * 1000 / (h * 1e-6) # W/m2 approx
                    
                    # 2. Mech Perf (Stress)
                    layer_configs = [
                        (0.001, MECHANICAL_PROPS, alpha_fix),
                        (50e-6, MECHANICAL_PROPS, alpha_fix),
                        (h * 1e-6, MECHANICAL_PROPS, alpha_fix)
                    ]
                    # Stress augmente souvent avec épaisseur dans TBC (plus de levier / mismatch cumulé) 
                    # ou diminue ? Dépend du modèle. Ici CLT simple.
                    
                    try:
                        res = solve_multilayer_problem(layer_configs, 0.1, 10.0, 1000, method='clt')
                        stress = np.max(np.abs(res['stress_profile']['sigma_33'])) / 1e6
                        
                        pareto_results.append({
                            'h_um': h,
                            'flux': flux_proxy,
                            'stress': stress,
                            'score': (flux_proxy/1e6) + (stress/100) # Dummy score
                        })
                    except:
                        pass
                
                st.session_state['pareto_data'] = pd.DataFrame(pareto_results)
        
        with cols_p[1]:
            if 'pareto_data' in st.session_state:
                df = st.session_state['pareto_data']
                
                # Plot
                fig_par = px.scatter(df, x='stress', y='flux', color='h_um',
                                     title="Front de Pareto : Isolation vs Contrainte",
                                     labels={'stress': 'Contrainte Max (MPa) [A minimiser]', 
                                             'flux': 'Flux Thermique (W/m²) [A minimiser]',
                                             'h_um': 'Épaisseur TBC (µm)'},
                                     color_continuous_scale='Viridis')
                
                fig_par.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
                
                fig_par.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9')
                )
                
                st.plotly_chart(fig_par, use_container_width=True)
                
                st.success("💡 **Le point Utopia** (idéal) est en bas à gauche (Faible Flux, Faible Stress).")
            else:
                st.info("Lancez l'optimisation pour visualiser les compromis de design.")

if __name__ == "__main__":
    render()
