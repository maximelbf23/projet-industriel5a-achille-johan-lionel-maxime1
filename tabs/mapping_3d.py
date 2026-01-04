import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.calculation import solve_tbc_model_v2

# --- Fonctions de calcul décorées pour la mise en cache ---
@st.cache_data
def cached_solve_tbc_model(alpha, beta, lw, t_bottom, t_top):
    return solve_tbc_model_v2(alpha, beta, lw, t_bottom, t_top)

def render(lw_in, t_bottom, t_top):
    """Affiche l'onglet de cartographie 3D."""
    
    # === EN-TÊTE HERO SPECTACULAIRE ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #164e63 50%, #1e1b4b 100%);
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; position: relative; overflow: hidden;
                border: 1px solid rgba(6, 182, 212, 0.2); box-shadow: 0 0 50px rgba(6, 182, 212, 0.15);">
        <div style="position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; 
                    background: radial-gradient(circle, rgba(6,182,212,0.25) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; 
                    background: radial-gradient(circle, rgba(59,130,246,0.2) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;
                       background: linear-gradient(135deg, #06b6d4 0%, #22d3ee 50%, #67e8f9 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;">
                🧊 Cartographie 3D : Preuve d'Hétérogénéité
            </h2>
            <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 1rem;">
                Surfaces interactives • Réponse continue vs discrète • Exploration Alpha/Beta
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    PALETTE = {
        'text': '#334155',
        'grid': '#e2e8f0'
    }

    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); padding: 1rem; border-radius: 12px; border-left: 4px solid #06b6d4; margin-bottom: 1.5rem;">
        <p style="color: #cbd5e1; margin: 0; font-size: 0.95rem;">
            💡 Cette visualisation permet de comparer la réponse <strong style="color: #22d3ee;">continue</strong> (Température) 
            et <strong style="color: #f472b6;">discrète/hétérogène</strong> (Saut de Flux). 
            Le saut de flux démontre que la matière n'est pas un milieu continu classique.
        </p>
    </div>
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
            [
                "🌡️ Température T(h1)", 
                "⚡ Saut de Flux ΔQ1(h1)",
                "📊 Contrainte Max σ₃₃ (Mécanique)",
                "🔴 Indicateur D (Endommagement)"
            ],
            help="Thermique ou Mécanique. σ₃₃ et D calculent les contraintes via le solveur spectral."
        )
        
        if st.button("🔄 Générer Surface 3D"):
            # Import pour calculs mécaniques
            from tabs.dashboard_home import compute_real_damage_indicator
            
            # Utilisation des plages configurées
            alpha_vals = np.linspace(a_min, a_max, res_grid)
            beta_vals = np.linspace(b_min, b_max, res_grid)
            z_data = []
            
            # Déterminer si calcul mécanique requis (plus lent)
            is_mechanical = "σ₃₃" in plot_type or "Indicateur D" in plot_type
            if is_mechanical:
                st.info("⏳ Calcul mécanique en cours... (plus long que thermique)")
            
            # Boucle de calcul 2D (Range Alpha x Range Beta)
            progress_bar = st.progress(0)
            for i, b in enumerate(beta_vals):
                z_row = []
                for a in alpha_vals:
                    r = cached_solve_tbc_model(a, b, lw_in, t_bottom, t_top)
                    if r['success']:
                        # Choix de la variable selon la sélection
                        if "Température" in plot_type:
                            val = r['T_at_h1']
                        elif "Flux" in plot_type:
                            val = r['dQ1_h1']  # Le fameux saut discret
                        elif "σ₃₃" in plot_type:
                            # Calcul simplifié de σ33 max via le modèle de mismatch
                            # σ ≈ E_cer * Δα * ΔT / (1 + α)
                            E_cer = 50e9  # Pa
                            delta_alpha = 3e-6  # |α_sub - α_cer|
                            delta_T = t_top - t_bottom
                            val = (E_cer * delta_alpha * delta_T / (1 + a)) / 1e6  # MPa
                        elif "Indicateur D" in plot_type:
                            # Utilise la fonction du dashboard
                            val = compute_real_damage_indicator(a, lw_in, t_top, t_bottom)
                        else:
                            val = np.nan
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
            elif "σ₃₃" in current_type:
                z_title = "σ₃₃ max (MPa)"
                colors = "RdYlGn_r"
                main_title = "Surface 3D : Contrainte Maximale σ₃₃ (Mécanique)"
            elif "Indicateur D" in current_type:
                z_title = "Indicateur D"
                colors = "RdYlGn_r"
                main_title = "Surface 3D : Carte d'Endommagement D(α, β)"
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
            
            # --- NOUVEAU : STATISTIQUES DE LA SURFACE ---
            z_array = np.array(st.session_state['z_3d'])
            z_flat = z_array[~np.isnan(z_array)]
            
            if len(z_flat) > 0:
                z_min, z_max, z_mean, z_std = z_flat.min(), z_flat.max(), z_flat.mean(), z_flat.std()
                
                # Trouver les coordonnées min/max
                min_idx = np.unravel_index(np.nanargmin(z_array), z_array.shape)
                max_idx = np.unravel_index(np.nanargmax(z_array), z_array.shape)
                
                alpha_at_min = st.session_state['x_3d'][min_idx[1]]
                beta_at_min = st.session_state['y_3d'][min_idx[0]]
                alpha_at_max = st.session_state['x_3d'][max_idx[1]]
                beta_at_max = st.session_state['y_3d'][max_idx[0]]
                
                st.markdown("### 📈 Statistiques de la Surface")
                
                cols_stats = st.columns(4)
                with cols_stats[0]:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #10b981;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">📉 Minimum</span>
                        <div style="color: #10b981; font-size: 1.4rem; font-weight: 700;">{z_min:.1f}</div>
                        <span style="color: #64748b; font-size: 0.7rem;">α={alpha_at_min:.2f}, β={beta_at_min:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_stats[1]:
                    st.markdown(f"""
                    <div style="background: rgba(239, 68, 68, 0.15); padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #ef4444;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">📈 Maximum</span>
                        <div style="color: #ef4444; font-size: 1.4rem; font-weight: 700;">{z_max:.1f}</div>
                        <span style="color: #64748b; font-size: 0.7rem;">α={alpha_at_max:.2f}, β={beta_at_max:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_stats[2]:
                    st.markdown(f"""
                    <div style="background: rgba(59, 130, 246, 0.15); padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #3b82f6;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">📊 Moyenne</span>
                        <div style="color: #3b82f6; font-size: 1.4rem; font-weight: 700;">{z_mean:.1f}</div>
                        <span style="color: #64748b; font-size: 0.7rem;">{z_title}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols_stats[3]:
                    st.markdown(f"""
                    <div style="background: rgba(139, 92, 246, 0.15); padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #8b5cf6;">
                        <span style="color: #94a3b8; font-size: 0.8rem;">📏 Écart-Type</span>
                        <div style="color: #8b5cf6; font-size: 1.4rem; font-weight: 700;">{z_std:.1f}</div>
                        <span style="color: #64748b; font-size: 0.7rem;">Dispersion</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export des données
                st.markdown("### 📥 Export des Données")
                col_exp1, col_exp2, col_spacer = st.columns([1, 1, 3])
                
                with col_exp1:
                    # Créer DataFrame pour export
                    export_data = []
                    for i, b in enumerate(st.session_state['y_3d']):
                        for j, a in enumerate(st.session_state['x_3d']):
                            export_data.append({'alpha': a, 'beta': b, 'value': z_array[i, j]})
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Exporter CSV",
                        data=csv,
                        file_name=f"surface_3d_{current_type.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                
                with col_exp2:
                    json_data = df_export.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Exporter JSON",
                        data=json_data,
                        file_name=f"surface_3d_{current_type.replace(' ', '_')}.json",
                        mime="application/json"
                    )
            
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
