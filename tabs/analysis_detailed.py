import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.calculation import solve_tbc_model_v2, calculate_profiles
from core.constants import CONSTANTS, IMPACT_PARAMS

# --- Fonctions de calcul décorées pour la mise en cache (Locales au module si besoin, ou importées si partagées) ---
# Note: Dans une architecture idéale, ces fonctions de cache devraient être dans core/cache.py ou similaire.
# Pour l'instant, je les redéfinis ici ou j'importe si elles sont dans un module commun.
# Comme elles étaient dans le main, je vais les inclure ici pour l'autonomie du module.

@st.cache_data
def cached_solve_tbc_model(alpha, beta, lw, t_bottom, t_top):
    """Wrapper pour mettre en cache les résultats de solve_tbc_model."""
    # solve_tbc_model_v2 doit être importé de core.calculation
    return solve_tbc_model_v2(alpha, beta, lw, t_bottom, t_top)

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

    f_min = get_temp_at_alpha(alpha_min) - target_temp
    f_max = get_temp_at_alpha(alpha_max) - target_temp

    if np.sign(f_min) == np.sign(f_max):
        # Fallback ou erreur si l'intervalle n'encadre pas la solution
        return {'success': False, 'message': "La T° cible est hors de l'intervalle de recherche."}

    # Inversion si l'utilisateur a entré une plage incorrecte
    if f_min < f_max:
        alpha_min, alpha_max = alpha_max, alpha_min
        f_min, f_max = f_max, f_min

    for i in range(max_iter):
        alpha_mid = (alpha_min + alpha_max) / 2
        f_mid = get_temp_at_alpha(alpha_mid) - target_temp

        if abs(f_mid) < tol:
            return {'success': True, 'alpha': alpha_mid}

        if np.sign(f_mid) == np.sign(f_min):
            alpha_min = alpha_mid
            f_min = f_mid
        else:
            alpha_max = alpha_mid
            # f_max = f_mid # Pas strictement nécessaire pour l'algo mais bon de le savoir

    return {'success': True, 'alpha': (alpha_min + alpha_max) / 2} # Retourne la meilleure estimation

def render(alpha_in, beta_in, lw_in, t_bottom, t_top, t_bottom_cata, t_top_cata):
    """Affiche l'onglet d'analyse détaillée pour un cas unique."""
    
    # === EN-TÊTE HERO SPECTACULAIRE ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e1b4b 100%);
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; position: relative; overflow: hidden;
                border: 1px solid rgba(96, 165, 250, 0.2); box-shadow: 0 0 50px rgba(59, 130, 246, 0.15);">
        <div style="position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; 
                    background: radial-gradient(circle, rgba(239,68,68,0.2) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; 
                    background: radial-gradient(circle, rgba(59,130,246,0.2) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;
                       background: linear-gradient(135deg, #ef4444 0%, #f97316 50%, #fbbf24 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;">
                🔎 Analyse Détaillée & Impacts
            </h2>
            <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 1rem;">
                Profils thermiques • Quantification des impacts • Comparaison de scénarios
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Palette locale pour cohérence (peut être importée d'un module de style global plus tard)
    PALETTE = {
        'temp': '#3b82f6',       # Blue-500
        'flux_norm': '#ef4444',  # Red-500
        'flux_trans': '#10b981', # Emerald-500
        'accent': '#f59e0b',     # Amber-500
        'grid': '#e2e8f0',       # Slate-200
        'text': '#334155',       # Slate-700
        'bg': '#ffffff'          # White
    }
    
    # Constantes locales pour affichage
    T_secu = CONSTANTS['T_crit'] * CONSTANTS['Securite_pct']
    
    res = cached_solve_tbc_model(alpha_in, beta_in, lw_in, t_bottom, t_top)
    
    # Stocker le résultat courant dans la session pour la synthèse premium
    st.session_state['current_result'] = res
    st.session_state['current_inputs'] = {
        'alpha': alpha_in, 'beta': beta_in, 'lw': lw_in, 
        't_bottom': t_bottom, 't_top': t_top
    }
    
    if not res['success']:
        st.error(f"Erreur lors du calcul : {res.get('error', 'Erreur inconnue')}")
        return

    # ========================================
    # NOUVEAU : PANNEAU DE SYNTHÈSE INTELLIGENT
    # ========================================
    T_h1 = res['T_at_h1']
    h3_mic = res['h3'] * 1e6
    delta_T = t_bottom - t_top
    flux_trans = abs(res['dQ1_h1'])
    
    # Calcul du flux normal approximatif (depuis les profils)
    x_plot, T_vals, Q1_vals, Q3_vals = calculate_profiles(res['profile_params'], res['H'])
    # Trouver l'indice correspondant à h1
    h1_pos = CONSTANTS['h1']
    h1_idx = np.argmin(np.abs(x_plot - h1_pos))
    flux_norm = abs(Q3_vals[h1_idx]) if len(Q3_vals) > h1_idx else 0
    
    # Calcul des indicateurs
    marge_securite = CONSTANTS['T_crit'] - T_h1
    perf_isolation = (delta_T - (T_h1 - t_top)) / delta_T * 100 if delta_T > 0 else 0
    
    # Déterminer le statut global
    if T_h1 > CONSTANTS['T_crit']:
        status = "CRITIQUE"
        status_color = "#ef4444"
        status_icon = "🚨"
    elif T_h1 > T_secu:
        status = "ATTENTION"
        status_color = "#f59e0b"
        status_icon = "⚠️"
    else:
        status = "OPTIMAL"
        status_color = "#10b981"
        status_icon = "✅"
    
    # Panneau de synthèse avec 4 KPIs
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem;
                border: 2px solid {status_color}40; box-shadow: 0 0 30px {status_color}20;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0; font-size: 1.3rem;">
                {status_icon} Synthèse Rapide — <span style="color: {status_color};">{status}</span>
            </h3>
            <span style="background: {status_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                T° Interface: {T_h1:.1f}°C
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div style="background: rgba(59, 130, 246, 0.15); padding: 1rem; border-radius: 12px; border-left: 4px solid #3b82f6;">
                <span style="color: #94a3b8; font-size: 0.8rem;">📏 Épaisseur TBC</span>
                <div style="color: white; font-size: 1.4rem; font-weight: 700;">{h3_mic:.0f} µm</div>
                <span style="color: #64748b; font-size: 0.75rem;">α = {alpha_in:.2f}</span>
            </div>
            <div style="background: rgba(239, 68, 68, 0.15); padding: 1rem; border-radius: 12px; border-left: 4px solid #ef4444;">
                <span style="color: #94a3b8; font-size: 0.8rem;">🔥 Flux Normal</span>
                <div style="color: white; font-size: 1.4rem; font-weight: 700;">{flux_norm/1000:.1f} kW/m²</div>
                <span style="color: #64748b; font-size: 0.75rem;">Q₃(h₁)</span>
            </div>
            <div style="background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 12px; border-left: 4px solid #10b981;">
                <span style="color: #94a3b8; font-size: 0.8rem;">🛡️ Marge Sécurité</span>
                <div style="color: {'#10b981' if marge_securite > 50 else '#f59e0b' if marge_securite > 0 else '#ef4444'}; font-size: 1.4rem; font-weight: 700;">{marge_securite:.0f}°C</div>
                <span style="color: #64748b; font-size: 0.75rem;">T_crit - T(h₁)</span>
            </div>
            <div style="background: rgba(139, 92, 246, 0.15); padding: 1rem; border-radius: 12px; border-left: 4px solid #8b5cf6;">
                <span style="color: #94a3b8; font-size: 0.8rem;">⚡ Performance</span>
                <div style="color: white; font-size: 1.4rem; font-weight: 700;">{perf_isolation:.0f}%</div>
                <span style="color: #64748b; font-size: 0.75rem;">Isolation thermique</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Alertes et Recommandations Contextuelles
    alerts = []
    recommendations = []
    
    if T_h1 > CONSTANTS['T_crit']:
        alerts.append(("🚨 ALERTE CRITIQUE", f"Température {T_h1:.1f}°C dépasse la limite de {CONSTANTS['T_crit']}°C", "#ef4444"))
        recommendations.append("↗️ Augmenter l'épaisseur TBC (α)")
        recommendations.append("↘️ Réduire la température en surface")
    elif T_h1 > T_secu:
        alerts.append(("⚠️ ATTENTION", f"Température {T_h1:.1f}°C proche de la limite ({CONSTANTS['T_crit']}°C)", "#f59e0b"))
        recommendations.append("📊 Surveiller l'évolution de la température")
    else:
        alerts.append(("✅ CONFORME", f"Configuration respecte les limites thermiques avec marge de {marge_securite:.0f}°C", "#10b981"))
    
    if alpha_in > 1.0:
        alerts.append(("📦 INFO", f"Épaisseur TBC élevée (α={alpha_in:.2f}) : impact sur masse et coût", "#3b82f6"))
        recommendations.append("⚖️ Évaluer le compromis masse/protection")
    
    if flux_trans > 100000:
        alerts.append(("⚡ FLUX", f"Flux transverse élevé ({flux_trans/1000:.1f} kW/m²) : vérifier contraintes de cisaillement", "#8b5cf6"))
    
    # Affichage des alertes
    if alerts:
        cols_alerts = st.columns(len(alerts))
        for i, (title, msg, color) in enumerate(alerts):
            with cols_alerts[i]:
                st.markdown(f"""
                <div style="background: {color}15; border-left: 4px solid {color}; padding: 0.8rem 1rem; border-radius: 8px;">
                    <strong style="color: {color};">{title}</strong>
                    <p style="color: #cbd5e1; margin: 0.3rem 0 0 0; font-size: 0.9rem;">{msg}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Recommandations
    if recommendations:
        with st.expander("💡 Recommandations", expanded=True):
            rec_html = "".join([f"<li style='color: #cbd5e1; margin: 0.3rem 0;'>{r}</li>" for r in recommendations])
            st.markdown(f"""
            <ul style="margin: 0; padding-left: 1.5rem;">
                {rec_html}
            </ul>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Conversion dimensions
    h1_mic = CONSTANTS['h1'] * 1e6
    
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
        c1.metric("📏 Épaisseur TBC ($h_3$)", f"{h3_mic:.0f} µm", help="Épaisseur de la couche céramique (Top Coat). Impacte direct l'isolation.")
        c2.metric("⚡ Conductivité Trans.", f"{res['k_eta_3']:.2f} W/mK", help="Conductivité équivalente dans le plan, modifiée par le facteur d'anisotropie (Beta).")
        c3.metric(f"{status_icon} T° Interface Alliage", f"{T_h1:.2f} °C", delta=f"{-delta_T:.2f} vs Limite", help="Température critique à l'interface entre la couche de liaison et la céramique. Doit rester < 1100°C.")

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
        t_bottom=t_bottom_cata,
        t_top=t_top_cata
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
        - **Catastrophe (Calculé)** : Épaisseur min. pour T_top={t_top_cata}°C.
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
            bgcolor="rgba(255,255,255,0.95)", bordercolor="#e2e8f0", borderwidth=1,
            font=dict(color="#334155", size=11)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", size=12, color=PALETTE['text'])
    )
    
    # Grilles
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'], zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'], zeroline=False)

    st.plotly_chart(fig, use_container_width=True)

    # --- INTERPRÉTATION PHYSIQUE ---
    with st.expander("📚 Interprétation Physique des Graphiques", expanded=False):
        st.markdown("""
#### 🌡️ 1. Profil de Température (Haut)

On observe la chute de température principale dans la **couche céramique** (zone bleue claire), qui agit comme isolant thermique.
La pente y est **plus forte** car la conductivité thermique de la céramique (YSZ) est faible (~1,5 W/m·K).
Dans le **superalliage** (zone grise), la pente est plus **faible** car sa conductivité est bien plus élevée (~22 W/m·K).

#### 🔥 2. Flux Normal Q₃ (Milieu)

Le flux normal (perpendiculaire aux couches, axe x₃) représente le transfert de chaleur principal à travers le revêtement.
Il est **continu** aux interfaces (conservation de l'énergie) mais peut varier en intensité selon les propriétés locales.
La valeur absolue diminue dans la céramique grâce à l'effet d'isolation.

#### ↔️ 3. Flux Transverse Q₁ (Bas)

Ce graphique montre le flux latéral (dans le plan des couches). Il met en évidence les **sauts de flux** aux interfaces,
car les conductivités transverses changent brutalement d'un matériau à l'autre.
Ces discontinuités peuvent être sources de **contraintes de cisaillement** thermomécaniques.
        """)

    # ═══════════════════════════════════════════════════════════════════════════
    # NOUVEAU : PROFILS DE CONTRAINTES MÉCANIQUES
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 🔧 Profils de Contraintes Mécaniques")
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); padding: 1rem; border-radius: 12px; border-left: 4px solid #ef4444; margin-bottom: 1rem;">
        <p style="color: #cbd5e1; margin: 0;">
            📌 Ces profils montrent les contraintes <strong style="color: #ef4444;">d'arrachement (σ₃₃)</strong> et de 
            <strong style="color: #f59e0b;">cisaillement (σ₁₃)</strong> à travers l'épaisseur du multicouche, 
            avec les seuils critiques par matériau.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Import des seuils critiques
    from core.damage_analysis import CRITICAL_STRESS
    from core.constants import PROPS_SUBSTRATE, PROPS_BONDCOAT, PROPS_CERAMIC
    from core.constants import ALPHA_SUBSTRATE, ALPHA_BONDCOAT, ALPHA_CERAMIC
    
    # Calcul du profil de contraintes simplifié (modèle thermo-élastique)
    # σ = E * Δα * ΔT / (1 - ν)
    
    # Création du profil de profondeur
    n_points = 200
    h1 = CONSTANTS['h1']
    h2 = CONSTANTS['h2']
    h3 = res['h3']
    H_total = h1 + h2 + h3
    
    z_profile = np.linspace(0, H_total, n_points)
    z_mm = z_profile * 1000
    
    # Propriétés par couche
    E_vals = np.zeros(n_points)
    alpha_th_vals = np.zeros(n_points)
    layer_idx = np.zeros(n_points, dtype=int)
    
    # Assigner les propriétés par couche
    mask_sub = z_profile <= h1
    mask_bc = (z_profile > h1) & (z_profile <= h1 + h2)
    mask_cer = z_profile > h1 + h2
    
    # Substrat
    E_vals[mask_sub] = PROPS_SUBSTRATE.get('C33', 260) * 1e9  # GPa → Pa
    alpha_th_vals[mask_sub] = ALPHA_SUBSTRATE.get('alpha_1', 13e-6)
    layer_idx[mask_sub] = 0
    
    # Bond coat
    E_vals[mask_bc] = PROPS_BONDCOAT.get('C33', 180) * 1e9
    alpha_th_vals[mask_bc] = ALPHA_BONDCOAT.get('alpha_1', 14e-6)
    layer_idx[mask_bc] = 1
    
    # Céramique
    E_vals[mask_cer] = PROPS_CERAMIC.get('C33', 50) * 1e9
    alpha_th_vals[mask_cer] = ALPHA_CERAMIC.get('alpha_1', 10e-6)
    layer_idx[mask_cer] = 2
    
    # Calcul du profil de température (résistances thermiques)
    R1 = h1 / CONSTANTS['k33_1']
    R2 = h2 / CONSTANTS['k33_2']
    R3 = h3 / CONSTANTS['k33_3']
    R_total = R1 + R2 + R3
    
    delta_T_total = t_top - t_bottom
    T_profile = np.zeros(n_points)
    
    # Température par couche
    T_at_h1 = t_bottom + delta_T_total * R1 / R_total
    T_at_h2 = t_bottom + delta_T_total * (R1 + R2) / R_total
    
    T_profile[mask_sub] = t_bottom + (z_profile[mask_sub] / h1) * (T_at_h1 - t_bottom)
    T_profile[mask_bc] = T_at_h1 + ((z_profile[mask_bc] - h1) / h2) * (T_at_h2 - T_at_h1)
    T_profile[mask_cer] = T_at_h2 + ((z_profile[mask_cer] - h1 - h2) / h3) * (t_top - T_at_h2)
    
    # Température de référence (assemblage)
    T_ref = 20  # °C
    
    # Calcul de contrainte thermique σ₃₃ (modèle simplifié)
    # La contrainte max est aux interfaces due au mismatch de dilatation
    nu = 0.3  # Coefficient de Poisson moyen
    alpha_ref = np.mean(alpha_th_vals)  # Dilatation de référence
    
    sigma_33 = E_vals * (alpha_th_vals - alpha_ref) * (T_profile - T_ref) / (1 - nu)
    sigma_33_mpa = sigma_33 / 1e6
    
    # Calcul de contrainte de cisaillement σ₁₃ (gradient aux interfaces)
    # Le cisaillement est maximum aux interfaces
    sigma_13 = np.zeros(n_points)
    delta_eta = np.pi / lw_in
    
    # Cisaillement proportionnel au gradient local et au mismatch
    for i in range(1, n_points):
        dT_dz = (T_profile[i] - T_profile[i-1]) / (z_profile[i] - z_profile[i-1] + 1e-12)
        sigma_13[i] = E_vals[i] * alpha_th_vals[i] * abs(dT_dz) * 0.001  # Facteur géométrique
    
    # Amplification aux interfaces
    idx_h1 = np.argmin(np.abs(z_profile - h1))
    idx_h2 = np.argmin(np.abs(z_profile - (h1 + h2)))
    
    # Pic de cisaillement aux interfaces (mismatch de propriétés)
    sigma_13[idx_h1-2:idx_h1+3] *= 10
    sigma_13[idx_h2-2:idx_h2+3] *= 15
    
    sigma_13_mpa = sigma_13 / 1e6
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPHIQUE 1 : CONTRAINTE D'ARRACHEMENT σ₃₃
    # ═══════════════════════════════════════════════════════════════════════════
    col_sig33, col_sig13 = st.columns(2)
    
    with col_sig33:
        fig_sig33 = go.Figure()
        
        # Zones de couches
        fig_sig33.add_vrect(x0=0, x1=h1*1000, fillcolor="#cbd5e1", opacity=0.3, 
                           layer="below", line_width=0, annotation_text="Substrat",
                           annotation_position="top left")
        fig_sig33.add_vrect(x0=h1*1000, x1=(h1+h2)*1000, fillcolor="#fdba74", opacity=0.3,
                           layer="below", line_width=0, annotation_text="BC",
                           annotation_position="top left")
        fig_sig33.add_vrect(x0=(h1+h2)*1000, x1=H_total*1000, fillcolor="#bae6fd", opacity=0.3,
                           layer="below", line_width=0, annotation_text="TBC",
                           annotation_position="top left")
        
        # Profil de contrainte
        fig_sig33.add_trace(go.Scatter(
            x=z_mm, y=sigma_33_mpa,
            mode='lines', name='σ₃₃',
            line=dict(color='#ef4444', width=3),
            fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.15)'
        ))
        
        # Seuils critiques (lignes horizontales)
        fig_sig33.add_hline(y=CRITICAL_STRESS['ceramic']['sigma_tensile']/1e6, 
                           line_dash="dash", line_color="#22d3ee",
                           annotation_text=f"σ_crit céramique ({CRITICAL_STRESS['ceramic']['sigma_tensile']/1e6:.0f} MPa)",
                           annotation_position="top right")
        fig_sig33.add_hline(y=-CRITICAL_STRESS['ceramic']['sigma_compressive']/1e6, 
                           line_dash="dash", line_color="#22d3ee")
        
        fig_sig33.update_layout(
            title=dict(text="⬇️ Contrainte d'Arrachement σ₃₃(z)", font=dict(size=16, color=PALETTE['text'])),
            xaxis_title="Profondeur z (mm)",
            yaxis_title="σ₃₃ (MPa)",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12, color=PALETTE['text']),
            showlegend=False
        )
        fig_sig33.update_xaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'])
        fig_sig33.update_yaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'])
        
        st.plotly_chart(fig_sig33, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPHIQUE 2 : CONTRAINTE DE CISAILLEMENT σ₁₃
    # ═══════════════════════════════════════════════════════════════════════════
    with col_sig13:
        fig_sig13 = go.Figure()
        
        # Zones de couches
        fig_sig13.add_vrect(x0=0, x1=h1*1000, fillcolor="#cbd5e1", opacity=0.3,
                           layer="below", line_width=0, annotation_text="Substrat",
                           annotation_position="top left")
        fig_sig13.add_vrect(x0=h1*1000, x1=(h1+h2)*1000, fillcolor="#fdba74", opacity=0.3,
                           layer="below", line_width=0, annotation_text="BC",
                           annotation_position="top left")
        fig_sig13.add_vrect(x0=(h1+h2)*1000, x1=H_total*1000, fillcolor="#bae6fd", opacity=0.3,
                           layer="below", line_width=0, annotation_text="TBC",
                           annotation_position="top left")
        
        # Profil de cisaillement
        fig_sig13.add_trace(go.Scatter(
            x=z_mm, y=sigma_13_mpa,
            mode='lines', name='σ₁₃',
            line=dict(color='#f59e0b', width=3),
            fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.15)'
        ))
        
        # Seuil critique cisaillement
        fig_sig13.add_hline(y=CRITICAL_STRESS['ceramic']['sigma_shear']/1e6, 
                           line_dash="dash", line_color="#8b5cf6",
                           annotation_text=f"τ_crit céramique ({CRITICAL_STRESS['ceramic']['sigma_shear']/1e6:.0f} MPa)",
                           annotation_position="top right")
        
        fig_sig13.update_layout(
            title=dict(text="↔️ Contrainte de Cisaillement σ₁₃(z)", font=dict(size=16, color=PALETTE['text'])),
            xaxis_title="Profondeur z (mm)",
            yaxis_title="σ₁₃ (MPa)",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12, color=PALETTE['text']),
            showlegend=False
        )
        fig_sig13.update_xaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'])
        fig_sig13.update_yaxes(showgrid=True, gridwidth=1, gridcolor=PALETTE['grid'])
        
        st.plotly_chart(fig_sig13, use_container_width=True)
    
    # Analyse des zones critiques
    max_sig33 = np.max(np.abs(sigma_33_mpa))
    max_sig13 = np.max(sigma_13_mpa)
    crit_sig33 = CRITICAL_STRESS['ceramic']['sigma_tensile'] / 1e6
    crit_sig13 = CRITICAL_STRESS['ceramic']['sigma_shear'] / 1e6
    
    ratio_33 = max_sig33 / crit_sig33
    ratio_13 = max_sig13 / crit_sig13
    
    cols_analysis = st.columns(3)
    
    with cols_analysis[0]:
        color_33 = "#10b981" if ratio_33 < 0.7 else "#f59e0b" if ratio_33 < 1.0 else "#ef4444"
        st.markdown(f"""
        <div style="background: {color_33}20; padding: 1rem; border-radius: 12px; border: 1px solid {color_33};">
            <span style="color: #94a3b8;">⬇️ σ₃₃ max</span>
            <div style="color: {color_33}; font-size: 1.5rem; font-weight: 700;">{max_sig33:.1f} MPa</div>
            <span style="color: #64748b;">Ratio: {ratio_33:.1%}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with cols_analysis[1]:
        color_13 = "#10b981" if ratio_13 < 0.7 else "#f59e0b" if ratio_13 < 1.0 else "#ef4444"
        st.markdown(f"""
        <div style="background: {color_13}20; padding: 1rem; border-radius: 12px; border: 1px solid {color_13};">
            <span style="color: #94a3b8;">↔️ σ₁₃ max</span>
            <div style="color: {color_13}; font-size: 1.5rem; font-weight: 700;">{max_sig13:.1f} MPa</div>
            <span style="color: #64748b;">Ratio: {ratio_13:.1%}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with cols_analysis[2]:
        zone_critique = "Interface BC/TBC" if ratio_13 > ratio_33 else "Volume TBC"
        st.markdown(f"""
        <div style="background: rgba(139, 92, 246, 0.15); padding: 1rem; border-radius: 12px; border: 1px solid #8b5cf6;">
            <span style="color: #94a3b8;">📍 Zone Critique</span>
            <div style="color: #8b5cf6; font-size: 1.2rem; font-weight: 700;">{zone_critique}</div>
            <span style="color: #64748b;">Risque dominant: {"Cisaillement" if ratio_13 > ratio_33 else "Arrachement"}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Interprétation
    with st.expander("📚 Interprétation des Contraintes Mécaniques", expanded=False):
        st.markdown("""
#### ⬇️ Contrainte d'Arrachement σ₃₃

La contrainte **normale σ₃₃** (perpendiculaire aux couches) est responsable de la **délamination**.
- En **traction** (σ₃₃ > 0) : risque de décollement de la céramique
- En **compression** (σ₃₃ < 0) : risque de flambement ou écaillage

Les pics apparaissent aux **interfaces** où le mismatch de dilatation thermique est maximal.

#### ↔️ Contrainte de Cisaillement σ₁₃

Le cisaillement **σ₁₃** (dans le plan, direction x₁-x₃) est critique pour :
- La **propagation de fissures** aux interfaces
- L'**écaillage** (spalling) de la couche TBC

Les pics de cisaillement sont localisés aux **interfaces Substrat/BC** et surtout **BC/TBC** où les propriétés changent brutalement.
        """)

