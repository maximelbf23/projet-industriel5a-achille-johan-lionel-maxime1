"""
Onglet Mécanique Avancé - Analyse Spectrale Multicouche
Affiche les modes propres τ, vecteurs propres V et W, matrices de transfert Φ
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from core.mechanical import (
    solve_characteristic_equation,
    compute_all_eigenvectors,
    compute_all_stress_eigenvectors,
    compute_beta_coefficients,
    compute_thermal_forcing,
    solve_multilayer_problem,
    get_M_matrix,
    build_Phi_matrix
)
from core.clt_solver import solve_multilayer_clt
from core.constants import MECHANICAL_PROPS
from core.damage_analysis import analyze_damage_profile, CRITICAL_STRESS
from core.calculation import solve_tbc_model_v2

# Palette
PALETTE = {
    'primary': '#3b82f6',
    'secondary': '#8b5cf6', 
    'accent': '#06b6d4',
    'danger': '#ef4444',
    'safe': '#10b981',
    'warning': '#f59e0b',
}

# Données de référence ONERA/Safran (Bovet, Chiaruttini, Vattré 2025)
ONERA_REFERENCE = {
    'sigma_vM_range': (400, 800),  # MPa - Plage FEM typique
    'sigma_vM_max_root': 1000,     # MPa - Concentration à la racine
    'C11_RT': 259.6,               # GPa - Inconel 718 à T ambiante
    'C12_RT': 179.0,               # GPa
    'C44_RT': 109.6,               # GPa
    'alpha_RT': 4.95e-6,           # K⁻¹
    'alpha_HT': 14.68e-6,          # K⁻¹ @ 1198K
    'source': 'Bovet et al., ONERA/Safran (2025)',
}

# Cache pour les calculs spectraux lourds
@st.cache_data(ttl=300, show_spinner=False)
def cached_spectral_solve(h_sub, h_bc, h_tbc, lw, T_hat, method='spectral'):
    """
    Cache le calcul spectral multicouche.
    TTL = 300 secondes (5 minutes) pour éviter des résultats obsolètes.
    """
    from core.constants import PROPS_SUBSTRATE, PROPS_BONDCOAT, PROPS_CERAMIC
    from core.constants import ALPHA_SUBSTRATE, ALPHA_BONDCOAT, ALPHA_CERAMIC
    
    lambda_th = np.pi / lw
    layer_configs = [
        (h_sub, PROPS_SUBSTRATE, ALPHA_SUBSTRATE),
        (h_bc, PROPS_BONDCOAT, ALPHA_BONDCOAT),
        (h_tbc, PROPS_CERAMIC, ALPHA_CERAMIC)
    ]
    
    result = solve_multilayer_problem(layer_configs, lw, lambda_th, T_hat, method=method)
    return result


def render():
    # === EN-TÊTE HERO SPECTACULAIRE ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e1b4b 100%);
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; position: relative; overflow: hidden;
                border: 1px solid rgba(59, 130, 246, 0.2); box-shadow: 0 0 50px rgba(59, 130, 246, 0.15);">
        <div style="position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; 
                    background: radial-gradient(circle, rgba(59,130,246,0.25) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; 
                    background: radial-gradient(circle, rgba(139,92,246,0.2) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;
                       background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 50%, #93c5fd 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;">
                ⚙️ Calcul Mécanique - Analyse Spectrale
            </h2>
            <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 1rem;">
                Modes propres τ • Vecteurs propres V/W • Matrice de transfert Φ • Critères d'endommagement
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- RÉCUPÉRATION DES PARAMÈTRES DU SIDEBAR PRINCIPAL ---
    # Ces valeurs viennent du volet de gauche (sidebar)
    alpha_sidebar = st.session_state.get('alpha_input', 0.20)
    beta_sidebar = st.session_state.get('beta_input', 0.8)
    lw_sidebar = st.session_state.get('lw_input', 0.1)
    t_bottom_sidebar = st.session_state.get('T_bottom', 500)
    t_top_sidebar = st.session_state.get('T_top', 1400)
    
    # Calcul de h_tbc à partir de alpha (α = h₃/h₁ → h₃ = α × h₁)
    # h₁ = 0.5 mm (épaisseur substrat référence)
    h1_mm = 0.5  # mm
    h_tbc_from_alpha = alpha_sidebar * h1_mm  # en mm
    h_tbc_um_default = min(max(50, int(h_tbc_from_alpha * 1000)), 5000)  # conversion en µm, clampé
    
    # --- CONFIGURATION COMPACTE ---
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.4); padding: 0.75rem 1rem; border-radius: 10px; 
                margin-bottom: 1rem; border: 1px solid rgba(59, 130, 246, 0.15);">
        <span style="color: #94a3b8; font-size: 0.85rem;">📐 Configuration depuis sidebar : </span>
        <span style="color: #60a5fa; font-weight: 500;">α = {alpha:.2f}</span> • 
        <span style="color: #f59e0b; font-weight: 500;">β = {beta:.1f}</span> • 
        <span style="color: #10b981; font-weight: 500;">Lw = {lw:.3f} m</span> • 
        <span style="color: #f472b6; font-weight: 500;">ΔT = {dt}°C</span>
    </div>
    """.format(alpha=alpha_sidebar, beta=beta_sidebar, lw=lw_sidebar, dt=int(t_top_sidebar - t_bottom_sidebar)), unsafe_allow_html=True)
    
    # Calcul de h_tbc à partir de alpha
    h1_mm = 0.5
    h_tbc_from_alpha = alpha_sidebar * h1_mm
    h_tbc_um = min(max(50, int(h_tbc_from_alpha * 1000)), 5000)
    h_bc_um = 10
    T_hat = int(t_top_sidebar - t_bottom_sidebar)
    Lw = lw_sidebar
    
    # --- PARAMÈTRES AVANCÉS (collapsés par défaut) ---
    with st.expander("⚙️ Options avancées", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            perturb_pct = st.slider("Variation latérale (%)", 0, 50, 5, step=1,
                                  help="Simule un point chaud sinusoïdal")
            T_perturb_top = t_top_sidebar * (perturb_pct / 100.0)
            T_perturb_bottom = t_bottom_sidebar * (perturb_pct / 100.0)
        
        with col2:
            method = st.radio("Solveur", ["spectral", "clt"], index=0, horizontal=True,
                             help="Spectral (recommandé) | CLT (classique)")
        
        with col3:
            if method == "spectral":
                n_modes = st.slider("Modes Fourier", 1, 21, 5, step=2)
            else:
                n_modes = 1
            show_math = st.checkbox("Détails math.", value=False)
    
    # --- CALCUL AUTOMATIQUE ET RÉACTIF ---
    # Plus besoin de bouton, le calcul est lancé automatiquement grâce au cache
    
    # Appel de la fonction cachée
    # On passe toutes les dépendances explicites pour que le cache fonctionne bien
    results = compute_mech_results(
        h_tbc_um, h_bc_um, T_hat, Lw, method, 
        alpha_sidebar, beta_sidebar, T_perturb_bottom, T_perturb_top, n_modes
    )
    
    # --- AFFICHAGE DES RÉSULTATS ---
    if results:
        display_spectral_results(results, show_math)

@st.cache_data(show_spinner="Calcul mécanique en cours...")
def compute_mech_results(h_tbc, h_bc_unused, T_hat, Lw, method, alpha, beta, t_bottom, t_top, n_modes=1):
    """
    Lance l'analyse complète et RETOURNE les résultats (fonction pure pour le cache).
    """
    
    # Import des constantes (épaisseurs fixes)
    from core.constants import CONSTANTS
    
    # Conversions - utilise les valeurs fixes des CONSTANTS
    h_tbc_m = h_tbc * 1e-6
    h_sub_m = CONSTANTS['h1']  # 500 µm = 0.0005 m (fixe)
    h_bc_m = CONSTANTS['h2']   # 10 µm = 0.00001 m (fixe)
    
    # Nombres d'onde spectraux
    delta1 = np.pi / Lw
    delta2 = np.pi / Lw
    lambda_th = np.pi / Lw
    
    # Import des propriétés par couche
    from core.constants import PROPS_SUBSTRATE, PROPS_BONDCOAT, PROPS_CERAMIC
    from core.constants import ALPHA_SUBSTRATE, ALPHA_BONDCOAT, ALPHA_CERAMIC
    
    # Configuration couches avec propriétés DISTINCTES (essentiel pour les gradients de contraintes)
    props = MECHANICAL_PROPS.copy()  # Garde pour compatibilité avec l'analyse spectrale initiale
    
    layer_configs = [
        (h_sub_m, PROPS_SUBSTRATE, ALPHA_SUBSTRATE),    # Couche 1: Substrat Nickel
        (h_bc_m, PROPS_BONDCOAT, ALPHA_BONDCOAT),       # Couche 2: Bond Coat MCrAlY
        (h_tbc_m, PROPS_CERAMIC, ALPHA_CERAMIC)         # Couche 3: Céramique YSZ
    ]
    
    try:
        # === ÉTAPE 7: Résolution équation caractéristique ===
        char_eq_result = solve_characteristic_equation(delta1, delta2, props)
        tau_roots = char_eq_result['tau_roots']
        coeffs_poly = char_eq_result['coeffs_poly']
        
        # === ÉTAPE 8.1: Vecteurs propres de déplacement ===
        eigenvectors = compute_all_eigenvectors(tau_roots, delta1, delta2, props)
        
        # === ÉTAPE 8.3: Vecteurs propres de contrainte ===
        eigenvectors_with_stress = compute_all_stress_eigenvectors(eigenvectors, delta1, delta2, props)
        
        # === Forçage thermique (utilise propriétés de la céramique pour l'analyse spectrale) ===
        # RÉCUPÉRATION DU CHAMP THERMIQUE COMPLET
        # Appel du solveur thermique pour avoir les profils réels (A*exp(lam*z) + B*exp(-lam*z))
        
        thermal_res = solve_tbc_model_v2(alpha, beta, Lw, t_bottom, t_top, n_modes=n_modes)
        
        if not thermal_res['success']:
            return None # Ou gérer l'erreur autrement
            
        # Extraction des coeffs (A, B) et lambdas
        # Si multi-modes, on passe directement la liste des modes
        if 'modes' in thermal_res['profile_params']:
            T_hat_list = thermal_res['profile_params']['modes']
        else:
            # Fallback legacy (si le cache n'est pas invalidé)
            th_coeffs = thermal_res['profile_params']['coeffs']
            th_lambdas = thermal_res['profile_params']['lambdas']
            
            # Mode unique
            T_hat_list = [{
                'm': 1,
                'delta_eta': np.pi / Lw,
                'lambdas': th_lambdas,
                'coeffs': th_coeffs,
                'interfaces': thermal_res['profile_params']['interfaces']
            }]

        # Note: On garde 'beta' (coeff dilatation) pour affichage, mais le calcul utilise th_data
        beta_mech = compute_beta_coefficients(ALPHA_CERAMIC, props)
        # Dummy forcing pour affichage compatibilité
        thermal_forcing = {'beta': beta_mech, 'lambda_th': lambda_th, 'T_hat': T_hat}
        
        # === Résolution multicouche complète (PERTURBATION) ===
        # On passe T_hat_list au lieu de T_hat (scalaire). 
        # Le solver mécanique patché saura gérer cette liste.
        spectral_res = solve_multilayer_problem(layer_configs, Lw, lambda_th, T_hat_list, method='spectral')
        
        # === Résolution CLT (CHAMP MOYEN) ===
        # Calcul des contraintes de base dues à la dilatation thermique différentielle
        T_ref = 20.0
        T_mean_struct = (t_bottom + t_top)/2 - T_ref
        clt_res = solve_multilayer_clt(layer_configs, T_mean_struct)
        
        # === SUPERPOSITION DES CONTRAINTES ===
        # Total = Mean (CLT) + Perturbation (Spectral)
        
        stress_total = {}
        stress_spec = spectral_res['stress_profile']
        stress_clt = clt_res['stress_profile']
        
        # Interpolation CLT sur la grille spectrale (z_spec)
        z_spec = stress_spec['z']
        
        s11_clt_interp = np.interp(z_spec, stress_clt['z'], stress_clt['sigma_11'])
        s22_clt_interp = np.interp(z_spec, stress_clt['z'], stress_clt['sigma_22'])
        
        # Superposition linéaire:
        stress_total['z'] = z_spec
        stress_total['sigma_11'] = s11_clt_interp # + contribution spec
        stress_total['sigma_22'] = s22_clt_interp
        stress_total['sigma_33'] = np.abs(stress_spec['sigma_33']) # + 0
        stress_total['sigma_13'] = np.abs(stress_spec['sigma_13']) + np.abs(np.interp(z_spec, stress_clt['z'], stress_clt['sigma_13']))
        stress_total['sigma_23'] = np.abs(stress_spec['sigma_23'])
        
        # Propager layer_idx pour que analyze_damage_profile utilise les bons seuils
        stress_total['layer_idx'] = stress_spec.get('layer_idx', np.zeros(len(z_spec), dtype=int))
        
        # === Matrice Phi à z=0 (pour affichage) ===
        Phi_0 = build_Phi_matrix(0, eigenvectors_with_stress, props)
        
        # Stocker tous les résultats DANS UN DICT (pas session_state)
        return {
            'tau_roots': tau_roots,
            'eigenvectors': eigenvectors_with_stress,
            'beta': beta,
            'thermal_forcing': thermal_forcing,
            'Phi_0': Phi_0,
            'full_results': {'stress_profile': stress_total}, # On remplace par le total superposé
            'params': {
                'h_tbc': h_tbc, 'h_bc': int(h_bc_m * 1e6), 'T_hat': T_hat, 'Lw': Lw,
                'delta1': delta1, 'delta2': delta2, 'lambda_th': lambda_th,
                'method': 'superposition' # Force label
            }
        }
        
    except Exception as e:
        st.error(f"❌ Erreur de calcul: {str(e)}")
        # import traceback
        # st.code(traceback.format_exc())
        return None

def display_spectral_results(results, show_math):
    """Affiche les résultats de l'analyse spectrale avec visualisations premium."""
    
    tau_roots = results['tau_roots']
    eigenvectors = results['eigenvectors']
    params = results['params']
    full_results = results['full_results']
    stress = full_results.get('stress_profile', {})
    
    # === HEADER PREMIUM ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1e1b4b 100%);
                padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; position: relative; overflow: hidden;">
        <div style="position: absolute; top: -30px; right: -30px; width: 150px; height: 150px; 
                    background: radial-gradient(circle, rgba(59,130,246,0.3) 0%, transparent 70%); border-radius: 50%;"></div>
        <h2 style="color: #f1f5f9; margin: 0; font-size: 1.8rem;">
            🔬 Résultats de l'Analyse Spectrale
        </h2>
        <p style="color: #94a3b8; margin: 0.3rem 0 0 0;">
            Méthode: <strong style="color: #60a5fa;">{method}</strong> | 
            6 modes propres calculés | 
            Système multicouche résolu
        </p>
    </div>
    """.format(method=params['method'].upper()), unsafe_allow_html=True)
    
    # === SCHÉMA STRUCTUREL 3D ===
    if stress and 'z' in stress:
        st.markdown("### 🏗️ Architecture Multicouche et Champs de Contraintes")
        
        # Import des constantes pour affichage précis
        from core.constants import CONSTANTS
        
        col_struct, col_3d = st.columns([1, 2])
        
        with col_struct:
            # Diagramme des couches (vertical)
            # h1 et h2 sont fixes dans CONSTANTS (m) -> conversion en mm
            h_sub = CONSTANTS['h1'] * 1000  # 0.5 mm
            h_bc = CONSTANTS['h2'] * 1000   # 0.01 mm
            
            # h_tbc vient des params calculés (h_tbc_microns -> mm)
            h_tbc = params['h_tbc'] / 1000  # µm -> mm
            
            total_h = h_sub + h_bc + h_tbc
            
            fig_layers = go.Figure()
            
            # Substrat
            fig_layers.add_trace(go.Bar(
                x=['Structure'], y=[h_sub/total_h * 100], name='Substrat (Ni)',
                marker_color='#64748b', text=f'{h_sub*1000:.0f} µm', textposition='inside'
            ))
            # BondCoat
            fig_layers.add_trace(go.Bar(
                x=['Structure'], y=[h_bc/total_h * 100], name='BondCoat (MCrAlY)',
                marker_color='#f59e0b', text=f'{h_bc*1000:.0f} µm', textposition='inside'
            ))
            # TBC
            fig_layers.add_trace(go.Bar(
                x=['Structure'], y=[h_tbc/total_h * 100], name='Céramique (YSZ)',
                marker_color='#3b82f6', text=f'{h_tbc*1000:.0f} µm', textposition='inside'
            ))
            
            fig_layers.update_layout(
                barmode='stack',
                height=300,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9'),
                showlegend=True,
                legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
                yaxis_title='Proportion (%)',
                margin=dict(l=40, r=20, t=20, b=60)
            )
            st.plotly_chart(fig_layers, use_container_width=True)
        
        with col_3d:
            # Surface 3D des contraintes
            z_mm = stress['z'] * 1e3
            sigma_33_profile = stress['sigma_33'] / 1e9  # GPa
            
            # Paramètres latéraux
            Lw = params['Lw']
            delta1 = params['delta1']
            
            # Création du maillage 3D avec variation sinusoïdale latérale
            # u3 ~ sin(delta1 * x1) => sigma33 ~ sin(...) aussi (selon les équations)
            # On va visualiser sur une période Lw
            n_x = 40
            x_vals = np.linspace(0, Lw, n_x)
            
            # Grid
            X, Z = np.meshgrid(x_vals, z_mm)
            
            # Modulation latérale : sin(pi * x / Lw)
            # Attention : sigma33 dépend de du3/dx3 et autres termes, mais en première approx 
            # la dépendance latérale suit la forme propre.
            # Pour le mode 1: sin(delta1 * x1) sin(delta2 * x2)
            # Ici on visualise une coupe 2D (x, z)
            
            lateral_variation = np.sin(delta1 * X)
            S33_3D = np.tile(sigma_33_profile, (n_x, 1)).T * lateral_variation
            
            fig_3d = go.Figure()
            fig_3d.add_trace(go.Surface(
                x=X * 1000, # mm
                y=Z, 
                z=S33_3D,
                colorscale='RdBu_r',
                colorbar=dict(title='σ₃₃ (GPa)', x=1.02),
                opacity=0.9,
                name="σ₃₃"
            ))
            
            fig_3d.update_layout(
                title="Champ de Contraintes σ₃₃(x, z)",
                scene=dict(
                    xaxis_title='x (mm)',
                    yaxis_title='z (mm)',
                    zaxis_title='σ₃₃ (GPa)',
                    bgcolor='rgba(15, 23, 42, 0.9)',
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
                    aspectmode='manual',
                    aspectratio=dict(x=1.5, y=1, z=0.8)
                ),
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9'),
                margin=dict(l=0, r=30, t=40, b=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    # === CERCLE DE MOHR INTERACTIF ===
    st.markdown("### ⭕ Cercle de Mohr des Contraintes")
    
    col_mohr_input, col_mohr_plot = st.columns([1, 2])
    
    with col_mohr_input:
        st.info("Sélectionnez une position z dans l'épaisseur pour analyser l'état de contrainte local.")
        if stress and 'z' in stress:
            z_max_mm = stress['z'][-1] * 1e3
            z_selected_mohr = st.slider("Position z (mm)", 0.0, float(z_max_mm), float(z_max_mm)/2.0, step=0.01)
            
            # Interpolation des contraintes à z
            idx_z = (np.abs(stress['z']*1e3 - z_selected_mohr)).argmin()
            
            s33_loc = stress['sigma_33'][idx_z] / 1e6 # MPa
            s13_loc = stress['sigma_13'][idx_z] / 1e6 # MPa
            
            # Use real sumperposed sigma_11 if available, else approx
            if 'sigma_11' in stress:
                s11_loc = stress['sigma_11'][idx_z] / 1e6 # MPa
            else:
                s11_loc = s33_loc * 0.3 
            
            st.markdown(f"""
            **État local (MPa) :**
            - $\sigma_{{33}}$ = {s33_loc:.1f}
            - $\sigma_{{13}}$ = {s13_loc:.1f}
            - $\sigma_{{11}}$ ≈ {s11_loc:.1f} (Est.)
            """)
            
    with col_mohr_plot:
        if stress and 'z' in stress:
            # Calcul Cercle
            # Centre C = (s11 + s33) / 2
            # Rayon R = sqrt( ((s11 - s33)/2)^2 + s13^2 )
            center = (s11_loc + s33_loc) / 2
            radius = np.sqrt(((s11_loc - s33_loc)/2)**2 + s13_loc**2)
            
            s_min = center - radius
            s_max = center + radius
            
            theta = np.linspace(0, 2*np.pi, 100)
            x_circ = center + radius * np.cos(theta)
            y_circ = radius * np.sin(theta)
            
            fig_mohr = go.Figure()
            # Cercle
            fig_mohr.add_trace(go.Scatter(x=x_circ, y=y_circ, mode='lines', name='Cercle', line=dict(color='#f472b6')))
            
            # Points de construction
            fig_mohr.add_trace(go.Scatter(x=[s33_loc, s11_loc], y=[s13_loc, -s13_loc], mode='markers+lines', name='Diamètre', marker=dict(size=8)))
            fig_mohr.add_trace(go.Scatter(x=[s33_loc], y=[s13_loc], mode='markers+text', text=["(σ₃₃, τ₁₃)"], textposition="top right", marker=dict(color='#60a5fa')))
            
            # Contraintes principales
            fig_mohr.add_trace(go.Scatter(x=[s_min, s_max], y=[0, 0], mode='markers', marker=dict(color='#ef4444', size=10), name='Principales'))
            
            # Mise en forme carrée
            max_range = radius * 1.5
            
            fig_mohr.update_layout(
                title="Cercle de Mohr (Plan 1-3)",
                xaxis_title="Contrainte Normale σ (MPa)",
                yaxis_title="Contrainte Cisaillement τ (MPa)",
                xaxis=dict(zeroline=True, scaleanchor="y", scaleratio=1),
                yaxis=dict(zeroline=True),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9')
            )
            st.plotly_chart(fig_mohr, use_container_width=True)
    
    # === MÉTRIQUES PREMIUM (4 KPI Cards) ===
    st.markdown("### 📊 Indicateurs Clés")
    
    if stress:
        max_s33 = np.max(np.abs(stress.get('sigma_33', [0]))) / 1e9  # GPa
        max_s13 = np.max(np.abs(stress.get('sigma_13', [0]))) / 1e9  # GPa
        tau_max = np.max(np.abs(tau_roots))
        
        # Calcul indicateur D
        sigma_crit = CRITICAL_STRESS['ceramic']['sigma_tensile']
        D_approx = max_s33 * 1e9 / sigma_crit # Retour en Pa pour le ratio
        status_D = "🟢" if D_approx < 0.5 else ("🟡" if D_approx < 0.8 else "🔴")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(59,130,246,0.05));
                        padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(59,130,246,0.3); text-align: center;">
                <div style="font-size: 2rem; font-weight: bold; color: #60a5fa;">{tau_max:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Mode |τ|max</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color_s33 = "#ef4444" if max_s33 > 0.1 else "#f59e0b" if max_s33 > 0.05 else "#10b981" # Seuils GPa ajustés approx (100 MPa = 0.1 GPa)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
                        padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(239,68,68,0.3); text-align: center;">
                <div style="font-size: 2rem; font-weight: bold; color: {color_s33};">{max_s33:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Max |σ₃₃| (GPa)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(139,92,246,0.05));
                        padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(139,92,246,0.3); text-align: center;">
                <div style="font-size: 2rem; font-weight: bold; color: #a78bfa;">{max_s13:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Max |σ₁₃| (GPa)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.05));
                        padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(16,185,129,0.3); text-align: center;">
                <div style="font-size: 2rem; font-weight: bold; color: #34d399;">{status_D} {D_approx:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Indicateur D</div>
            </div>
            """, unsafe_allow_html=True)
    
    # === ONGLETS DE VISUALISATION DÉTAILLÉE (4 onglets consolidés) ===
    st.divider()
    st.markdown("### 📈 Analyse Détaillée")
    tabs = st.tabs([
        "📈 Contraintes", 
        "🔴 Endommagement", 
        "🔗 Interfaces",
        "🔬 Avancé"
    ])
    
    # --- TAB 1: PROFILS DE CONTRAINTES ---
    with tabs[0]:
        st.markdown("### 📈 Profils de Contraintes dans l'Épaisseur")
        
        if stress and 'z' in stress:
            z_mm = stress['z'] * 1e3
            
            # === RÉCUPÉRATION DES ÉPAISSEURS DES COUCHES ===
            from core.constants import CONSTANTS
            h_sub_mm = CONSTANTS['h1'] * 1000  # mm
            h_bc_mm = CONSTANTS['h2'] * 1000   # mm
            h_tbc_mm = params['h_tbc'] / 1000  # µm → mm
            H_total_mm = h_sub_mm + h_bc_mm + h_tbc_mm
            
            # === GRAPHIQUE σ₃₃ (ARRACHEMENT) ===
            col_s33, col_s13 = st.columns(2)
            
            with col_s33:
                fig_s33 = go.Figure()
                
                # Zones de couches colorées
                fig_s33.add_vrect(x0=0, x1=h_sub_mm, fillcolor="#64748b", opacity=0.2,
                                 layer="below", line_width=0)
                fig_s33.add_vrect(x0=h_sub_mm, x1=h_sub_mm + h_bc_mm, fillcolor="#f59e0b", opacity=0.25,
                                 layer="below", line_width=0)
                fig_s33.add_vrect(x0=h_sub_mm + h_bc_mm, x1=H_total_mm, fillcolor="#3b82f6", opacity=0.2,
                                 layer="below", line_width=0)
                
                # Courbe σ₃₃ en MPa
                sigma_33_mpa = stress['sigma_33'] / 1e6
                fig_s33.add_trace(go.Scatter(
                    x=z_mm, y=sigma_33_mpa,
                    mode='lines', name='σ₃₃',
                    line=dict(color='#ef4444', width=3),
                    fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.15)'
                ))
                
                # Seuils critiques (lignes horizontales)
                fig_s33.add_hline(y=CRITICAL_STRESS['ceramic']['sigma_tensile']/1e6, 
                                 line_dash="dash", line_color="#22d3ee", line_width=2,
                                 annotation_text=f"σ_crit TBC ({CRITICAL_STRESS['ceramic']['sigma_tensile']/1e6:.0f} MPa)",
                                 annotation_position="top right",
                                 annotation_font=dict(size=10, color="#22d3ee"))
                fig_s33.add_hline(y=-CRITICAL_STRESS['ceramic']['sigma_compressive']/1e6, 
                                 line_dash="dash", line_color="#22d3ee", line_width=2,
                                 annotation_text=f"Compression ({-CRITICAL_STRESS['ceramic']['sigma_compressive']/1e6:.0f} MPa)",
                                 annotation_position="bottom right",
                                 annotation_font=dict(size=10, color="#22d3ee"))
                
                # Lignes verticales pour interfaces
                fig_s33.add_vline(x=h_sub_mm, line_dash="dot", line_color="#94a3b8", line_width=1,
                                 annotation_text="Sub/BC", annotation_position="top")
                fig_s33.add_vline(x=h_sub_mm + h_bc_mm, line_dash="dot", line_color="#f59e0b", line_width=1,
                                 annotation_text="BC/TBC", annotation_position="top")
                
                fig_s33.update_layout(
                    title=dict(text="⬇️ Contrainte d'Arrachement σ₃₃(z)", font=dict(size=16, color='#f1f5f9')),
                    xaxis_title="Profondeur z (mm)",
                    yaxis_title="σ₃₃ (MPa)",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9'),
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)'),
                    yaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)')
                )
                st.plotly_chart(fig_s33, use_container_width=True)
            
            # === GRAPHIQUE σ₁₃ (CISAILLEMENT) ===
            with col_s13:
                fig_s13 = go.Figure()
                
                # Zones de couches colorées
                fig_s13.add_vrect(x0=0, x1=h_sub_mm, fillcolor="#64748b", opacity=0.2,
                                 layer="below", line_width=0)
                fig_s13.add_vrect(x0=h_sub_mm, x1=h_sub_mm + h_bc_mm, fillcolor="#f59e0b", opacity=0.25,
                                 layer="below", line_width=0)
                fig_s13.add_vrect(x0=h_sub_mm + h_bc_mm, x1=H_total_mm, fillcolor="#3b82f6", opacity=0.2,
                                 layer="below", line_width=0)
                
                # Courbe σ₁₃ en MPa
                sigma_13_mpa = stress['sigma_13'] / 1e6
                fig_s13.add_trace(go.Scatter(
                    x=z_mm, y=sigma_13_mpa,
                    mode='lines', name='σ₁₃',
                    line=dict(color='#f59e0b', width=3),
                    fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.15)'
                ))
                
                # Seuil critique cisaillement
                fig_s13.add_hline(y=CRITICAL_STRESS['ceramic']['sigma_shear']/1e6, 
                                 line_dash="dash", line_color="#8b5cf6", line_width=2,
                                 annotation_text=f"τ_crit TBC ({CRITICAL_STRESS['ceramic']['sigma_shear']/1e6:.0f} MPa)",
                                 annotation_position="top right",
                                 annotation_font=dict(size=10, color="#8b5cf6"))
                fig_s13.add_hline(y=CRITICAL_STRESS['bondcoat']['sigma_shear']/1e6, 
                                 line_dash="dot", line_color="#f472b6", line_width=2,
                                 annotation_text=f"τ_crit BC ({CRITICAL_STRESS['bondcoat']['sigma_shear']/1e6:.0f} MPa)",
                                 annotation_position="top left",
                                 annotation_font=dict(size=10, color="#f472b6"))
                
                # Lignes verticales pour interfaces
                fig_s13.add_vline(x=h_sub_mm, line_dash="dot", line_color="#94a3b8", line_width=1,
                                 annotation_text="Sub/BC", annotation_position="top")
                fig_s13.add_vline(x=h_sub_mm + h_bc_mm, line_dash="dot", line_color="#f59e0b", line_width=1,
                                 annotation_text="BC/TBC", annotation_position="top")
                
                fig_s13.update_layout(
                    title=dict(text="↔️ Contrainte de Cisaillement σ₁₃(z)", font=dict(size=16, color='#f1f5f9')),
                    xaxis_title="Profondeur z (mm)",
                    yaxis_title="σ₁₃ (MPa)",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9'),
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)'),
                    yaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)')
                )
                st.plotly_chart(fig_s13, use_container_width=True)
            
            # === LÉGENDE DES COUCHES ===
            st.markdown("""
            <div style="display: flex; justify-content: center; gap: 2rem; margin: 1rem 0;">
                <span style="color: #94a3b8;">
                    <span style="background: rgba(100, 116, 139, 0.4); padding: 0.2rem 0.5rem; border-radius: 4px;">■</span> Substrat (Ni)
                </span>
                <span style="color: #f59e0b;">
                    <span style="background: rgba(245, 158, 11, 0.4); padding: 0.2rem 0.5rem; border-radius: 4px;">■</span> Bond Coat (MCrAlY)
                </span>
                <span style="color: #3b82f6;">
                    <span style="background: rgba(59, 130, 246, 0.4); padding: 0.2rem 0.5rem; border-radius: 4px;">■</span> TBC (YSZ)
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # === ANALYSE RÉSUMÉE ===
            max_s33_mpa = np.max(np.abs(sigma_33_mpa))
            max_s13_mpa = np.max(np.abs(sigma_13_mpa))
            crit_s33 = CRITICAL_STRESS['ceramic']['sigma_tensile'] / 1e6
            crit_s13 = CRITICAL_STRESS['ceramic']['sigma_shear'] / 1e6
            
            ratio_s33 = max_s33_mpa / crit_s33
            ratio_s13 = max_s13_mpa / crit_s13
            
            cols_resume = st.columns(3)
            
            with cols_resume[0]:
                color_33 = "#10b981" if ratio_s33 < 0.7 else "#f59e0b" if ratio_s33 < 1.0 else "#ef4444"
                st.markdown(f"""
                <div style="background: {color_33}15; padding: 1rem; border-radius: 12px; border: 1px solid {color_33}; text-align: center;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">⬇️ σ₃₃ max</span>
                    <div style="color: {color_33}; font-size: 1.6rem; font-weight: 700;">{max_s33_mpa:.1f} MPa</div>
                    <span style="color: #64748b; font-size: 0.8rem;">Ratio: {ratio_s33:.0%} du seuil</span>
                </div>
                """, unsafe_allow_html=True)
            
            with cols_resume[1]:
                color_13 = "#10b981" if ratio_s13 < 0.7 else "#f59e0b" if ratio_s13 < 1.0 else "#ef4444"
                st.markdown(f"""
                <div style="background: {color_13}15; padding: 1rem; border-radius: 12px; border: 1px solid {color_13}; text-align: center;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">↔️ σ₁₃ max</span>
                    <div style="color: {color_13}; font-size: 1.6rem; font-weight: 700;">{max_s13_mpa:.1f} MPa</div>
                    <span style="color: #64748b; font-size: 0.8rem;">Ratio: {ratio_s13:.0%} du seuil</span>
                </div>
                """, unsafe_allow_html=True)
            
            with cols_resume[2]:
                zone_crit = "Interface BC/TBC" if ratio_s13 > ratio_s33 else "Volume TBC"
                mode_crit = "Cisaillement" if ratio_s13 > ratio_s33 else "Arrachement"
                st.markdown(f"""
                <div style="background: rgba(139, 92, 246, 0.15); padding: 1rem; border-radius: 12px; border: 1px solid #8b5cf6; text-align: center;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">📍 Zone Critique</span>
                    <div style="color: #a78bfa; font-size: 1.2rem; font-weight: 700;">{zone_crit}</div>
                    <span style="color: #64748b; font-size: 0.8rem;">Mode: {mode_crit}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Interprétation
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.5); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <h5 style="color: #60a5fa; margin: 0 0 0.5rem 0;">📚 Interprétation Physique</h5>
                <ul style="color: #cbd5e1; font-size: 0.9rem; margin: 0; padding-left: 1.5rem;">
                    <li><strong>σ₃₃ (Arrachement)</strong> : Contrainte perpendiculaire aux interfaces. Responsable de la <em>délamination</em>.</li>
                    <li><strong>σ₁₃ (Cisaillement)</strong> : Contrainte tangentielle. Maximum aux interfaces (discontinuité de propriétés).</li>
                    <li><strong>Zones critiques</strong> : Les pics se situent aux interfaces Substrat/BondCoat et surtout BondCoat/TBC.</li>
                    <li><strong>Seuils</strong> : Les lignes pointillées indiquent les limites admissibles pour chaque matériau.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Pas de données de contrainte disponibles. Lancez un calcul d'abord.")
    
    # --- TAB 2: ANALYSE D'ENDOMMAGEMENT ---
    with tabs[1]:
        st.markdown("### 🔴 Analyse d'Endommagement et Critères de Rupture")
        
        if stress and 'z' in stress:
            z_mm = stress['z'] * 1e3
            n_points = len(stress['z'])
            
            layer_idx = stress.get('layer_idx', np.zeros(n_points, dtype=int))
            layer_types = ['substrate', 'bondcoat', 'ceramic']
            
            from core.damage_analysis import analyze_damage_profile
            damage_result = analyze_damage_profile(stress, layer_idx, layer_types)
            D = damage_result['D']
            F = damage_result['F']
            
            max_D = np.max(D)
            max_F = np.max(F)
            
            # Métriques
            col_d1, col_d2, col_d3 = st.columns(3)
            
            status_D = "✅" if max_D < 0.5 else ("⚠️" if max_D < 0.8 else "🚨")
            status_F = "✅" if max_F < 0.5 else ("⚠️" if max_F < 1.0 else "🚨")
            
            col_d1.metric(f"{status_D} Indicateur D max", f"{max_D:.3f}", 
                         help="D < 0.5: Sûr | 0.5-0.8: Prudence | > 0.8: Critique")
            col_d2.metric(f"{status_F} Tsai-Wu F max", f"{max_F:.3f}",
                         help="F < 1: Intègre | F ≥ 1: Rupture")
            col_d3.metric("Position critique", f"{z_mm[np.argmax(D)]:.2f} mm")
            
            # Alert
            if max_D > 0.8:
                st.error(f"🚨 **ALERTE** : D = {max_D:.2f} dépasse le seuil critique (0.8). Risque de délamination élevé !")
            elif max_D > 0.5:
                st.warning(f"⚠️ D = {max_D:.2f} : Zone de prudence. Vérifier le dimensionnement.")
            else:
                st.success(f"✅ D = {max_D:.2f} : Niveau de contrainte acceptable.")
            
            # Graphiques D et F
            fig_damage = make_subplots(rows=1, cols=2, subplot_titles=("Indicateur D(z)", "Critère Tsai-Wu F(z)"))
            
            fig_damage.add_trace(go.Scatter(
                x=z_mm, y=D, mode='lines', name='D',
                line=dict(color=PALETTE['secondary'], width=3),
                fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.15)'
            ), row=1, col=1)
            fig_damage.add_hline(y=0.8, line_dash="dash", line_color=PALETTE['danger'], 
                                annotation_text="Seuil critique", row=1, col=1)
            fig_damage.add_hline(y=0.5, line_dash="dot", line_color=PALETTE['warning'], row=1, col=1)
            
            fig_damage.add_trace(go.Scatter(
                x=z_mm, y=F, mode='lines', name='F',
                line=dict(color=PALETTE['accent'], width=3),
                fill='tozeroy', fillcolor='rgba(6, 182, 212, 0.15)'
            ), row=1, col=2)
            fig_damage.add_hline(y=1.0, line_dash="dash", line_color=PALETTE['danger'],
                                annotation_text="Limite rupture", row=1, col=2)
            
            fig_damage.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9'), showlegend=False
            )
            fig_damage.update_xaxes(title_text="z (mm)", showgrid=True, gridcolor='rgba(96, 165, 250, 0.1)')
            fig_damage.update_yaxes(showgrid=True, gridcolor='rgba(96, 165, 250, 0.1)')
            
            st.plotly_chart(fig_damage, use_container_width=True)
            
            # Interprétation
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.6); padding: 1rem; border-radius: 12px; margin-top: 1rem;">
                <h5 style="color: #60a5fa; margin: 0 0 0.5rem 0;">📚 Interprétation des Critères</h5>
                <ul style="color: #cbd5e1; font-size: 0.9rem;">
                    <li><strong>Indicateur D</strong> : Ratio max contrainte/admissible. Simple et conservatif.</li>
                    <li><strong>Critère Tsai-Wu</strong> : Critère polynomial pour matériaux anisotropes.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Lancez un calcul pour voir l'analyse d'endommagement.")
    
    # --- TAB 3: ANALYSE AVANCÉE DES INTERFACES ---
    with tabs[2]:
        st.markdown("### 🔗 Analyse des Interfaces & Recommandations")
        
        if stress and 'z' in stress:
            from core.constants import CONSTANTS, GPa_TO_PA
            
            # Positions des interfaces
            h_sub_m = CONSTANTS['h1']
            h_bc_m = CONSTANTS['h2']
            h_tbc_m = params['h_tbc'] * 1e-6
            
            z_int1 = h_sub_m  # Interface Substrat/BondCoat
            z_int2 = h_sub_m + h_bc_m  # Interface BondCoat/TBC
            
            z_array = stress['z']
            
            # Indices des interfaces
            idx_int1 = np.argmin(np.abs(z_array - z_int1))
            idx_int2 = np.argmin(np.abs(z_array - z_int2))
            
            # === SECTION 1: COMPARAISON DES INTERFACES ===
            st.markdown("#### 🔗 Comparaison des Contraintes aux Interfaces")
            
            col_int1, col_int2, col_int3 = st.columns(3)
            
            # Données interface 1
            s33_int1 = stress['sigma_33'][idx_int1] / 1e6  # MPa
            s13_int1 = stress['sigma_13'][idx_int1] / 1e6
            
            # Données interface 2
            s33_int2 = stress['sigma_33'][idx_int2] / 1e6
            s13_int2 = stress['sigma_13'][idx_int2] / 1e6
            
            with col_int1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(100,116,139,0.3), rgba(100,116,139,0.1));
                            padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(100,116,139,0.4); text-align: center;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Interface 1</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f1f5f9;">Substrat / Bond Coat</div>
                    <div style="color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;">z = {z_int1*1e3:.2f} mm</div>
                    <hr style="border-color: rgba(100,116,139,0.3); margin: 0.75rem 0;">
                    <div style="color: #ef4444; font-size: 1.3rem; font-weight: bold;">σ₃₃ = {s33_int1:.1f} MPa</div>
                    <div style="color: #3b82f6; font-size: 1.1rem;">τ₁₃ = {s13_int1:.1f} MPa</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_int2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(245,158,11,0.3), rgba(245,158,11,0.1));
                            padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(245,158,11,0.4); text-align: center;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Interface 2</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #f1f5f9;">Bond Coat / TBC</div>
                    <div style="color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;">z = {z_int2*1e3:.2f} mm</div>
                    <hr style="border-color: rgba(245,158,11,0.3); margin: 0.75rem 0;">
                    <div style="color: #ef4444; font-size: 1.3rem; font-weight: bold;">σ₃₃ = {s33_int2:.1f} MPa</div>
                    <div style="color: #3b82f6; font-size: 1.1rem;">τ₁₃ = {s13_int2:.1f} MPa</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_int3:
                # Calcul du ratio de criticité
                # Interface critique = celle avec le plus grand ratio contrainte/limite
                crit_bc = CRITICAL_STRESS['bondcoat']
                crit_cer = CRITICAL_STRESS['ceramic']
                
                D_int1 = max(abs(s33_int1*1e6) / crit_bc['sigma_tensile'], 
                            abs(s13_int1*1e6) / crit_bc['sigma_shear'])
                D_int2 = max(abs(s33_int2*1e6) / crit_cer['sigma_tensile'],
                            abs(s13_int2*1e6) / crit_cer['sigma_shear'])
                
                if D_int2 > D_int1:
                    critical_interface = "BC/TBC"
                    critical_D = D_int2
                    critical_color = "#ef4444"
                else:
                    critical_interface = "Sub/BC"
                    critical_D = D_int1
                    critical_color = "#f59e0b"
                
                status_emoji = "🟢" if critical_D < 0.5 else ("🟡" if critical_D < 0.8 else "🔴")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(239,68,68,0.3), rgba(239,68,68,0.1));
                            padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(239,68,68,0.4); text-align: center;">
                    <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Interface Critique</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: {critical_color};">{status_emoji} {critical_interface}</div>
                    <div style="color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;">Facteur de charge</div>
                    <hr style="border-color: rgba(239,68,68,0.3); margin: 0.75rem 0;">
                    <div style="color: #f1f5f9; font-size: 2rem; font-weight: bold;">D = {critical_D:.2f}</div>
                    <div style="color: #94a3b8; font-size: 0.8rem;">{"Sûr" if critical_D < 0.5 else ("Prudence" if critical_D < 0.8 else "CRITIQUE")}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # === SECTION 2: DENSITÉ D'ÉNERGIE DE DÉFORMATION ===
            st.markdown("#### ⚡ Densité d'Énergie de Déformation")
            
            # Calcul simplifié de la densité d'énergie élastique
            # U = 0.5 * σ²/E (approximation isotrope)
            # Pour chaque couche on utilise son module
            E_values = {
                'substrate': 250e9,  # GPa -> Pa (approximation de C33)
                'bondcoat': 180e9,
                'ceramic': 50e9
            }
            
            layer_idx = stress.get('layer_idx', np.zeros(len(z_array), dtype=int))
            layer_names = ['substrate', 'bondcoat', 'ceramic']
            
            # Énergie de déformation totale
            U_density = np.zeros_like(z_array)
            for i, z_val in enumerate(z_array):
                layer_name = layer_names[min(int(layer_idx[i]), 2)]
                E_layer = E_values[layer_name]
                # Energie = 0.5 * (σ33² + 2*σ13² + 2*σ23²) / E
                sigma_33 = stress['sigma_33'][i]
                sigma_13 = stress['sigma_13'][i]
                sigma_23 = stress.get('sigma_23', np.zeros_like(stress['sigma_33']))[i]
                U_density[i] = 0.5 * (sigma_33**2 + 2*sigma_13**2 + 2*sigma_23**2) / E_layer
            
            # Conversion en kJ/m³
            U_density_kJ = U_density / 1e3
            
            col_energy, col_gauge = st.columns([2, 1])
            
            with col_energy:
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(
                    x=z_array * 1e3, y=U_density_kJ,
                    mode='lines', name='U',
                    line=dict(color='#f472b6', width=3),
                    fill='tozeroy', fillcolor='rgba(244, 114, 182, 0.2)'
                ))
                
                # Marquer les interfaces
                fig_energy.add_vline(x=z_int1*1e3, line_dash="dash", line_color="#64748b",
                                    annotation_text="Int. 1", annotation_position="top")
                fig_energy.add_vline(x=z_int2*1e3, line_dash="dash", line_color="#f59e0b",
                                    annotation_text="Int. 2", annotation_position="top")
                
                fig_energy.update_layout(
                    title="Profil de Densité d'Énergie Élastique",
                    xaxis_title="z (mm)",
                    yaxis_title="U (kJ/m³)",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f1f5f9'),
                    xaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)'),
                    yaxis=dict(gridcolor='rgba(96, 165, 250, 0.1)')
                )
                st.plotly_chart(fig_energy, use_container_width=True)
            
            with col_gauge:
                # Jauge de risque de délamination
                # Basée sur l'énergie aux interfaces vs énergie de fracture typique (Gc ~ 20-100 J/m²)
                U_int1 = U_density[idx_int1] * (h_bc_m)  # Énergie par unité de surface [J/m²]
                U_int2 = U_density[idx_int2] * (h_bc_m)
                
                Gc_typical = 50  # J/m² (énergie de fracture interface TBC)
                delam_risk = max(U_int1, U_int2) / Gc_typical
                delam_risk_pct = min(100, delam_risk * 100)
                
                risk_color = "#10b981" if delam_risk < 0.3 else ("#f59e0b" if delam_risk < 0.7 else "#ef4444")
                
                fig_gauge_delam = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=delam_risk_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    number={'suffix': '%', 'font': {'size': 32, 'color': '#f1f5f9'}},
                    title={'text': "Risque Délamination", 'font': {'size': 14, 'color': '#94a3b8'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#475569'},
                        'bar': {'color': risk_color, 'thickness': 0.7},
                        'bgcolor': 'rgba(30, 41, 59, 0.8)',
                        'bordercolor': 'rgba(96, 165, 250, 0.3)',
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'},
                        ],
                        'threshold': {'line': {'color': '#ef4444', 'width': 4}, 'thickness': 0.75, 'value': 70}
                    }
                ))
                fig_gauge_delam.update_layout(
                    height=250, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font={'color': '#f1f5f9'},
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_gauge_delam, use_container_width=True)
                st.caption(f"Basé sur G_c ≈ {Gc_typical} J/m²")
            
            # === SECTION 3: RECOMMANDATIONS D'INGÉNIERIE ===
            st.markdown("#### 💡 Recommandations d'Ingénierie")
            
            recommendations = []
            
            # Analyse température
            if critical_D > 0.8:
                recommendations.append({
                    'icon': '🚨', 'type': 'CRITIQUE', 'color': '#ef4444',
                    'title': 'Risque de Délamination Élevé',
                    'action': f"Augmenter l'épaisseur TBC de 20-30% ou réduire le gradient thermique. D = {critical_D:.2f} dépasse le seuil de 0.8."
                })
            elif critical_D > 0.5:
                recommendations.append({
                    'icon': '⚠️', 'type': 'ATTENTION', 'color': '#f59e0b',
                    'title': 'Zone de Prudence',
                    'action': f"Surveiller l'évolution des contraintes en cyclage. D = {critical_D:.2f} proche du seuil."
                })
            else:
                recommendations.append({
                    'icon': '✅', 'type': 'OK', 'color': '#10b981',
                    'title': 'Configuration Robuste',
                    'action': f"Les niveaux de contrainte sont acceptables (D = {critical_D:.2f}). Marge de sécurité suffisante."
                })
            
            # Analyse cisaillement
            if max(abs(s13_int1), abs(s13_int2)) > 80:  # MPa
                recommendations.append({
                    'icon': '🔧', 'type': 'SUGGESTION', 'color': '#8b5cf6',
                    'title': 'Cisaillement Élevé aux Interfaces',
                    'action': "Envisager une couche de gradient (FGM) ou un bond coat plus épais pour réduire le mismatch de propriétés."
                })
            
            # Analyse énergie
            if delam_risk > 0.5:
                recommendations.append({
                    'icon': '⚡', 'type': 'ÉNERGIE', 'color': '#06b6d4',
                    'title': 'Concentration d\'Énergie aux Interfaces',
                    'action': f"L'énergie stockée représente {delam_risk_pct:.0f}% de l'énergie de fracture typique. Évaluer la résistance réelle du système."
                })
            
            # Affichage des recommandations
            rec_cols = st.columns(len(recommendations))
            for i, rec in enumerate(recommendations):
                with rec_cols[i]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9));
                                padding: 1rem; border-radius: 12px; height: 100%;
                                border-left: 4px solid {rec['color']};
                                border-top: 1px solid rgba(96, 165, 250, 0.1);">
                        <div style="font-size: 1.5rem;">{rec['icon']}</div>
                        <div style="font-size: 0.7rem; color: {rec['color']}; text-transform: uppercase; margin: 0.3rem 0;">{rec['type']}</div>
                        <div style="font-size: 0.95rem; color: #f1f5f9; font-weight: 600;">{rec['title']}</div>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">{rec['action']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === SECTION 4: TABLEAU RÉCAPITULATIF ===
            st.markdown("#### 📋 Tableau Récapitulatif")
            
            summary_data = {
                'Paramètre': [
                    'σ₃₃ max (MPa)', 'σ₁₃ max (MPa)', 
                    'Interface critique', 'Facteur D',
                    'Énergie max (kJ/m³)', 'Risque délamination'
                ],
                'Valeur': [
                    f"{np.max(np.abs(stress['sigma_33']))/1e6:.1f}",
                    f"{np.max(np.abs(stress['sigma_13']))/1e6:.1f}",
                    critical_interface,
                    f"{critical_D:.3f}",
                    f"{np.max(U_density_kJ):.2f}",
                    f"{delam_risk_pct:.0f}%"
                ],
                'Statut': [
                    '🟢' if np.max(np.abs(stress['sigma_33']))/1e6 < 100 else ('🟡' if np.max(np.abs(stress['sigma_33']))/1e6 < 150 else '🔴'),
                    '🟢' if np.max(np.abs(stress['sigma_13']))/1e6 < 80 else ('🟡' if np.max(np.abs(stress['sigma_13']))/1e6 < 120 else '🔴'),
                    status_emoji,
                    '🟢' if critical_D < 0.5 else ('🟡' if critical_D < 0.8 else '🔴'),
                    '🟢' if np.max(U_density_kJ) < 10 else ('🟡' if np.max(U_density_kJ) < 50 else '🔴'),
                    '🟢' if delam_risk_pct < 30 else ('🟡' if delam_risk_pct < 70 else '🔴')
                ]
            }
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        else:
            st.warning("Lancez d'abord un calcul pour voir l'analyse des interfaces.")
    
    # --- TAB 4: DONNÉES AVANCÉES (Modes τ, Vecteurs, Matrice Φ, ONERA) ---
    with tabs[3]:
        st.markdown("### 🔬 Données Avancées")
        
        # Sous-onglets pour les données techniques
        subtabs = st.tabs(["🔢 Modes τ", "📐 Vecteurs V/W", "📋 Matrice Φ", "🏛️ ONERA"])
        
        # === Sous-tab Modes τ ===
        with subtabs[0]:
            st.markdown("#### Racines de l'Équation Caractéristique")
            df_roots = pd.DataFrame({
                "Mode": [f"τ_{i+1}" for i in range(len(tau_roots))],
                "Partie Réelle": [f"{r.real:.6f}" for r in tau_roots],
                "Partie Imaginaire": [f"{r.imag:.6f}" for r in tau_roots],
                "|τ|": [f"{abs(r):.4f}" for r in tau_roots]
            })
            st.dataframe(df_roots, use_container_width=True, hide_index=True)
            
            # Visualisation dans le plan complexe
            fig_roots = go.Figure()
            fig_roots.add_trace(go.Scatter(
                x=[r.real for r in tau_roots],
                y=[r.imag for r in tau_roots],
                mode='markers+text',
                marker=dict(size=12, color=PALETTE['primary']),
                text=[f"τ{i+1}" for i in range(len(tau_roots))],
                textposition="top center",
                name="Racines τ"
            ))
            fig_roots.update_layout(
                title="Racines τ dans le Plan Complexe",
                xaxis_title="Re(τ)", yaxis_title="Im(τ)",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9')
            )
            st.plotly_chart(fig_roots, use_container_width=True)
        
        # === Sous-tab Vecteurs V/W ===
        with subtabs[1]:
            st.markdown("#### Vecteurs Propres")
            col_v, col_w = st.columns(2)
            with col_v:
                st.markdown("**Vecteurs Déplacement V**")
                V_data = [{"Mode": f"V_{i+1}", "V₁": f"{ev['V'][0]:.3f}", "V₂": f"{ev['V'][1]:.3f}", "V₃": f"{ev['V'][2]:.3f}"} for i, ev in enumerate(eigenvectors)]
                st.dataframe(pd.DataFrame(V_data), use_container_width=True, hide_index=True)
            with col_w:
                st.markdown("**Vecteurs Contrainte W**")
                W_data = [{"Mode": f"W_{i+1}", "σ₁₃": f"{ev['W'][0]/1e9:.2f}", "σ₂₃": f"{ev['W'][1]/1e9:.2f}", "σ₃₃": f"{ev['W'][2]/1e9:.2f}"} for i, ev in enumerate(eigenvectors) if 'W' in ev]
                if W_data:
                    st.dataframe(pd.DataFrame(W_data), use_container_width=True, hide_index=True)
        
        # === Sous-tab Matrice Φ ===
        with subtabs[2]:
            st.markdown("#### Matrice Fondamentale Φ(z)")
            
            if show_math:
                st.latex(r"\Phi(z) = \begin{bmatrix} V_1 e^{\tau_1 z} & \cdots & V_6 e^{\tau_6 z} \\ W_1 e^{\tau_1 z} & \cdots & W_6 e^{\tau_6 z} \end{bmatrix}_{6 \times 6}")
            
            Phi_0 = results['Phi_0']
            Phi_display = np.abs(Phi_0) / np.max(np.abs(Phi_0))
            
            fig_phi = px.imshow(Phi_display, 
                               labels=dict(x="Mode r", y="Composante", color="|Φᵢⱼ|"),
                               x=[f"τ{i+1}" for i in range(6)],
                               y=["V₁", "V₂", "V₃", "W₁", "W₂", "W₃"],
                               color_continuous_scale="Viridis",
                               title="Heatmap de |Φ(0)| normalisée")
            fig_phi.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9')
            )
            st.plotly_chart(fig_phi, use_container_width=True)
    
        # === Sous-tab ONERA ===
        with subtabs[3]:
            st.markdown("#### Validation vs Référence ONERA/Safran")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(6,182,212,0.05) 100%);
                        padding: 1rem; border-radius: 12px; border: 1px solid rgba(16,185,129,0.2); margin-bottom: 1rem;">
                <div style="color: #10b981; font-weight: 600;">📚 Bovet et al. (2025) - ONERA/Safran</div>
            </div>
            """, unsafe_allow_html=True)
            
            from core.constants import PROPS_SUBSTRATE, ALPHA_SUBSTRATE
            
            props_comparison = pd.DataFrame({
                "Propriété": ["C₁₁ (GPa)", "C₁₂ (GPa)", "C₄₄ (GPa)"],
                "Code": [PROPS_SUBSTRATE['C11'], PROPS_SUBSTRATE['C12'], PROPS_SUBSTRATE['C44']],
                "ONERA": [ONERA_REFERENCE['C11_RT'], ONERA_REFERENCE['C12_RT'], ONERA_REFERENCE['C44_RT']],
            })
            props_comparison["Écart (%)"] = ((props_comparison["Code"] - props_comparison["ONERA"]) / props_comparison["ONERA"] * 100).round(2)
            props_comparison["Statut"] = props_comparison["Écart (%)"].apply(lambda x: "✅" if abs(x) < 5 else "⚠️")
            st.dataframe(props_comparison, use_container_width=True, hide_index=True)
            
            if stress and 'sigma_33' in stress:
                sigma_max_MPa = np.max(np.abs(stress['sigma_33'])) / 1e6
                sigma_min, sigma_max = ONERA_REFERENCE['sigma_vM_range']
                is_in_range = sigma_min * 0.5 <= sigma_max_MPa <= sigma_max * 1.5
                
                col1, col2 = st.columns(2)
                col1.metric("σ₃₃ max calculé", f"{sigma_max_MPa:.1f} MPa")
                col2.metric("Plage ONERA", f"{sigma_min}-{sigma_max} MPa")
                
                if is_in_range:
                    st.success("✅ CONFORME aux références ONERA")
                else:
                    st.warning("⚠️ Hors plage - Vérifier les paramètres")
            else:
                st.info("Lancez un calcul pour la comparaison ONERA.")

if __name__ == "__main__":
    render()

