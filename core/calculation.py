import numpy as np
import streamlit as st
from .constants import CONSTANTS

@st.cache_data
def solve_tbc_model_v2(alpha, beta_ceramique, lw_val, t_bottom, t_top, n_modes=1):
    """
    Résout le système d'équations thermiques pour une configuration donnée (Multi-modes).
    
    Méthode: Résolution spectrale (Étapes 1-3 du PDF ProjectEstaca.pdf)
    Généralisée pour M modes (m = 1, 3, ..., 2M-1).
    
    Pour des températures imposées constantes T_bottom/T_top, seuls les modes impairs
    sont excités (symétrie par rapport au milieu de la période 2Lw).
    Amplitude du mode m: A_m = 4/(m*pi) * T_amp
    """
    try:
        # =====================================================================
        # ÉTAPE 1: DÉFINITION GÉOMÉTRIQUE
        # =====================================================================
        h1, h2 = CONSTANTS['h1'], CONSTANTS['h2']
        x_i1 = h1
        x_i2 = h1 + h2
        h3 = alpha * h1
        H = h1 + h2 + h3
        
        # Propriétés
        k_33_1 = CONSTANTS['k33_1']
        k_eta_1 = k_33_1
        
        k_33_2 = CONSTANTS['k33_2']
        k_eta_2 = k_33_2
        
        k_33_3 = CONSTANTS['k33_3']
        beta_safe = max(beta_ceramique, 1e-6)
        k_eta_3 = k_33_3 / beta_safe
        
        modes_results = []
        
        # Boucle sur les modes impairs (m = 1, 3, 5, ...)
        # Si n_modes=1 -> m=1 uniquement (comportement original)
        for i_mode in range(n_modes):
            m = 2 * i_mode + 1
            
            # Amplitude thermique
            # Correction : Si n_modes=1, on veut que l'amplitude max SOIT t_top, pas 1.27*t_top (Gibbs)
            if n_modes == 1:
                fourier_factor = 1.0
            else:
                # Série de Fourier d'un signal carré (Square Wave)
                fourier_factor = 4 / (m * np.pi)
                
            t_bot_m = t_bottom * fourier_factor
            t_top_m = t_top * fourier_factor
            
            # Nombre d'onde spectral (mode m)
            delta_eta = (m * np.pi) / lw_val
            
            # =====================================================================
            # VALEURS PROPRES THERMIQUES λ (Scaling avec m)
            # =====================================================================
            l1 = delta_eta * np.sqrt(k_eta_1 / k_33_1)
            l2 = delta_eta * np.sqrt(k_eta_2 / k_33_2)
            l3 = delta_eta * np.sqrt(k_eta_3 / k_33_3)
            
            # Coefficients C (Flux)
            C1 = k_33_1 * l1
            C2 = k_33_2 * l2
            C3 = k_33_3 * l3

            # Assemblage Matrice Système (6x6)
            M = np.zeros((6, 6))
            F = np.zeros(6)

            M[0,0]=1; M[0,1]=1; F[0]=t_bot_m
            M[1,0]=np.exp(l1*x_i1); M[1,1]=np.exp(-l1*x_i1); M[1,2]=-np.exp(l2*x_i1); M[1,3]=-np.exp(-l2*x_i1)
            M[2,0]=C1*np.exp(l1*x_i1); M[2,1]=-C1*np.exp(-l1*x_i1); M[2,2]=-C2*np.exp(l2*x_i1); M[2,3]=C2*np.exp(-l2*x_i1)
            M[3,2]=np.exp(l2*x_i2); M[3,3]=np.exp(-l2*x_i2); M[3,4]=-np.exp(l3*x_i2); M[3,5]=-np.exp(-l3*x_i2)
            M[4,2]=C2*np.exp(l2*x_i2); M[4,3]=-C2*np.exp(-l2*x_i2); M[4,4]=-C3*np.exp(l3*x_i2); M[4,5]=C3*np.exp(-l3*x_i2)
            M[5,4]=np.exp(l3*H); M[5,5]=np.exp(-l3*H); F[5]=t_top_m

            try:
                coeffs = np.linalg.solve(M, F)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(M, F, rcond=None)[0]
                
            A1, B1, A2, B2, A3, B3 = coeffs

            # Stockage des résultats pour ce mode
            modes_results.append({
                'm': m,
                'coeffs': coeffs,
                'lambdas': (l1, l2, l3),
                'k_eta': (k_eta_1, k_eta_2, k_eta_3), # Constant mais nécessaire pour calcul flux
                'C_coeffs': (C1, C2, C3),
                'delta_eta': delta_eta,
                'interfaces': (x_i1, x_i2),
                't_amps': (t_bot_m, t_top_m)
            })

        # Pour compatibilité, on calcule les valeurs scalaires "équivalentes" (basées sur m=1 ou sommation au centre)
        # Ici on retourne le mode 1 pour l'affichage simple, mais la liste complète pour le solver méca
        
        mode1 = modes_results[0]
        # Recalcul de T aux interfaces pour le mode 1 (usage display simple)
        l1, l2 = mode1['lambdas'][:2]
        A1, B1, A2, B2 = mode1['coeffs'][:4]
        
        T_h1 = A1*np.exp(l1*x_i1) + B1*np.exp(-l1*x_i1)
        T_h2 = A2*np.exp(l2*x_i2) + B2*np.exp(-l2*x_i2)
        
        Q1_h1_minus = -k_eta_1 * mode1['delta_eta'] * T_h1
        Q1_h1_plus  = -k_eta_2 * mode1['delta_eta'] * T_h1
        dQ1_h1 = Q1_h1_plus - Q1_h1_minus
        
        Q1_h2_minus = -k_eta_2 * mode1['delta_eta'] * T_h2
        Q1_h2_plus  = -k_eta_3 * mode1['delta_eta'] * T_h2
        dQ1_h2 = Q1_h2_plus - Q1_h2_minus

        return {
            'success': True, 
            'H': H, 
            'h3': h3, 
            'k_eta_3': k_eta_3,
            'T_at_h1': T_h1, # Valeur du mode 1 (approx du pic)
            'T_at_h2': T_h2,
            'dQ1_h1': dQ1_h1, 
            'dQ1_h2': dQ1_h2,
            'params': {'alpha': alpha, 'beta': beta_ceramique, 'lw': lw_val, 'n_modes': n_modes},
            'profile_params': {
                'modes': modes_results, # Nouvelle structure liste
                'coeffs': mode1['coeffs'], # Legacy
                'lambdas': mode1['lambdas'], # Legacy
                'interfaces': (x_i1, x_i2),
                'k_eta': (k_eta_1, k_eta_2, k_eta_3),
                'C_coeffs': mode1['C_coeffs'], # Legacy
                'delta_eta': mode1['delta_eta'] # Legacy
            }
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

def calculate_profiles(profile_params, H, num_points=500):
    """
    Calcule les profils de température et de flux (Sommation Multi-modes).
    """
    x_arr = np.linspace(0, H, num_points)
    
    # Init accumulators
    T_total = np.zeros_like(x_arr)
    Q1_total = np.zeros_like(x_arr)
    Q3_total = np.zeros_like(x_arr)
    
    # Check if multi-mode
    if 'modes' in profile_params:
        modes = profile_params['modes']
    else:
        # Backward compatibility wrapper
        modes = [{
            'coeffs': profile_params['coeffs'],
            'lambdas': profile_params['lambdas'],
            'k_eta': profile_params['k_eta'],
            'C_coeffs': profile_params['C_coeffs'],
            'delta_eta': profile_params['delta_eta'],
            'm': 1 # Assume mode 1
        }]
    
    x_i1, x_i2 = profile_params['interfaces']
    condlist = [x_arr <= x_i1, (x_arr > x_i1) & (x_arr <= x_i2), x_arr > x_i2]

    # Sum Fourier series
    for mode in modes:
        coeffs = mode['coeffs']
        l1, l2, l3 = mode['lambdas']
        k_eta_1, k_eta_2, k_eta_3 = mode['k_eta']
        C1, C2, C3 = mode['C_coeffs']
        delta_eta = mode['delta_eta']
        A1, B1, A2, B2, A3, B3 = coeffs
        
        # Temp function for this mode
        T_funcs = [
            lambda x: A1*np.exp(l1*x) + B1*np.exp(-l1*x),
            lambda x: A2*np.exp(l2*x) + B2*np.exp(-l2*x),
            lambda x: A3*np.exp(l3*x) + B3*np.exp(-l3*x)
        ]
        T_mode = np.piecewise(x_arr, condlist, T_funcs)
        
        # Fluxes
        Q1_funcs = [
            lambda x: -k_eta_1 * delta_eta * T_funcs[0](x),
            lambda x: -k_eta_2 * delta_eta * T_funcs[1](x),
            lambda x: -k_eta_3 * delta_eta * T_funcs[2](x)
        ]
        Q1_mode = np.piecewise(x_arr, condlist, Q1_funcs)
        
        Q3_funcs = [
            lambda x: -(C1*A1*np.exp(l1*x) - C1*B1*np.exp(-l1*x)),
            lambda x: -(C2*A2*np.exp(l2*x) - C2*B2*np.exp(-l2*x)),
            lambda x: -(C3*A3*np.exp(l3*x) - C3*B3*np.exp(-l3*x))
        ]
        Q3_mode = np.piecewise(x_arr, condlist, Q3_funcs)
        
        # Add contribution
        # Note: Here we are plotting the profile along z at x_transverse corresponding to peak
        # In PDF: T(x,z) = sum T_m(z) * sin(delta_m * x)
        # Peak is at x = Lw/2 (center) if simple sine.
        # But for square wave (modes 1, 3, 5...), sin(m*pi/2) alternates: 1, -1, 1, -1...
        # So at center x=L/2: sin(m*pi/2) = (-1)^((m-1)/2)
        m = mode.get('m', 1)
        phase = (-1)**((m-1)//2)
        
        T_total += T_mode * phase
        Q1_total += Q1_mode * phase
        Q3_total += Q3_mode * phase
        
    return x_arr, T_total, Q1_total, Q3_total


def calculate_sensitivity(alpha, beta, lw, t_bottom, t_top, delta_pct=0.1):
    """
    Calcule la sensibilité de T_h1 et dQ1_h1 aux variations de alpha, beta, lw.
    Retourne un dictionnaire avec les sensibilités normalisées.
    """
    results = {}
    
    # Base case
    base_res = solve_tbc_model_v2(alpha, beta, lw, t_bottom, t_top)
    if not base_res['success']:
        return None
        
    base_T = base_res['T_at_h1']
    
    # Paramètres à varier
    params = ['alpha', 'beta', 'lw']
    labels = ['Épaisseur (α)', 'Anisotropie (β)', 'Longueur d\'Onde (Lw)']
    current_vals = {'alpha': alpha, 'beta': beta, 'lw': lw}
    
    for param, label in zip(params, labels):
        # Variation +delta
        vals_plus = current_vals.copy()
        vals_plus[param] *= (1 + delta_pct)
        res_plus = solve_tbc_model_v2(vals_plus['alpha'], vals_plus['beta'], vals_plus['lw'], t_bottom, t_top)
        
        # Variation -delta
        vals_minus = current_vals.copy()
        vals_minus[param] *= (1 - delta_pct)
        res_minus = solve_tbc_model_v2(vals_minus['alpha'], vals_minus['beta'], vals_minus['lw'], t_bottom, t_top)
        
        if res_plus['success'] and res_minus['success']:
            T_plus = res_plus['T_at_h1']
            T_minus = res_minus['T_at_h1']
            
            # Calcul de sensibilité : Variation de T pour +/- delta_pct variation
            # Delta T total sur l'intervalle [x-d, x+d] -> représente la pente
            delta_T = T_plus - T_minus
            
            # Impact sur Flux Transverse (dQ1)
            dQ_plus = abs(res_plus['dQ1_h1'])
            dQ_minus = abs(res_minus['dQ1_h1'])
            delta_Q = dQ_plus - dQ_minus
            
            results[param] = {
                'label': label,
                'delta_T': delta_T, # Changement absolu sur plage +/- delta
                'delta_Q': delta_Q,
                'T_plus': T_plus,
                'T_minus': T_minus
            }
            
    return results

