import numpy as np
import streamlit as st
from .constants import CONSTANTS

@st.cache_data
def solve_tbc_model(alpha, beta_ceramique, lw_val, t_bottom, t_top):
    """
    Résout le système d'équations thermiques pour une configuration donnée.
    Retourne un dictionnaire de résultats et de paramètres pour le traçage.
    """
    try:
        # 1. Géométrie
        h1, h2 = CONSTANTS['h1'], CONSTANTS['h2']
        x_i1 = h1
        x_i2 = h1 + h2
        h3 = alpha * h1
        H = h1 + h2 + h3
        
        # 2. Propriétés Thermiques
        k_eta_1 = CONSTANTS['k33_1']
        k_eta_2 = CONSTANTS['k33_2']
        
        beta_safe = max(beta_ceramique, 1e-6)
        k_eta_3 = CONSTANTS['k33_3'] / beta_safe
        
        delta_eta = np.pi / lw_val
        
        # Valeurs propres (Lambdas)
        l1 = delta_eta * np.sqrt(k_eta_1 / CONSTANTS['k33_1'])
        l2 = delta_eta * np.sqrt(k_eta_2 / CONSTANTS['k33_2'])
        l3 = delta_eta * np.sqrt(k_eta_3 / CONSTANTS['k33_3'])
        
        # Coefficients C (Flux)
        C1 = CONSTANTS['k33_1'] * l1
        C2 = CONSTANTS['k33_2'] * l2
        C3 = CONSTANTS['k33_3'] * l3

        # 3. Assemblage Matrice Système (6x6)
        M = np.zeros((6, 6))
        F = np.zeros(6)

        M[0,0]=1; M[0,1]=1; F[0]=t_bottom
        M[1,0]=np.exp(l1*x_i1); M[1,1]=np.exp(-l1*x_i1); M[1,2]=-np.exp(l2*x_i1); M[1,3]=-np.exp(-l2*x_i1)
        M[2,0]=C1*np.exp(l1*x_i1); M[2,1]=-C1*np.exp(-l1*x_i1); M[2,2]=-C2*np.exp(l2*x_i1); M[2,3]=C2*np.exp(-l2*x_i1)
        M[3,2]=np.exp(l2*x_i2); M[3,3]=np.exp(-l2*x_i2); M[3,4]=-np.exp(l3*x_i2); M[3,5]=-np.exp(-l3*x_i2)
        M[4,2]=C2*np.exp(l2*x_i2); M[4,3]=-C2*np.exp(-l2*x_i2); M[4,4]=-C3*np.exp(l3*x_i2); M[4,5]=C3*np.exp(-l3*x_i2)
        M[5,4]=np.exp(l3*H); M[5,5]=np.exp(-l3*H); F[5]=t_top

        coeffs = np.linalg.solve(M, F)
        A1, B1, A2, B2, A3, B3 = coeffs

        T_h1 = A1*np.exp(l1*x_i1) + B1*np.exp(-l1*x_i1)
        T_h2 = A2*np.exp(l2*x_i2) + B2*np.exp(-l2*x_i2)
        
        Q1_h1_minus = -k_eta_1 * delta_eta * T_h1
        Q1_h1_plus  = -k_eta_2 * delta_eta * T_h1
        dQ1_h1 = Q1_h1_plus - Q1_h1_minus
        
        Q1_h2_minus = -k_eta_2 * delta_eta * T_h2
        Q1_h2_plus  = -k_eta_3 * delta_eta * T_h2
        dQ1_h2 = Q1_h2_plus - Q1_h2_minus

        return {
            'success': True, 'H': H, 'h3': h3, 'k_eta_3': k_eta_3,
            'T_at_h1': T_h1, 'T_at_h2': T_h2,
            'dQ1_h1': dQ1_h1, 'dQ1_h2': dQ1_h2,
            'params': {'alpha': alpha, 'beta': beta_ceramique, 'lw': lw_val},
            'profile_params': {
                'coeffs': tuple(coeffs),
                'lambdas': (l1, l2, l3),
                'interfaces': (x_i1, x_i2),
                'k_etas': (k_eta_1, k_eta_2, k_eta_3),
                'C_coeffs': (C1, C2, C3),
                'delta_eta': delta_eta
            }
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

def calculate_profiles(profile_params, H, num_points=500):
    """
    Calcule les profils de température et de flux à partir des paramètres résolus.
    """
    x_arr = np.linspace(0, H, num_points)
    
    # Unpack params
    coeffs = profile_params['coeffs']
    l1, l2, l3 = profile_params['lambdas']
    x_i1, x_i2 = profile_params['interfaces']
    k_eta_1, k_eta_2, k_eta_3 = profile_params['k_etas']
    C1, C2, C3 = profile_params['C_coeffs']
    delta_eta = profile_params['delta_eta']
    A1, B1, A2, B2, A3, B3 = coeffs
    
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
    
    return x_arr, T, Q1, Q3


def calculate_sensitivity(alpha, beta, lw, t_bottom, t_top, delta_pct=0.1):
    """
    Calcule la sensibilité de T_h1 et dQ1_h1 aux variations de alpha, beta, lw.
    Retourne un dictionnaire avec les sensibilités normalisées.
    """
    results = {}
    
    # Base case
    base_res = solve_tbc_model(alpha, beta, lw, t_bottom, t_top)
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
        res_plus = solve_tbc_model(vals_plus['alpha'], vals_plus['beta'], vals_plus['lw'], t_bottom, t_top)
        
        # Variation -delta
        vals_minus = current_vals.copy()
        vals_minus[param] *= (1 - delta_pct)
        res_minus = solve_tbc_model(vals_minus['alpha'], vals_minus['beta'], vals_minus['lw'], t_bottom, t_top)
        
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

