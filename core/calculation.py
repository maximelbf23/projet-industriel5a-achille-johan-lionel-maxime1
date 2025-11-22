import streamlit as st
import numpy as np
from core.constants import CONSTANTS

@st.cache_data
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
