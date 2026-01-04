
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

from core.mechanical import solve_multilayer_problem
from core.constants import PROPS_CERAMIC, ALPHA_CERAMIC, PROPS_SUBSTRATE, ALPHA_SUBSTRATE

def verify_free_expansion():
    """
    Vérifie qu'une couche libre se dilate sans contrainte (ou presque).
    Ceci valide la soustraction du terme thermique dans le calcul de contrainte.
    """
    print("\n=== Test 1: Vérification de la Soustraction Thermique (Free Expansion) ===")
    
    # Configuration: 1 seule couche de céramique
    # L large pour approcher le cas 1D uniforme
    h = 0.001 # 1 mm
    Lw = 1.0 # 1 m (très large)
    T_hat = 100 # 100 K
    
    delta = np.pi / Lw # très petit
    lambda_th = delta # 
    
    layer_configs = [
        (h, PROPS_CERAMIC, ALPHA_CERAMIC)
    ]
    
    # Résolution
    result = solve_multilayer_problem(layer_configs, Lw, lambda_th, T_hat)
    stress = result['stress_profile']
    
    sigma_33_max = np.max(np.abs(stress['sigma_33'])) / 1e6 # MPa
    
    print(f"Sigma_33 max calculé: {sigma_33_max:.4f} MPa")
    
    if sigma_33_max < 1.0:
        print("✅ SUCCESS: Contrainte quasi-nulle (< 1 MPa). La dilatation libre est respectée.")
    else:
        print(f"❌ FAILURE: Contrainte élevée ({sigma_33_max:.1f} MPa).")

def verify_multimode_effect():
    """
    Vérifie que l'approche Multi-Mode donne des résultats différents (et plus riches)
    que l'approche Mono-Mode simplifiée.
    """
    print("\n=== Test 2: Vérification Multi-Mode vs Mono-Mode ===")
    
    # Utilisation d'une couche plus épaisse pour que les effets spatiaux de exp(lambda*z) soient visibles
    h = 0.05 # 5 cm
    Lw = 0.05 # 5 cm
    T_hat_global = 100
    lambda_global = np.pi / Lw # ~ 62.8
    
    # lambda * h ~ 3.14. exp(3.14) ~ 23. C'est significatif.
    
    # Cas 1: Multi-Mode (Simulé)
    # On définit 2 modes avec des contributions distinctes
    th_data_multi = {
        'A': 80,
        'B': 20,
        'lambda': lambda_global * 1.5 
    }
    
    layer_configs_multi = [
        (h, PROPS_SUBSTRATE, ALPHA_SUBSTRATE, th_data_multi)
    ]
    
    res_multi = solve_multilayer_problem(layer_configs_multi, Lw, lambda_global, T_hat_global, method='spectral')
    s33_multi = res_multi['stress_profile']['sigma_33']
    
    # Cas 2: Mono-Mode (Legacy)
    # Utilise T_hat_global et lambda_global
    layer_configs_single = [
        (h, PROPS_SUBSTRATE, ALPHA_SUBSTRATE) # Pas de th_data
    ]
    
    res_single = solve_multilayer_problem(layer_configs_single, Lw, lambda_global, T_hat_global, method='spectral')
    s33_single = res_single['stress_profile']['sigma_33']
    
    # Comparaison
    diff_max = np.max(np.abs(s33_multi - s33_single)) / 1e6 # MPa
    max_multi = np.max(np.abs(s33_multi)) / 1e6
    max_single = np.max(np.abs(s33_single)) / 1e6
    
    print(f"Max Sigma_33 Multi-Mode:  {max_multi:.2f} MPa")
    print(f"Max Sigma_33 Single-Mode: {max_single:.2f} MPa")
    print(f"Différence Max:           {diff_max:.2f} MPa")
    
    if diff_max > 0.1: 
        print("✅ SUCCESS: Différence significative trouvée. Le mode Multi-Mode est actif.")
    else:
        print("❌ FAILURE: Les résultats sont trop proches. Vérifier l'implémentation.")

if __name__ == "__main__":
    verify_free_expansion()
    verify_multimode_effect()
