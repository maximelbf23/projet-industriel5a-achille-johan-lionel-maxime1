"""
Damage Criteria, Sensitivity Analysis, and Optimization Module
for TBC Multilayer Systems
"""
import numpy as np
# scipy is imported lazily inside functions to avoid import errors


# =============================================================================
# PHASE 1: Damage Criteria
# =============================================================================

# Critical stress values for each material (Pa)
# Values adjusted for TBC systems designed to survive thermal cycling
# Reference: TBC design guidelines for gas turbine applications
CRITICAL_STRESS = {
    'substrate': {
        'sigma_tensile': 1000e6,    # Superalliage Ni: ~1000 MPa yield
        'sigma_compressive': 1200e6,
        'sigma_shear': 600e6,
    },
    'bondcoat': {
        'sigma_tensile': 500e6,     # MCrAlY: ~500 MPa yield
        'sigma_compressive': 700e6,
        'sigma_shear': 300e6,
    },
    'ceramic': {
        # YSZ TBC peut supporter des contraintes modérées grâce à:
        # - Structure poreuse/colonnaire qui absorbe les contraintes
        # - Tolérance à la fissuration segmentée
        'sigma_tensile': 150e6,     # ~150 MPa (contrainte de design TBC)
        'sigma_compressive': 500e6, # Céramique très résistante en compression
        'sigma_shear': 120e6,       # ~120 MPa cisaillement
    }
}


def compute_damage_indicator(sigma_13, sigma_23, sigma_33, layer_type='ceramic'):
    """
    Calcule l'indicateur d'endommagement D pour chaque point.
    
    D = max(|σ_ij| / σ_crit_ij)
    
    D < 1: Sûr
    D = 1: Limite
    D > 1: Endommagement probable
    
    Args:
        sigma_13, sigma_23, sigma_33: Arrays de contraintes (Pa)
        layer_type: 'substrate', 'bondcoat', ou 'ceramic'
    
    Returns:
        D: Array d'indicateurs d'endommagement
    """
    crit = CRITICAL_STRESS.get(layer_type, CRITICAL_STRESS['ceramic'])
    
    # D = max des ratios |σ| / σ_crit
    D_shear = np.maximum(np.abs(sigma_13), np.abs(sigma_23)) / crit['sigma_shear']
    
    # Pour σ_33, on distingue traction (positif) et compression (négatif)
    D_normal = np.where(
        sigma_33 >= 0,
        np.abs(sigma_33) / crit['sigma_tensile'],
        np.abs(sigma_33) / crit['sigma_compressive']
    )
    
    D = np.maximum(D_shear, D_normal)
    return D


def compute_tsai_wu_criterion(sigma_13, sigma_23, sigma_33, layer_type='ceramic'):
    """
    Critère de Tsai-Wu pour matériaux composites/anisotropes.
    
    F = F_i * σ_i + F_ij * σ_i * σ_j
    
    F < 1: Sûr
    F = 1: Limite de rupture
    F > 1: Rupture
    
    Simplified version for through-thickness stresses.
    """
    crit = CRITICAL_STRESS.get(layer_type, CRITICAL_STRESS['ceramic'])
    
    # Coefficients F_i (asymétrie traction/compression)
    F_3 = 1/crit['sigma_tensile'] - 1/crit['sigma_compressive']
    
    # Coefficients F_ij
    F_33 = 1 / (crit['sigma_tensile'] * crit['sigma_compressive'])
    F_44 = 1 / (crit['sigma_shear']**2)  # Pour σ_23
    F_55 = 1 / (crit['sigma_shear']**2)  # Pour σ_13
    
    # Critère de Tsai-Wu
    F = (F_3 * sigma_33 + 
         F_33 * sigma_33**2 + 
         F_44 * sigma_23**2 + 
         F_55 * sigma_13**2)
    
    return F


def analyze_damage_profile(stress_profile, layer_idx_array, layer_types=['substrate', 'bondcoat', 'ceramic']):
    """
    Analyse le profil d'endommagement pour tout le multicouche.
    
    Returns:
        dict avec D (damage indicator), F (Tsai-Wu), et zones critiques
    """
    n_points = len(stress_profile['z'])
    D = np.zeros(n_points)
    F = np.zeros(n_points)
    
    for k, layer_type in enumerate(layer_types):
        mask = layer_idx_array == k
        if not np.any(mask):
            continue
        
        sigma_13 = stress_profile['sigma_13'][mask]
        sigma_23 = stress_profile['sigma_23'][mask]
        sigma_33 = stress_profile['sigma_33'][mask]
        
        D[mask] = compute_damage_indicator(sigma_13, sigma_23, sigma_33, layer_type)
        F[mask] = compute_tsai_wu_criterion(sigma_13, sigma_23, sigma_33, layer_type)
    
    # Identification des zones critiques (D > 0.8)
    critical_zones = np.where(D > 0.8)[0]
    
    return {
        'D': D,  # Damage indicator
        'F': F,  # Tsai-Wu criterion
        'max_D': np.max(D),
        'max_F': np.max(F),
        'critical_zones': critical_zones,
        'z_critical': stress_profile['z'][critical_zones] if len(critical_zones) > 0 else np.array([])
    }


# =============================================================================
# PHASE 2: Sensitivity Analysis (Parametric Sweep)
# =============================================================================

def sensitivity_sweep(solve_func, param_name, param_range, base_params):
    """
    Effectue un balayage paramétrique pour analyser la sensibilité.
    
    Args:
        solve_func: Fonction de résolution (retourne stress_profile)
        param_name: Nom du paramètre à varier
        param_range: Array de valeurs à tester
        base_params: Dict des paramètres de base
    
    Returns:
        Dict avec résultats pour chaque valeur du paramètre
    """
    results = {
        'param_values': param_range,
        'max_sigma_13': [],
        'max_sigma_33': [],
        'max_D': [],
        'max_F': [],
    }
    
    for val in param_range:
        params = base_params.copy()
        params[param_name] = val
        
        try:
            stress_profile = solve_func(**params)
            
            results['max_sigma_13'].append(np.max(np.abs(stress_profile['sigma_13'])))
            results['max_sigma_33'].append(np.max(np.abs(stress_profile['sigma_33'])))
            
            # Damage analysis
            damage = analyze_damage_profile(
                stress_profile, 
                stress_profile.get('layer_idx', np.zeros(len(stress_profile['z']))),
                ['substrate', 'bondcoat', 'ceramic']
            )
            results['max_D'].append(damage['max_D'])
            results['max_F'].append(damage['max_F'])
        except Exception as e:
            print(f"Error at {param_name}={val}: {e}")
            results['max_sigma_13'].append(np.nan)
            results['max_sigma_33'].append(np.nan)
            results['max_D'].append(np.nan)
            results['max_F'].append(np.nan)
    
    return {k: np.array(v) if isinstance(v, list) else v for k, v in results.items()}


def sensitivity_heatmap_2d(solve_func, param1_name, param1_range, param2_name, param2_range, base_params):
    """
    Génère une heatmap 2D de sensibilité pour deux paramètres.
    
    Returns:
        X, Y: Meshgrids des paramètres
        Z: Matrice des valeurs (ex: max_D)
    """
    n1 = len(param1_range)
    n2 = len(param2_range)
    
    Z_sigma = np.zeros((n2, n1))
    Z_D = np.zeros((n2, n1))
    
    for i, val1 in enumerate(param1_range):
        for j, val2 in enumerate(param2_range):
            params = base_params.copy()
            params[param1_name] = val1
            params[param2_name] = val2
            
            try:
                stress_profile = solve_func(**params)
                Z_sigma[j, i] = np.max(np.abs(stress_profile['sigma_33'])) / 1e6  # MPa
                
                damage = analyze_damage_profile(
                    stress_profile,
                    stress_profile.get('layer_idx', np.zeros(len(stress_profile['z']))),
                    ['substrate', 'bondcoat', 'ceramic']
                )
                Z_D[j, i] = damage['max_D']
            except:
                Z_sigma[j, i] = np.nan
                Z_D[j, i] = np.nan
    
    X, Y = np.meshgrid(param1_range, param2_range)
    return X, Y, Z_sigma, Z_D


# =============================================================================
# PHASE 3: Optimization
# =============================================================================

def optimize_tbc_thickness(solve_func, h_range, base_params, objective='min_damage'):
    """
    Optimise l'épaisseur de la couche TBC pour minimiser l'endommagement.
    
    Args:
        solve_func: Fonction de résolution
        h_range: (h_min, h_max) en mètres
        base_params: Paramètres de base
        objective: 'min_damage' ou 'min_stress'
    
    Returns:
        Dict avec h_optimal, damage_min, etc.
    """
    def objective_func(h_tbc):
        params = base_params.copy()
        params['h_ceramic'] = h_tbc
        
        try:
            result = solve_func(**params)
            stress = result['stress_profile']
            
            if objective == 'min_damage':
                damage = analyze_damage_profile(
                    stress,
                    stress.get('layer_idx', np.zeros(len(stress['z']))),
                    ['substrate', 'bondcoat', 'ceramic']
                )
                return damage['max_D']
            else:  # min_stress
                return np.max(np.abs(stress['sigma_33'])) / 1e6
        except:
            return 1e10  # Very high value if solver fails
    
    # Optimization using scipy
    result = minimize_scalar(
        objective_func,
        bounds=h_range,
        method='bounded'
    )
    
    return {
        'h_optimal': result.x,
        'objective_value': result.fun,
        'success': result.success,
        'message': result.message if hasattr(result, 'message') else 'OK'
    }


def multi_objective_optimization(solve_func, bounds, base_params):
    """
    Optimisation multi-objectif: minimiser endommagement ET épaisseur.
    
    Args:
        bounds: Dict avec limites pour chaque paramètre
        
    Returns:
        Optimal parameters
    """
    def combined_objective(x):
        h_bondcoat, h_ceramic = x
        
        params = base_params.copy()
        params['h_bondcoat'] = h_bondcoat
        params['h_ceramic'] = h_ceramic
        
        try:
            result = solve_func(**params)
            stress = result['stress_profile']
            
            damage = analyze_damage_profile(
                stress,
                stress.get('layer_idx', np.zeros(len(stress['z']))),
                ['substrate', 'bondcoat', 'ceramic']
            )
            
            # Objectif combiné: minimiser D + pénalité pour épaisseur
            thickness_penalty = (h_bondcoat + h_ceramic) * 1e4  # Convertir en mm
            return damage['max_D'] + 0.1 * thickness_penalty
        except:
            return 1e10
    
    x0 = [bounds['h_bondcoat'][0], bounds['h_ceramic'][0]]
    bound_list = [bounds['h_bondcoat'], bounds['h_ceramic']]
    
    result = minimize(
        combined_objective,
        x0,
        method='L-BFGS-B',
        bounds=bound_list
    )
    
    return {
        'h_bondcoat_optimal': result.x[0],
        'h_ceramic_optimal': result.x[1],
        'objective_value': result.fun,
        'success': result.success
    }
