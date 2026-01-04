"""
Module de validation pour le solveur mécanique multicouche TBC.

Ce module fournit:
1. Données de référence expérimentales (littérature TBC)
2. Solutions analytiques pour cas limites (benchmark)
3. Critères de cohérence physique

Références:
- Padture, N.P. et al. "Thermal barrier coatings...", Science 296, 280 (2002)
- Evans, A.G. et al. "Mechanics of delamination...", Surf. Coat. Tech. 177-178, 7 (2004)
- Timoshenko & Goodier, Theory of Elasticity, Chapter 13 (bicouche analytique)
"""
import numpy as np


# ============================================================================
# DONNÉES EXPÉRIMENTALES DE RÉFÉRENCE
# Sources: Publications sur systèmes TBC réels
# ============================================================================

REFERENCE_DATA = {
    'typical_tbc_system': {
        'description': "YSZ (250µm) / MCrAlY (50µm) / Superalliage, ΔT=600°C",
        'h_ceramic_um': 250,
        'h_bondcoat_um': 50,
        'h_substrate_um': 500,
        'delta_T': 600,
        # Contraintes de référence (MPa) - basées sur mesures et FEM
        'sigma_33_ceramic_MPa': {
            'min': -50,    # Compression en refroidissement
            'max': 100,    # Traction au chauffage rapide
            'typical': 30  # Contrainte moyenne en cyclage
        },
        'sigma_interface_MPa': {
            'TGO_bondcoat': (50, 200),     # Zone critique de spallation
            'bondcoat_substrate': (10, 80)  # Interface moins critique
        },
        'shear_stress_MPa': {
            'typical_max': 50,
            'critical': 120  # Limite avant délaminage
        }
    },
    
    'high_temperature_tbc': {
        'description': "TBC haute température pour turbines aéronautiques, ΔT=900°C",
        'h_ceramic_um': 300,
        'h_bondcoat_um': 75,
        'delta_T': 900,
        'sigma_33_ceramic_MPa': {
            'min': -80,
            'max': 150,
            'typical': 60
        }
    }
}


# ============================================================================
# SOLUTIONS ANALYTIQUES DE RÉFÉRENCE
# ============================================================================

def analytical_bilayer_interface_stress(E1, E2, alpha1, alpha2, h1, h2, dT, nu=0.3):
    """
    Solution analytique exacte pour la contrainte d'interface dans un bicouche libre.
    
    Basé sur la théorie de Timoshenko pour les structures bimétalliques.
    Hypothèses:
    - Surfaces supérieure et inférieure libres
    - Couches parfaitement collées à l'interface
    - Déformation plane (état biaxial)
    
    Args:
        E1, E2: Modules d'Young des couches 1 (bas) et 2 (haut) [Pa]
        alpha1, alpha2: Coefficients de dilatation [1/K]
        h1, h2: Épaisseurs [m]
        dT: Variation de température [K ou °C]
        nu: Coefficient de Poisson (supposé identique)
    
    Returns:
        sigma_interface: Contrainte normale à l'interface [Pa]
    """
    # Modules effectifs biaxiaux
    E1_eff = E1 / (1 - nu)
    E2_eff = E2 / (1 - nu)
    
    # Différence de dilatation thermique
    delta_alpha = alpha1 - alpha2
    
    # Rapports géométriques et de rigidité
    n = E2_eff / E1_eff  # Ratio de rigidité
    m = h2 / h1          # Ratio d'épaisseur
    
    # Formule de Timoshenko pour la contrainte d'interface
    # Dérivée de la condition d'équilibre et de compatibilité
    numerator = E1_eff * delta_alpha * dT
    denominator = 1 + (1 + m) * (1 + n * m**3) / (n * m * (3 * (1 + m)**2))
    
    sigma_interface = numerator / denominator
    
    return sigma_interface


def analytical_trilayer_stress(layers, dT, nu=0.3):
    """
    Solution semi-analytique pour un système tricouche (substrat/bondcoat/céramique).
    
    Utilise l'approche de superposition:
    1. Calculer la déformation thermique moyenne pondérée
    2. Calculer la contrainte de mismatch dans chaque couche
    
    Args:
        layers: Liste de dicts avec 'E', 'alpha', 'h' pour chaque couche
        dT: Variation de température
        nu: Coefficient de Poisson
    
    Returns:
        dict avec contraintes dans chaque couche
    """
    # Rigidités effectives biaxiales
    E_eff = [lay['E'] / (1 - nu) for lay in layers]
    
    # Calcul de la déformation thermique moyenne pondérée
    sum_E_h = sum(E_eff[i] * layers[i]['h'] for i in range(len(layers)))
    sum_E_alpha_h = sum(E_eff[i] * layers[i]['alpha'] * layers[i]['h'] for i in range(len(layers)))
    
    # Déformation moyenne du système
    epsilon_avg = sum_E_alpha_h / sum_E_h * dT
    
    # Contrainte dans chaque couche: σ = E_eff * (ε_avg - α * ΔT)
    stresses = []
    for i, lay in enumerate(layers):
        sigma = E_eff[i] * (epsilon_avg - lay['alpha'] * dT)
        stresses.append({
            'layer': i,
            'sigma_xx': sigma,  # Contrainte dans le plan
            'sigma_33_avg': 0   # σ_33 moyen = 0 pour couche libre
        })
    
    return {
        'epsilon_avg': epsilon_avg,
        'layer_stresses': stresses
    }


def analytical_interface_shear_estimate(E, h, delta_alpha, dT, Lw, nu=0.3):
    """
    Estimation de la contrainte de cisaillement près des interfaces.
    
    Basé sur le modèle de shear-lag:
    τ_max ≈ E * Δα * ΔT * h / Lw
    
    Cette formule donne l'ordre de grandeur, pas la valeur exacte.
    """
    E_eff = E / (1 - nu)
    tau_max = E_eff * abs(delta_alpha) * abs(dT) * h / Lw
    return tau_max


# ============================================================================
# FONCTIONS DE VALIDATION
# ============================================================================

def validate_stress_magnitudes(stress_profile, reference_key='typical_tbc_system'):
    """
    Valide les ordres de grandeur des contraintes calculées.
    
    Vérifie:
    1. Contraintes dans une plage physiquement réaliste
    2. Conditions aux limites (σ ≈ 0 aux surfaces libres)
    3. Cohérence avec données expérimentales
    
    Args:
        stress_profile: Dict avec 'z', 'sigma_13', 'sigma_23', 'sigma_33'
        reference_key: Clé dans REFERENCE_DATA
    
    Returns:
        dict avec is_valid, warnings, comparison
    """
    ref = REFERENCE_DATA.get(reference_key, REFERENCE_DATA['typical_tbc_system'])
    
    # Extraction des valeurs maximales
    sigma_33_max = np.max(np.abs(stress_profile['sigma_33'])) / 1e6  # MPa
    sigma_33_values = stress_profile['sigma_33'] / 1e6
    sigma_shear_max = np.max([
        np.max(np.abs(stress_profile['sigma_13'])),
        np.max(np.abs(stress_profile['sigma_23']))
    ]) / 1e6
    
    warnings = []
    is_valid = True
    
    # ---- Vérification 1: Ordres de grandeur ----
    ref_range = ref['sigma_33_ceramic_MPa']
    
    if sigma_33_max < 0.1:
        warnings.append(f"⚠️ σ_33 très faible ({sigma_33_max:.2f} MPa). Problème de calcul possible.")
        is_valid = False
    elif sigma_33_max > ref_range['max'] * 10:
        warnings.append(f"⚠️ σ_33 anormalement élevée ({sigma_33_max:.2f} MPa > {ref_range['max']*10} MPa).")
        is_valid = False
    
    # ---- Vérification 2: Conditions aux limites ----
    sigma_33_boundaries = [stress_profile['sigma_33'][0], stress_profile['sigma_33'][-1]]
    bc_tolerance = 5e6  # 5 MPa de tolérance numérique
    
    for i, sigma_bc in enumerate(sigma_33_boundaries):
        if abs(sigma_bc) > bc_tolerance:
            warnings.append(f"⚠️ σ_33 ≠ 0 au bord {['inférieur','supérieur'][i]}: {sigma_bc/1e6:.2f} MPa")
            is_valid = False
    
    # ---- Vérification 3: Cisaillement ----
    if 'shear_stress_MPa' in ref:
        shear_crit = ref['shear_stress_MPa']['critical']
        if sigma_shear_max > shear_crit:
            warnings.append(f"⚠️ Cisaillement max ({sigma_shear_max:.2f} MPa) > critique ({shear_crit} MPa)")
    
    # ---- Vérification 4: Profil physiquement cohérent ----
    # σ_33 devrait être max au centre et ~0 aux bords
    z_array = stress_profile['z']
    z_mid = (z_array[0] + z_array[-1]) / 2
    idx_mid = np.argmin(np.abs(z_array - z_mid))
    
    sigma_33_at_mid = abs(stress_profile['sigma_33'][idx_mid]) / 1e6
    sigma_33_at_edges = (abs(sigma_33_boundaries[0]) + abs(sigma_33_boundaries[1])) / 2 / 1e6
    
    if sigma_33_at_mid < sigma_33_at_edges and sigma_33_max > 1:
        warnings.append("ℹ️ Profil σ_33 atypique: max aux bords au lieu du centre.")
    
    return {
        'is_valid': is_valid,
        'warnings': warnings,
        'comparison': {
            'sigma_33_max_MPa': sigma_33_max,
            'sigma_33_range_MPa': (np.min(sigma_33_values), np.max(sigma_33_values)),
            'reference_typical_MPa': ref_range['typical'],
            'sigma_shear_max_MPa': sigma_shear_max,
            'bc_residual_MPa': {
                'bottom': sigma_33_boundaries[0] / 1e6,
                'top': sigma_33_boundaries[1] / 1e6
            }
        }
    }


def run_analytical_benchmark(solve_func=None):
    """
    Exécute un benchmark contre la solution analytique bicouche.
    
    Compare la solution numérique à la formule de Timoshenko exacte.
    
    Args:
        solve_func: Fonction de résolution (optionnel, sinon import auto)
    
    Returns:
        dict avec sigma_analytical, sigma_numerical, error_percent, is_acceptable
    """
    # Configuration du cas benchmark: bicouche simple
    layers = [
        {'h': 0.001, 'E': 200e9, 'alpha': 13e-6, 'name': 'Substrat'},
        {'h': 0.0002, 'E': 50e9, 'alpha': 10e-6, 'name': 'Céramique'}
    ]
    delta_T = 500
    
    # Solution analytique exacte
    sigma_analytical = analytical_bilayer_interface_stress(
        E1=layers[0]['E'], E2=layers[1]['E'],
        alpha1=layers[0]['alpha'], alpha2=layers[1]['alpha'],
        h1=layers[0]['h'], h2=layers[1]['h'],
        dT=delta_T
    )
    
    # Solution numérique
    if solve_func is None:
        try:
            from core.mechanical import solve_multilayer_problem
            solve_func = solve_multilayer_problem
        except ImportError:
            return {
                'error': "Impossible d'importer solve_multilayer_problem",
                'is_acceptable': False
            }
    
    # Construire la configuration des couches
    layer_configs = []
    for lay in layers:
        # Matrice de rigidité isotrope simplifiée depuis E
        E = lay['E']
        nu = 0.3
        G = E / (2 * (1 + nu))
        
        props = {
            'C11': E * (1 - nu) / ((1 + nu) * (1 - 2*nu)),
            'C12': E * nu / ((1 + nu) * (1 - 2*nu)),
            'C13': E * nu / ((1 + nu) * (1 - 2*nu)),
            'C21': E * nu / ((1 + nu) * (1 - 2*nu)),
            'C22': E * (1 - nu) / ((1 + nu) * (1 - 2*nu)),
            'C23': E * nu / ((1 + nu) * (1 - 2*nu)),
            'C31': E * nu / ((1 + nu) * (1 - 2*nu)),
            'C32': E * nu / ((1 + nu) * (1 - 2*nu)),
            'C33': E * (1 - nu) / ((1 + nu) * (1 - 2*nu)),
            'C44': G, 'C55': G, 'C66': G
        }
        alpha_dict = {'alpha_1': lay['alpha'], 'alpha_2': lay['alpha'], 'alpha_3': lay['alpha']}
        layer_configs.append((lay['h'], props, alpha_dict))
    
    try:
        result = solve_func(layer_configs, lw=0.1, lambda_th=np.pi/0.1, T_hat=delta_T)
        stress = result['stress_profile']
        
        # Trouver l'interface (z = h1)
        z_interface = layers[0]['h']
        idx_interface = np.argmin(np.abs(stress['z'] - z_interface))
        
        # Prendre la moyenne près de l'interface pour éviter les discontinuités numériques
        idx_range = slice(max(0, idx_interface-2), min(len(stress['z']), idx_interface+3))
        sigma_numerical = np.mean(stress['sigma_33'][idx_range])
        
    except Exception as e:
        return {
            'error': f"Erreur solver: {str(e)}",
            'is_acceptable': False
        }
    
    # Comparaison
    error_abs = abs(sigma_numerical - sigma_analytical)
    error_pct = error_abs / abs(sigma_analytical) * 100 if abs(sigma_analytical) > 1e-10 else 0
    
    return {
        'sigma_analytical_MPa': sigma_analytical / 1e6,
        'sigma_numerical_MPa': sigma_numerical / 1e6,
        'error_absolute_MPa': error_abs / 1e6,
        'error_percent': error_pct,
        'is_acceptable': error_pct < 25,  # Tolérance de 25%
        'benchmark_config': {
            'layers': layers,
            'delta_T': delta_T
        }
    }


def validate_boundary_conditions(stress_profile, tolerance_MPa=5.0):
    """
    Vérifie strictement que les conditions aux limites sont satisfaites.
    
    Pour surfaces libres: σ_i3 = 0 en z=0 et z=H
    
    Args:
        stress_profile: Profil de contraintes
        tolerance_MPa: Tolérance acceptable en MPa
    
    Returns:
        dict avec is_satisfied et détails
    """
    tol = tolerance_MPa * 1e6  # Convertir en Pa
    
    # Valeurs aux bords
    bc_values = {
        'bottom': {
            'sigma_13': stress_profile['sigma_13'][0],
            'sigma_23': stress_profile['sigma_23'][0],
            'sigma_33': stress_profile['sigma_33'][0]
        },
        'top': {
            'sigma_13': stress_profile['sigma_13'][-1],
            'sigma_23': stress_profile['sigma_23'][-1],
            'sigma_33': stress_profile['sigma_33'][-1]
        }
    }
    
    errors = []
    for boundary, values in bc_values.items():
        for component, value in values.items():
            if abs(value) > tol:
                errors.append({
                    'boundary': boundary,
                    'component': component,
                    'value_MPa': value / 1e6,
                    'tolerance_MPa': tolerance_MPa
                })
    
    return {
        'is_satisfied': len(errors) == 0,
        'boundary_values': bc_values,
        'errors': errors,
        'tolerance_MPa': tolerance_MPa
    }


def generate_validation_report(stress_profile, layer_types=['substrate', 'bondcoat', 'ceramic']):
    """
    Génère un rapport de validation complet.
    
    Returns:
        dict avec tous les résultats de validation
    """
    report = {
        'timestamp': np.datetime64('now'),
        'magnitudes': validate_stress_magnitudes(stress_profile),
        'boundary_conditions': validate_boundary_conditions(stress_profile),
        'overall_valid': True
    }
    
    # Synthèse
    if not report['magnitudes']['is_valid']:
        report['overall_valid'] = False
    if not report['boundary_conditions']['is_satisfied']:
        report['overall_valid'] = False
    
    # Résumé des warnings
    all_warnings = report['magnitudes']['warnings'].copy()
    for err in report['boundary_conditions']['errors']:
        all_warnings.append(
            f"BC violation: {err['component']}({err['boundary']}) = {err['value_MPa']:.2f} MPa"
        )
    
    report['all_warnings'] = all_warnings
    
    return report
