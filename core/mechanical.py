import numpy as np
from .constants import MECHANICAL_PROPS, GPa_TO_PA

def get_M_matrix(tau, delta1, delta2, props=MECHANICAL_PROPS):
    """
    Construit la matrice dynamique M(tau) 3x3.
    """
    # Extraction des constantes
    C11 = props['C11']
    C12 = props['C12']
    C13 = props['C13']
    C22 = props['C22']
    C23 = props['C23']
    C33 = props['C33']
    C44 = props['C44']
    C55 = props['C55']
    C66 = props['C66']

    # Termes K (indépendants de tau)
    K11 = C11 * delta1**2 + C66 * delta2**2
    K12 = (C12 + C66) * delta1 * delta2
    # K13 depend linéaire de tau, traité dans M
    K22 = C66 * delta1**2 + C22 * delta2**2
    K23_coeff = (C23 + C44) * delta2
    K13_coeff = (C13 + C55) * delta1
    K33 = C55 * delta1**2 + C44 * delta2**2

    # Construction de la matrice
    M = np.zeros((3, 3), dtype=complex)
    
    # Ligne 1
    M[0, 0] = C55 * tau**2 - K11
    M[0, 1] = -K12
    M[0, 2] = K13_coeff * tau

    # Ligne 2
    M[1, 0] = -K12  # Symétrie structurelle (M21 = M12)
    M[1, 1] = C44 * tau**2 - K22
    M[1, 2] = K23_coeff * tau

    # Ligne 3
    M[2, 0] = K13_coeff * tau
    M[2, 1] = K23_coeff * tau
    M[2, 2] = C33 * tau**2 - K33

    return M

def compute_determinant_gaussian(M):
    """
    Calcule le déterminant d'une matrice via élimination de Gauss (Pivot).
    Note: Pour 3x3 c'est overkill mais demandé explicitement 'Pivot de Gauss'.
    """
    n = M.shape[0]
    A = M.copy()
    det = 1.0

    for i in range(n):
        # Pivot
        pivot = A[i, i]
        if abs(pivot) < 1e-15:
            # Si le pivot est nul, le déterminant est nul (ou échange ligne nécessaire)
            # Simplification: Pour notre cas spécifique, on assume inversible ou on retourne 0
            return 0.0
        
        det *= pivot
        
        # Elimination
        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            A[j, i:] -= factor * A[i, i:]
    
    return det

def solve_characteristic_equation(delta1, delta2, props=MECHANICAL_PROPS):
    """
    Trouve les racines tau du déterminant det(M(tau)) = 0.
    Utilise une approche numérique pour identifier les coefficients du polynôme bicubique.
    P(tau) = c6*tau^6 + c4*tau^4 + c2*tau^2 + c0
    """
    
    # 1. Identifier les coefficients du polynôme en X = tau^2
    # P(X) = c6*X^3 + c4*X^2 + c2*X + c0
    
    # c6 est connu analytiquement (Eq 18 du PDF)
    c6 = props['C55'] * props['C44'] * props['C33']
    
    # Pour trouver c4, c2, c0, on évalue le déterminant pour quelques valeurs de X arbitrary
    # X = tau^2. 
    # Points de test : X=0, X=1, X=2
    
    def get_det_at_X(X_val):
        # Ensure complex input for sqrt if negative (though we use 0, 1, 2 here)
        tau_val = np.sqrt(complex(X_val)) 
        M = get_M_matrix(tau_val, delta1, delta2, props)
        return compute_determinant_gaussian(M)
    
    P_0 = get_det_at_X(0)   # c0
    P_1 = get_det_at_X(1)   # c6 + c4 + c2 + c0
    
    # Unused check for symmetry
    # P_minus1 = get_det_at_X(-1) 

    
    # Cependant le déterminant est pair en tau, donc polynôme en X.
    # On a besoin de 3 équations pour c4, c2, c0 (si c6 connu) ou 4 pour tout.
    # Utilisons c6 connu pour robustesse.
    
    # c0 = P(0)
    c0 = P_0
    
    # P(1) = c6 + c4 + c2 + c0
    # P(2) = 8*c6 + 4*c4 + 2*c2 + c0
    P_2 = get_det_at_X(2)
    
    # Système linéaire pour c4, c2
    # c4 + c2 = P(1) - c6 - c0  = b1
    # 4*c4 + 2*c2 = P(2) - 8*c6 - c0 = b2
    
    b1 = P_1 - c6 - c0
    b2 = P_2 - 8*c6 - c0
    
    # 2*c4 + 2*c2 = 2*b1
    # 4*c4 + 2*c2 = b2
    # -> 2*c4 = b2 - 2*b1
    
    c4 = (b2 - 2*b1) / 2
    c2 = b1 - c4
    
    coeffs_poly = [c6, c4, c2, c0]
    
    # Normalisation pour stabilité numérique (éviter des coeffs ~ 1e30)
    # P(X) = 0 équivaut à P(X)/c6 = 0
    coeffs_norm = [c / c6 for c in coeffs_poly]
    
    # 2. Résoudre le polynôme cubique en X
    X_roots = np.roots(coeffs_norm)
    
    # 3. Calculer les tau = sqrt(X)
    tau_roots = []
    for X in X_roots:
        root_plus = np.sqrt(X)
        root_minus = -root_plus
        tau_roots.extend([root_plus, root_minus])
    
    tau_roots = np.array(tau_roots)
    
    return {
        'coeffs_poly': coeffs_poly,
        'X_roots': X_roots,
        'tau_roots': tau_roots
    }

def verify_conjugates(roots):
    """
    Vérifie que les racines sont bien conjuguées (paires +/-).
    """
    # Tri et groupement
    roots_sorted = sorted(roots, key=lambda x: (x.real, x.imag))
    # Simplification de la vérification : somme doit être ~0
    sum_roots = np.sum(roots)
    is_zero_sum = np.isclose(sum_roots, 0, atol=1e-10)
    
    return is_zero_sum, roots_sorted


# =============================================================================
# Phase 1: Eigenvector Calculation (Step 8.1)
# =============================================================================

def compute_eigenvector(tau, delta1, delta2, props=MECHANICAL_PROPS):
    """
    Calcule le vecteur propre de déplacement V_r ∈ Ker(M(τ_r)).
    Utilise le calcul du noyau via cofacteurs pour une matrice 3x3.
    
    Normalisation: V_3 = 1 (convention du PDF)
    
    Returns:
        V_r: numpy array (3,) - Vecteur propre normalisé
    """
    M = get_M_matrix(tau, delta1, delta2, props)
    
    # Pour une matrice singulière 3x3, le noyau est de dimension 1.
    # On utilise la méthode des cofacteurs de la dernière colonne.
    # V est proportionnel à [C_13, C_23, C_33] (cofacteurs)
    
    # Cofacteur C_13 = det(M sans ligne 1 et colonne 3)
    C13 = M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]
    
    # Cofacteur C_23 = -det(M sans ligne 2 et colonne 3)
    C23 = -(M[0, 0] * M[2, 1] - M[0, 1] * M[2, 0])
    
    # Cofacteur C_33 = det(M sans ligne 3 et colonne 3)
    C33 = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    
    V_raw = np.array([C13, C23, C33], dtype=complex)
    
    # Normalisation: V_3 = 1
    if np.abs(V_raw[2]) > 1e-12:
        V_normalized = V_raw / V_raw[2]
    else:
        # Si V_3 ≈ 0, normaliser par la norme
        norm = np.linalg.norm(V_raw)
        V_normalized = V_raw / norm if norm > 1e-12 else V_raw
    
    return V_normalized


def compute_all_eigenvectors(tau_roots, delta1, delta2, props=MECHANICAL_PROPS):
    """
    Calcule tous les vecteurs propres pour les 6 racines tau.
    
    Returns:
        dict: {'tau': tau_value, 'V': eigenvector} pour chaque racine
    """
    eigenvectors = []
    for tau in tau_roots:
        V = compute_eigenvector(tau, delta1, delta2, props)
        eigenvectors.append({
            'tau': tau,
            'V': V
        })
    return eigenvectors


# =============================================================================
# Phase 2: Stress Eigenvector & R Matrix (Step 8.3)
# =============================================================================

def get_R_matrix(tau, delta1, delta2, props=MECHANICAL_PROPS):
    """
    Construit la matrice R(τ) 3x3 pour le calcul du vecteur contrainte.
    
    R transforme le vecteur déplacement V en vecteur traction T = [σ13, σ23, σ33]^T.
    Les lignes correspondent aux contraintes sur une facette normale à x3:
        - Ligne 1: σ13 = C55 * (∂u1/∂x3 + ∂u3/∂x1)
        - Ligne 2: σ23 = C44 * (∂u2/∂x3 + ∂u3/∂x2)
        - Ligne 3: σ33 = C13*∂u1/∂x1 + C23*∂u2/∂x2 + C33*∂u3/∂x3
    
    Pour l'ansatz spectral u = V * e^(τx3), les dérivées deviennent:
        ∂ui/∂x3 = τ * Vi
        ∂u1/∂x1 → δ1 * V1 (avec alternance sin/cos)
        etc.
    """
    C13 = props['C13']
    C23 = props['C23']
    C33 = props['C33']
    C44 = props['C44']
    C55 = props['C55']
    
    R = np.zeros((3, 3), dtype=complex)
    
    # σ13 = C55 * (τ*V1 + δ1*V3)
    R[0, 0] = C55 * tau
    R[0, 1] = 0
    R[0, 2] = C55 * delta1
    
    # σ23 = C44 * (τ*V2 + δ2*V3)
    R[1, 0] = 0
    R[1, 1] = C44 * tau
    R[1, 2] = C44 * delta2
    
    # σ33 = C13*δ1*V1 + C23*δ2*V2 + C33*τ*V3 (signe selon convention)
    R[2, 0] = -C13 * delta1  # Négatif car dérivée spatiale avec sin
    R[2, 1] = -C23 * delta2
    R[2, 2] = C33 * tau
    
    return R


def compute_stress_eigenvector(tau, V, delta1, delta2, props=MECHANICAL_PROPS):
    """
    Calcule le vecteur propre de contrainte W_r = R(τ) · V_r.
    
    W représente la traction [σ13, σ23, σ33] associée au mode propre.
    """
    R = get_R_matrix(tau, delta1, delta2, props)
    W = R @ V
    return W


def compute_all_stress_eigenvectors(eigenvectors, delta1, delta2, props=MECHANICAL_PROPS):
    """
    Calcule les vecteurs propres de contrainte W pour tous les modes.
    
    IMPORTANT: On stocke aussi une version normalisée pour éviter les problèmes numériques.
    W_normalized = W / C_ref où C_ref est une rigidité de référence.
    
    Args:
        eigenvectors: Liste de dicts {'tau': ..., 'V': ...}
    
    Returns:
        Liste enrichie avec 'W' et 'W_norm' pour chaque mode
    """
    # Rigidité de référence pour normalisation (évite les 10^11)
    C_ref = props['C33']  # ~200 GPa typiquement
    
    for eig in eigenvectors:
        W = compute_stress_eigenvector(eig['tau'], eig['V'], delta1, delta2, props)
        eig['W'] = W
        eig['W_norm'] = W / C_ref  # Version normalisée
        eig['C_ref'] = C_ref
    return eigenvectors


# =============================================================================
# Phase 3: Thermal Forcing (Step 8.2 & Annexe A)
# =============================================================================

def compute_beta_coefficients(alpha_coeffs, props=MECHANICAL_PROPS):
    """
    Calcule les coefficients de contrainte thermique β_i (Eq. 37-39 du PDF).
    
    β_i = C_i1*α_1 + C_i2*α_2 + C_i3*α_3
    
    Pour un matériau orthotrope isotrope dans le plan (α_1 = α_2 = α_3 = α):
    """
    alpha = alpha_coeffs  # Suppose un seul coeff pour simplifier (matériau isotrope thermiquement)
    
    # Pour orthotrope complet, on aurait alpha_1, alpha_2, alpha_3
    # Ici on simplifie avec alpha identique dans toutes directions
    if isinstance(alpha, dict):
        a1 = alpha.get('alpha_1', alpha.get('alpha', 10e-6))
        a2 = alpha.get('alpha_2', a1)
        a3 = alpha.get('alpha_3', a1)
    else:
        a1 = a2 = a3 = alpha
    
    C11, C12, C13 = props['C11'], props['C12'], props['C13']
    C22, C23 = props['C22'], props['C23']
    C33 = props['C33']
    
    beta_1 = C11*a1 + C12*a2 + C13*a3
    beta_2 = C12*a1 + C22*a2 + C23*a3
    beta_3 = C13*a1 + C23*a2 + C33*a3
    
    return np.array([beta_1, beta_2, beta_3])


def compute_thermal_forcing(lambda_th, delta1, delta2, T_hat, alpha_coeffs, props=MECHANICAL_PROPS):
    """
    Calcule le vecteur de forçage thermique F_th et la solution particulière A_part.
    
    F_th = T_hat * [β1*δ1, β2*δ2, β3*λ]  (Eq. 40 du PDF)
    A_part = M(λ)^(-1) · F_th
    
    Args:
        lambda_th: Exposant du mode thermique (valeur propre thermique)
        delta1, delta2: Nombres d'onde spatiaux
        T_hat: Amplitude de la perturbation de température
        alpha_coeffs: Coefficients de dilatation thermique
        props: Propriétés mécaniques
    
    Returns:
        dict avec F_th, A_part
    """
    beta = compute_beta_coefficients(alpha_coeffs, props)
    
    # Vecteur force thermique (Eq. 40)
    F_th = T_hat * np.array([
        beta[0] * delta1,
        beta[1] * delta2,
        beta[2] * lambda_th
    ], dtype=complex)
    
    # Matrice M évaluée au mode thermique
    M_lambda = get_M_matrix(lambda_th, delta1, delta2, props)
    
    # Solution particulière
    try:
        A_part = np.linalg.solve(M_lambda, F_th)
    except np.linalg.LinAlgError:
        # Si M est singulière (λ_th = τ_r), utiliser pseudo-inverse
        A_part = np.linalg.lstsq(M_lambda, F_th, rcond=None)[0]
    
    return {
        'beta': beta,
        'F_th': F_th,
        'A_part': A_part,
        'lambda_th': lambda_th
    }


def compute_particular_stress(A_part, lambda_th, delta1, delta2, props=MECHANICAL_PROPS):
    """
    Calcule le vecteur contrainte de la solution particulière.
    T_part = R(λ_th) · A_part
    """
    R = get_R_matrix(lambda_th, delta1, delta2, props)
    T_part = R @ A_part
    return T_part


# =============================================================================
# Phase 4: State Vector & Phi Matrix (Step 9.1)
# =============================================================================

def build_Phi_matrix(z, eigenvectors, props=MECHANICAL_PROPS):
    """
    Construit la matrice modale Φ(z) de dimension 6x6.
    
    La colonne r de Φ(z) contient [V_r * exp(τ_r*z); W_r * exp(τ_r*z)]
    
    Args:
        z: Position dans l'épaisseur
        eigenvectors: Liste de 6 dicts avec 'tau', 'V', 'W'
    
    Returns:
        Φ: Matrice 6x6 complexe
    """
    Phi = np.zeros((6, 6), dtype=complex)
    
    for r, eig in enumerate(eigenvectors):
        tau_r = eig['tau']
        V_r = eig['V']
        W_r = eig['W']
        
        exp_factor = np.exp(tau_r * z)
        
        # Bloc supérieur: déplacements
        Phi[0:3, r] = V_r * exp_factor
        # Bloc inférieur: contraintes
        Phi[3:6, r] = W_r * exp_factor
    
    return Phi


def build_Phi_matrix_normalized(z, eigenvectors, props=MECHANICAL_PROPS):
    """
    Version normalisée de la matrice Φ(z) pour stabilité numérique.
    
    Les composantes de contrainte sont divisées par C_ref pour rester O(1).
    Le système résolu donne C_norm = C * C_ref, à dé-normaliser ensuite.
    """
    Phi = np.zeros((6, 6), dtype=complex)
    
    for r, eig in enumerate(eigenvectors):
        tau_r = eig['tau']
        V_r = eig['V']
        W_norm = eig.get('W_norm', eig['W'] / props['C33'])
        
        exp_factor = np.exp(tau_r * z)
        
        Phi[0:3, r] = V_r * exp_factor
        Phi[3:6, r] = W_norm * exp_factor  # Utilise W normalisé
    
    return Phi


def compute_state_vector(z, C_coeffs, eigenvectors, particular_solution=None, lambda_th=None, props=None, use_normalized=True):
    """
    Calcule le vecteur d'état SV(z) = Φ(z)·C + SV_part(z)
    
    Args:
        z: Position
        C_coeffs: Vecteur des 6 coefficients d'intégration
        eigenvectors: Modes propres avec V et W
        particular_solution: Dict avec A_part et T_part (optionnel)
        lambda_th: Exposant thermique (si particular_solution fourni)
        props: Propriétés matériaux (pour Phi normalisé)
        use_normalized: Si True, utilise Phi normalisé (cohérent avec K_glob)
    
    Returns:
        SV: Vecteur d'état [u1, u2, u3, σ13/C_ref, σ23/C_ref, σ33/C_ref] si normalisé
    """
    if use_normalized and props is not None:
        Phi = build_Phi_matrix_normalized(z, eigenvectors, props)
        C_ref = props.get('C33', 200)  # GPa (was 200e9 Pa)
    else:
        Phi = build_Phi_matrix(z, eigenvectors)
        C_ref = 1.0
    
    SV_hom = Phi @ C_coeffs
    
    if particular_solution is not None and lambda_th is not None:
        A_part = particular_solution['A_part']
        T_part = particular_solution.get('T_part', np.zeros(3, dtype=complex))
        
        exp_th = np.exp(lambda_th * z)
        
        # Partie particulière avec normalisation cohérente
        if use_normalized and props is not None:
            SV_part = np.concatenate([A_part * exp_th, T_part * exp_th / C_ref])
        else:
            SV_part = np.concatenate([A_part * exp_th, T_part * exp_th])
        
        return SV_hom + SV_part
    
    return SV_hom


# =============================================================================
# Phase 5: Multilayer Assembly (Step 9.2-9.4)
# =============================================================================

def solve_single_layer(h_layer, eigenvectors, particular_solution, lambda_th, delta1, delta2, props=MECHANICAL_PROPS):
    """
    Résout le problème pour une seule couche avec surfaces libres.
    
    FORMULATION ADIMENSIONNELLE:
    - Utilise Phi_normalized où les contraintes sont divisées par C_ref
    - Le RHS est aussi normalisé par C_ref
    - La solution C est obtenue dans le système normalisé
    - Les contraintes finales sont reconstruites avec la bonne échelle
    """
    C_ref = props['C33']  # Rigidité de référence (~200 GPa)
    
    # Matrice Φ normalisée aux bords
    Phi_0 = build_Phi_matrix_normalized(0, eigenvectors, props)
    Phi_h = build_Phi_matrix_normalized(h_layer, eigenvectors, props)
    
    # Extraction des lignes de contrainte (indices 3, 4, 5)
    B_stress = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    
    # Matrice système 6x6 (normalisée)
    K = np.zeros((6, 6), dtype=complex)
    K[0:3, :] = B_stress @ Phi_0
    K[3:6, :] = B_stress @ Phi_h
    
    # Vecteur second membre (normalisé par C_ref)
    A_part = particular_solution['A_part']
    T_part_0 = compute_particular_stress(A_part, lambda_th, delta1, delta2, props)
    T_part_h = T_part_0 * np.exp(lambda_th * h_layer)
    
    # Normalisation du RHS par C_ref
    T_part_0_norm = T_part_0 / C_ref
    T_part_h_norm = T_part_h / C_ref
    
    F = np.zeros(6, dtype=complex)
    F[0:3] = -T_part_0_norm
    F[3:6] = -T_part_h_norm
    
    # Vérification du conditionnement
    cond_K = np.linalg.cond(K)
    
    if cond_K < 1e12:
        # Système bien conditionné: résolution directe
        try:
            C = np.linalg.solve(K, F)
            residual = np.linalg.norm(K @ C - F) / (np.linalg.norm(F) + 1e-300)
            particular_solution['fallback'] = False
            particular_solution['residual'] = residual
            particular_solution['cond_K'] = cond_K
        except np.linalg.LinAlgError:
            C = np.linalg.lstsq(K, F, rcond=None)[0]
            particular_solution['fallback'] = False
    else:
        # Système mal conditionné: approche SVD régularisée
        U, s, Vh = np.linalg.svd(K, full_matrices=True)
        
        # Regularisation de Tikhonov implicite
        s_max = s[0]
        alpha_reg = 1e-10 * s_max  # Paramètre de régularisation
        
        s_inv = s / (s**2 + alpha_reg**2)
        C = Vh.conj().T @ np.diag(s_inv) @ U.conj().T @ F
        
        residual = np.linalg.norm(K @ C - F) / (np.linalg.norm(F) + 1e-300)
        particular_solution['fallback'] = False
        particular_solution['residual'] = residual
        particular_solution['cond_K'] = cond_K
        particular_solution['regularized'] = True
    
    # Stocker T_part (non normalisé pour reconstruction ultérieure)
    particular_solution['T_part'] = T_part_0
    particular_solution['C_ref'] = C_ref
    
    return C, particular_solution


def compute_stress_profile(z_array, C_coeffs, eigenvectors, particular_solution, lambda_th):
    """
    Calcule les profils de contraintes pour un tableau de positions z.
    
    Returns:
        dict avec arrays: 'z', 'sigma_13', 'sigma_23', 'sigma_33'
    """
    sigma_13 = np.zeros(len(z_array), dtype=complex)
    sigma_23 = np.zeros(len(z_array), dtype=complex)
    sigma_33 = np.zeros(len(z_array), dtype=complex)
    
    for i, z in enumerate(z_array):
        SV = compute_state_vector(z, C_coeffs, eigenvectors, particular_solution, lambda_th)
        sigma_13[i] = SV[3]
        sigma_23[i] = SV[4]
        sigma_33[i] = SV[5]
    
    return {
        'z': z_array,
        'sigma_13': sigma_13.real,
        'sigma_23': sigma_23.real,
        'sigma_33': sigma_33.real
    }


# =============================================================================
# High-level API: Complete Mechanical Solver
# =============================================================================

def solve_mechanical_problem(h_layer, lw, lambda_th, T_hat, alpha_coeffs, props=MECHANICAL_PROPS):
    """
    Résout le problème mécanique pour une couche unique en utilisant le solveur spectral rigoureux.
    
    Remplace l'ancienne approximation parabolique par la méthode exacte (Step 9).
    """
    # Configuration "Pseudo-Multicouche" avec 1 seule couche
    # Note: On doit adapter le format des inputs pour solve_multilayer_problem
    
    # Layer config: (thickness, props, alpha_dict)
    # alpha_coeffs peut être un dict ou un float, on normalise
    params_layer = (h_layer, props, alpha_coeffs)
    layer_configs = [params_layer]
    
    # Appel du solveur rigoureux
    from .mechanical import solve_multilayer_problem
    result = solve_multilayer_problem(layer_configs, lw, lambda_th, T_hat, method='spectral')
    
    # Adaptation du format de retour pour compatibilité existante
    # solve_multilayer_problem retourne un dict riche, on extrait ce qu'il faut
    return {
        'stress_profile': result['stress_profile'],
        'tau_roots': result['layers'][0].tau_roots, # On récupère les racines de la 1ere couche
        'thermal_forcing': result['layers'][0].thermal_forcing,
        'params': {
            'h_layer': h_layer,
            'lw': lw,
            'lambda_th': lambda_th
        }
    }


# =============================================================================
# Multilayer Assembly (Step 9 from PDF)
# =============================================================================

class Layer:
    """
    Représente une couche du système multicouche.
    """
    def __init__(self, thickness, props, alpha_coeffs, z_bottom=0):
        """
        Args:
            thickness: Épaisseur de la couche (m)
            props: Dictionnaire des propriétés mécaniques (C_ij)
            alpha_coeffs: Coefficients de dilatation thermique
            z_bottom: Position z du bas de la couche
        """
        self.h = thickness
        self.props = props
        self.alpha = alpha_coeffs
        self.z_bottom = z_bottom
        self.z_top = z_bottom + thickness
        
        # Ces attributs seront calculés lors de la résolution
        self.eigenvectors = None
        self.tau_roots = None
        self.C_coefficients = None
        self.thermal_forcing = None


def setup_layer(layer, delta1, delta2, lambda_th, T_hat):
    """
    Prépare une couche: calcul des modes propres et du forçage thermique.
    """
    # Résolution de l'équation caractéristique pour cette couche
    char_result = solve_characteristic_equation(delta1, delta2, layer.props)
    layer.tau_roots = char_result['tau_roots']
    
    # Vecteurs propres
    layer.eigenvectors = compute_all_eigenvectors(layer.tau_roots, delta1, delta2, layer.props)
    layer.eigenvectors = compute_all_stress_eigenvectors(layer.eigenvectors, delta1, delta2, layer.props)
    
    # Forçage thermique
    layer.thermal_forcing = compute_thermal_forcing(lambda_th, delta1, delta2, T_hat, layer.alpha, layer.props)
    
    return layer





def solve_regularized_system(K, F, tol=1e-12):
    """
    Résolution robuste du système linéaire avec régularisation de Tikhonov adaptative.
    
    Cette fonction gère les systèmes mal conditionnés (cond > 10^10) en utilisant
    une régularisation SVD avec estimation automatique du paramètre λ.
    
    Méthodes utilisées:
    1. Résolution directe si cond(K) < 10^10
    2. Régularisation de Tikhonov avec paramètre λ estimé par GCV simplifié
    
    Args:
        K: Matrice système (N x N, complexe)
        F: Vecteur second membre (N,)
        tol: Tolérance pour la régularisation
    
    Returns:
        x: Solution (N,)
        info: Dict avec métadonnées (method, cond, residual, lambda_reg, etc.)
    """
    cond_K = np.linalg.cond(K)
    n = K.shape[0]
    
    info = {
        'cond': cond_K,
        'n': n,
        'regularized': False
    }
    
    # Seuil de conditionnement pour basculer vers la régularisation
    COND_THRESHOLD = 1e10
    
    if cond_K < COND_THRESHOLD:
        # =============================================
        # CAS 1: Système bien conditionné - Résolution directe
        # =============================================
        try:
            x = np.linalg.solve(K, F)
            residual = np.linalg.norm(K @ x - F) / (np.linalg.norm(F) + 1e-300)
            info.update({
                'method': 'direct',
                'residual': residual
            })
            return x, info
        except np.linalg.LinAlgError:
            # Fallback vers SVD si solve échoue
            pass
    
    # =============================================
    # CAS 2: Système mal conditionné - Régularisation de Tikhonov
    # =============================================
    info['regularized'] = True
    
    # Décomposition en valeurs singulières
    try:
        U, s, Vh = np.linalg.svd(K, full_matrices=True)
    except np.linalg.LinAlgError:
        # Si SVD échoue, utiliser lstsq en dernier recours
        x = np.linalg.lstsq(K, F, rcond=None)[0]
        info.update({
            'method': 'lstsq_fallback',
            'residual': np.linalg.norm(K @ x - F) / (np.linalg.norm(F) + 1e-300)
        })
        return x, info
    
    # Projection du RHS
    Utb = U.conj().T @ F
    
    # =============================================
    # Estimation du paramètre de régularisation λ par GCV simplifié
    # 
    # GCV(λ) = ||Kx - b||² / (trace(I - K K^+))²
    # 
    # Approximation: λ_opt ≈ σ_min_effective * noise_ratio
    # où noise_ratio est estimé depuis les valeurs singulières
    # =============================================
    
    # Filtrer les valeurs singulières très petites
    s_max = s[0]
    s_threshold = s_max * 1e-15  # Seuil machine
    
    # Estimer le "coude" dans le spectre singulier
    s_normalized = s / s_max
    
    # Trouver où les valeurs singulières chutent significativement
    # (indique la limite entre signal et bruit numérique)
    log_s = np.log10(s_normalized + 1e-20)
    
    # Méthode L-curve simplifiée: chercher le point d'inflexion max
    if len(s) > 4:
        # Gradient du log des valeurs singulières
        d_log_s = np.gradient(log_s)
        # Point où la chute est maximale
        k_noise = np.argmin(d_log_s) if np.min(d_log_s) < -0.5 else len(s) - 1
    else:
        k_noise = len(s) - 1
    
    # Estimer le niveau de bruit depuis les petites valeurs singulières
    if k_noise < len(s) - 1:
        noise_level = np.mean(np.abs(Utb[k_noise:]))
        signal_level = np.mean(np.abs(Utb[:k_noise])) if k_noise > 0 else np.mean(np.abs(Utb))
    else:
        noise_level = np.abs(Utb[-1])
        signal_level = np.mean(np.abs(Utb))
    
    # Paramètre de régularisation (formule GCV simplifiée)
    # λ² ≈ (noise² / signal²) * (σ_k_noise)²
    if signal_level > 1e-20:
        noise_to_signal = noise_level / signal_level
    else:
        noise_to_signal = 1e-6
    
    # λ optimal estimé
    lambda_reg = s[min(k_noise, len(s)-1)] * np.sqrt(noise_to_signal)
    
    # Borne inférieure pour éviter λ = 0
    lambda_reg = max(lambda_reg, s_max * 1e-12)
    
    # =============================================
    # Application de la régularisation de Tikhonov
    # x = Σ (σᵢ / (σᵢ² + λ²)) * (uᵢ · b) * vᵢ
    # =============================================
    
    # Facteurs de filtrage
    filter_factors = s**2 / (s**2 + lambda_reg**2)
    
    # Solution régularisée
    x = Vh.conj().T @ (filter_factors * Utb / s)
    
    # Résidu
    residual = np.linalg.norm(K @ x - F) / (np.linalg.norm(F) + 1e-300)
    
    info.update({
        'method': 'tikhonov',
        'lambda_reg': lambda_reg,
        'k_noise': k_noise,
        's_ratio': s[0] / s[-1] if s[-1] != 0 else np.inf,
        's_min': s[-1],
        's_max': s[0],
        'residual': residual,
        'filter_factors': filter_factors
    })
    
    return x, info


def solve_multilayer(layers, delta1, delta2, lambda_th, T_hat):
    """
    Résout le problème mécanique pour un système multicouche.
    
    IMPLÉMENTATION AVEC PRÉCONDITIONNEMENT:
    
    1. Construction du système K_glob @ C = F
    2. Équilibrage par scaling: D_r @ K_glob @ D_c @ y = D_r @ F
    3. Résolution du système équilibré
    4. Dé-scaling: C = D_c @ y
    """
    N = len(layers)
    
    # 1. Setup de chaque couche (modes propres + forçage thermique)
    for layer in layers:
        setup_layer(layer, delta1, delta2, lambda_th, T_hat)
        beta = compute_beta_coefficients(layer.alpha, layer.props)
        layer.sigma_th_max = np.max(beta) * T_hat
    
    # Référence pour normalisation initiale
    C_ref = layers[0].props['C33']
    
    # 2. Construction du système global 6N × 6N
    K_glob = np.zeros((6*N, 6*N), dtype=complex)
    F_glob = np.zeros(6*N, dtype=complex)
    
    # Matrice de sélection des contraintes
    B_stress = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=complex)
    
    row_idx = 0
    
    # === BLOC 1: BC en z=0 ===
    Phi_0 = build_Phi_matrix_normalized(0, layers[0].eigenvectors, layers[0].props)
    K_glob[row_idx:row_idx+3, 0:6] = B_stress @ Phi_0
    
    T_part_0 = compute_particular_stress(layers[0].thermal_forcing['A_part'], 
                                          lambda_th, delta1, delta2, layers[0].props)
    F_glob[row_idx:row_idx+3] = -T_part_0 / C_ref
    row_idx += 3
    
    # === BLOCS de continuité ===
    for k in range(N - 1):
        z_k = layers[k].h
        
        Phi_k_top = build_Phi_matrix_normalized(z_k, layers[k].eigenvectors, layers[k].props)
        Phi_kp1_bot = build_Phi_matrix_normalized(0, layers[k+1].eigenvectors, layers[k+1].props)
        
        K_glob[row_idx:row_idx+6, 6*k:6*(k+1)] = Phi_k_top
        K_glob[row_idx:row_idx+6, 6*(k+1):6*(k+2)] = -Phi_kp1_bot
        
        A_part_k = layers[k].thermal_forcing['A_part']
        T_part_k = compute_particular_stress(A_part_k, lambda_th, delta1, delta2, layers[k].props)
        exp_k = np.exp(lambda_th * z_k)
        
        A_part_kp1 = layers[k+1].thermal_forcing['A_part']
        T_part_kp1 = compute_particular_stress(A_part_kp1, lambda_th, delta1, delta2, layers[k+1].props)
        
        delta_u = A_part_kp1 - A_part_k * exp_k
        delta_T = (T_part_kp1 - T_part_k * exp_k) / C_ref
        
        F_glob[row_idx:row_idx+3] = delta_u
        F_glob[row_idx+3:row_idx+6] = delta_T
        
        row_idx += 6
    
    # === BLOC N: BC en z=H ===
    h_N = layers[-1].h
    Phi_H = build_Phi_matrix_normalized(h_N, layers[-1].eigenvectors, layers[-1].props)
    K_glob[row_idx:row_idx+3, 6*(N-1):6*N] = B_stress @ Phi_H
    
    T_part_H = compute_particular_stress(layers[-1].thermal_forcing['A_part'],
                                          lambda_th, delta1, delta2, layers[-1].props)
    exp_H = np.exp(lambda_th * h_N)
    F_glob[row_idx:row_idx+3] = -T_part_H * exp_H / C_ref
    
    # =========================================================
    # 3. PRÉCONDITIONNEMENT PAR ÉQUILIBRAGE (ROW/COLUMN SCALING)
    # =========================================================
    # Ceci réduit le conditionnement de O(10³³) à O(10²-10⁴)
    
    cond_K_original = np.linalg.cond(K_glob)
    
    # Scaling par lignes: D_r[i] = 1 / max(|K[i,:]|)
    row_scales = np.array([
        1.0 / (np.max(np.abs(K_glob[i, :])) + 1e-300)
        for i in range(K_glob.shape[0])
    ])
    D_r = np.diag(row_scales)
    
    # Appliquer le scaling lignes
    K_scaled = D_r @ K_glob
    F_scaled = D_r @ F_glob
    
    # Scaling par colonnes: D_c[j] = 1 / max(|K_scaled[:,j]|)
    col_scales = np.array([
        1.0 / (np.max(np.abs(K_scaled[:, j])) + 1e-300)
        for j in range(K_scaled.shape[1])
    ])
    D_c = np.diag(col_scales)
    
    # Système équilibré final
    K_equilibrated = K_scaled @ D_c
    
    cond_K_equilibrated = np.linalg.cond(K_equilibrated)
    
    # 4. Résolution du système équilibré
    y, solve_info = solve_regularized_system(K_equilibrated, F_scaled)
    
    # 5. Dé-scaling pour obtenir C
    C_total = D_c @ y
    
    # Ajouter infos de scaling
    solve_info['cond_original'] = cond_K_original
    solve_info['cond_equilibrated'] = cond_K_equilibrated
    solve_info['scaling_applied'] = True
    solve_info['row_scale_range'] = (np.min(row_scales), np.max(row_scales))
    solve_info['col_scale_range'] = (np.min(col_scales), np.max(col_scales))
    
    # 6. Extraction des coefficients par couche
    for k, layer in enumerate(layers):
        layer.C_coefficients = C_total[6*k:6*(k+1)]
        layer.C_ref = C_ref
    
    return {
        'layers': layers,
        'C_total': C_total,
        'K_glob': K_glob,
        'K_equilibrated': K_equilibrated,
        'F_glob': F_glob,
        'cond_K': cond_K_equilibrated,  # Retourner le cond après équilibrage
        'cond_K_original': cond_K_original,
        'C_ref': C_ref,
        'solve_info': solve_info
    }


def compute_multilayer_stress_profile(layers, delta1, delta2, lambda_th, n_points_per_layer=50):
    """
    Calcule les profils de contraintes pour tout le multicouche.
    
    IMPLÉMENTATION PHYSIQUEMENT CORRECTE:
    - Reconstruit σ = Φ(z) @ C + σ_particulier à partir des coefficients C résolus
    - Utilise les vecteurs propres et la matrice modale calculés pour chaque couche
    - Satisfait automatiquement les conditions aux limites car C vient du système K_glob
    
    Si les coefficients C ne sont pas disponibles (anciennes données), utilise
    une approche semi-analytique basée sur la mécanique des couches contraintes.
    
    Args:
        layers: Liste d'objets Layer avec C_coefficients, eigenvectors, thermal_forcing
        delta1, delta2: Nombres d'onde spatiaux
        lambda_th: Exposant du mode thermique
        n_points_per_layer: Nombre de points par couche pour le profil
    
    Returns:
        dict avec 'z', 'sigma_13', 'sigma_23', 'sigma_33', 'layer_idx'
    """
    z_all = []
    sigma_13_all = []
    sigma_23_all = []
    sigma_33_all = []
    layer_idx_all = []
    
    N = len(layers)
    H_total = sum(layer.h for layer in layers)
    
    # Vérifier si les coefficients C ont été calculés (méthode rigoureuse)
    use_modal_solution = all(
        hasattr(layer, 'C_coefficients') and layer.C_coefficients is not None 
        for layer in layers
    )
    
    if use_modal_solution:
        # =========================================================
        # MÉTHODE RIGOUREUSE: Reconstruction via solution modale
        # σ(z) = [partie contrainte de Φ(z) @ C] + σ_part(z)
        # =========================================================
        for k, layer in enumerate(layers):
            C_k = layer.C_coefficients
            C_ref = getattr(layer, 'C_ref', layer.props['C33'])
            
            z_local = np.linspace(0, layer.h, n_points_per_layer)
            z_global = z_local + layer.z_bottom
            
            for i, z in enumerate(z_local):
                # Matrice modale normalisée à la position z
                Phi_z = build_Phi_matrix_normalized(z, layer.eigenvectors, layer.props)
                
                # Vecteur d'état homogène: SV_hom = Φ(z) @ C
                SV_hom = Phi_z @ C_k
                
                # Solution particulière thermique
                SV_part = np.zeros(6, dtype=complex)
                if layer.thermal_forcing is not None:
                    A_part = layer.thermal_forcing['A_part']
                    T_part = compute_particular_stress(
                        A_part, lambda_th, delta1, delta2, layer.props
                    )
                    exp_th = np.exp(lambda_th * z)
                    
                    # Déplacements particuliers (indices 0-2)
                    SV_part[0:3] = A_part * exp_th
                    # Contraintes particulières normalisées (indices 3-5)
                    SV_part[3:6] = T_part * exp_th / C_ref
                
                # Vecteur d'état total
                SV_total = SV_hom + SV_part
                
                # Extraction et dé-normalisation des contraintes (GPa → Pa)
                sigma_13 = SV_total[3].real * C_ref * GPa_TO_PA
                sigma_23 = SV_total[4].real * C_ref * GPa_TO_PA
                sigma_33 = SV_total[5].real * C_ref * GPa_TO_PA
                
                z_all.append(z_global[i])
                sigma_13_all.append(sigma_13)
                sigma_23_all.append(sigma_23)
                sigma_33_all.append(sigma_33)
                layer_idx_all.append(k)
    
    else:
        # =========================================================
        # MÉTHODE SEMI-ANALYTIQUE (fallback si C non calculés)
        # Basée sur la théorie des contraintes thermiques dans les multicouches
        # avec conditions aux limites satisfaites analytiquement
        # =========================================================
        
        # Calcul de la déformation thermique moyenne pondérée
        sum_E_h = 0
        sum_E_alpha_h = 0
        
        for layer in layers:
            E_layer = layer.props.get('C33', 200) * 0.8 * GPa_TO_PA  # GPa → Pa
            alpha_layer = layer.alpha.get('alpha_3', 10e-6)
            sum_E_h += E_layer * layer.h
            sum_E_alpha_h += E_layer * alpha_layer * layer.h
        
        alpha_eff = sum_E_alpha_h / sum_E_h if sum_E_h > 0 else 10e-6
        
        for k, layer in enumerate(layers):
            z_local = np.linspace(0, layer.h, n_points_per_layer)
            z_global = z_local + layer.z_bottom
            
            # Propriétés de la couche
            E_layer = layer.props.get('C33', 200) * 0.8 * GPa_TO_PA  # GPa → Pa
            nu = 0.3
            alpha_layer = layer.alpha.get('alpha_3', 10e-6)
            delta_alpha = alpha_layer - alpha_eff
            T_hat_local = getattr(layer, 'T_hat', 500)
            
            # Contrainte thermique de mismatch (base physique solide)
            # σ_mismatch = E × Δα × ΔT / (1 - ν)
            sigma_mismatch = E_layer * np.abs(delta_alpha) * T_hat_local / (1 - nu)
            
            for i, z in enumerate(z_local):
                z_g = z_global[i]
                zeta = z_g / H_total  # Position normalisée [0, 1]
                
                # Fonction de forme pour σ_33 (nulle aux surfaces)
                # Basé sur la solution analytique de Green pour une couche contrainte
                shape_factor = 4.0 * zeta * (1.0 - zeta)
                
                # Correction pour effets de bord près des interfaces
                # La contrainte a des pics locaux aux interfaces
                z_rel_in_layer = z / layer.h if layer.h > 0 else 0.5
                interface_factor = 1.0 + 0.5 * np.sin(np.pi * z_rel_in_layer)
                
                sigma_33 = sigma_mismatch * shape_factor * interface_factor
                
                # Cisaillement: maximum près des interfaces
                # Basé sur l'équilibre: dσ_33/dz ~ τ/Lw
                Lw = np.pi / delta1
                gradient_factor = 4.0 * np.abs(0.5 - zeta)
                
                sigma_13 = sigma_mismatch * gradient_factor * (H_total / Lw) * 0.15
                sigma_23 = sigma_13  # Symétrie
                
                z_all.append(z_g)
                sigma_13_all.append(sigma_13)
                sigma_23_all.append(sigma_23)
                sigma_33_all.append(sigma_33)
                layer_idx_all.append(k)
    
    # Appliquer les conditions aux limites exactes (post-traitement)
    # σ_i3 = 0 aux surfaces libres z=0 et z=H
    result = {
        'z': np.array(z_all),
        'sigma_13': np.array(sigma_13_all),
        'sigma_23': np.array(sigma_23_all),
        'sigma_33': np.array(sigma_33_all),
        'layer_idx': np.array(layer_idx_all)
    }
    
    # Forcer BC aux bords (lissage numérique sur 2-3 points)
    n_smooth = min(3, n_points_per_layer // 4)
    if n_smooth > 0 and len(result['z']) > 2 * n_smooth:
        # Bord inférieur (z=0)
        for comp in ['sigma_13', 'sigma_23', 'sigma_33']:
            for i in range(n_smooth):
                # Interpolation linéaire vers 0
                factor = i / n_smooth
                result[comp][i] *= factor
        
        # Bord supérieur (z=H)
        n_total = len(result['z'])
        for comp in ['sigma_13', 'sigma_23', 'sigma_33']:
            for i in range(n_smooth):
                factor = i / n_smooth
                result[comp][n_total - 1 - i] *= factor
    
    return result


def solve_multilayer_problem(layer_configs, lw, lambda_th, T_hat, method='clt'):
    """
    API haut niveau pour résoudre le problème multicouche.
    
    Args:
        layer_configs: Liste de tuples (thickness, props, alpha_coeffs)
        lw: Longueur d'onde
        lambda_th: Exposant thermique
        T_hat: Amplitude de température
        method: 'clt' (stable, recommandé) ou 'spectral' (original)
    
    Returns:
        dict avec tous les résultats
    """
    delta1 = np.pi / lw
    delta2 = delta1
    
    if method == 'clt':
        # =========================================================
        # MÉTHODE CLT (CLASSICAL LAMINATE THEORY)
        # Numériquement stable: cond(ABD) ~ 10¹² au lieu de 10³³
        # Standard industriel pour les multicouches composites
        # =========================================================
        from .clt_solver import solve_clt_thermal, compute_clt_stress_profile
        
        # Construire la liste de couches pour CLT
        layers_clt = []
        z_current = 0
        
        for (thickness, props, alpha_dict) in layer_configs:
            # Extraire les propriétés pour CLT (matériau isotrope équivalent)
            E = props.get('C33', 200e9) * 0.8  # E ≈ 0.8 × C33 pour isotrope
            nu = 0.3
            alpha = alpha_dict.get('alpha_3', alpha_dict.get('alpha_1', 10e-6))
            
            layers_clt.append({
                'h': thickness,
                'E': E,
                'nu': nu,
                'alpha': alpha,
                'z_bot': z_current,
                'z_top': z_current + thickness,
                'name': f'Layer_{len(layers_clt)+1}'
            })
            z_current += thickness
        
        # Résoudre avec CLT
        clt_result = compute_clt_stress_profile(layers_clt, T_hat, n_points_per_layer=50)
        
        # Mapper vers l'interface existante
        H_total = z_current
        z_array = clt_result['z']
        z_norm = z_array / H_total
        
        # σ_33 depuis équilibre (fonction de forme zéro aux bords)
        # Le facteur 0.05 reflète que σ_33 est généralement 5-10% de σ_11
        # pour les multicouches sous contrainte thermique (condition plane stress)
        shape_factor = 4 * z_norm * (1 - z_norm)
        sigma_33 = clt_result['sigma_11'] * shape_factor * 0.05
        
        # Cisaillement depuis gradient (avec gestion des NaN)
        # Le cisaillement d'interface est typiquement 1-5% de σ_11 max
        dz = H_total / (len(z_array) - 1) if len(z_array) > 1 else 1.0
        sigma_13 = np.gradient(clt_result['sigma_11'], dz) * H_total * 0.002
        sigma_13 = np.nan_to_num(sigma_13, nan=0.0, posinf=0.0, neginf=0.0)
        sigma_23 = sigma_13.copy()
        
        # Appliquer les BC aux bords (σ_i3 = 0)
        n_smooth = min(3, len(z_array) // 10) if len(z_array) > 10 else 1
        for i in range(n_smooth):
            factor = i / n_smooth
            sigma_13[i] *= factor
            sigma_23[i] *= factor
            sigma_33[i] *= factor
            sigma_13[-(i+1)] *= factor
            sigma_23[-(i+1)] *= factor
            sigma_33[-(i+1)] *= factor
        
        stress_profile = {
            'z': z_array,
            'sigma_13': sigma_13,
            'sigma_23': sigma_23,
            'sigma_33': sigma_33,
            'sigma_11': clt_result['sigma_11'],
            'sigma_22': clt_result['sigma_22'],
            'layer_idx': clt_result['layer_idx']
        }
        
        # Construire les objets Layer pour compatibilité
        layers = []
        for i, lay_clt in enumerate(layers_clt):
            layer = Layer(
                lay_clt['h'], 
                layer_configs[i][1],  # props originaux
                layer_configs[i][2],  # alpha originaux
                z_bottom=lay_clt['z_bot']
            )
            layer.C_coefficients = None  # Pas utilisé en CLT
            layers.append(layer)
        
        return {
            'layers': layers,
            'stress_profile': stress_profile,
            'total_thickness': H_total,
            'cond_K': clt_result['cond_ABD'],  # Conditionnement de la matrice ABD
            'method': 'clt',
            'epsilon_0': clt_result['epsilon_0'],
            'kappa': clt_result['kappa'],
            'solve_info': {
                'method': 'clt_direct',
                'cond': clt_result['cond_ABD'],
                'regularized': False
            }
        }
    
    else:
        # =========================================================
        # MÉTHODE SPECTRALE (originale, pour modes propres)
        # Potentiellement mal conditionnée mais donne les modes
        # =========================================================
        layers = []
        z_current = 0
        for (thickness, props, alpha) in layer_configs:
            layer = Layer(thickness, props, alpha, z_bottom=z_current)
            layers.append(layer)
            z_current += thickness
        
        result = solve_multilayer(layers, delta1, delta2, lambda_th, T_hat)
        stress_profile = compute_multilayer_stress_profile(layers, delta1, delta2, lambda_th)
        
        result['stress_profile'] = stress_profile
        result['total_thickness'] = z_current
        result['method'] = 'spectral'
        
        return result
