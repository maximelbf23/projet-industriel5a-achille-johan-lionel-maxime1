"""
Module Mécanique - Méthodologie PDF
====================================

Ce module implémente la résolution du problème thermoélastique multicouche
en suivant exactement la méthodologie de `equilibre_local_corrige.pdf`.

Structure:
    Section 2 : Forme de la solution générale
    Section 5 : Opérateurs L_jk et termes thermiques Q_α
    Section 6 : Matrice Γ(τ) et système homogène
    Section 7 : Assemblage 9×9 avec chargement thermique

Auteur: Projet Industriel 5A
Référence: equilibre_local_corrige.pdf
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .constants import MECHANICAL_PROPS, GPa_TO_PA

# =============================================================================
# SECTION 2: Forme de la solution générale (Eq. 44-48)
# =============================================================================

def build_displacement_field(A_amplitudes: np.ndarray, tau_roots: np.ndarray, x3: float) -> np.ndarray:
    """
    Construit le champ de déplacement U(x₃) pour une position donnée.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 2
    ================================================
    
    U_α(x₃) = Σ_{r=1}^{3} A_α^r × exp(τ_r × x₃)
    
    où:
        - A_α^r sont les amplitudes pour direction α et mode r
        - τ_r sont les valeurs propres caractéristiques (3 modes à Re(τ) < 0)
        - x₃ est la position dans l'épaisseur
    
    Args:
        A_amplitudes: Matrice (3, 3) où A[α, r] = A_α^{r+1}
                      Lignes: directions α ∈ {0, 1, 2} → {1, 2, 3}
                      Colonnes: modes r ∈ {0, 1, 2} → {1, 2, 3}
        tau_roots: Vecteur (3,) des valeurs propres [τ₁, τ₂, τ₃]
        x3: Position dans l'épaisseur de la couche
    
    Returns:
        U: Vecteur (3,) des déplacements [U₁, U₂, U₃]
    """
    U = np.zeros(3, dtype=complex)
    
    for alpha in range(3):  # Directions 1, 2, 3
        for r in range(3):  # Modes 1, 2, 3
            U[alpha] += A_amplitudes[alpha, r] * np.exp(tau_roots[r] * x3)
    
    return U


def build_displacement_derivatives(A_amplitudes: np.ndarray, tau_roots: np.ndarray, x3: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les dérivées premières et secondes du champ de déplacement.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 3
    ================================================
    
    Dérivée première:
        dU_α/dx₃ = Σ_{r=1}^{3} τ_r × A_α^r × exp(τ_r × x₃)
    
    Dérivée seconde:
        d²U_α/dx₃² = Σ_{r=1}^{3} τ_r² × A_α^r × exp(τ_r × x₃)
    
    Args:
        A_amplitudes: Matrice (3, 3) des amplitudes
        tau_roots: Vecteur (3,) des valeurs propres
        x3: Position dans l'épaisseur
    
    Returns:
        dU_dx3: Vecteur (3,) des dérivées premières
        d2U_dx3: Vecteur (3,) des dérivées secondes
    """
    dU_dx3 = np.zeros(3, dtype=complex)
    d2U_dx3 = np.zeros(3, dtype=complex)
    
    for alpha in range(3):
        for r in range(3):
            exp_term = np.exp(tau_roots[r] * x3)
            dU_dx3[alpha] += tau_roots[r] * A_amplitudes[alpha, r] * exp_term
            d2U_dx3[alpha] += (tau_roots[r]**2) * A_amplitudes[alpha, r] * exp_term
    
    return dU_dx3, d2U_dx3


# =============================================================================
# SECTION 5: Opérateurs L_jk et termes thermiques Q_α (Eq. 132-149)
# =============================================================================

def compute_L_operators(tau: complex, delta1: float, delta2: float, 
                        props: Dict = MECHANICAL_PROPS) -> Dict[str, complex]:
    """
    Calcule les opérateurs L_jk de la matrice dynamique.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 5
    ================================================
    
    Termes diagonaux (couplage direct):
        L_11 = C₅₅·τ² - (C₁₁·δ₁² + C₆₆·δ₂²)
        L_22 = C₄₄·τ² - (C₂₂·δ₂² + C₆₆·δ₁²)
        L_33 = C₃₃·τ² - (C₅₅·δ₁² + C₄₄·δ₂²)
    
    Termes croisés (couplage dans le plan):
        L_12 = L_21 = -(C₁₂ + C₆₆)·δ₁·δ₂
    
    Termes hors-plan (faisant intervenir τ):
        L_13 = +(C₁₃ + C₅₅)·δ₁·τ
        L_23 = +(C₂₃ + C₄₄)·δ₂·τ
        L_31 = -(C₁₃ + C₅₅)·δ₁·τ  (antisymétrique!)
        L_32 = -(C₂₃ + C₄₄)·δ₂·τ  (antisymétrique!)
    
    Args:
        tau: Valeur propre τ_r
        delta1, delta2: Nombres d'onde δ₁, δ₂
        props: Propriétés mécaniques {C11, C12, ..., C66}
    
    Returns:
        Dict avec L_11, L_12, ..., L_33
    """
    # Extraction des constantes élastiques (notation Voigt)
    C11, C12, C13 = props['C11'], props['C12'], props['C13']
    C22, C23, C33 = props['C22'], props['C23'], props['C33']
    C44, C55, C66 = props['C44'], props['C55'], props['C66']
    
    tau2 = tau**2
    
    # Termes diagonaux
    L_11 = C55 * tau2 - (C11 * delta1**2 + C66 * delta2**2)
    L_22 = C44 * tau2 - (C22 * delta2**2 + C66 * delta1**2)
    L_33 = C33 * tau2 - (C55 * delta1**2 + C44 * delta2**2)
    
    # Termes croisés dans le plan (symétriques)
    L_12 = -(C12 + C66) * delta1 * delta2
    L_21 = L_12  # Symétrie
    
    # Termes de couplage hors-plan (antisymétriques!)
    L_13 = +(C13 + C55) * delta1 * tau
    L_23 = +(C23 + C44) * delta2 * tau
    L_31 = -(C13 + C55) * delta1 * tau  # Signe négatif!
    L_32 = -(C23 + C44) * delta2 * tau  # Signe négatif!
    
    return {
        'L_11': L_11, 'L_12': L_12, 'L_13': L_13,
        'L_21': L_21, 'L_22': L_22, 'L_23': L_23,
        'L_31': L_31, 'L_32': L_32, 'L_33': L_33
    }


def compute_Q_thermal_vector(delta1: float, delta2: float, 
                              T: complex, dT_dx3: complex,
                              alpha_coeffs: Dict, 
                              props: Dict = MECHANICAL_PROPS) -> np.ndarray:
    """
    Calcule le vecteur de sollicitation thermique Q = [Q₁, Q₂, Q₃].
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 7 (Eq. 266-269)
    ===============================================================
    
    Q₁(x₃ = h̄) = (C₁₁·α₁₁ + C₁₂·α₂₂)·δ₁·T(x₃ = h̄)
    Q₂(x₃ = h̄) = (C₂₂·α₂₂ + C₁₂·α₁₁)·δ₂·T(x₃ = h̄)
    Q₃(x₃ = h̄) = (C₁₃·α₁₁ + C₂₃·α₂₂ + C₃₃·α₃₃)·dT/dx₃|_{x₃ = h̄}
    
    Args:
        delta1, delta2: Nombres d'onde
        T: Température à la position h̄ de la couche
        dT_dx3: Gradient de température à la position h̄
        alpha_coeffs: Coefficients de dilatation {alpha_1, alpha_2, alpha_3}
        props: Propriétés mécaniques
    
    Returns:
        Q: Vecteur (3,) des termes thermiques [Q₁, Q₂, Q₃]
    """
    # Extraction des alphas
    if isinstance(alpha_coeffs, dict):
        a1 = alpha_coeffs.get('alpha_1', alpha_coeffs.get('alpha', 10e-6))
        a2 = alpha_coeffs.get('alpha_2', a1)
        a3 = alpha_coeffs.get('alpha_3', a1)
    else:
        a1 = a2 = a3 = alpha_coeffs
    
    # Extraction des rigidités
    C11, C12, C13 = props['C11'], props['C12'], props['C13']
    C22, C23, C33 = props['C22'], props['C23'], props['C33']
    
    # Calcul des termes Q selon la formulation PDF
    Q1 = (C11 * a1 + C12 * a2) * delta1 * T
    Q2 = (C22 * a2 + C12 * a1) * delta2 * T
    Q3 = (C13 * a1 + C23 * a2 + C33 * a3) * dT_dx3
    
    return np.array([Q1, Q2, Q3], dtype=complex)


# =============================================================================
# SECTION 6: Matrice Γ(τ) et système homogène (Eq. 176-208)
# =============================================================================

def get_Gamma_matrix(tau: complex, delta1: float, delta2: float,
                     props: Dict = MECHANICAL_PROPS) -> np.ndarray:
    """
    Construit la matrice dynamique Γ(τ) 3×3.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 6 (Eq. 199-208)
    ===============================================================
    
    Le système homogène s'écrit: Γ(τ) · A = 0
    
    Les valeurs propres τ_r sont telles que det(Γ(τ_r)) = 0.
    
    Composantes de Γ:
        Γ₁₁ = C₅₅·τ² - (C₁₁·δ₁² + C₆₆·δ₂²)
        Γ₂₂ = C₄₄·τ² - (C₂₂·δ₂² + C₆₆·δ₁²)
        Γ₃₃ = C₃₃·τ² - (C₅₅·δ₁² + C₄₄·δ₂²)
        Γ₁₂ = Γ₂₁ = -(C₁₂ + C₆₆)·δ₁·δ₂
        Γ₁₃ = +(C₁₃ + C₅₅)·δ₁·τ
        Γ₂₃ = +(C₂₃ + C₄₄)·δ₂·τ
        Γ₃₁ = -(C₁₃ + C₅₅)·δ₁·τ  ← ANTISYMÉTRIQUE
        Γ₃₂ = -(C₂₃ + C₄₄)·δ₂·τ  ← ANTISYMÉTRIQUE
    
    Args:
        tau: Valeur propre τ
        delta1, delta2: Nombres d'onde δ₁, δ₂
        props: Propriétés mécaniques
    
    Returns:
        Gamma: Matrice 3×3 complexe
    """
    L = compute_L_operators(tau, delta1, delta2, props)
    
    Gamma = np.array([
        [L['L_11'], L['L_12'], L['L_13']],
        [L['L_21'], L['L_22'], L['L_23']],
        [L['L_31'], L['L_32'], L['L_33']]
    ], dtype=complex)
    
    return Gamma


def solve_characteristic_polynomial(delta1: float, delta2: float,
                                     props: Dict = MECHANICAL_PROPS) -> Dict:
    """
    Résout l'équation caractéristique det(Γ(τ)) = 0.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 2
    ================================================
    
    Le déterminant de Γ(τ) est un polynôme d'ordre 6 en τ:
        P(τ) = c₆τ⁶ + c₄τ⁴ + c₂τ² + c₀ = 0
    
    En posant X = τ², on obtient un polynôme cubique:
        P(X) = c₆X³ + c₄X² + c₂X + c₀ = 0
    
    Les 6 racines τ viennent par paires ±√X.
    On sélectionne les 3 racines avec Re(τ) < 0 (condition de radiation).
    
    Returns:
        dict avec:
            - 'tau_all': Les 6 racines τ
            - 'tau_selected': Les 3 racines sélectionnées (Re < 0)
            - 'X_roots': Les 3 racines en X = τ²
    """
    C11, C12, C13 = props['C11'], props['C12'], props['C13']
    C22, C23, C33 = props['C22'], props['C23'], props['C33']
    C44, C55, C66 = props['C44'], props['C55'], props['C66']
    
    # Coefficient dominant c₆ = C₅₅ × C₄₄ × C₃₃
    c6 = C55 * C44 * C33
    
    # Évaluation numérique des coefficients par interpolation
    def get_det_at_X(X_val):
        tau_val = np.sqrt(complex(X_val))
        Gamma = get_Gamma_matrix(tau_val, delta1, delta2, props)
        return np.linalg.det(Gamma)
    
    # Évaluations pour interpolation
    P_0 = get_det_at_X(0)
    P_1 = get_det_at_X(1)
    P_2 = get_det_at_X(2)
    
    c0 = P_0
    b1 = P_1 - c6 - c0
    b2 = P_2 - 8*c6 - c0
    c4 = (b2 - 2*b1) / 2
    c2 = b1 - c4
    
    # Résolution du polynôme cubique
    coeffs_norm = [1, c4/c6, c2/c6, c0/c6]
    X_roots = np.roots(coeffs_norm)
    
    # Calcul des 6 racines τ = ±√X
    tau_all = []
    for X in X_roots:
        root_plus = np.sqrt(X)
        root_minus = -root_plus
        tau_all.extend([root_plus, root_minus])
    
    tau_all = np.array(tau_all)
    
    # Sélection des 3 racines avec Re(τ) < 0 (condition de radiation)
    tau_selected = np.array([t for t in tau_all if t.real < 0])
    
    # Si pas assez de racines négatives, prendre par partie imaginaire
    if len(tau_selected) < 3:
        tau_sorted = sorted(tau_all, key=lambda x: (x.real, -abs(x.imag)))
        tau_selected = np.array(tau_sorted[:3])
    
    return {
        'tau_all': tau_all,
        'tau_selected': tau_selected[:3],
        'X_roots': X_roots
    }


# =============================================================================
# SECTION 7: Assemblage 9×9 avec chargement thermique (Eq. 210-261)
# =============================================================================

def assemble_K_dyn_9x9(tau_roots: np.ndarray, delta1: float, delta2: float,
                        props: Dict = MECHANICAL_PROPS) -> np.ndarray:
    """
    Assemble la matrice dynamique bloc-diagonale 9×9.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 7 (Eq. 220-235)
    ===============================================================
    
    La matrice K_dyn est bloc-diagonale:
    
        K_dyn = | Γ(τ₁)    0       0    |
                |   0    Γ(τ₂)    0    |
                |   0      0    Γ(τ₃)  |
    
    Chaque bloc Γ(τ_r) est une matrice 3×3.
    
    Args:
        tau_roots: Vecteur (3,) des valeurs propres [τ₁, τ₂, τ₃]
        delta1, delta2: Nombres d'onde
        props: Propriétés mécaniques
    
    Returns:
        K_dyn: Matrice 9×9 complexe bloc-diagonale
    """
    K_dyn = np.zeros((9, 9), dtype=complex)
    
    for r in range(3):
        # Bloc diagonal r (indices 3r:3r+3, 3r:3r+3)
        Gamma_r = get_Gamma_matrix(tau_roots[r], delta1, delta2, props)
        start = 3 * r
        end = 3 * (r + 1)
        K_dyn[start:end, start:end] = Gamma_r
    
    return K_dyn


def assemble_F_thermal_9x1(Q_vector: np.ndarray) -> np.ndarray:
    """
    Assemble le vecteur de sollicitation thermique 9×1.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 7 (Eq. 247-259)
    ===============================================================
    
    Le vecteur F_th est constitué de 3 copies identiques de Q:
    
        F_th = | Q₁ |
               | Q₂ |
               | Q₃ |
               | Q₁ |
               | Q₂ |
               | Q₃ |
               | Q₁ |
               | Q₂ |
               | Q₃ |
    
    Note: Les termes sont identiques pour chaque bloc si le chargement
    thermique est évalué à la même position h̄ pour tous les modes.
    
    Args:
        Q_vector: Vecteur (3,) = [Q₁, Q₂, Q₃]
    
    Returns:
        F_th: Vecteur 9×1
    """
    F_th = np.zeros(9, dtype=complex)
    
    for r in range(3):
        start = 3 * r
        F_th[start:start+3] = Q_vector
    
    return F_th


def solve_amplitude_system(K_dyn: np.ndarray, F_th: np.ndarray) -> np.ndarray:
    """
    Résout le système K_dyn · A = F_th pour les 9 amplitudes.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 7
    ================================================
    
    Le système est:
        K_dyn · A = F_th
    
    où A = [A₁¹, A₂¹, A₃¹, A₁², A₂², A₃², A₁³, A₂³, A₃³]ᵀ
    
    Args:
        K_dyn: Matrice 9×9
        F_th: Vecteur 9×1
    
    Returns:
        A: Vecteur (9,) des amplitudes
    """
    try:
        A = np.linalg.solve(K_dyn, F_th)
    except np.linalg.LinAlgError:
        # Système singulier: utiliser pseudo-inverse
        A = np.linalg.lstsq(K_dyn, F_th, rcond=None)[0]
    
    return A


def reshape_amplitudes_to_matrix(A_vector: np.ndarray) -> np.ndarray:
    """
    Convertit le vecteur d'amplitudes 9×1 en matrice 3×3.
    
    Args:
        A_vector: [A₁¹, A₂¹, A₃¹, A₁², A₂², A₃², A₁³, A₂³, A₃³]
    
    Returns:
        A_matrix: Matrice (3, 3) où A[α, r] = A_α^{r+1}
    """
    # Le vecteur est organisé par blocs de modes r
    # A = [bloc_mode_1, bloc_mode_2, bloc_mode_3]
    # bloc_mode_r = [A_1^r, A_2^r, A_3^r]
    
    A_matrix = np.zeros((3, 3), dtype=complex)
    
    for r in range(3):
        for alpha in range(3):
            A_matrix[alpha, r] = A_vector[3*r + alpha]
    
    return A_matrix


# =============================================================================
# API HAUT NIVEAU: Solveur complet suivant la méthodologie PDF
# =============================================================================

def solve_layer_pdf_method(h_layer: float, lw: float, 
                            T_profile: callable,
                            alpha_coeffs: Dict,
                            props: Dict = MECHANICAL_PROPS,
                            h_bar: Optional[float] = None) -> Dict:
    """
    Résout le problème thermoélastique pour une couche unique
    en suivant la méthodologie complète du PDF.
    
    MÉTHODOLOGIE:
    1. Calcul des nombres d'onde δ₁, δ₂
    2. Résolution polynôme caractéristique → τ₁, τ₂, τ₃
    3. Assemblage K_dyn (9×9 bloc-diagonal)
    4. Calcul termes thermiques Q
    5. Résolution système K_dyn · A = F_th
    6. Reconstruction champ de déplacement U(x₃)
    
    Args:
        h_layer: Épaisseur de la couche (m)
        lw: Longueur d'onde caractéristique (m)
        T_profile: Fonction T(x3) retournant (T, dT/dx3) à la position x3
        alpha_coeffs: Coefficients de dilatation thermique
        props: Propriétés mécaniques
        h_bar: Position d'évaluation (défaut: milieu de la couche)
    
    Returns:
        dict avec:
            - 'tau_roots': Les 3 valeurs propres sélectionnées
            - 'A_amplitudes': Matrice 3×3 des amplitudes
            - 'K_dyn': Matrice 9×9
            - 'F_th': Vecteur thermique
            - 'displacement_func': Fonction U(x3)
    """
    # Étape 1: Nombres d'onde
    delta1 = np.pi / lw
    delta2 = np.pi / lw
    
    # Étape 2: Résolution polynôme caractéristique
    char_result = solve_characteristic_polynomial(delta1, delta2, props)
    tau_roots = char_result['tau_selected']
    
    # Étape 3: Assemblage K_dyn 9×9
    K_dyn = assemble_K_dyn_9x9(tau_roots, delta1, delta2, props)
    
    # Étape 4: Calcul termes thermiques à h̄
    if h_bar is None:
        h_bar = h_layer / 2  # Milieu de la couche par défaut
    
    T_val, dT_dx3_val = T_profile(h_bar)
    Q_vector = compute_Q_thermal_vector(delta1, delta2, T_val, dT_dx3_val, 
                                         alpha_coeffs, props)
    
    # Étape 5: Assemblage F_th et résolution
    F_th = assemble_F_thermal_9x1(Q_vector)
    A_vector = solve_amplitude_system(K_dyn, F_th)
    A_amplitudes = reshape_amplitudes_to_matrix(A_vector)
    
    # Étape 6: Fonction de déplacement
    def displacement_func(x3):
        return build_displacement_field(A_amplitudes, tau_roots, x3)
    
    return {
        'tau_roots': tau_roots,
        'A_amplitudes': A_amplitudes,
        'A_vector': A_vector,
        'K_dyn': K_dyn,
        'F_th': F_th,
        'Q_vector': Q_vector,
        'delta1': delta1,
        'delta2': delta2,
        'displacement_func': displacement_func,
        'h_layer': h_layer,
        'props': props
    }


def compute_stress_from_displacement(result: Dict, x3: float) -> np.ndarray:
    """
    Calcule les contraintes à partir du champ de déplacement.
    
    RÉFÉRENCE: Loi de comportement σ = C : ε - β·T
    
    Pour les composantes de cisaillement et normale:
        σ₁₃ = C₅₅(∂U₁/∂x₃ + δ₁·U₃)
        σ₂₃ = C₄₄(∂U₂/∂x₃ + δ₂·U₃)  
        σ₃₃ = -C₁₃·δ₁·U₁ - C₂₃·δ₂·U₂ + C₃₃·∂U₃/∂x₃
    
    Args:
        result: Dict retourné par solve_layer_pdf_method
        x3: Position dans l'épaisseur
    
    Returns:
        sigma: Vecteur [σ₁₃, σ₂₃, σ₃₃]
    """
    A_amplitudes = result['A_amplitudes']
    tau_roots = result['tau_roots']
    delta1 = result['delta1']
    delta2 = result['delta2']
    props = result['props']
    
    # Déplacements et dérivées
    U = build_displacement_field(A_amplitudes, tau_roots, x3)
    dU_dx3, _ = build_displacement_derivatives(A_amplitudes, tau_roots, x3)
    
    # Extraction des rigidités
    C13, C23, C33 = props['C13'], props['C23'], props['C33']
    C44, C55 = props['C44'], props['C55']
    
    # Calcul des contraintes
    sigma_13 = C55 * (dU_dx3[0] + delta1 * U[2])
    sigma_23 = C44 * (dU_dx3[1] + delta2 * U[2])
    sigma_33 = -C13 * delta1 * U[0] - C23 * delta2 * U[1] + C33 * dU_dx3[2]
    
    return np.array([sigma_13, sigma_23, sigma_33], dtype=complex)
