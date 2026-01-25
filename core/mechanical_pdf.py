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

================================================================================
RÉDUCTION DU NOMBRE D'ÉQUATIONS: 27 → 18 (pour 3 couches)
================================================================================

FORMULATION THÉORIQUE COMPLÈTE (27 équations pour N=3 couches):
---------------------------------------------------------------
┌─────────────────────────────────────────────────────────────────────────────┐
│  TYPE D'ÉQUATION                        │  NOMBRE  │  FORMULE              │
├─────────────────────────────────────────┼──────────┼───────────────────────┤
│  Équilibre volumique (div σ = 0)       │   9      │  3 directions × 3 couches │
│  Continuité aux interfaces             │  12      │  2 interfaces × 6 conditions │
│  Conditions aux limites (σ=0)          │   6      │  2 surfaces × 3 composantes │
├─────────────────────────────────────────┼──────────┼───────────────────────┤
│  TOTAL THÉORIQUE                        │  27      │  9 + 12 + 6           │
└─────────────────────────────────────────┴──────────┴───────────────────────┘

RÉDUCTION PAR L'APPROCHE MODALE (méthode spectrale):
----------------------------------------------------
Les 9 ÉQUATIONS D'ÉQUILIBRE VOLUMIQUE sont IMPLICITEMENT SATISFAITES !

Pourquoi ? Car on écrit la solution sous la forme:
    U_α(x₃) = Σ_{r} A_α^r × exp(τ_r × x₃)

où les τ_r sont les VALEURS PROPRES de la matrice Γ(τ), définies par:
    det(Γ(τ_r)) = 0

Cette condition garantit que pour chaque mode r:
    Γ(τ_r) · [A_1^r, A_2^r, A_3^r]ᵀ = 0

Ce qui est EXACTEMENT l'équation d'équilibre volumique div(σ) = 0 !

SYSTÈME FINAL À RÉSOUDRE (18 équations pour N=3 couches):
---------------------------------------------------------
┌─────────────────────────────────────────────────────────────────────────────┐
│  TYPE D'ÉQUATION                        │  NOMBRE  │  IMPLÉMENTATION        │
├─────────────────────────────────────────┼──────────┼────────────────────────┤
│  Équilibre volumique                    │   0      │  ⚡ IMPLICITE (τ_r)    │
│  Continuité aux interfaces             │  12      │  Φ(h_k) = Φ(0)         │
│  Conditions aux limites                │   6      │  σ(0)=0, σ(H)=0        │
├─────────────────────────────────────────┼──────────┼────────────────────────┤
│  TOTAL RÉSOLU                           │  18      │  = 6N pour N couches   │
│  INCONNUES                              │  18      │  = 6N coefficients C_r │
└─────────────────────────────────────────┴──────────┴────────────────────────┘

RÉSUMÉ:
-------
• Théorie: 27 équations = 9 (équilibre) + 12 (interfaces) + 6 (limites)
• Spectral: 18 équations = 0 (équilibre implicite) + 12 (interfaces) + 6 (limites)
• Ce module (1 couche): 9 équations = 3 modes × 3 directions

La PUISSANCE de la méthode spectrale est de réduire le système de 9 équations
en utilisant la structure modale de la solution.
================================================================================
"""

import warnings                                                 # Pour gérer les warnings numériques
import numpy as np                                              # Bibliothèque NumPy pour calcul matriciel
from typing import Dict, List, Tuple, Optional                  # Types pour annotations (documentation)
from .constants import MECHANICAL_PROPS, GPa_TO_PA              # Import des propriétés mécaniques par défaut

# =============================================================================
# SECTION 2: Forme de la solution générale (Eq. 44-48 du PDF)
# =============================================================================
#
# La solution générale pour les déplacements dans une couche (i) est:
#   U_α(x₃) = Σ_{r=1}^{3} A_α^r × exp(τ_r × x₃)
#
# où:
#   - α ∈ {1, 2, 3} est la direction du déplacement
#   - r ∈ {1, 2, 3} est l'indice du mode propre
#   - A_α^r est l'amplitude du mode r dans la direction α
#   - τ_r est la r-ième valeur propre caractéristique
#   - x₃ est la coordonnée verticale (épaisseur)
# =============================================================================

def build_displacement_field(A_amplitudes: np.ndarray, tau_roots: np.ndarray, x3: float) -> np.ndarray:
    """
    Construit le champ de déplacement U(x₃) pour une position donnée.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 2 (Eq. 44-48)
    ============================================================
    
    Formule mathématique (pour chaque direction α):
        U_α(x₃) = Σ_{r=1}^{3} A_α^r × exp(τ_r × x₃)
    
    Cette formule développe la somme sur les 3 modes propres sélectionnés.
    
    Args:
        A_amplitudes: Matrice (3, 3) des amplitudes
                      - Lignes: directions α ∈ {0, 1, 2} → correspond à {x₁, x₂, x₃}
                      - Colonnes: modes r ∈ {0, 1, 2} → correspond aux modes {1, 2, 3}
                      - A_amplitudes[α, r] = A_α^{r+1} (indice Python commence à 0)
        tau_roots: Vecteur (3,) des valeurs propres [τ₁, τ₂, τ₃]
                   Sélectionnées avec Re(τ) < 0 (condition de radiation)
        x3: Position dans l'épaisseur de la couche (en mètres)
    
    Returns:
        U: Vecteur (3,) des déplacements [U₁, U₂, U₃] (valeurs complexes)
    """
    # Initialisation du vecteur déplacement à zéro
    # dtype=complex car τ peut être complexe → exp(τ*x₃) sera complexe
    U = np.zeros(3, dtype=complex)
    
    # Double boucle : pour chaque direction α et chaque mode r
    for alpha in range(3):          # α = 0, 1, 2 (directions x₁, x₂, x₃)
        for r in range(3):          # r = 0, 1, 2 (modes 1, 2, 3)
            # Terme de la somme: A_α^r × exp(τ_r × x₃)
            # np.exp calcule l'exponentielle complexe si τ est complexe
            U[alpha] += A_amplitudes[alpha, r] * np.exp(tau_roots[r] * x3)
    
    # Retourne le vecteur des 3 composantes de déplacement
    return U


def build_displacement_derivatives(A_amplitudes: np.ndarray, tau_roots: np.ndarray, x3: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les dérivées premières et secondes du champ de déplacement.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 3 (Eq. 61-72)
    =============================================================
    
    Formules mathématiques:
    
    Dérivée première (nécessaire pour les déformations ε et contraintes σ):
        dU_α/dx₃ = Σ_{r=1}^{3} τ_r × A_α^r × exp(τ_r × x₃)
        
    Dérivée seconde (nécessaire pour l'équation d'équilibre):
        d²U_α/dx₃² = Σ_{r=1}^{3} τ_r² × A_α^r × exp(τ_r × x₃)
    
    Note: La dérivation de exp(τ×x₃) donne τ×exp(τ×x₃)
          La dérivée seconde donne τ²×exp(τ×x₃)
    
    Args:
        A_amplitudes: Matrice (3, 3) des amplitudes [voir build_displacement_field]
        tau_roots: Vecteur (3,) des valeurs propres [τ₁, τ₂, τ₃]
        x3: Position dans l'épaisseur de la couche (en mètres)
    
    Returns:
        dU_dx3: Vecteur (3,) des dérivées premières [dU₁/dx₃, dU₂/dx₃, dU₃/dx₃]
        d2U_dx3: Vecteur (3,) des dérivées secondes [d²U₁/dx₃², d²U₂/dx₃², d²U₃/dx₃²]
    """
    # Initialisation des vecteurs de dérivées à zéro
    dU_dx3 = np.zeros(3, dtype=complex)    # Dérivées premières
    d2U_dx3 = np.zeros(3, dtype=complex)   # Dérivées secondes
    
    # Double boucle sur les directions et les modes
    for alpha in range(3):                  # Pour chaque direction α
        for r in range(3):                  # Pour chaque mode r
            # Calcul du terme exponentiel (commun aux deux dérivées)
            exp_term = np.exp(tau_roots[r] * x3)
            
            # Dérivée première: facteur τ_r devant
            # d/dx₃[A×exp(τ×x₃)] = A×τ×exp(τ×x₃)
            dU_dx3[alpha] += tau_roots[r] * A_amplitudes[alpha, r] * exp_term
            
            # Dérivée seconde: facteur τ_r² devant
            # d²/dx₃²[A×exp(τ×x₃)] = A×τ²×exp(τ×x₃)
            d2U_dx3[alpha] += (tau_roots[r]**2) * A_amplitudes[alpha, r] * exp_term
    
    # Retourne les deux vecteurs de dérivées
    return dU_dx3, d2U_dx3


# =============================================================================
# SECTION 5: Opérateurs L_jk et termes thermiques Q_α (Eq. 132-149 du PDF)
# =============================================================================
#
# Les opérateurs L_jk définissent la matrice dynamique Γ(τ).
# Ils proviennent de l'injection de la solution U = A×exp(τ×x₃) dans
# l'équation d'équilibre div(σ) = 0.
#
# Notation Voigt (tenseur → matrice):
#   C₁₁₁₁ → C11,  C₁₁₂₂ → C12,  C₁₁₃₃ → C13
#   C₂₂₂₂ → C22,  C₂₂₃₃ → C23,  C₃₃₃₃ → C33
#   C₁₃₁₃ → C55,  C₂₃₂₃ → C44,  C₁₂₁₂ → C66
# =============================================================================

def compute_L_operators(tau: complex, delta1: float, delta2: float, 
                        props: Dict = MECHANICAL_PROPS) -> Dict[str, complex]:
    """
    Calcule les 9 opérateurs L_jk de la matrice dynamique Γ(τ).
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 5 (Eq. 132-149)
    ===============================================================
    
    TERMES DIAGONAUX (rigidité effective dans chaque direction):
    -------------------------------------------------------------
    L_11 = C₅₅·τ² - (C₁₁·δ₁² + C₆₆·δ₂²)
           │         │
           │         └── Rigidité membranaire (dans le plan x₁-x₂)
           └── Rigidité hors-plan (cisaillement x₁-x₃)
           
    L_22 = C₄₄·τ² - (C₂₂·δ₂² + C₆₆·δ₁²)
           │         │
           │         └── Rigidité membranaire (échangé x₁ ↔ x₂)
           └── Rigidité hors-plan (cisaillement x₂-x₃)
           
    L_33 = C₃₃·τ² - (C₅₅·δ₁² + C₄₄·δ₂²)
           │         │
           │         └── Rigidité de cisaillement combinée
           └── Module normal (compression x₃)
    
    TERMES CROISÉS DANS LE PLAN (couplage x₁-x₂, SYMÉTRIQUES):
    ----------------------------------------------------------
    L_12 = L_21 = -(C₁₂ + C₆₆)·δ₁·δ₂
           │
           └── Couplage Poisson + cisaillement dans le plan
               Le signe négatif vient de l'équation d'équilibre
    
    TERMES DE COUPLAGE HORS-PLAN (ANTISYMÉTRIQUES !):
    -------------------------------------------------
    L_13 = +(C₁₃ + C₅₅)·δ₁·τ   ← Signe POSITIF
    L_31 = -(C₁₃ + C₅₅)·δ₁·τ   ← Signe NÉGATIF (antisymétrique)
    
    L_23 = +(C₂₃ + C₄₄)·δ₂·τ   ← Signe POSITIF
    L_32 = -(C₂₃ + C₄₄)·δ₂·τ   ← Signe NÉGATIF (antisymétrique)
    
    ⚠️  L'ANTISYMÉTRIE est ESSENTIELLE pour la physique correcte !
        Elle provient de l'équation d'équilibre en direction x₃.
    
    Args:
        tau: Valeur propre τ (généralement complexe)
        delta1: Nombre d'onde δ₁ = π/Lw dans la direction x₁
        delta2: Nombre d'onde δ₂ = π/Lw dans la direction x₂
        props: Dictionnaire des propriétés mécaniques {C11, C12, ..., C66} en GPa
    
    Returns:
        Dict avec les 9 opérateurs {'L_11': ..., 'L_12': ..., ..., 'L_33': ...}
    """
    # =========================================================================
    # ÉTAPE 1: Extraction des constantes élastiques (notation Voigt)
    # =========================================================================
    C11, C12, C13 = props['C11'], props['C12'], props['C13']  # Modules axiaux et couplage Poisson
    C22, C23, C33 = props['C22'], props['C23'], props['C33']  # Idem pour directions 2 et 3
    C44, C55, C66 = props['C44'], props['C55'], props['C66']  # Modules de cisaillement
    
    # Précalcul de τ² (utilisé dans tous les termes diagonaux)
    tau2 = tau**2   # τ² = τ × τ (peut être complexe si τ est complexe)
    
    # =========================================================================
    # ÉTAPE 2: Calcul des termes DIAGONAUX
    # =========================================================================
    # Ces termes représentent la rigidité effective dans chaque direction
    
    # L_11: Équilibre en direction x₁
    # C55·τ² : contribution du cisaillement x₁-x₃ (hors-plan)
    # C11·δ₁² : compression/extension dans x₁ (membranaire)
    # C66·δ₂² : cisaillement dans le plan x₁-x₂
    L_11 = C55 * tau2 - (C11 * delta1**2 + C66 * delta2**2)
    
    # L_22: Équilibre en direction x₂ (symétrique à L_11 avec échange 1↔2)
    L_22 = C44 * tau2 - (C22 * delta2**2 + C66 * delta1**2)
    
    # L_33: Équilibre en direction x₃ (normale)
    L_33 = C33 * tau2 - (C55 * delta1**2 + C44 * delta2**2)
    
    # =========================================================================
    # ÉTAPE 3: Calcul des termes CROISÉS DANS LE PLAN (symétriques)
    # =========================================================================
    # Couplage entre les directions x₁ et x₂
    # Le signe négatif vient de la formulation de l'équation d'équilibre
    L_12 = -(C12 + C66) * delta1 * delta2   # Couplage Poisson + cisaillement
    L_21 = L_12                              # Symétrie: L_12 = L_21
    
    # =========================================================================
    # ÉTAPE 4: Calcul des termes HORS-PLAN (antisymétriques !)
    # =========================================================================
    # Ces termes couplent les déplacements horizontaux (U₁, U₂) avec U₃
    # ⚠️ ATTENTION: L_13 ≠ L_31 et L_23 ≠ L_32 (antisymétrie)
    
    # Couplage x₁ ↔ x₃
    L_13 = +(C13 + C55) * delta1 * tau   # Signe POSITIF (ligne 1, colonne 3)
    L_31 = -(C13 + C55) * delta1 * tau   # Signe NÉGATIF (ligne 3, colonne 1)
    #        │
    #        └── L_31 = -L_13 (antisymétrique car τ apparaît une fois)
    
    # Couplage x₂ ↔ x₃
    L_23 = +(C23 + C44) * delta2 * tau   # Signe POSITIF
    L_32 = -(C23 + C44) * delta2 * tau   # Signe NÉGATIF
    
    # =========================================================================
    # ÉTAPE 5: Retour des 9 opérateurs dans un dictionnaire
    # =========================================================================
    return {
        'L_11': L_11, 'L_12': L_12, 'L_13': L_13,   # Ligne 1
        'L_21': L_21, 'L_22': L_22, 'L_23': L_23,   # Ligne 2
        'L_31': L_31, 'L_32': L_32, 'L_33': L_33    # Ligne 3
    }


def compute_Q_thermal_vector(delta1: float, delta2: float, 
                              T: complex, dT_dx3: complex,
                              alpha_coeffs: Dict, 
                              props: Dict = MECHANICAL_PROPS) -> np.ndarray:
    """
    Calcule le vecteur de sollicitation thermique Q = [Q₁, Q₂, Q₃].
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 7 (Eq. 266-269)
    ===============================================================
    
    Les termes Q_α représentent le FORÇAGE THERMIQUE dans l'équation
    d'équilibre mécanique. Ils proviennent de la dilatation thermique.
    
    FORMULES (évaluées à la position x₃ = h̄):
    -----------------------------------------
    Q₁ = (C₁₁·α₁₁ + C₁₂·α₂₂)·δ₁·T
         │         │         │   │
         │         │         │   └── Température à la position h̄
         │         │         └── Nombre d'onde (harmonique spatiale)
         │         └── Couplage dilatation x₂ via Poisson
         └── Dilatation directe x₁
    
    Q₂ = (C₂₂·α₂₂ + C₁₂·α₁₁)·δ₂·T
         Symétrique à Q₁ avec échange des indices 1 ↔ 2
    
    Q₃ = (C₁₃·α₁₁ + C₂₃·α₂₂ + C₃₃·α₃₃)·dT/dx₃
         │                           │
         │                           └── Gradient de température (non T!)
         └── Contributions des 3 dilatations via couplage Poisson
    
    Note: Q₃ dépend du GRADIENT dT/dx₃ car la direction 3 est normale.
    
    Args:
        delta1, delta2: Nombres d'onde δ₁, δ₂ = π/Lw (rad/m)
        T: Température à la position h̄ (°C ou K)
        dT_dx3: Gradient de température dT/dx₃ à la position h̄ (K/m)
        alpha_coeffs: Coefficients de dilatation thermique
                      Dict {'alpha_1': ..., 'alpha_2': ..., 'alpha_3': ...} en 1/K
                      Ou valeur float si isotrope thermiquement
        props: Propriétés mécaniques {C11, C12, ..., C33} en GPa
    
    Returns:
        Q: Vecteur (3,) des termes thermiques [Q₁, Q₂, Q₃]
    """
    # =========================================================================
    # ÉTAPE 1: Extraction des coefficients de dilatation thermique
    # =========================================================================
    # Gestion des deux formats possibles: dict ou float
    if isinstance(alpha_coeffs, dict):
        # Format dictionnaire: {'alpha_1': ..., 'alpha_2': ..., 'alpha_3': ...}
        a1 = alpha_coeffs.get('alpha_1', alpha_coeffs.get('alpha', 10e-6))
        a2 = alpha_coeffs.get('alpha_2', a1)   # Défaut: même que a1 si absent
        a3 = alpha_coeffs.get('alpha_3', a1)   # Défaut: même que a1 si absent
    else:
        # Format scalaire: matériau isotrope thermiquement
        a1 = a2 = a3 = alpha_coeffs            # α₁ = α₂ = α₃ = α
    
    # =========================================================================
    # ÉTAPE 2: Extraction des rigidités nécessaires
    # =========================================================================
    C11, C12, C13 = props['C11'], props['C12'], props['C13']
    C22, C23, C33 = props['C22'], props['C23'], props['C33']
    
    # =========================================================================
    # ÉTAPE 3: Calcul des termes Q selon les formules du PDF
    # =========================================================================
    
    # Q₁: Terme thermique pour l'équilibre en direction x₁
    # (C11·α1 + C12·α2) = contrainte thermique de référence en x₁
    # δ₁ × T = amplitude de la perturbation harmonique
    Q1 = (C11 * a1 + C12 * a2) * delta1 * T
    
    # Q₂: Terme thermique pour l'équilibre en direction x₂
    # Symétrique à Q1 avec échange des indices
    Q2 = (C22 * a2 + C12 * a1) * delta2 * T
    
    # Q₃: Terme thermique pour l'équilibre en direction x₃ (normale)
    # ⚠️ Utilise dT/dx₃ (gradient) au lieu de T !
    # Car la direction x₃ est perpendiculaire aux couches
    Q3 = (C13 * a1 + C23 * a2 + C33 * a3) * dT_dx3
    
    # =========================================================================
    # ÉTAPE 4: Retour du vecteur Q
    # =========================================================================
    return np.array([Q1, Q2, Q3], dtype=complex)


# =============================================================================
# SECTION 6: Matrice Γ(τ) et système homogène (Eq. 176-208 du PDF)
# =============================================================================
#
# La matrice dynamique Γ(τ) assemble les opérateurs L_jk en une matrice 3×3.
# Le système homogène Γ(τ)·A = 0 définit les modes propres.
#
# Propriétés importantes:
#   - Γ₁₂ = Γ₂₁ (symétrie)
#   - Γ₁₃ = -Γ₃₁ et Γ₂₃ = -Γ₃₂ (ANTISYMÉTRIE)
#
# Les valeurs propres τ_r sont telles que det(Γ(τ_r)) = 0
# =============================================================================

def get_Gamma_matrix(tau: complex, delta1: float, delta2: float,
                     props: Dict = MECHANICAL_PROPS) -> np.ndarray:
    """
    Construit la matrice dynamique Γ(τ) de dimension 3×3.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 6 (Eq. 199-208)
    ===============================================================
    
    STRUCTURE DE LA MATRICE:
    ------------------------
        Γ(τ) = | L_11   L_12   L_13 |
               | L_21   L_22   L_23 |
               | L_31   L_32   L_33 |
    
    PROPRIÉTÉS DE SYMÉTRIE:
    -----------------------
        • Γ₁₂ = Γ₂₁     (symétrique)
        • Γ₁₃ = -Γ₃₁    (antisymétrique !)
        • Γ₂₃ = -Γ₃₂    (antisymétrique !)
    
    SYSTÈME HOMOGÈNE:
    -----------------
        Γ(τ) · A = 0
        
    Pour avoir une solution non triviale A ≠ 0, il faut:
        det(Γ(τ)) = 0
    
    Ceci définit les valeurs propres τ_r.
    
    Args:
        tau: Valeur propre τ (généralement complexe)
        delta1, delta2: Nombres d'onde δ₁, δ₂
        props: Propriétés mécaniques
    
    Returns:
        Gamma: Matrice 3×3 complexe
    """
    # =========================================================================
    # ÉTAPE 1: Calcul des opérateurs L_jk via la fonction dédiée
    # =========================================================================
    L = compute_L_operators(tau, delta1, delta2, props)
    
    # =========================================================================
    # ÉTAPE 2: Assemblage de la matrice Γ(τ) 3×3
    # =========================================================================
    # Organisation:
    #   Ligne 0 (équilibre x₁): [L_11, L_12, L_13]
    #   Ligne 1 (équilibre x₂): [L_21, L_22, L_23]
    #   Ligne 2 (équilibre x₃): [L_31, L_32, L_33]
    Gamma = np.array([
        [L['L_11'], L['L_12'], L['L_13']],   # Équation d'équilibre direction 1
        [L['L_21'], L['L_22'], L['L_23']],   # Équation d'équilibre direction 2
        [L['L_31'], L['L_32'], L['L_33']]    # Équation d'équilibre direction 3
    ], dtype=complex)
    
    return Gamma


def solve_characteristic_polynomial(delta1: float, delta2: float,
                                     props: Dict = MECHANICAL_PROPS) -> Dict:
    """
    Résout l'équation caractéristique det(Γ(τ)) = 0 pour trouver les τ_r.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 2
    =================================================
    
    THÉORIE:
    --------
    Le déterminant de Γ(τ) est un polynôme d'ordre 6 en τ:
        P(τ) = c₆τ⁶ + c₄τ⁴ + c₂τ² + c₀ = 0
    
    Note: Seules les puissances PAIRES de τ apparaissent
          (le polynôme est pair en τ).
    
    CHANGEMENT DE VARIABLE:
    -----------------------
    En posant X = τ², on obtient un polynôme CUBIQUE en X:
        P(X) = c₆X³ + c₄X² + c₂X + c₀ = 0
    
    Ce polynôme a 3 racines X₁, X₂, X₃ (potentiellement complexes).
    
    RACINES EN τ:
    -------------
    Pour chaque racine X_i, on calcule τ = ±√X_i.
    Cela donne 6 racines τ au total (±τ₁, ±τ₂, ±τ₃).
    
    CONDITION DE RADIATION:
    -----------------------
    On sélectionne les 3 racines avec Re(τ) < 0.
    Cela assure la décroissance des déplacements vers l'infini.
    
    MÉTHODE NUMÉRIQUE:
    ------------------
    Au lieu d'utiliser les formules analytiques complexes pour c₂, c₄,
    on utilise une interpolation numérique en 3 points (X = 0, 1, 2).
    
    Args:
        delta1, delta2: Nombres d'onde δ₁, δ₂
        props: Propriétés mécaniques
    
    Returns:
        dict avec:
            - 'tau_all': Les 6 racines τ
            - 'tau_selected': Les 3 racines avec Re(τ) < 0
            - 'X_roots': Les 3 racines en X
    """
    # =========================================================================
    # ÉTAPE 1: Coefficient dominant c₆ (analytique exact)
    # =========================================================================
    # c₆ = C₅₅ × C₄₄ × C₃₃ (produit des modules de cisaillement et normal)
    C44, C55, C33 = props['C44'], props['C55'], props['C33']
    c6 = C55 * C44 * C33
    
    # =========================================================================
    # ÉTAPE 2: Coefficients c₀, c₂, c₄ par interpolation numérique
    # =========================================================================
    # On évalue det(Γ(√X)) en X = 0, 1, 2 puis on résout pour les coefficients
    
    def get_det_at_X(X_val):
        """Évalue det(Γ(τ)) pour τ = √X.
        
        Note: Pour X=0, τ=0 et déterminer peut générer des warnings
        numériques (divisions par zéro) qui sont bénins car c₀ = det(Γ(0))
        est bien défini analytiquement.
        """
        tau_val = np.sqrt(complex(X_val))          # τ = √X (complexe si X < 0)
        Gamma = get_Gamma_matrix(tau_val, delta1, delta2, props)
        return np.linalg.det(Gamma)                # Déterminant 3×3
    
    # Évaluations du polynôme P(X) = c₆X³ + c₄X² + c₂X + c₀
    # Suppression des warnings numériques bénins lors de l'évaluation à X=0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        P_0 = get_det_at_X(0)    # P(0) = c₀ (peut générer warning bénin)
        P_1 = get_det_at_X(1)    # P(1) = c₆ + c₄ + c₂ + c₀
        P_2 = get_det_at_X(2)    # P(2) = 8c₆ + 4c₄ + 2c₂ + c₀
    
    # Résolution du système pour c₀, c₂, c₄:
    c0 = P_0                         # Directement depuis P(0)
    b1 = P_1 - c6 - c0               # = c₄ + c₂
    b2 = P_2 - 8*c6 - c0             # = 4c₄ + 2c₂
    c4 = (b2 - 2*b1) / 2             # Élimination de c₂
    c2 = b1 - c4                     # Substitution
    
    # =========================================================================
    # ÉTAPE 3: Résolution du polynôme cubique en X
    # =========================================================================
    # Normalisation par c₆: X³ + (c₄/c₆)X² + (c₂/c₆)X + (c₀/c₆) = 0
    coeffs_norm = [1, c4/c6, c2/c6, c0/c6]   # Coefficients normalisés
    X_roots = np.roots(coeffs_norm)           # Racines du polynôme cubique
    
    # =========================================================================
    # ÉTAPE 4: Calcul des 6 racines τ = ±√X
    # =========================================================================
    tau_all = []
    for X in X_roots:
        root_plus = np.sqrt(X)       # +√X
        root_minus = -root_plus      # -√X
        tau_all.extend([root_plus, root_minus])
    
    tau_all = np.array(tau_all)
    
    # =========================================================================
    # ÉTAPE 5: Sélection des 3 racines avec Re(τ) < 0 (condition de radiation)
    # =========================================================================
    tau_selected = np.array([t for t in tau_all if t.real < 0])
    
    # Cas dégénéré: si moins de 3 racines à partie réelle négative
    if len(tau_selected) < 3:
        # Tri par partie réelle croissante, puis par |Im| décroissant
        tau_sorted = sorted(tau_all, key=lambda x: (x.real, -abs(x.imag)))
        tau_selected = np.array(tau_sorted[:3])
    
    # =========================================================================
    # ÉTAPE 6: Retour des résultats
    # =========================================================================
    return {
        'tau_all': tau_all,             # Les 6 racines
        'tau_selected': tau_selected[:3],  # Les 3 racines sélectionnées
        'X_roots': X_roots              # Les 3 racines en X = τ²
    }


# =============================================================================
# SECTION 7: Assemblage 9×9 avec chargement thermique (Eq. 210-261 du PDF)
# =============================================================================
#
# Le système complet pour une couche utilise une matrice bloc-diagonale 9×9.
#
# Structure:
#   K_Dyn = | Γ(τ₁)    0       0    |      Vecteur inconnues:    A = | A₁¹ |
#           |   0    Γ(τ₂)    0    |                                 | A₂¹ |
#           |   0      0    Γ(τ₃)  |                                 | A₃¹ |
#                                                                     | A₁² |
#   [9×9]   × [9×1] = [9×1]                                          | ... |
#                                                                     | A₃³ |
#
# Les blocs sont indépendants car les modes sont découplés.
# =============================================================================

def assemble_K_dyn_9x9(tau_roots: np.ndarray, delta1: float, delta2: float,
                        props: Dict = MECHANICAL_PROPS) -> np.ndarray:
    """
    Assemble la matrice dynamique bloc-diagonale K_Dyn de dimension 9×9.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 7 (Eq. 220-235)
    ===============================================================
    
    STRUCTURE BLOC-DIAGONALE:
    -------------------------
    La matrice K_Dyn est composée de 3 blocs Γ(τ_r) sur la diagonale:
    
        K_Dyn = | Γ(τ₁)     0        0    |  (lignes 0-2)
                |   0     Γ(τ₂)     0    |  (lignes 3-5)
                |   0       0     Γ(τ₃)  |  (lignes 6-8)
    
    Les blocs hors-diagonale sont ZÉRO car les modes propres sont
    indépendants les uns des autres.
    
    TAILLE:
    -------
    - 3 modes × 3 directions = 9 inconnues
    - Matrice 9×9
    
    Args:
        tau_roots: Vecteur (3,) des valeurs propres [τ₁, τ₂, τ₃]
        delta1, delta2: Nombres d'onde δ₁, δ₂
        props: Propriétés mécaniques
    
    Returns:
        K_dyn: Matrice 9×9 complexe bloc-diagonale
    """
    # =========================================================================
    # ÉTAPE 1: Initialisation de la matrice 9×9 à zéro
    # =========================================================================
    K_dyn = np.zeros((9, 9), dtype=complex)   # Matrice nulle 9×9
    
    # =========================================================================
    # ÉTAPE 2: Remplissage des blocs diagonaux
    # =========================================================================
    for r in range(3):                        # Pour chaque mode r = 0, 1, 2
        # Construction du bloc Γ(τ_r) 3×3
        Gamma_r = get_Gamma_matrix(tau_roots[r], delta1, delta2, props)
        
        # Indices du bloc r dans la matrice 9×9
        start = 3 * r       # Début: 0, 3, 6
        end = 3 * (r + 1)   # Fin: 3, 6, 9
        
        # Insertion du bloc sur la diagonale
        # K_dyn[start:end, start:end] = sous-matrice 3×3 aux lignes/colonnes start:end
        K_dyn[start:end, start:end] = Gamma_r
    
    # =========================================================================
    # ÉTAPE 3: Les blocs hors-diagonale restent à zéro (pas de couplage)
    # =========================================================================
    # Rien à faire car np.zeros initialise déjà à zéro
    
    return K_dyn


def assemble_F_thermal_9x1(Q_vector: np.ndarray) -> np.ndarray:
    """
    Assemble le vecteur de sollicitation thermique F_Th de dimension 9×1.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 7 (Eq. 247-259)
    ===============================================================
    
    STRUCTURE DU VECTEUR:
    ---------------------
    Le vecteur F_Th est constitué de 3 copies du vecteur Q = [Q₁, Q₂, Q₃]:
    
        F_Th = | Q₁ |  ─┐
               | Q₂ |   │ Bloc mode 1
               | Q₃ |  ─┘
               | Q₁ |  ─┐
               | Q₂ |   │ Bloc mode 2
               | Q₃ |  ─┘
               | Q₁ |  ─┐
               | Q₂ |   │ Bloc mode 3
               | Q₃ |  ─┘
    
    Note: Les termes Q sont identiques pour chaque mode car le chargement
    thermique est évalué à la même position h̄ pour tous les modes.
    
    EXPLICATION PHYSIQUE:
    ---------------------
    Chaque mode propre τ_r ressent le même chargement thermique.
    La différence vient de la matrice Γ(τ_r) qui varie avec τ_r.
    
    Args:
        Q_vector: Vecteur (3,) des termes thermiques [Q₁, Q₂, Q₃]
    
    Returns:
        F_th: Vecteur (9,) de sollicitation thermique
    """
    # =========================================================================
    # ÉTAPE 1: Initialisation du vecteur 9×1 à zéro
    # =========================================================================
    F_th = np.zeros(9, dtype=complex)
    
    # =========================================================================
    # ÉTAPE 2: Copie du vecteur Q dans chaque bloc
    # =========================================================================
    for r in range(3):                # Pour chaque mode r = 0, 1, 2
        start = 3 * r                 # Début du bloc: 0, 3, 6
        F_th[start:start+3] = Q_vector   # Copie de [Q₁, Q₂, Q₃]
    
    return F_th


def solve_amplitude_system(K_dyn: np.ndarray, F_th: np.ndarray) -> np.ndarray:
    """
    Résout le système linéaire K_Dyn · A = F_Th pour obtenir les 9 amplitudes.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf, Section 7
    =================================================
    
    SYSTÈME À RÉSOUDRE:
    -------------------
        K_Dyn · A = F_Th
    
    où:
        K_Dyn: Matrice 9×9 bloc-diagonale
        A: Vecteur 9×1 des amplitudes inconnues
        F_Th: Vecteur 9×1 du chargement thermique
    
    ORGANISATION DU VECTEUR A:
    --------------------------
        A = [A₁¹, A₂¹, A₃¹,   ← Mode 1 (directions 1, 2, 3)
             A₁², A₂², A₃²,   ← Mode 2
             A₁³, A₂³, A₃³]   ← Mode 3
    
    MÉTHODE DE RÉSOLUTION:
    ----------------------
    1. Si K_Dyn est inversible: résolution directe par np.linalg.solve
    2. Si K_Dyn est singulière: utilisation de la pseudo-inverse (lstsq)
    
    Args:
        K_dyn: Matrice 9×9 du système
        F_th: Vecteur 9×1 second membre
    
    Returns:
        A: Vecteur (9,) des 9 amplitudes
    """
    # =========================================================================
    # Résolution du système linéaire
    # =========================================================================
    try:
        # Méthode directe: K_Dyn⁻¹ × F_Th
        # Utilise la décomposition LU pour efficacité
        A = np.linalg.solve(K_dyn, F_th)
    except np.linalg.LinAlgError:
        # Système singulier ou mal conditionné
        # Utilise la pseudo-inverse (moindres carrés)
        A = np.linalg.lstsq(K_dyn, F_th, rcond=None)[0]
    
    return A


def reshape_amplitudes_to_matrix(A_vector: np.ndarray) -> np.ndarray:
    """
    Convertit le vecteur d'amplitudes 9×1 en matrice 3×3.
    
    Cette fonction réorganise les amplitudes pour faciliter
    le calcul du champ de déplacement.
    
    TRANSFORMATION:
    ---------------
    Vecteur (9,):         →        Matrice (3, 3):
    [A₁¹, A₂¹, A₃¹,                A[α,r] = A_α^{r+1}
     A₁², A₂², A₃²,       →        
     A₁³, A₂³, A₃³]                Lignes: directions α
                                   Colonnes: modes r
    
    UTILISATION:
    ------------
    U_α(x₃) = Σ_r A[α,r] × exp(τ_r × x₃)
    
    Args:
        A_vector: Vecteur (9,) [A₁¹, A₂¹, A₃¹, A₁², A₂², A₃², A₁³, A₂³, A₃³]
    
    Returns:
        A_matrix: Matrice (3, 3) où A_matrix[α, r] = amplitude direction α, mode r
    """
    # =========================================================================
    # Initialisation de la matrice 3×3
    # =========================================================================
    A_matrix = np.zeros((3, 3), dtype=complex)
    
    # =========================================================================
    # Réorganisation: vecteur bloc → matrice (direction, mode)
    # =========================================================================
    for r in range(3):              # Colonnes: modes r = 0, 1, 2
        for alpha in range(3):      # Lignes: directions α = 0, 1, 2
            # Position dans le vecteur: bloc_r × 3 + position_dans_bloc
            # A_vector = [bloc0, bloc1, bloc2]
            # bloc_r = [A_1^r, A_2^r, A_3^r]
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
    en suivant la méthodologie complète du PDF étape par étape.
    
    RÉFÉRENCE: equilibre_local_corrige.pdf (toutes sections)
    =========================================================
    
    WORKFLOW COMPLET:
    -----------------
    ┌─────────────────────────────────────────────────────────┐
    │  ÉTAPE 1: Calcul des nombres d'onde δ₁, δ₂             │
    │           δ = π / Lw                                    │
    └────────────────────────┬────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │  ÉTAPE 2: Résolution polynôme caractéristique           │
    │           det(Γ(τ)) = 0 → τ₁, τ₂, τ₃                   │
    └────────────────────────┬────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │  ÉTAPE 3: Assemblage matrice K_Dyn (9×9 bloc-diagonal) │
    │           K_Dyn = diag(Γ(τ₁), Γ(τ₂), Γ(τ₃))           │
    └────────────────────────┬────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │  ÉTAPE 4: Calcul termes thermiques Q à h̄              │
    │           Q = [Q₁, Q₂, Q₃]                             │
    └────────────────────────┬────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │  ÉTAPE 5: Résolution système K_Dyn · A = F_Th          │
    │           → 9 amplitudes A_α^r                          │
    └────────────────────────┬────────────────────────────────┘
                             ▼
    ┌─────────────────────────────────────────────────────────┐
    │  ÉTAPE 6: Fonction de déplacement U(x₃)                │
    │           U_α(x₃) = Σ A_α^r × exp(τ_r × x₃)            │
    └─────────────────────────────────────────────────────────┘
    
    Args:
        h_layer: Épaisseur de la couche (m)
        lw: Longueur d'onde caractéristique Lw (m)
        T_profile: Fonction T(x3) → (T, dT/dx3) 
                   Retourne température et gradient à la position x3
        alpha_coeffs: Coefficients de dilatation thermique
        props: Propriétés mécaniques {C11, ..., C66} en GPa
        h_bar: Position d'évaluation du chargement thermique
               Défaut: milieu de la couche (h_layer/2)
    
    Returns:
        dict avec:
            - 'tau_roots': Les 3 valeurs propres sélectionnées
            - 'A_amplitudes': Matrice 3×3 des amplitudes
            - 'K_dyn': Matrice 9×9
            - 'F_th': Vecteur thermique 9×1
            - 'Q_vector': Vecteur [Q₁, Q₂, Q₃]
            - 'displacement_func': Fonction U(x3) → [U₁, U₂, U₃]
    """
    # =========================================================================
    # ÉTAPE 1: Calcul des nombres d'onde
    # =========================================================================
    # δ = π/Lw (correspond au mode fondamental de la décomposition de Fourier)
    delta1 = np.pi / lw     # Nombre d'onde dans la direction x₁
    delta2 = np.pi / lw     # Nombre d'onde dans la direction x₂ (même valeur)
    
    # =========================================================================
    # ÉTAPE 2: Résolution du polynôme caractéristique → τ₁, τ₂, τ₃
    # =========================================================================
    char_result = solve_characteristic_polynomial(delta1, delta2, props)
    tau_roots = char_result['tau_selected']   # 3 racines avec Re(τ) < 0
    
    # =========================================================================
    # ÉTAPE 3: Assemblage de la matrice K_Dyn (9×9 bloc-diagonal)
    # =========================================================================
    K_dyn = assemble_K_dyn_9x9(tau_roots, delta1, delta2, props)
    
    # =========================================================================
    # ÉTAPE 4: Calcul des termes thermiques à la position h̄
    # =========================================================================
    if h_bar is None:
        h_bar = h_layer / 2   # Par défaut: milieu de la couche
    
    # Évaluation du profil thermique à h̄
    T_val, dT_dx3_val = T_profile(h_bar)   # (température, gradient)
    
    # Calcul du vecteur Q = [Q₁, Q₂, Q₃]
    Q_vector = compute_Q_thermal_vector(delta1, delta2, T_val, dT_dx3_val, 
                                         alpha_coeffs, props)
    
    # =========================================================================
    # ÉTAPE 5: Assemblage F_Th et résolution du système
    # =========================================================================
    F_th = assemble_F_thermal_9x1(Q_vector)      # Vecteur 9×1
    A_vector = solve_amplitude_system(K_dyn, F_th)   # Résolution
    A_amplitudes = reshape_amplitudes_to_matrix(A_vector)   # Matrice 3×3
    
    # =========================================================================
    # ÉTAPE 6: Construction de la fonction de déplacement
    # =========================================================================
    def displacement_func(x3):
        """Calcule le déplacement [U₁, U₂, U₃] à la position x₃."""
        return build_displacement_field(A_amplitudes, tau_roots, x3)
    
    # =========================================================================
    # Retour de tous les résultats
    # =========================================================================
    return {
        'tau_roots': tau_roots,           # Valeurs propres
        'A_amplitudes': A_amplitudes,     # Amplitudes (matrice 3×3)
        'A_vector': A_vector,             # Amplitudes (vecteur 9×1)
        'K_dyn': K_dyn,                   # Matrice dynamique 9×9
        'F_th': F_th,                     # Vecteur thermique 9×1
        'Q_vector': Q_vector,             # Termes thermiques [Q₁, Q₂, Q₃]
        'delta1': delta1,                 # Nombre d'onde δ₁
        'delta2': delta2,                 # Nombre d'onde δ₂
        'displacement_func': displacement_func,   # Fonction U(x₃)
        'h_layer': h_layer,               # Épaisseur
        'props': props                    # Propriétés mécaniques
    }


def compute_stress_from_displacement(result: Dict, x3: float, in_pascal: bool = False) -> np.ndarray:
    """
    Calcule les contraintes à partir du champ de déplacement.
    
    RÉFÉRENCE: Loi de comportement thermo-élastique (Étape 4 du PDF)
    ================================================================
    
    FORMULES DES CONTRAINTES:
    -------------------------
    Les contraintes d'interface (cisaillement et normale) sont:
    
    σ₁₃ = C₅₅ × (∂U₁/∂x₃ + δ₁ × U₃)
          │      │           │
          │      │           └── Gradient de U₃ dans le plan
          │      └── Cisaillement pur x₁-x₃
          └── Module de cisaillement
    
    σ₂₃ = C₄₄ × (∂U₂/∂x₃ + δ₂ × U₃)
          Symétrique à σ₁₃ avec échange 1 ↔ 2
    
    σ₃₃ = -C₁₃ × δ₁ × U₁ - C₂₃ × δ₂ × U₂ + C₃₃ × ∂U₃/∂x₃
          │               │               │
          │               │               └── Compression normale
          │               └── Couplage Poisson x₂ → x₃
          └── Couplage Poisson x₁ → x₃ (signe négatif)
    
    Note: Les signes négatifs devant C₁₃ et C₂₃ proviennent de la
    dérivation des fonctions trigonométriques (cos → -sin).
    
    Args:
        result: Dict retourné par solve_layer_pdf_method
        x3: Position dans l'épaisseur (m)
        in_pascal: Si True, retourne les contraintes en Pa (défaut: False → GPa)
    
    Returns:
        sigma: Vecteur [σ₁₃, σ₂₃, σ₃₃] des contraintes
               - En GPa si in_pascal=False (défaut)
               - En Pa si in_pascal=True
    """
    # =========================================================================
    # Extraction des données depuis le dict result
    # =========================================================================
    A_amplitudes = result['A_amplitudes']   # Matrice 3×3 des amplitudes
    tau_roots = result['tau_roots']         # Valeurs propres [τ₁, τ₂, τ₃]
    delta1 = result['delta1']               # Nombre d'onde δ₁
    delta2 = result['delta2']               # Nombre d'onde δ₂
    props = result['props']                 # Propriétés mécaniques
    
    # =========================================================================
    # Calcul des déplacements et dérivées à la position x₃
    # =========================================================================
    U = build_displacement_field(A_amplitudes, tau_roots, x3)        # [U₁, U₂, U₃]
    dU_dx3, _ = build_displacement_derivatives(A_amplitudes, tau_roots, x3)  # [dU₁/dx₃, ...]
    
    # =========================================================================
    # Extraction des rigidités nécessaires
    # =========================================================================
    C13, C23, C33 = props['C13'], props['C23'], props['C33']   # Couplages Poisson + normal
    C44, C55 = props['C44'], props['C55']                       # Modules de cisaillement
    
    # =========================================================================
    # Calcul des 3 contraintes d'interface
    # =========================================================================
    
    # σ₁₃: Cisaillement transverse dans le plan x₁-x₃
    sigma_13 = C55 * (dU_dx3[0] + delta1 * U[2])
    #                 │           │
    #                 │           └── δ₁ × U₃: contribution du gradient de u₃ dans x₁
    #                 └── ∂U₁/∂x₃: cisaillement pur
    
    # σ₂₃: Cisaillement transverse dans le plan x₂-x₃
    sigma_23 = C44 * (dU_dx3[1] + delta2 * U[2])
    
    # σ₃₃: Contrainte normale (compression/traction dans x₃)
    sigma_33 = -C13 * delta1 * U[0] - C23 * delta2 * U[1] + C33 * dU_dx3[2]
    #           │                     │                     │
    #           │                     │                     └── Compression normale
    #           │                     └── Couplage Poisson x₂ → x₃
    #           └── Couplage Poisson x₁ → x₃
    
    # =========================================================================
    # Retour du vecteur des contraintes (avec conversion optionnelle)
    # =========================================================================
    sigma = np.array([sigma_13, sigma_23, sigma_33], dtype=complex)
    
    # Conversion GPa → Pa si demandé (les C_ij sont en GPa dans constants.py)
    if in_pascal:
        sigma = sigma * GPa_TO_PA  # Multiplication par 1e9
    
    return sigma

