#!/usr/bin/env python3
"""
Test du module mechanical_pdf.py
================================

Vérifie l'implémentation selon la méthodologie PDF.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from core.mechanical_pdf import (
    compute_L_operators,
    get_Gamma_matrix,
    solve_characteristic_polynomial,
    compute_Q_thermal_vector,
    assemble_K_dyn_9x9,
    assemble_F_thermal_9x1,
    solve_amplitude_system,
    build_displacement_field,
    solve_layer_pdf_method,
    compute_stress_from_displacement
)
from core.constants import MECHANICAL_PROPS, ALPHA_SUBSTRATE

def test_gamma_matrix_properties():
    """Test les propriétés de la matrice Γ."""
    print("Test 1: Propriétés de Γ(τ)...")
    
    delta1 = delta2 = np.pi / 0.1  # lw = 10 cm
    tau = 1.0 + 0.5j
    
    Gamma = get_Gamma_matrix(tau, delta1, delta2)
    
    # Vérifier symétrie Γ₁₂ = Γ₂₁
    assert np.isclose(Gamma[0, 1], Gamma[1, 0]), "Γ₁₂ ≠ Γ₂₁"
    
    # Vérifier antisymétrie Γ₁₃ = -Γ₃₁
    assert np.isclose(Gamma[0, 2], -Gamma[2, 0]), "Γ₁₃ ≠ -Γ₃₁"
    
    # Vérifier antisymétrie Γ₂₃ = -Γ₃₂
    assert np.isclose(Gamma[1, 2], -Gamma[2, 1]), "Γ₂₃ ≠ -Γ₃₂"
    
    print("  ✅ Symétrie Γ₁₂ = Γ₂₁: OK")
    print("  ✅ Antisymétrie Γ₁₃ = -Γ₃₁: OK")
    print("  ✅ Antisymétrie Γ₂₃ = -Γ₃₂: OK")


def test_characteristic_polynomial():
    """Test la résolution du polynôme caractéristique."""
    print("\nTest 2: Polynôme caractéristique...")
    
    delta1 = delta2 = np.pi / 0.1
    
    result = solve_characteristic_polynomial(delta1, delta2)
    
    tau_selected = result['tau_selected']
    print(f"  τ₁ = {tau_selected[0]:.4f}")
    print(f"  τ₂ = {tau_selected[1]:.4f}")
    print(f"  τ₃ = {tau_selected[2]:.4f}")
    
    # Vérifier que det(Γ(τ_r)) ≈ 0 pour chaque racine
    for i, tau in enumerate(tau_selected):
        Gamma = get_Gamma_matrix(tau, delta1, delta2)
        det_val = np.linalg.det(Gamma)
        assert abs(det_val) < 1e-6 * abs(Gamma[0,0]**3), f"det(Γ(τ_{i+1})) ≠ 0"
    
    print("  ✅ det(Γ(τᵣ)) = 0 pour tous les modes")


def test_K_dyn_9x9_structure():
    """Test la structure bloc-diagonale de K_dyn."""
    print("\nTest 3: Structure K_dyn 9×9...")
    
    delta1 = delta2 = np.pi / 0.1
    result = solve_characteristic_polynomial(delta1, delta2)
    tau_roots = result['tau_selected']
    
    K_dyn = assemble_K_dyn_9x9(tau_roots, delta1, delta2)
    
    # Vérifier que K_dyn est 9×9
    assert K_dyn.shape == (9, 9), f"K_dyn shape incorrect: {K_dyn.shape}"
    
    # Vérifier les blocs hors-diagonaux sont nuls
    for r1 in range(3):
        for r2 in range(3):
            if r1 != r2:
                block = K_dyn[3*r1:3*(r1+1), 3*r2:3*(r2+1)]
                assert np.allclose(block, 0), f"Bloc ({r1},{r2}) non nul"
    
    print("  ✅ Dimension: 9×9")
    print("  ✅ Structure bloc-diagonale: OK")


def test_thermal_terms():
    """Test les termes thermiques Q."""
    print("\nTest 4: Termes thermiques Q...")
    
    delta1 = delta2 = np.pi / 0.1
    T = 100.0  # 100°C
    dT_dx3 = 1000.0  # 1000 K/m
    
    Q = compute_Q_thermal_vector(delta1, delta2, T, dT_dx3, ALPHA_SUBSTRATE)
    
    print(f"  Q₁ = {Q[0].real:.2e}")
    print(f"  Q₂ = {Q[1].real:.2e}")
    print(f"  Q₃ = {Q[2].real:.2e}")
    
    # Q₁ et Q₂ doivent être du même ordre (symétrie isotrope)
    assert np.isclose(Q[0], Q[1], rtol=0.01), "Q₁ ≠ Q₂ pour matériau isotrope"
    
    print("  ✅ Q₁ ≈ Q₂ (symétrie): OK")


def test_full_solver():
    """Test le solveur complet."""
    print("\nTest 5: Solveur complet PDF...")
    
    h_layer = 0.0005  # 500 µm
    lw = 0.1          # 10 cm
    
    def T_profile(x3):
        """Profil thermique linéaire simple."""
        T = 500 + 1800 * x3 / h_layer
        dT_dx3 = 1800 / h_layer
        return T, dT_dx3
    
    result = solve_layer_pdf_method(
        h_layer=h_layer,
        lw=lw,
        T_profile=T_profile,
        alpha_coeffs=ALPHA_SUBSTRATE
    )
    
    print(f"  τ₁ = {result['tau_roots'][0]:.4f}")
    print(f"  τ₂ = {result['tau_roots'][1]:.4f}")
    print(f"  τ₃ = {result['tau_roots'][2]:.4f}")
    
    # Test déplacement à différentes positions
    U_0 = result['displacement_func'](0)
    U_h = result['displacement_func'](h_layer)
    
    print(f"  U₃(0) = {U_0[2].real:.2e}")
    print(f"  U₃(h) = {U_h[2].real:.2e}")
    
    # Calcul des contraintes
    sigma_0 = compute_stress_from_displacement(result, 0)
    sigma_h = compute_stress_from_displacement(result, h_layer)
    
    print(f"  σ₃₃(0) = {sigma_0[2].real:.2e} GPa")
    print(f"  σ₃₃(h) = {sigma_h[2].real:.2e} GPa")
    
    print("  ✅ Solveur exécuté avec succès")


def run_all_tests():
    print("=" * 60)
    print("TESTS - MODULE MECHANICAL_PDF.PY")
    print("Méthodologie: equilibre_local_corrige.pdf")
    print("=" * 60)
    
    try:
        test_gamma_matrix_properties()
        test_characteristic_polynomial()
        test_K_dyn_9x9_structure()
        test_thermal_terms()
        test_full_solver()
        
        print("\n" + "=" * 60)
        print("✅ TOUS LES TESTS RÉUSSIS")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test échoué: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
