#!/usr/bin/env python3
"""
Benchmark approfondi vs résultats ONERA/Safran (Bovet, Chiaruttini, Vattré 2025)

Référence: "Full-scale crystal plasticity modeling and data-driven learning 
           of microstructure effects in polycrystalline turbine blades"

Conditions de l'étude ONERA (Section 5.1):
- Température: 400°C à 810°C (ΔT ≈ 410°C à 810°C selon la zone)
- Rotation: 26,000 rpm
- Pression aérodynamique: 0.3 MPa (intrados)
- Durée de montée: 50 secondes
- Géométrie: Aube de 60 mm de hauteur

Résultats de référence (Section 5):
- σ_vonMises dans la partie haute de l'aube: 400-800 MPa
- Zones de concentration (racine): > 1 GPa
- Résistance à la traction Inconel 718 (RT): 1365 MPa
- Résistance à la traction (1088K): ~720 MPa après adoucissement
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from core.mechanical import solve_multilayer_problem
from core.constants import (
    PROPS_SUBSTRATE, PROPS_BONDCOAT, PROPS_CERAMIC,
    ALPHA_SUBSTRATE, ALPHA_BONDCOAT, ALPHA_CERAMIC
)

# =============================================================================
# PARAMÈTRES DE RÉFÉRENCE ONERA
# =============================================================================

ONERA_REFERENCE = {
    # Conditions de chargement
    'T_min': 400,           # °C - Température min (pied d'aube)
    'T_max': 810,           # °C - Température max (tête d'aube)
    'delta_T_typique': 500, # °C - ΔT typique pour calcul thermique multicouche
    
    # Résultats de contraintes FEM (Section 5)
    'sigma_vM_range': (400, 800),  # MPa - Plage normale
    'sigma_vM_max_root': 1000,     # MPa - Concentration à la racine
    
    # Propriétés matériau (Tab. 3)
    'C11_RT': 259.6,     # GPa
    'C12_RT': 179.0,     # GPa  
    'C44_RT': 109.6,     # GPa
    'C11_HT': 40.2,      # GPa @ 1198K
    'C12_HT': 27.7,      # GPa @ 1198K
    'C44_HT': 17.0,      # GPa @ 1198K
    'alpha_RT': 4.95e-6, # K⁻¹
    'alpha_HT': 14.68e-6,# K⁻¹ @ 1198K
    
    # Résistance mécanique
    'tensile_strength_RT': 1365,  # MPa
    'tensile_strength_1088K': 720, # MPa (après adoucissement ~47%)
}


def run_benchmark_case(delta_T, lw=0.1, description=""):
    """
    Exécute un cas de benchmark et retourne les résultats.
    """
    layer_configs = [
        (0.0005, PROPS_SUBSTRATE.copy(), ALPHA_SUBSTRATE.copy()),
        (0.0001, PROPS_BONDCOAT.copy(), ALPHA_BONDCOAT.copy()),
        (0.0002, PROPS_CERAMIC.copy(), ALPHA_CERAMIC.copy()),
    ]
    
    lambda_th = np.pi / lw
    result = solve_multilayer_problem(layer_configs, lw=lw, lambda_th=lambda_th, T_hat=delta_T)
    stress = result['stress_profile']
    
    sigma_33_max = np.max(np.abs(stress['sigma_33'])) / 1e6
    sigma_13_max = np.max(np.abs(stress['sigma_13'])) / 1e6
    sigma_23_max = np.max(np.abs(stress['sigma_23'])) / 1e6
    
    # Estimation von Mises simplifiée (σ_33 domine dans notre modèle)
    sigma_vM_estimated = np.sqrt(sigma_33_max**2 + 3*(sigma_13_max**2 + sigma_23_max**2))
    
    return {
        'description': description,
        'delta_T': delta_T,
        'sigma_33_max': sigma_33_max,
        'sigma_13_max': sigma_13_max,
        'sigma_23_max': sigma_23_max,
        'sigma_vM_estimated': sigma_vM_estimated
    }


def main():
    print("=" * 70)
    print("BENCHMARK APPROFONDI vs RÉSULTATS ONERA/SAFRAN")
    print("Référence: Bovet, Chiaruttini, Vattré (2025)")
    print("=" * 70)
    print()
    
    # =========================================================================
    # VÉRIFICATION DES PROPRIÉTÉS MATÉRIAUX
    # =========================================================================
    print("1. VÉRIFICATION DES PROPRIÉTÉS MATÉRIAUX (Inconel 718)")
    print("-" * 50)
    print()
    print(f"{'Propriété':<12} {'Code':>10} {'ONERA Tab.3':>15} {'Écart':>10}")
    print("-" * 50)
    
    props_check = [
        ('C11 (GPa)', PROPS_SUBSTRATE['C11'], ONERA_REFERENCE['C11_RT']),
        ('C12 (GPa)', PROPS_SUBSTRATE['C12'], ONERA_REFERENCE['C12_RT']),
        ('C44 (GPa)', PROPS_SUBSTRATE['C44'], ONERA_REFERENCE['C44_RT']),
        ('α (×10⁻⁶)', ALPHA_SUBSTRATE['alpha_1']*1e6, 
         (ONERA_REFERENCE['alpha_RT'] + ONERA_REFERENCE['alpha_HT'])/2 * 1e6),
    ]
    
    for name, code_val, ref_val in props_check:
        ecart = (code_val - ref_val) / ref_val * 100
        status = "✅" if abs(ecart) < 10 else "⚠️"
        print(f"{name:<12} {code_val:>10.1f} {ref_val:>15.1f} {ecart:>+9.1f}% {status}")
    
    print()
    
    # =========================================================================
    # CAS DE BENCHMARK
    # =========================================================================
    print("2. CAS DE BENCHMARK THERMOMÉCANIQUE")
    print("-" * 50)
    print()
    
    cases = [
        (300, "ΔT modéré (pied d'aube)"),
        (500, "ΔT typique (zone médiane)"),
        (700, "ΔT sévère (tête d'aube)"),
        (810, "ΔT max ONERA"),
    ]
    
    results = []
    for delta_T, desc in cases:
        res = run_benchmark_case(delta_T, description=desc)
        results.append(res)
    
    print(f"{'ΔT (°C)':<10} {'σ₃₃ max':>12} {'σ_vM est.':>12} {'Plage ONERA':>16} {'Statut':>8}")
    print("-" * 60)
    
    ref_min, ref_max = ONERA_REFERENCE['sigma_vM_range']
    
    for res in results:
        sigma_vM = res['sigma_vM_estimated']
        
        if sigma_vM < ref_min * 0.5:
            status = "⚠️ Sous"
        elif sigma_vM > ref_max * 1.5:
            status = "⚠️ Sur"
        else:
            status = "✅ OK"
        
        print(f"{res['delta_T']:<10} {res['sigma_33_max']:>10.1f} MPa {sigma_vM:>10.1f} MPa "
              f"{ref_min}-{ref_max} MPa {status:>8}")
    
    print()
    
    # =========================================================================
    # ANALYSE DÉTAILLÉE DU CAS TYPIQUE (ΔT = 500°C)
    # =========================================================================
    print("3. ANALYSE DÉTAILLÉE - CAS TYPIQUE (ΔT = 500°C)")
    print("-" * 50)
    print()
    
    typical_case = results[1]  # ΔT = 500°C
    
    print(f"Contraintes calculées:")
    print(f"  σ₃₃ max (arrachement)  = {typical_case['sigma_33_max']:.1f} MPa")
    print(f"  σ₁₃ max (cisaillement) = {typical_case['sigma_13_max']:.1f} MPa")
    print(f"  σ₂₃ max (cisaillement) = {typical_case['sigma_23_max']:.1f} MPa")
    print(f"  σ_vM estimé            = {typical_case['sigma_vM_estimated']:.1f} MPa")
    print()
    
    # Comparaison avec référence ONERA
    ref_typical = 30  # MPa - valeur typique de la littérature TBC pour σ₃₃
    ref_fem = (ref_min + ref_max) / 2  # Valeur moyenne FEM
    
    print(f"Comparaison avec références:")
    print(f"  ONERA FEM (σ_vM)       = {ref_min}-{ref_max} MPa")
    print(f"  Littérature TBC (σ₃₃)  = 30-100 MPa (typique)")
    print()
    
    # =========================================================================
    # RATIO PAR RAPPORT À LA LIMITE D'ÉCOULEMENT
    # =========================================================================
    print("4. MARGE DE SÉCURITÉ PAR RAPPORT À LA RUPTURE")
    print("-" * 50)
    print()
    
    sigma_yield_HT = ONERA_REFERENCE['tensile_strength_1088K'] * 0.9  # ~650 MPa
    
    for res in results:
        ratio = res['sigma_vM_estimated'] / sigma_yield_HT * 100
        margin = 100 - ratio
        status = "✅" if margin > 20 else ("⚠️" if margin > 0 else "❌")
        print(f"ΔT = {res['delta_T']}°C: σ/σ_yield = {ratio:.1f}% → Marge = {margin:.1f}% {status}")
    
    print()
    
    # =========================================================================
    # CONCLUSION
    # =========================================================================
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    
    max_sigma = max(r['sigma_vM_estimated'] for r in results)
    
    if ref_min * 0.7 <= max_sigma <= ref_max * 1.3:
        print("✅ Les contraintes calculées sont COHÉRENTES avec les résultats FEM ONERA")
        print(f"   Plage obtenue: {results[0]['sigma_vM_estimated']:.0f}-{max_sigma:.0f} MPa")
        print(f"   Plage ONERA:   {ref_min}-{ref_max} MPa")
    else:
        print("⚠️ Écart significatif avec les résultats ONERA")
        print("   → Vérifier les conditions aux limites et le chargement")
    
    print()
    print("Note: Le modèle spectral calcule principalement σ₃₃ (arrachement),")
    print("      tandis que les FEM ONERA reportent σ_vonMises (état 3D complet).")
    print("      La comparaison directe nécessite des précautions d'interprétation.")


if __name__ == '__main__':
    main()
