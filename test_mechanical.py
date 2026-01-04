"""
Test script for complete mechanical model.
"""
from core.mechanical import solve_mechanical_problem
from core.constants import THERMAL_EXPANSION
import numpy as np

# Test parameters
h_layer = 0.0005  # 500 µm
lw = 0.1          # 10 cm wavelength
lambda_th = 30.0  # Thermal mode (arbitrary)
T_hat = 100.0     # 100°C perturbation amplitude

print("=" * 60)
print("TEST: Complete Mechanical Model")
print("=" * 60)

try:
    result = solve_mechanical_problem(
        h_layer=h_layer,
        lw=lw,
        lambda_th=lambda_th,
        T_hat=T_hat,
        alpha_coeffs=THERMAL_EXPANSION
    )
    
    print("\n✅ Solver executed successfully!")
    
    # Display tau roots
    print("\n--- Tau Roots ---")
    for i, tau in enumerate(result['tau_roots']):
        print(f"  τ_{i+1} = {tau:.4f}")
    
    # Display stress at boundaries
    stress = result['stress_profile']
    print("\n--- Stress at Boundaries ---")
    print(f"  σ_33(z=0) = {stress['sigma_33'][0]:.2e} Pa")
    print(f"  σ_33(z=h) = {stress['sigma_33'][-1]:.2e} Pa")
    
    # Check boundary conditions (should be ~0 for free surfaces)
    bc_error = max(abs(stress['sigma_33'][0]), abs(stress['sigma_33'][-1]))
    if bc_error < 1e-3:
        print(f"\n✅ Boundary conditions satisfied (error = {bc_error:.2e})")
    else:
        print(f"\n⚠️ Boundary condition error = {bc_error:.2e}")
    
    # Max stress
    print("\n--- Maximum Stresses ---")
    print(f"  max|σ_13| = {np.max(np.abs(stress['sigma_13'])):.2e} Pa")
    print(f"  max|σ_23| = {np.max(np.abs(stress['sigma_23'])):.2e} Pa")
    print(f"  max|σ_33| = {np.max(np.abs(stress['sigma_33'])):.2e} Pa")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
