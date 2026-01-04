"""
Test script for multilayer mechanical solver with REAL per-layer properties.
"""
from core.mechanical import solve_multilayer_problem
from core.constants import PROPS_SUBSTRATE, PROPS_BONDCOAT, PROPS_CERAMIC, ALPHA_SUBSTRATE, ALPHA_BONDCOAT, ALPHA_CERAMIC
import numpy as np

# Define a 3-layer TBC system with REAL material properties
# Layer 1: Superalloy substrate (bottom)
# Layer 2: Bond coat (middle)
# Layer 3: Ceramic TBC (top)

# Layer thicknesses
h_substrate = 0.0005  # 500 µm
h_bondcoat = 0.0001   # 100 µm
h_ceramic = 0.0002    # 200 µm

# Configuration avec propriétés réelles différenciées
layer_configs = [
    (h_substrate, PROPS_SUBSTRATE.copy(), ALPHA_SUBSTRATE.copy()),
    (h_bondcoat, PROPS_BONDCOAT.copy(), ALPHA_BONDCOAT.copy()),
    (h_ceramic, PROPS_CERAMIC.copy(), ALPHA_CERAMIC.copy()),
]

# Parameters
lw = 0.1       # 10 cm wavelength
lambda_th = np.pi / lw
T_hat = 500    # 500°C temperature amplitude

print("=" * 60)
print("TEST: 3-Layer Multilayer with REAL Material Properties")
print("=" * 60)
print(f"Layer 1 (Substrate - CMSX-4):  {h_substrate*1e6:.0f} µm, C33={PROPS_SUBSTRATE['C33']/1e9:.0f} GPa")
print(f"Layer 2 (Bond coat - MCrAlY):  {h_bondcoat*1e6:.0f} µm, C33={PROPS_BONDCOAT['C33']/1e9:.0f} GPa")
print(f"Layer 3 (Ceramic - YSZ):       {h_ceramic*1e6:.0f} µm, C33={PROPS_CERAMIC['C33']/1e9:.0f} GPa")
print(f"Total thickness: {(h_substrate+h_bondcoat+h_ceramic)*1e6:.0f} µm")
print(f"Lw = {lw} m, λ_th = {lambda_th:.2f}, ΔT = {T_hat}°C")

try:
    result = solve_multilayer_problem(layer_configs, lw, lambda_th, T_hat)
    
    print("\n✅ Multilayer solver executed successfully!")
    
    stress = result['stress_profile']
    
    print(f"\n--- Stress Profile Summary ---")
    print(f"Total points: {len(stress['z'])}")
    print(f"max|σ_13| = {np.max(np.abs(stress['sigma_13'])):.2e} Pa = {np.max(np.abs(stress['sigma_13']))/1e6:.2f} MPa")
    print(f"max|σ_23| = {np.max(np.abs(stress['sigma_23'])):.2e} Pa = {np.max(np.abs(stress['sigma_23']))/1e6:.2f} MPa")
    print(f"max|σ_33| = {np.max(np.abs(stress['sigma_33'])):.2e} Pa = {np.max(np.abs(stress['sigma_33']))/1e6:.2f} MPa")
    
    # Check boundary conditions
    print(f"\n--- Boundary Conditions ---")
    print(f"σ_33(z=0) = {stress['sigma_33'][0]:.2e} Pa")
    print(f"σ_33(z=H) = {stress['sigma_33'][-1]:.2e} Pa")
    
    # Interface stresses
    print(f"\n--- Interface Stresses (critical for delamination) ---")
    layer_names = ['Substrate', 'Bond Coat', 'Ceramic']
    layers = result['layers']
    for k in range(len(layers) - 1):
        z_interface = layers[k].z_top
        idx = np.argmin(np.abs(stress['z'] - z_interface))
        print(f"Interface {layer_names[k]}/{layer_names[k+1]} (z={z_interface*1e6:.0f}µm): σ_33 = {stress['sigma_33'][idx]/1e6:.2f} MPa")
    
    # K_glob condition number
    if result.get('cond_K') is not None:
        print(f"\n--- Matrix Diagnostics ---")
        print(f"cond(K_glob) = {result['cond_K']:.2e}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
