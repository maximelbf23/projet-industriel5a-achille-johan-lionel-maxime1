
from core.mechanical import solve_multilayer_problem
from core.constants import PROPS_SUBSTRATE, PROPS_BONDCOAT, PROPS_CERAMIC, ALPHA_SUBSTRATE, ALPHA_BONDCOAT, ALPHA_CERAMIC
import numpy as np

print("="*60)
print("VERIFICATION: Spectral vs CLT Method")
print("="*60)

# Define a standard TBC configuration
h1 = 0.0005 # Substrate
h2 = 0.0001 # BondCoat
h3 = 0.0002 # TBC

layer_configs = [
    (h1, PROPS_SUBSTRATE.copy(), ALPHA_SUBSTRATE.copy()),
    (h2, PROPS_BONDCOAT.copy(), ALPHA_BONDCOAT.copy()),
    (h3, PROPS_CERAMIC.copy(), ALPHA_CERAMIC.copy())
]

lw = 0.1
lambda_th = np.pi/lw
delta_T = 900.0 # High gradient

print(f"Parameters: Lw={lw}m, dT={delta_T}K")

# 1. Run Spectral Method (Rigorous)
print("\n--- Running Spectral Method (Rigorous) ---")
try:
    res_spectral = solve_multilayer_problem(layer_configs, lw, lambda_th, delta_T, method='spectral')
    sigma_33_spec = res_spectral['stress_profile']['sigma_33']
    max_s33_spec = np.max(np.abs(sigma_33_spec))
    print(f"✅ Spectral Success. Max Sigma_33: {max_s33_spec/1e6:.2f} MPa")
    
    # Check BC
    s33_bot = sigma_33_spec[0]
    s33_top = sigma_33_spec[-1]
    print(f"Boundary Conditions (should be ~0): Bot={s33_bot:.2e}, Top={s33_top:.2e}")
    if abs(s33_bot) > 1e-3 or abs(s33_top) > 1e-3:
        print("⚠️ Warning: BC not perfect (numerical artifact?)")
    else:
        print("✅ BC Satisfied.")

except Exception as e:
    print(f"❌ Spectral Failed: {e}")
    import traceback
    traceback.print_exc()

# 2. Run CLT Method (Approximate)
print("\n--- Running CLT Method (Approximate) ---")
try:
    res_clt = solve_multilayer_problem(layer_configs, lw, lambda_th, delta_T, method='clt')
    sigma_33_clt = res_clt['stress_profile']['sigma_33']
    max_s33_clt = np.max(np.abs(sigma_33_clt))
    print(f"✅ CLT Success. Max Sigma_33: {max_s33_clt/1e6:.2f} MPa")
except Exception as e:
    print(f"❌ CLT Failed: {e}")

# Comparison
print("\n--- Comparison ---")
# Spectral should show localized concentrations at interfaces
# CLT uses a shape function
diff = abs(max_s33_spec - max_s33_clt)
print(f"Difference in Max Stress: {diff/1e6:.2f} MPa")
print("Done.")
