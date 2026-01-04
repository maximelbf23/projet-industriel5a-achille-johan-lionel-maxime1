
import numpy as np
from core.calculation import solve_tbc_model_v2
from core.mechanical import solve_multilayer_problem
from core.constants import MECHANICAL_PROPS, PROPS_SUBSTRATE, PROPS_BONDCOAT, PROPS_CERAMIC
from core.constants import ALPHA_SUBSTRATE, ALPHA_BONDCOAT, ALPHA_CERAMIC

def test_multimode_convergence():
    print("Testing Multi-mode Convergence...")
    
    # Parameters
    alpha = 0.2
    beta = 0.8
    lw = 0.1
    t_bottom = 500
    t_top = 1000
    
    # Layers config (standard)
    h_sub = 0.0005
    h_bc = 0.00001
    h_tbc = alpha * h_sub
    
    # We will compute max stress for different number of modes
    modes_to_test = [1, 3, 5, 9, 15]
    
    layer_configs_base = [
        (h_sub, PROPS_SUBSTRATE, ALPHA_SUBSTRATE),
        (h_bc, PROPS_BONDCOAT, ALPHA_BONDCOAT),
        (h_tbc, PROPS_CERAMIC, ALPHA_CERAMIC)
    ]
    
    print(f"{'Modes':<10} | {'Max Sigma_33 (MPa)':<20} | {'Max Sigma_13 (MPa)':<20}")
    print("-" * 55)
    
    results = []
    
    for n in modes_to_test:
        # 1. Thermal Solver
        thermal_res = solve_tbc_model_v2(alpha, beta, lw, t_bottom, t_top, n_modes=n)
        
        if not thermal_res['success']:
            print(f"Thermal solver failed for n={n}")
            continue
            
        T_hat_list = thermal_res['profile_params']['modes']
        
        # 2. Mech Solver
        mech_res = solve_multilayer_problem(
            layer_configs_base, 
            lw, 
            np.pi/lw, # lambda_th dummy
            T_hat_list, # The list of modes
            method='spectral'
        )
        
        stress = mech_res['stress_profile']
        max_s33 = np.max(np.abs(stress['sigma_33'])) / 1e6
        max_s13 = np.max(np.abs(stress['sigma_13'])) / 1e6
        
        print(f"{n:<10} | {max_s33:<20.2f} | {max_s13:<20.2f}")
        results.append(max_s33)
        
    # Check convergence behavior
    diff = abs(results[-1] - results[-2])
    print(f"\nDifference between {modes_to_test[-2]} and {modes_to_test[-1]} modes: {diff:.3f} MPa")
    
    if diff < 5.0:
        print("✅ Convergence looks good (stable within 5 MPa)")
    else:
        print("⚠️ Convergence not fully reached yet (edge effects strong?)")

if __name__ == "__main__":
    test_multimode_convergence()
