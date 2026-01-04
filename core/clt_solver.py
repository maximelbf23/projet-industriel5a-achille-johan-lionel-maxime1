"""
Classical Laminate Theory (CLT) based solver for multilayer thermal stresses.

This is a numerically STABLE approach used in industry for composite analysis.
Reference: Jones, R.M. "Mechanics of Composite Materials", 2nd Ed.

Key equations:
- Thermal resultants: N_th = Σ [Q]_k × α_k × ΔT × h_k
- A-B-D matrix: relates forces/moments to strains/curvatures
- Layer stresses: σ_k = [Q]_k × (ε_0 + z×κ - α_k×ΔT)
"""

import numpy as np

# STANDARD UNIT CONVERSION
GPa_TO_PA = 1e9

def compute_Q_matrix(E1, E2, nu12, G12):
    """
    Compute reduced stiffness matrix Q for orthotropic material.
    
    For isotropic: E1=E2=E, nu12=nu, G12=E/(2(1+nu))
    """
    nu21 = nu12 * E2 / E1
    denom = 1 - nu12 * nu21
    
    Q = np.array([
        [E1/denom, nu12*E2/denom, 0],
        [nu12*E2/denom, E2/denom, 0],
        [0, 0, G12]
    ])
    return Q


def compute_ABD_matrix(layers):
    """
    Compute the A-B-D stiffness matrix for a laminate.
    
    A = Extensional stiffness (N/m)
    B = Coupling stiffness (N)
    D = Bending stiffness (N·m)
    
    Args:
        layers: List of dicts with {h, E, nu, alpha, z_mid}
    
    Returns:
        A, B, D: 3x3 matrices
    """
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    
    for layer in layers:
        E = layer['E']
        nu = layer['nu']
        G = E / (2 * (1 + nu))  # Isotropic approximation
        h = layer['h']
        z_bot = layer['z_bot']
        z_top = layer['z_top']
        
        Q = compute_Q_matrix(E, E, nu, G)
        
        # Integration through thickness
        A += Q * (z_top - z_bot)
        B += 0.5 * Q * (z_top**2 - z_bot**2)
        D += (1/3) * Q * (z_top**3 - z_bot**3)
    
    return A, B, D


def compute_thermal_resultants(layers, delta_T):
    """
    Compute thermal force and moment resultants.
    
    N_th = Σ [Q]_k × α_k × ΔT(z) × dz
    M_th = Σ [Q]_k × α_k × ΔT(z) × z × dz
    
    For uniform ΔT:
    N_th = Σ [Q]_k × α_k × ΔT × h_k
    M_th = Σ [Q]_k × α_k × ΔT × h_k × z_mid_k
    """
    N_th = np.zeros(3)
    M_th = np.zeros(3)
    
    for layer in layers:
        E = layer['E']
        nu = layer['nu']
        G = E / (2 * (1 + nu))
        alpha = layer['alpha']
        h = layer['h']
        z_mid = (layer['z_bot'] + layer['z_top']) / 2
        
        Q = compute_Q_matrix(E, E, nu, G)
        
        # Thermal strain vector [α, α, 0] for isotropic
        alpha_vec = np.array([alpha, alpha, 0])
        
        # Thermal stress resultant contribution
        Q_alpha = Q @ alpha_vec
        N_th += Q_alpha * delta_T * h
        M_th += Q_alpha * delta_T * h * z_mid
    
    return N_th, M_th


def solve_clt_thermal(layers, delta_T):
    """
    Solve for thermal stresses using Classical Laminate Theory.
    
    For free expansion (no external loads, no constraints):
    - Total resultants = 0 (N = M = 0)
    - ε_0 and κ adjust to satisfy this
    - Internal stresses arise from constraint of different layers
    
    Returns:
        epsilon_0: Mid-plane strains
        kappa: Curvatures
        layer_stresses: Stress in each layer
    """
    # Compute ABD matrix
    A, B, D = compute_ABD_matrix(layers)
    
    # Compute thermal resultants
    N_th, M_th = compute_thermal_resultants(layers, delta_T)
    
    # For free thermal expansion with no external loads:
    # [N]   [A B] [ε_0]   [N_th]
    # [M] = [B D] [κ  ] + [M_th] = 0
    #
    # Therefore:
    # [A B] [ε_0]   [N_th]
    # [B D] [κ  ] = [M_th]
    
    # Assemble full ABD matrix
    ABD = np.block([
        [A, B],
        [B, D]
    ])
    
    # RHS (Positive sign for thermal expansion in free state)
    rhs = np.concatenate([N_th, M_th])
    
    # Solve
    cond_ABD = np.linalg.cond(ABD)
    
    if cond_ABD < 1e12:
        solution = np.linalg.solve(ABD, rhs)
    else:
        # Use pseudo-inverse for ill-conditioned cases
        solution = np.linalg.lstsq(ABD, rhs, rcond=None)[0]
    
    epsilon_0 = solution[:3]
    kappa = solution[3:]
    
    # Compute stresses in each layer
    layer_stresses = []
    
    for layer in layers:
        E = layer['E']
        nu = layer['nu']
        G = E / (2 * (1 + nu))
        alpha = layer['alpha']
        z_mid = (layer['z_bot'] + layer['z_top']) / 2
        
        Q = compute_Q_matrix(E, E, nu, G)
        alpha_vec = np.array([alpha, alpha, 0])
        
        # Total strain at layer midplane
        epsilon_total = epsilon_0 + z_mid * kappa
        
        # Mechanical strain = Total - Thermal
        epsilon_mech = epsilon_total - alpha_vec * delta_T
        
        # Stress
        sigma = Q @ epsilon_mech
        
        layer_stresses.append({
            'z_mid': z_mid,
            'sigma_1': sigma[0],  # In-plane stress 1
            'sigma_2': sigma[1],  # In-plane stress 2
            'tau_12': sigma[2],   # In-plane shear
            'layer_name': layer.get('name', 'Unknown')
        })
    
    return {
        'epsilon_0': epsilon_0,
        'kappa': kappa,
        'layer_stresses': layer_stresses,
        'cond_ABD': cond_ABD,
        'N_th': N_th,
        'M_th': M_th
    }


def compute_clt_stress_profile(layers, delta_T, n_points_per_layer=50):
    """
    Compute stress profile through the laminate thickness.
    """
    result = solve_clt_thermal(layers, delta_T)
    epsilon_0 = result['epsilon_0']
    kappa = result['kappa']
    
    z_all = []
    sigma_11_all = []
    sigma_22_all = []
    tau_12_all = []
    layer_idx_all = []
    
    for k, layer in enumerate(layers):
        E = layer['E']
        nu = layer['nu']
        G = E / (2 * (1 + nu))
        alpha = layer['alpha']
        
        Q = compute_Q_matrix(E, E, nu, G)
        alpha_vec = np.array([alpha, alpha, 0])
        
        z_local = np.linspace(layer['z_bot'], layer['z_top'], n_points_per_layer)
        
        for z in z_local:
            # Total strain at this z
            epsilon_total = epsilon_0 + z * kappa
            
            # Mechanical strain
            epsilon_mech = epsilon_total - alpha_vec * delta_T
            
            # Stress
            sigma = Q @ epsilon_mech
            
            z_all.append(z)
            sigma_11_all.append(sigma[0])
            sigma_22_all.append(sigma[1])
            tau_12_all.append(sigma[2])
            layer_idx_all.append(k)
    
    return {
        'z': np.array(z_all),
        'sigma_11': np.array(sigma_11_all),
        'sigma_22': np.array(sigma_22_all),
        'tau_12': np.array(tau_12_all),
        'layer_idx': np.array(layer_idx_all),
        **result
    }


def solve_multilayer_clt(layer_configs, delta_T, n_points_per_layer=50):
    """
    High-level API for CLT multilayer thermal stress analysis.
    
    Args:
        layer_configs: List of (h, props, alpha_dict) tuples
        delta_T: Temperature change
        n_points_per_layer: Number of points for profile
    
    Returns:
        Stress profile compatible with existing damage analysis
    """
    # Build layer list with z positions
    layers = []
    z_current = 0
    
    for h, props, alpha_dict in layer_configs:
        E = props.get('C33', 200e9) * 0.8  # Approximate E from C33
        nu = 0.3
        alpha = alpha_dict.get('alpha_3', 10e-6)
        
        layers.append({
            'h': h,
            'E': E,
            'nu': nu,
            'alpha': alpha,
            'z_bot': z_current,
            'z_top': z_current + h,
            'name': f'Layer_{len(layers)+1}'
        })
        z_current += h
    
    # Solve using CLT
    result = compute_clt_stress_profile(layers, delta_T, n_points_per_layer)
    
    # Map to the existing interface (sigma_13, sigma_23, sigma_33)
    # In CLT: sigma_11, sigma_22 are in-plane, tau_12 is in-plane shear
    # For through-thickness we use:
    # - sigma_33 ≈ 0 (plane stress assumption in CLT)
    # - sigma_13, sigma_23 ≈ 0 (no transverse shear in basic CLT)
    # 
    # For comparison, we'll use sigma_11 as the dominant stress
    
    # Generate σ_33 from equilibrium considerations
    # In reality, σ_33 develops near interfaces due to mismatch
    # We'll use the in-plane stress as proxy for interface normal stress
    
    H_total = z_current
    z_norm = result['z'] / H_total
    
    # Shape function for σ_33 (zero at surfaces)
    shape_factor = 4 * z_norm * (1 - z_norm)
    
    # Use max in-plane stress as basis for interface stress
    sigma_33 = result['sigma_11'] * shape_factor * 0.3  # Scale factor for interface
    
    # Shear stresses: calcul du gradient PAR COUCHE pour éviter les pics aux interfaces
    # Le gradient global de sigma_11 crée des valeurs aberrantes aux discontinuités
    sigma_13 = np.zeros_like(result['z'])
    
    layer_idx = result['layer_idx']
    for k in range(int(np.max(layer_idx)) + 1):
        mask = layer_idx == k
        if np.sum(mask) > 1:
            z_layer = result['z'][mask]
            s11_layer = result['sigma_11'][mask]
            
            # Gradient uniquement à l'intérieur de la couche
            if len(z_layer) > 1:
                grad = np.gradient(s11_layer, z_layer)
                # Facteur d'échelle pour conversion vers cisaillement transverse
                # Typiquement sigma_13 << sigma_11 dans CLT
                sigma_13[mask] = grad * H_total * 0.001  # Réduire le facteur (était 0.01)
            else:
                sigma_13[mask] = 0
    
    # Lisser aux interfaces (transition douce sur 2 points)
    n_smooth = 2
    for k in range(int(np.max(layer_idx))):
        # Trouver les indices de transition
        trans_idx = np.where((layer_idx[:-1] == k) & (layer_idx[1:] == k+1))[0]
        for idx in trans_idx:
            # Moyenne les valeurs autour de l'interface
            if idx > 0 and idx < len(sigma_13) - 1:
                sigma_13[idx] = (sigma_13[idx-1] + sigma_13[idx+1]) / 2
                sigma_13[idx+1] = sigma_13[idx]
    
    sigma_23 = sigma_13  # Symmetric assumption
    
    return {
        'stress_profile': {
            'z': result['z'],
            'sigma_13': sigma_13 * GPa_TO_PA,
            'sigma_23': sigma_23 * GPa_TO_PA,
            'sigma_33': sigma_33 * GPa_TO_PA,
            'sigma_11': result['sigma_11'] * GPa_TO_PA,  # Convert to Pa
            'sigma_22': result['sigma_22'] * GPa_TO_PA,
            'layer_idx': result['layer_idx']
        },
        'layers': layers,
        'cond_ABD': result['cond_ABD'],
        'epsilon_0': result['epsilon_0'],
        'kappa': result['kappa']
    }
