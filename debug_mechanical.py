"""
Debug: Check the K matrix and solution in solve_single_layer
"""
from core.mechanical import (
    solve_characteristic_equation, compute_all_eigenvectors,
    compute_all_stress_eigenvectors, compute_thermal_forcing,
    build_Phi_matrix, compute_particular_stress, solve_single_layer
)
from core.constants import THERMAL_EXPANSION, CONSTANTS
import numpy as np

# Parameters
alpha_in = 0.20
h1 = CONSTANTS['h1']
h3 = alpha_in * h1
lw = 0.1
delta_T = 900
lambda_th = np.pi / lw
delta1 = delta2 = np.pi / lw

print("=" * 60)
print("DEBUG: Single Layer Solver")
print("=" * 60)

# Setup
char_result = solve_characteristic_equation(delta1, delta2)
tau_roots = char_result['tau_roots']

eigenvectors = compute_all_eigenvectors(tau_roots, delta1, delta2)
eigenvectors = compute_all_stress_eigenvectors(eigenvectors, delta1, delta2)

thermal = compute_thermal_forcing(lambda_th, delta1, delta2, delta_T, THERMAL_EXPANSION)

# Check Phi matrices
print("\n--- Phi(0) ---")
Phi_0 = build_Phi_matrix(0, eigenvectors)
print(f"Shape: {Phi_0.shape}")
print(f"Max value: {np.max(np.abs(Phi_0)):.2e}")

print("\n--- Phi(h3) ---")
Phi_h = build_Phi_matrix(h3, eigenvectors)
print(f"Max value: {np.max(np.abs(Phi_h)):.2e}")

# Check exp factors
print("\n--- Exp factors at z=h3 ---")
for i, eig in enumerate(eigenvectors):
    tau = eig['tau']
    exp_val = np.exp(tau * h3)
    print(f"  exp(τ_{i+1} * h3) = exp({tau.real:.2f} * {h3:.6f}) = {exp_val:.4e}")

# Check the K matrix condition
print("\n--- K matrix in solve_single_layer ---")
B_stress = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

K = np.zeros((6, 6), dtype=complex)
K[0:3, :] = B_stress @ Phi_0
K[3:6, :] = B_stress @ Phi_h

print(f"K max: {np.max(np.abs(K)):.2e}")
print(f"K cond: {np.linalg.cond(K):.2e}")
print(f"K det: {np.linalg.det(K):.2e}")

# Check the RHS
A_part = thermal['A_part']
T_part_0 = compute_particular_stress(A_part, lambda_th, delta1, delta2)
T_part_h = T_part_0 * np.exp(lambda_th * h3)

print(f"\nT_part_0 = {T_part_0}")
print(f"T_part_h = {T_part_h}")

F = np.zeros(6, dtype=complex)
F[0:3] = -T_part_0
F[3:6] = -T_part_h

print(f"\nRHS max: {np.max(np.abs(F)):.2e}")

# Solve
C = np.linalg.solve(K, F)
print(f"\nC coefficients: {C}")
print(f"C max: {np.max(np.abs(C)):.2e}")
