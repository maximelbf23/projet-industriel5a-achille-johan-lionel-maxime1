# ==========================================
# DÉFINITION DES PARAMÈTRES PHYSIQUES (FIXES)
# ==========================================
CONSTANTS = {
    'h1': 0.0005,      # Superalliage (m)
    'h2': 0.00001,     # Liaison (m)
    'k33_1': 20.0,     # Superalliage (W/mK)
    'k33_2': 8.0,      # Liaison (W/mK)
    'k33_3': 1.5,      # Céramique (W/mK) - Base
    'T_bottom': 500,   # T(x3=0)
    'T_top': 1400,     # T(x3=H)
    'T_crit': 1100,    # Température critique
    'Securite_pct': 0.8
}

# --- PROPRIÉTÉS MÉCANIQUES (Orthotrope Zircone approx) ---
# Matrice de rigidité C (GPa) - Changé de Pa pour éviter les grands nombres dans les déterminants
# Pour convertir en Pa: multiplier par GPa_TO_PA
GPa_TO_PA = 1e9  # Facteur de conversion

MECHANICAL_PROPS = {
    'C11': 200, 'C12': 50,  'C13': 50,
    'C21': 50,  'C22': 200, 'C23': 50,
    'C31': 50,  'C32': 50,  'C33': 200,
    'C44': 75,  'C55': 75,  'C66': 75
}

# --- PROPRIÉTÉS MÉCANIQUES PAR COUCHE (TBC système) ---
# Référence: Bovet, Chiaruttini, Vattré - ONERA/Safran (2025)
# "Full-scale crystal plasticity modeling..." - Table 3

# Couche 1: Substrat superalliage base Nickel (Inconel 718)
# Valeurs à température ambiante (T0) - Source: ONERA/Safran Tab. 3
PROPS_SUBSTRATE = {
    'C11': 260, 'C12': 179, 'C13': 179,  # c11=259.6, c12=179.0 GPa
    'C21': 179, 'C22': 260, 'C23': 179,
    'C31': 179, 'C32': 179, 'C33': 260,
    'C44': 110, 'C55': 110, 'C66': 110   # c44=109.6 GPa
}

# Couche 2: Bond Coat (MCrAlY - NiCoCrAlY)
# Valeurs estimées (littérature TBC)
PROPS_BONDCOAT = {
    'C11': 180, 'C12': 80,  'C13': 80,
    'C21': 80,  'C22': 180, 'C23': 80,
    'C31': 80,  'C32': 80,  'C33': 180,
    'C44': 60,  'C55': 60,  'C66': 60
}

# Couche 3: Céramique TBC (YSZ - 7% Y2O3-ZrO2)
# Valeurs typiques YSZ poreux (littérature TBC)
PROPS_CERAMIC = {
    'C11': 50,  'C12': 10,  'C13': 10,
    'C21': 10,  'C22': 50,  'C23': 10,
    'C31': 10,  'C32': 10,  'C33': 50,
    'C44': 20,  'C55': 20,  'C66': 20
}

# Coefficients de dilatation thermique par couche (1/K)
# Référence: ONERA Tab. 3 - αT varie de 4.95e-6 (RT) à 14.68e-6 (1198K)
# Valeurs moyennes utilisées pour conditions de service (~800-1000K)
ALPHA_SUBSTRATE = {'alpha_1': 12e-6, 'alpha_2': 12e-6, 'alpha_3': 12e-6}  # Inconel 718 ~800K
ALPHA_BONDCOAT = {'alpha_1': 14e-6, 'alpha_2': 14e-6, 'alpha_3': 14e-6}   # MCrAlY
ALPHA_CERAMIC = {'alpha_1': 10e-6, 'alpha_2': 10e-6, 'alpha_3': 10e-6}    # YSZ

# --- COEFFICIENTS DE DILATATION THERMIQUE (legacy) ---
THERMAL_EXPANSION = {
    # Coefficients alpha (1/K) pour chaque couche
    'alpha_1': 13e-6,   # Superalliage (Nickel base)
    'alpha_2': 14e-6,   # Couche liaison (MCrAlY)
    'alpha_3': 10e-6,   # Céramique TBC (YSZ)
}

# --- TÂCHE 1 : CONSTANTES POUR L'IMPACT ---
IMPACT_PARAMS = {
    'rho_ceram': 8700,      # Masse volumique (kg/m^3)
    'cost_per_vol': 32000000,  # Coût estimé par volume (€/m^3)
    'co2_per_kg': 15.5,      # Empreinte carbone (kgCO2/kg matière)
    
    # --- Paramètres pour une géométrie d'aube simplifiée ---
    'blade_height': 0.1,    # Hauteur de l'aube (m)
    'blade_chord': 0.05     # Corde de l'aube (m)
}
