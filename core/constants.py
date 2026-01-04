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
# Couche 1: Substrat superalliage base Nickel (ex: CMSX-4)
PROPS_SUBSTRATE = {
    'C11': 250, 'C12': 160, 'C13': 160,
    'C21': 160, 'C22': 250, 'C23': 160,
    'C31': 160, 'C32': 160, 'C33': 250,
    'C44': 130, 'C55': 130, 'C66': 130
}

# Couche 2: Bond Coat (MCrAlY - NiCoCrAlY)
PROPS_BONDCOAT = {
    'C11': 180, 'C12': 80,  'C13': 80,
    'C21': 80,  'C22': 180, 'C23': 80,
    'C31': 80,  'C32': 80,  'C33': 180,
    'C44': 60,  'C55': 60,  'C66': 60
}

# Couche 3: Céramique TBC (YSZ - 7% Y2O3-ZrO2)
PROPS_CERAMIC = {
    'C11': 50,  'C12': 10,  'C13': 10,
    'C21': 10,  'C22': 50,  'C23': 10,
    'C31': 10,  'C32': 10,  'C33': 50,
    'C44': 20,  'C55': 20,  'C66': 20
}

# Coefficients de dilatation thermique par couche (1/K)
ALPHA_SUBSTRATE = {'alpha_1': 13e-6, 'alpha_2': 13e-6, 'alpha_3': 13e-6}
ALPHA_BONDCOAT = {'alpha_1': 14e-6, 'alpha_2': 14e-6, 'alpha_3': 14e-6}
ALPHA_CERAMIC = {'alpha_1': 10e-6, 'alpha_2': 10e-6, 'alpha_3': 10e-6}

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
