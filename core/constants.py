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

# --- TÂCHE 1 : CONSTANTES POUR L'IMPACT ---
IMPACT_PARAMS = {
    'rho_ceram': 8700,      # Masse volumique (kg/m^3)
    'cost_per_vol': 4800000,  # Coût estimé par volume (€/m^3)
    'co2_per_kg': 15.5,      # Empreinte carbone (kgCO2/kg matière)
    
    # --- Paramètres pour une géométrie d'aube simplifiée ---
    'blade_height': 0.1,    # Hauteur de l'aube (m)
    'blade_chord': 0.05     # Corde de l'aube (m)
}
