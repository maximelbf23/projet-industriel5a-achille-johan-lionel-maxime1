import streamlit as st

def render():
    # === EN-TÊTE HERO SPECTACULAIRE ===
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e1b4b 100%);
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; position: relative; overflow: hidden;
                border: 1px solid rgba(245, 158, 11, 0.2); box-shadow: 0 0 50px rgba(245, 158, 11, 0.15);">
        <div style="position: absolute; top: -40px; right: -40px; width: 180px; height: 180px; 
                    background: radial-gradient(circle, rgba(245,158,11,0.25) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; 
                    background: radial-gradient(circle, rgba(234,179,8,0.2) 0%, transparent 70%); 
                    border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;
                       background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 50%, #fde047 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text;">
                📖 Démarche & Théorie
            </h2>
            <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 1rem;">
                Modélisation semi-analytique • Méthode spectrale • 8 étapes de résolution • Projet 5A ESTACA/ONERA
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.08)); 
                padding: 1.5rem; border-radius: 16px; border-left: 4px solid #3b82f6; margin-bottom: 2rem;">
        <h3 style="color: #60a5fa; margin: 0 0 0.5rem 0;">🎯 Contexte Industriel</h3>
        <p style="color: #cbd5e1; font-size: 1rem; margin: 0;">
            Les <strong>aubes de turbines</strong> des moteurs aéronautiques sont soumises à des conditions extrêmes 
            (T > 1200°C, gradients thermiques, contraintes cycliques). Les <strong>barrières thermiques (TBC)</strong> 
            protègent le substrat métallique mais les <strong>interfaces multicouches</strong> sont des zones 
            critiques de <strong>concentration de contraintes</strong> et de risque de délamination.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture TBC
    st.markdown("### 🏗️ Architecture d'une Aube Multicouche")
    col_arch1, col_arch2 = st.columns([2, 1])
    
    with col_arch1:
        st.code("""
    ┌───────────────────────────────────────────────────────┐
    │                      GAZ CHAUD                         │  T_top ≈ 1200-1400°C
    │                   (combustion)                         │
    ├───────────────────────────────────────────────────────┤
    │  ████████████████████████████████████████████████████ │  Couche 3: Céramique (YSZ)
    │  ██████████████████ TBC (h₃) ████████████████████████ │  Zircone stabilisée Yttrine
    │  ████████████████████████████████████████████████████ │  k₃₃ ≈ 1.5 W/m·K (isolant)
    ├───────────────────────────────────────────────────────┤  ← Interface critique h₁
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ BondCoat (h₂) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  Couche 2: MCrAlY
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  Protection oxydation
    ├───────────────────────────────────────────────────────┤
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  Couche 1: Superalliage (Ni)
    │  ░░░░░░░░░░░░░░░░░░ SUBSTRAT (h₁) ░░░░░░░░░░░░░░░░░░ │  Structure porteuse
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  k₃₃ ≈ 20 W/m·K
    ├───────────────────────────────────────────────────────┤
    │                    REFROIDISSEMENT                     │  T_bottom ≈ 600-800°C
    │                   (air comprimé)                       │
    └───────────────────────────────────────────────────────┘
        """, language=None)
    
    with col_arch2:
        st.markdown("""
        **Paramètres clés :**
        
        - **α = h₃/h₁** : Ratio épaisseur TBC/Substrat
        - **β = k₁₁/k₃₃** : Anisotropie thermique
        - **Lw** : Longueur d'onde perturbation
        - **N** : Nombre de couches
        
        **Interfaces critiques :**
        - TBC / BondCoat (h₁)
        - BondCoat / Substrat
        """)
    
    st.markdown("---")
    
    # Méthodologie en 8 étapes
    st.markdown("### 📋 Méthodologie Semi-Analytique (8 Étapes)")
    
    # THERMIQUE
    st.markdown("#### 🔥 Partie Thermique (Étapes 1-3)")
    
    with st.expander("**Étape 1** — Définition Géométrique et Thermique", expanded=True):
        col_e1a, col_e1b = st.columns(2)
        with col_e1a:
            st.markdown("""
            **Géométrie :**
            - Plaque multicouche de N couches empilées selon x₃
            - Domaine : (x₁, x₂, x₃) ∈ [0, L₁] × [0, L₂] × [0, H]
            - Épaisseur totale : H = Σhᵢ
            """)
        with col_e1b:
            st.markdown("""
            **Conductivité anisotrope par couche :**
            """)
            st.latex(r"k^i = \begin{pmatrix} k_{11}^i & 0 & 0 \\ 0 & k_{22}^i & 0 \\ 0 & 0 & k_{33}^i \end{pmatrix}")
            st.markdown("avec k₁₁ = k₂₂ ≠ k₃₃ (isotropie transverse)")
    
    with st.expander("**Étape 2** — Représentation Spectrale de la Température", expanded=False):
        st.markdown("On développe T en **série double de Fourier** (séparation des variables) :")
        st.latex(r"T(x_\alpha, x_3) = \sum_{m_\alpha, m_\beta} \hat{T}_{m_\alpha m_\beta}(x_3) \sin(\delta_\alpha x_\alpha) \sin(\delta_\beta x_\beta)")
        st.latex(r"\text{avec } \delta_\alpha = \frac{m_\alpha \pi}{L_\alpha}")
        st.info("💡 Cette méthode réduit un problème 3D à une collection de problèmes 1D (un par mode de Fourier).")
    
    with st.expander("**Étape 3** — Résolution de la Conduction dans Chaque Couche", expanded=False):
        st.markdown("**Équation de conduction 1D** pour chaque coefficient de Fourier :")
        st.latex(r"-\frac{d}{dx_3}\left(k_{33}^i \frac{d\hat{T}}{dx_3}\right) + k_{\eta\eta}^i \delta_\eta^2 \hat{T} = 0")
        st.markdown("**Solution générale** (exponentielle) :")
        st.latex(r"\hat{T}^i(x_3) = A^i e^{\lambda^i x_3} + B^i e^{-\lambda^i x_3}, \quad \lambda^i = \delta_\eta \sqrt{\frac{k_{\eta\eta}^i}{k_{33}^i}}")
        st.markdown("""
        **Conditions de raccord aux interfaces :**
        - Continuité de T : [T] = 0
        - Continuité du flux : [k₃₃ ∂₃T] = 0
        """)
    
    # MÉCANIQUE
    st.markdown("#### ⚙️ Partie Mécanique (Étapes 4-8)")
    
    with st.expander("**Étape 4** — Loi de Comportement Thermoélastique Anisotrope", expanded=False):
        st.markdown("**Tenseur des contraintes** avec couplage thermique :")
        st.latex(r"\sigma_{ij} = C_{ijkl}(x_3) \left( \varepsilon_{kl} - \alpha_{kl}(x_3) T(x) \right)")
        st.markdown("""
        - **Cᵢⱼₖₗ** : Tenseur de rigidité (orthotrope, 9 constantes indépendantes)
        - **αₖₗ** : Coefficients de dilatation thermique (anisotropes)
        - **εₖₗ** : Déformation infinitésimale = ½(∂ᵢuⱼ + ∂ⱼuᵢ)
        """)
    
    with st.expander("**Étape 5** — Séparation de Variables sur les Déplacements", expanded=False):
        st.markdown("**Ansatz de séparation** sur les champs de déplacement :")
        st.latex(r"u_\alpha(x_\alpha, x_3) = U_\alpha(x_3) \cos(\delta_\alpha x_\alpha) \sin(\delta_\beta x_\beta)")
        st.latex(r"u_3(x_\alpha, x_3) = U_3(x_3) \sin(\delta_\alpha x_\alpha) \sin(\delta_\beta x_\beta)")
        st.markdown("En injectant dans l'équilibre local ∂ⱼσᵢⱼ = 0, on obtient un **système d'EDO couplées** pour Uₖ(x₃).")
    
    with st.expander("**Étape 6** — Ansatz Exponentiel et Problème aux Valeurs Propres", expanded=True):
        st.markdown("**Ansatz exponentiel** pour les solutions homogènes :")
        st.latex(r"U_\alpha(x_3) = A_\alpha e^{\tau x_3}, \quad U_3(x_3) = A_3 e^{\tau x_3}")
        st.markdown("Cela conduit au **système eigenvalue** :")
        st.latex(r"M(\tau) \cdot \mathbf{A} = 0")
        st.markdown("avec la **matrice caractéristique M(τ)** :")
        st.latex(r"""M(\tau) = \begin{pmatrix} 
        \tau^2 C_{55} - \delta_1^2 C_{11} - \delta_2^2 C_{66} & -\delta_1\delta_2(C_{12}+C_{66}) & \tau\delta_1(C_{13}+C_{55}) \\
        -\delta_1\delta_2(C_{21}+C_{66}) & \tau^2 C_{44} - \delta_2^2 C_{22} - \delta_1^2 C_{66} & \tau\delta_2(C_{23}+C_{44}) \\
        -\tau\delta_1(C_{31}+C_{55}) & -\tau\delta_2(C_{32}+C_{44}) & \tau^2 C_{33} - \delta_1^2 C_{55} - \delta_2^2 C_{44}
        \end{pmatrix}""")
        st.warning("⚠️ Les valeurs propres τᵣ sont les **6 racines** de det(M(τ)) = 0, formant des paires conjuguées (±τ).")
    
    with st.expander("**Étape 7** — Conditions aux Interfaces et aux Bords", expanded=False):
        st.markdown("**Aux interfaces** (x₃ = xᵢ entre couches i et i+1) :")
        st.latex(r"[U_k] = 0 \quad \text{(continuité des déplacements)}")
        st.latex(r"[C_{\alpha 3 \alpha 3}(\partial_3 U_\alpha + \delta_\alpha U_3)] = 0 \quad \text{(continuité des tractions)}")
        st.markdown("**Aux extrémités** (x₃ = 0 et x₃ = H) — surfaces libres :")
        st.latex(r"\sigma_{i3}(x_3 = 0) = \sigma_{i3}(x_3 = H) = 0")
    
    with st.expander("**Étape 8** — Assemblage et Résolution Globale", expanded=False):
        st.markdown("**Système global** :")
        st.latex(r"M_{global} \cdot A_{global} = F_{thermique}")
        st.markdown("""
        - **M_global** : Matrice d'assemblage (blocs par couche et interface)
        - **A_global** : Amplitudes inconnues {Aᵢ,ʳₖ} (3×6×N coefficients)
        - **F_thermique** : Second membre dû au chargement T(x₃)
        
        **Procédure de résolution :**
        1. Calcul des racines τᵢʳ pour chaque couche
        2. Assemblage des expressions de déplacement/contrainte
        3. Application des conditions d'interface et de bord
        4. Résolution du système linéaire
        5. Reconstruction des champs u(x) et σᵢⱼ(x)
        """)
    
    # Critères d'endommagement
    st.markdown("---")
    st.markdown("### 🔴 Critères d'Endommagement")
    
    col_crit1, col_crit2 = st.columns(2)
    with col_crit1:
        st.markdown("**Indicateur D (Thermique + Mécanique) :**")
        st.latex(r"D = \max(D_{th}, D_{mec})")
        st.markdown("avec :")
        st.latex(r"D_{th} = \begin{cases} 1 + \frac{T_{int} - T_{crit}}{200} & \text{si } T_{int} > T_{crit} \\ 0 & \text{sinon} \end{cases}")
        st.latex(r"D_{mec} = \max\left(\frac{\sigma_{th}}{\sigma_{crit}}, \frac{\tau}{\tau_{crit}}\right)")
        st.markdown("- D < 0.5 : ✅ Sûr\n- 0.5 < D < 0.8 : ⚠️ Prudence\n- D > 0.8 : 🚨 Critique\n- D ≥ 1.0 : 💀 Rupture")
    
    with col_crit2:
        st.markdown("**Critère de Tsai-Wu (Rupture Anisotrope) :**")
        st.latex(r"F = F_3\sigma_{33} + F_{33}\sigma_{33}^2 + F_{44}\sigma_{23}^2 + F_{55}\sigma_{13}^2")
        st.markdown("- F < 1 : Structure intègre\n- F ≥ 1 : Rupture probable")
    
    # =============================================
    # NOUVEAU : CALCULATRICE INTERACTIVE
    # =============================================
    st.markdown("---")
    st.markdown("### 🧮 Calculatrice Interactive")
    st.markdown("""
    <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 1rem;">
        <p style="color: #cbd5e1; margin: 0;">
            💡 Testez les équations thermiques avec vos propres valeurs pour mieux comprendre leur comportement.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    calc_tab1, calc_tab2, calc_tab3 = st.tabs(["🌡️ Flux Thermique", "📏 Épaisseur Requise", "⚡ Résistance Thermique"])
    
    with calc_tab1:
        st.markdown("**Calcul du Flux Thermique (Fourier)**")
        st.latex(r"q = \frac{k \cdot \Delta T}{e}")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            k_calc = st.number_input("Conductivité k (W/m·K)", value=1.5, min_value=0.1, max_value=50.0, step=0.1, key="calc_k")
            delta_t_calc = st.number_input("Delta T (°C)", value=500.0, min_value=1.0, max_value=2000.0, step=10.0, key="calc_dt")
            e_calc = st.number_input("Épaisseur e (µm)", value=300.0, min_value=10.0, max_value=5000.0, step=10.0, key="calc_e")
        
        with col_c2:
            flux_result = k_calc * delta_t_calc / (e_calc * 1e-6)
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.2); padding: 1.5rem; border-radius: 12px; text-align: center; margin-top: 1rem;">
                <span style="color: #94a3b8; font-size: 0.9rem;">📊 Flux Thermique</span>
                <div style="color: #10b981; font-size: 2rem; font-weight: 700;">{flux_result/1e6:.2f} MW/m²</div>
                <span style="color: #64748b; font-size: 0.8rem;">{flux_result:.0f} W/m²</span>
            </div>
            """, unsafe_allow_html=True)
    
    with calc_tab2:
        st.markdown("**Calcul de l'Épaisseur Requise**")
        st.latex(r"e = \frac{k \cdot \Delta T}{q_{max}}")
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            k_req = st.number_input("Conductivité k (W/m·K)", value=1.5, min_value=0.1, max_value=50.0, step=0.1, key="req_k")
            dt_req = st.number_input("Delta T Requis (°C)", value=600.0, min_value=1.0, max_value=2000.0, step=10.0, key="req_dt")
            q_max = st.number_input("Flux Max Admissible (MW/m²)", value=2.5, min_value=0.1, max_value=10.0, step=0.1, key="req_q")
        
        with col_e2:
            e_required = (k_req * dt_req) / (q_max * 1e6) * 1e6  # en µm
            st.markdown(f"""
            <div style="background: rgba(139, 92, 246, 0.2); padding: 1.5rem; border-radius: 12px; text-align: center; margin-top: 1rem;">
                <span style="color: #94a3b8; font-size: 0.9rem;">📏 Épaisseur Minimale</span>
                <div style="color: #8b5cf6; font-size: 2rem; font-weight: 700;">{e_required:.0f} µm</div>
                <span style="color: #64748b; font-size: 0.8rem;">{e_required/1000:.2f} mm</span>
            </div>
            """, unsafe_allow_html=True)
    
    with calc_tab3:
        st.markdown("**Résistance Thermique Multicouche**")
        st.latex(r"R_{total} = \sum_{i=1}^{N} \frac{e_i}{k_i}")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Couche 1 : Substrat (Ni)**")
            e1_r = st.number_input("Épaisseur (µm)", value=500.0, key="r_e1")
            k1_r = st.number_input("Conductivité (W/m·K)", value=20.0, key="r_k1")
            
            st.markdown("**Couche 2 : BondCoat**")
            e2_r = st.number_input("Épaisseur (µm)", value=50.0, key="r_e2")
            k2_r = st.number_input("Conductivité (W/m·K)", value=10.0, key="r_k2")
            
            st.markdown("**Couche 3 : TBC (Céramique)**")
            e3_r = st.number_input("Épaisseur (µm)", value=300.0, key="r_e3")
            k3_r = st.number_input("Conductivité (W/m·K)", value=1.5, key="r_k3")
        
        with col_r2:
            R1 = (e1_r * 1e-6) / k1_r
            R2 = (e2_r * 1e-6) / k2_r
            R3 = (e3_r * 1e-6) / k3_r
            R_total = R1 + R2 + R3
            
            # Pourcentage de contribution
            pct1 = R1/R_total * 100 if R_total > 0 else 0
            pct2 = R2/R_total * 100 if R_total > 0 else 0
            pct3 = R3/R_total * 100 if R_total > 0 else 0
            
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.2); padding: 1.5rem; border-radius: 12px; margin-top: 1rem;">
                <div style="text-align: center; margin-bottom: 1rem;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">🔥 Résistance Thermique Totale</span>
                    <div style="color: #f59e0b; font-size: 2rem; font-weight: 700;">{R_total*1e4:.3f} ×10⁻⁴ m²K/W</div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; text-align: center;">
                    <div style="background: rgba(148, 163, 184, 0.15); padding: 0.5rem; border-radius: 8px;">
                        <div style="color: #94a3b8; font-size: 0.75rem;">Substrat</div>
                        <div style="color: white; font-weight: 600;">{pct1:.1f}%</div>
                    </div>
                    <div style="background: rgba(251, 146, 60, 0.15); padding: 0.5rem; border-radius: 8px;">
                        <div style="color: #fb923c; font-size: 0.75rem;">BondCoat</div>
                        <div style="color: white; font-weight: 600;">{pct2:.1f}%</div>
                    </div>
                    <div style="background: rgba(96, 165, 250, 0.15); padding: 0.5rem; border-radius: 8px;">
                        <div style="color: #60a5fa; font-size: 0.75rem;">TBC</div>
                        <div style="color: white; font-weight: 600;">{pct3:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"💡 La couche TBC représente **{pct3:.0f}%** de la résistance thermique totale, confirmant son rôle d'isolant principal.")
    
    # Références
    st.markdown("---")
    st.markdown("### 📚 Références")
    st.markdown("""
    - **02_Projet_25-26_5A-IDSA_MAT_AVattré.pdf** — Description du projet (Vattré, ONERA)
    - **ProjectEstaca.pdf** — Formulation mathématique complète (8 étapes)
    - **resolution_mécanique_5A.pdf** — Détails de la résolution spectrale
    - Jones, R.M. "Mechanics of Composite Materials" — Méthode CLT
    - Padture et al. "Thermal barrier coatings for gas-turbine engine applications" — TBC Design
    """)

# Si on veut tester le module
if __name__ == "__main__":
    import streamlit as st
    render()
