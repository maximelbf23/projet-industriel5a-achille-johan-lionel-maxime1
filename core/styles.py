
def load_css():
    """
    Returns the CSS string for the application.
    """
    return """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ============================================ */
    /* GLOBAL DARK THEME - ENHANCED                */
    /* ============================================ */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #030712 0%, #0f172a 40%, #1e1b4b 70%, #0f172a 100%);
        color: #f1f5f9;
    }
    
    /* Masquer la barre d'en-tête Streamlit (header blanc avec bouton Deploy) */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Alternative: rendre le header transparent au lieu de le masquer */
    /* 
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }
    */
    
    .stApp {
        background: 
            radial-gradient(ellipse at 20% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 60%),
            linear-gradient(180deg, #030712 0%, #0f172a 50%, #030712 100%);
        min-height: 100vh;
    }

    /* Subtle animated background particles effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, rgba(96, 165, 250, 0.3), transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(139, 92, 246, 0.2), transparent),
            radial-gradient(1px 1px at 90px 40px, rgba(34, 211, 238, 0.3), transparent),
            radial-gradient(2px 2px at 130px 80px, rgba(96, 165, 250, 0.2), transparent),
            radial-gradient(1px 1px at 160px 30px, rgba(139, 92, 246, 0.3), transparent);
        background-size: 200px 100px;
        animation: starfield 20s linear infinite;
        opacity: 0.4;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes starfield {
        0% { transform: translateY(0); }
        100% { transform: translateY(-100px); }
    }

    /* ============================================ */
    /* ANIMATED GRADIENT HERO - PREMIUM            */
    /* ============================================ */
    .hero-container {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.12) 0%, 
            rgba(6, 182, 212, 0.08) 25%,
            rgba(139, 92, 246, 0.1) 50%,
            rgba(236, 72, 153, 0.06) 75%,
            rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 28px;
        padding: 3rem;
        margin-bottom: 2.5rem;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 0 60px rgba(59, 130, 246, 0.15),
            0 25px 50px -12px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 200%;
        height: 100%;
        background: linear-gradient(
            90deg, 
            transparent 0%, 
            rgba(255,255,255,0.03) 25%,
            rgba(255,255,255,0.08) 50%,
            rgba(255,255,255,0.03) 75%,
            transparent 100%);
        animation: shimmer 4s ease-in-out infinite;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
        animation: float 8s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-50%); opacity: 0.5; }
        50% { opacity: 1; }
        100% { transform: translateX(50%); opacity: 0.5; }
    }

    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        25% { transform: translate(5%, 5%) rotate(2deg); }
        50% { transform: translate(0, 10%) rotate(0deg); }
        75% { transform: translate(-5%, 5%) rotate(-2deg); }
    }
    
    .gradient-title {
        background: linear-gradient(135deg, 
            #60a5fa 0%, 
            #22d3ee 25%, 
            #a78bfa 50%, 
            #f472b6 75%, 
            #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0.75rem;
        animation: gradient-shift 6s ease infinite;
        background-size: 300% 300%;
        text-shadow: 0 0 40px rgba(96, 165, 250, 0.3);
        position: relative;
        z-index: 1;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.3px;
        position: relative;
        z-index: 1;
    }
    
    .hero-stats {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.25rem 1.75rem;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        border: 1px solid rgba(96, 165, 250, 0.15);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stat-item::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.3), rgba(139, 92, 246, 0.3));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .stat-item:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.2);
    }

    .stat-item:hover::before {
        opacity: 1;
    }
    
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    /* ============================================ */
    /* TYPOGRAPHY - REFINED                        */
    /* ============================================ */
    h1 {
        color: #f1f5f9 !important;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }

    h2 {
        color: #e2e8f0 !important;
        font-weight: 700;
        margin-top: 1.5rem;
        position: relative;
    }
    
    h3 {
        color: #cbd5e1 !important;
        font-weight: 600;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6) 1;
        padding-bottom: 12px;
        margin-top: 28px;
    }
    
    h4 {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    
    p, span, label {
        color: #94a3b8;
    }

    /* ============================================ */
    /* GLASSMORPHISM METRIC CARDS - ULTRA PREMIUM  */
    /* ============================================ */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, 
            rgba(30, 41, 59, 0.8) 0%, 
            rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(96, 165, 250, 0.15);
        padding: 24px 28px;
        border-radius: 20px;
        box-shadow: 
            0 0 40px rgba(59, 130, 246, 0.1),
            0 15px 35px -5px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6, #f472b6);
        background-size: 300% 100%;
        opacity: 0;
        transition: opacity 0.4s ease;
        animation: border-gradient 3s linear infinite;
    }

    @keyframes border-gradient {
        0% { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }

    div[data-testid="stMetric"]::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 0%,
            rgba(255, 255, 255, 0.03) 50%,
            transparent 100%
        );
        transform: rotate(45deg);
        transition: all 0.6s ease;
        opacity: 0;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 
            0 0 60px rgba(59, 130, 246, 0.25),
            0 25px 50px -10px rgba(0, 0, 0, 0.5);
        border-color: rgba(96, 165, 250, 0.4);
    }
    
    div[data-testid="stMetric"]:hover::before {
        opacity: 1;
    }

    div[data-testid="stMetric"]:hover::after {
        opacity: 1;
        left: 100%;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #64748b !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.9rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f1f5f9, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    div[data-testid="stMetricDelta"] {
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ============================================ */
    /* PREMIUM WARNING/INFO BOXES - ENHANCED       */
    /* ============================================ */
    .warning-box {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.12) 0%, 
            rgba(6, 182, 212, 0.08) 100%);
        border-left: 4px solid;
        border-image: linear-gradient(180deg, #60a5fa, #22d3ee) 1;
        padding: 24px 28px;
        color: #93c5fd;
        font-weight: 500;
        border-radius: 0 20px 20px 0;
        margin: 24px 0;
        backdrop-filter: blur(16px);
        box-shadow: 
            0 0 40px rgba(59, 130, 246, 0.12),
            0 10px 30px -10px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }

    .warning-box::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        animation: pulse-soft 4s ease-in-out infinite;
    }

    @keyframes pulse-soft {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
    }
    
    .success-box {
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.12) 0%, 
            rgba(5, 150, 105, 0.08) 100%);
        border-left: 4px solid;
        border-image: linear-gradient(180deg, #10b981, #34d399) 1;
        padding: 24px 28px;
        color: #6ee7b7;
        font-weight: 500;
        border-radius: 0 20px 20px 0;
        margin: 24px 0;
        backdrop-filter: blur(16px);
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.1);
    }
    
    .danger-box {
        background: linear-gradient(135deg, 
            rgba(239, 68, 68, 0.12) 0%, 
            rgba(220, 38, 38, 0.08) 100%);
        border-left: 4px solid;
        border-image: linear-gradient(180deg, #ef4444, #f87171) 1;
        padding: 24px 28px;
        color: #fca5a5;
        font-weight: 500;
        border-radius: 0 20px 20px 0;
        margin: 24px 0;
        backdrop-filter: blur(16px);
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.1);
    }

    /* ============================================ */
    /* PREMIUM TABS - REFINED                      */
    /* ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.6);
        padding: 10px;
        border-radius: 20px;
        border: 1px solid rgba(96, 165, 250, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        height: 52px;
        background: transparent;
        border-radius: 14px;
        padding: 12px 22px;
        color: #64748b;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: none !important;
        position: relative;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(139, 92, 246, 0.05));
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: 14px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
    }

    .stTabs [data-baseweb="tab"]:hover::before {
        opacity: 1;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.25) 0%, 
            rgba(139, 92, 246, 0.15) 100%) !important;
        color: #60a5fa !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        box-shadow: 
            0 0 30px rgba(59, 130, 246, 0.2),
            0 8px 20px -5px rgba(59, 130, 246, 0.3);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ============================================ */
    /* SIDEBAR - ULTRA CLEAN MINIMAL DESIGN        */
    /* ============================================ */
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1rem !important;
    }
    
    /* ===== SIDEBAR TITLE ===== */
    section[data-testid="stSidebar"] h1 {
        color: #f8fafc !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ===== SECTION HEADERS ===== */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] .stSubheader {
        color: #60a5fa !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #334155 !important;
    }
    
    /* ===== ALL LABELS ===== */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSlider label p,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    /* ===== SLIDER TRACK - SIMPLE SOLID ===== */
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div {
        background: #334155 !important;
        height: 6px !important;
        border-radius: 3px !important;
    }
    
    /* Slider active/filled portion */
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div:first-child {
        background: #3b82f6 !important;
    }
    
    /* ===== SLIDER THUMB - SIMPLE WHITE ===== */
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #ffffff !important;
        border: 2px solid #3b82f6 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        width: 18px !important;
        height: 18px !important;
    }
    
    /* ===== SLIDER VALUE - HIGH CONTRAST ===== */
    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
        background: #0f172a !important;
        color: #22d3ee !important;
        border: 2px solid #3b82f6 !important;
        padding: 10px 18px !important;
        border-radius: 8px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important;
        min-width: 70px !important;
        text-align: center !important;
        text-shadow: 0 0 8px rgba(34, 211, 238, 0.4) !important;
    }
    
    /* Position slider value above */
    section[data-testid="stSidebar"] .stSlider {
        padding-top: 10px !important;
    }
    
    /* ===== MIN/MAX VALUES ===== */
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #64748b !important;
        font-size: 0.75rem !important;
    }
    
    /* ===== NUMBER INPUTS - MAXIMUM CONTRAST ===== */
    /* Target ALL input variations in sidebar */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stNumberInput > div > div > input,
    section[data-testid="stSidebar"] input[type="number"],
    section[data-testid="stSidebar"] [data-baseweb="input"] input {
        background: #0a0f1a !important;
        border: 2px solid #3b82f6 !important;
        color: #4ade80 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        padding: 14px !important;
        text-align: center !important;
        -webkit-text-fill-color: #4ade80 !important;
    }
    
    section[data-testid="stSidebar"] input:focus,
    section[data-testid="stSidebar"] .stNumberInput input:focus {
        border-color: #60a5fa !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.5), 0 0 20px rgba(74, 222, 128, 0.3) !important;
        color: #4ade80 !important;
        -webkit-text-fill-color: #4ade80 !important;
    }
    
    /* Number +/- buttons - bright blue */
    section[data-testid="stSidebar"] .stNumberInput button,
    section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
    section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"] {
        background: #3b82f6 !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
    }
    
    section[data-testid="stSidebar"] .stNumberInput button:hover {
        background: #60a5fa !important;
        color: #ffffff !important;
    }
    
    /* ===== BUTTONS ===== */
    section[data-testid="stSidebar"] .stButton > button {
        background: #334155 !important;
        border: 1px solid #475569 !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #475569 !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
    }
    
    /* ===== DIVIDERS ===== */
    section[data-testid="stSidebar"] hr {
        border: none !important;
        height: 1px !important;
        background: #334155 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ===== CAPTIONS / HELP TEXT ===== */
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
    }
    
    /* ===== TEXT CONTENT ===== */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #cbd5e1 !important;
    }
    
    /* Hide streamlit elements (keep header visible for sidebar toggle) */
    section[data-testid="stSidebar"] [data-testid="stStatusWidget"],
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* ============================================ */
    /* FORM ELEMENTS - HIGH CONTRAST EVERYWHERE    */
    /* ============================================ */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stNumberInput input,
    input[type="number"],
    [data-baseweb="input"] input {
        background: #0a0f1a !important;
        border: 2px solid #3b82f6 !important;
        color: #4ade80 !important;
        -webkit-text-fill-color: #4ade80 !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 12px !important;
        text-align: center !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stNumberInput input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3), 0 0 15px rgba(74, 222, 128, 0.2) !important;
        color: #4ade80 !important;
        -webkit-text-fill-color: #4ade80 !important;
    }
    
    /* Number input buttons everywhere */
    .stNumberInput button {
        background: #3b82f6 !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    .stNumberInput button:hover {
        background: #60a5fa !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(96, 165, 250, 0.3) !important;
        color: #f1f5f9 !important;
        border-radius: 10px !important;
    }

    /* ============================================ */
    /* BUTTONS - ULTRA PREMIUM                     */
    /* ============================================ */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 0 30px rgba(59, 130, 246, 0.3),
            0 8px 20px -5px rgba(59, 130, 246, 0.4);
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 0 50px rgba(59, 130, 246, 0.4),
            0 15px 30px -5px rgba(59, 130, 246, 0.5);
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
    }

    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 50%, #22d3ee 100%);
        box-shadow: 
            0 0 30px rgba(6, 182, 212, 0.3),
            0 8px 20px -5px rgba(6, 182, 212, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 
            0 0 50px rgba(6, 182, 212, 0.4),
            0 15px 30px -5px rgba(6, 182, 212, 0.5);
        background: linear-gradient(135deg, #22d3ee 0%, #06b6d4 50%, #0891b2 100%);
    }

    /* ============================================ */
    /* DATAFRAMES & TABLES - REFINED               */
    /* ============================================ */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(96, 165, 250, 0.15);
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.08);
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: rgba(15, 23, 42, 0.8);
    }

    /* Table header style */
    .stDataFrame th {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.1)) !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* ============================================ */
    /* EXPANDERS - PREMIUM                         */
    /* ============================================ */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8)) !important;
        border-radius: 14px !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(96, 165, 250, 0.15) !important;
        transition: all 0.3s ease !important;
    }

    .streamlit-expanderHeader:hover {
        border-color: rgba(96, 165, 250, 0.3) !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.1) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(96, 165, 250, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 14px 14px !important;
    }

    /* ============================================ */
    /* DIVIDERS - PREMIUM                          */
    /* ============================================ */
    hr {
        margin: 3rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(96, 165, 250, 0.3) 20%,
            rgba(139, 92, 246, 0.3) 50%,
            rgba(96, 165, 250, 0.3) 80%,
            transparent 100%);
    }

    /* ============================================ */
    /* ALERTS & INFO BOXES - REFINED               */
    /* ============================================ */
    .stAlert {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(96, 165, 250, 0.2) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(16px);
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.08);
    }

    /* ============================================ */
    /* MAIN CONTAINER                              */
    /* ============================================ */
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 5rem;
        max-width: 1500px;
    }
    
    /* ============================================ */
    /* CUSTOM ANIMATIONS - ENHANCED                */
    /* ============================================ */
    @keyframes pulse-glow {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.3),
                        0 0 40px rgba(59, 130, 246, 0.1); 
        }
        50% { 
            box-shadow: 0 0 40px rgba(59, 130, 246, 0.5),
                        0 0 80px rgba(59, 130, 246, 0.2); 
        }
    }
    
    .pulse-animation {
        animation: pulse-glow 2.5s ease-in-out infinite;
    }
    
    @keyframes fade-in-up {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    .fade-in {
        animation: fade-in-up 0.6s ease-out;
    }

    @keyframes glow-pulse {
        0%, 100% { filter: drop-shadow(0 0 5px rgba(96, 165, 250, 0.5)); }
        50% { filter: drop-shadow(0 0 20px rgba(96, 165, 250, 0.8)); }
    }
    
    /* ============================================ */
    /* PROGRESS BARS - PREMIUM                     */
    /* ============================================ */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6) !important;
        background-size: 200% 100%;
        animation: progress-gradient 2s linear infinite;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
    }

    @keyframes progress-gradient {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    /* ============================================ */
    /* RISK INDICATOR GAUGE - ENHANCED             */
    /* ============================================ */
    .risk-gauge {
        width: 100%;
        height: 14px;
        background: rgba(15, 23, 42, 0.8);
        border-radius: 7px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .risk-gauge-fill {
        height: 100%;
        border-radius: 7px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1), background 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .risk-gauge-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255, 255, 255, 0.3) 50%,
            transparent 100%
        );
        animation: gauge-shimmer 2s ease-in-out infinite;
    }

    @keyframes gauge-shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .risk-low { 
        background: linear-gradient(90deg, #10b981, #34d399); 
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    .risk-medium { 
        background: linear-gradient(90deg, #f59e0b, #fbbf24); 
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.4);
    }
    .risk-high { 
        background: linear-gradient(90deg, #ef4444, #f87171); 
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }

    /* ============================================ */
    /* SCROLLBAR STYLING                           */
    /* ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6, #8b5cf6);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #60a5fa, #a78bfa);
    }

    /* ============================================ */
    /* PLOTLY CHART CONTAINERS                     */
    /* ============================================ */
    .js-plotly-plot, .plotly {
        border-radius: 16px;
        overflow: hidden;
    }

    /* ============================================ */
    /* RADIO BUTTONS & CHECKBOXES                  */
    /* ============================================ */
    .stRadio > div {
        background: rgba(15, 23, 42, 0.5);
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid rgba(96, 165, 250, 0.1);
    }

    .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    /* ============================================ */
    /* MULTISELECT                                 */
    /* ============================================ */
    .stMultiSelect [data-baseweb="tag"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        border-radius: 8px !important;
    }

    /* ============================================ */
    /* UNIVERSAL DARK MODE - FORCE ALL BACKGROUNDS */
    /* ============================================ */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="popover"] > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > div {
        background: rgba(15, 23, 42, 0.95) !important;
        color: #f1f5f9 !important;
        border-color: rgba(96, 165, 250, 0.2) !important;
    }

    /* Force dark on ALL table cells */
    .stDataFrame td, .stDataFrame th,
    table td, table th,
    [data-testid="stTable"] td,
    [data-testid="stTable"] th {
        background: rgba(15, 23, 42, 0.9) !important;
        color: #e2e8f0 !important;
        border-color: rgba(96, 165, 250, 0.15) !important;
    }

    /* Table headers with gradient */
    .stDataFrame th, table th {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(139, 92, 246, 0.15)) !important;
        font-weight: 600 !important;
    }

    /* Expander content ALWAYS dark */
    .streamlit-expanderContent,
    [data-testid="stExpander"] > div:last-child {
        background: rgba(15, 23, 42, 0.95) !important;
        border-color: rgba(96, 165, 250, 0.15) !important;
    }

    /* ============================================ */
    /* SLIDER VALUE FIX - NO OVERLAP               */
    /* ============================================ */
    .stSlider [data-testid="stThumbValue"] {
        top: -35px !important;
        position: relative !important;
        z-index: 100 !important;
    }

    .stSlider [data-baseweb="slider"] {
        margin-top: 25px !important;
    }

    /* ============================================ */
    /* ENTRY ANIMATIONS                            */
    /* ============================================ */
    div[data-testid="stMetric"],
    .stDataFrame,
    .js-plotly-plot {
        animation: fadeInUp 0.5s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ============================================ */
    /* DROPDOWN MENUS DARK                         */
    /* ============================================ */
    [data-baseweb="menu"],
    [data-baseweb="popover"],
    ul[role="listbox"] {
        background: rgba(15, 23, 42, 0.98) !important;
        border: 1px solid rgba(96, 165, 250, 0.2) !important;
        backdrop-filter: blur(20px) !important;
    }

    [data-baseweb="menu"] li,
    ul[role="listbox"] li {
        color: #e2e8f0 !important;
    }

    [data-baseweb="menu"] li:hover,
    ul[role="listbox"] li:hover {
        background: rgba(59, 130, 246, 0.2) !important;
    }

    /* ============================================ */
    /* SPECTACULAR DASHBOARD STYLES                */
    /* ============================================ */
    
    /* Animated KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.95) 0%, rgba(15,23,42,0.98) 100%);
        border-radius: 20px;
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(96, 165, 250, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6);
        background-size: 200% 100%;
        animation: shimmer-bar 3s linear infinite;
    }
    
    .kpi-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 0 60px rgba(59, 130, 246, 0.25),
            0 25px 50px -12px rgba(0, 0, 0, 0.5);
        border-color: rgba(96, 165, 250, 0.5);
    }
    
    @keyframes shimmer-bar {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Neon Glow Text */
    .neon-text {
        text-shadow: 
            0 0 10px rgba(96, 165, 250, 0.5),
            0 0 20px rgba(96, 165, 250, 0.3),
            0 0 30px rgba(96, 165, 250, 0.2);
    }
    
    .neon-text-danger {
        text-shadow: 
            0 0 10px rgba(239, 68, 68, 0.5),
            0 0 20px rgba(239, 68, 68, 0.3),
            0 0 30px rgba(239, 68, 68, 0.2);
    }
    
    .neon-text-success {
        text-shadow: 
            0 0 10px rgba(16, 185, 129, 0.5),
            0 0 20px rgba(16, 185, 129, 0.3),
            0 0 30px rgba(16, 185, 129, 0.2);
    }
    
    /* Glassmorphism Ultra */
    .glass-ultra {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    /* Animated Border */
    .animated-border {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
    }
    
    .animated-border::before {
        content: '';
        position: absolute;
        inset: -2px;
        background: linear-gradient(45deg, #3b82f6, #06b6d4, #8b5cf6, #f472b6, #3b82f6);
        background-size: 400% 400%;
        border-radius: 18px;
        z-index: -1;
        animation: border-rotate 4s linear infinite;
    }
    
    @keyframes border-rotate {
        0% { background-position: 0% 50%; }
        100% { background-position: 400% 50%; }
    }
    
    /* Pulse Ring Animation */
    .pulse-ring {
        position: relative;
    }
    
    .pulse-ring::after {
        content: '';
        position: absolute;
        inset: -10px;
        border: 2px solid rgba(59, 130, 246, 0.4);
        border-radius: 50%;
        animation: pulse-ring-anim 2s ease-out infinite;
    }
    
    @keyframes pulse-ring-anim {
        0% { transform: scale(0.8); opacity: 1; }
        100% { transform: scale(1.4); opacity: 0; }
    }
    
    /* Floating Animation */
    .float-animation {
        animation: float-updown 6s ease-in-out infinite;
    }
    
    @keyframes float-updown {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Gradient Text Animation */
    .gradient-text-animated {
        background: linear-gradient(135deg, #60a5fa, #22d3ee, #a78bfa, #f472b6, #60a5fa);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 4s ease infinite;
    }
    
    /* Dashboard Grid */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Sparkline Container */
    .sparkline-container {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 8px;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    
    /* Value Change Indicator */
    .value-up {
        color: #10b981;
    }
    
    .value-up::before {
        content: '▲ ';
        font-size: 0.75em;
    }
    
    .value-down {
        color: #ef4444;
    }
    
    .value-down::before {
        content: '▼ ';
        font-size: 0.75em;
    }
    
    /* Chart Container Premium */
    .chart-container-premium {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(96, 165, 250, 0.15);
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.08);
    }

</style>
"""

