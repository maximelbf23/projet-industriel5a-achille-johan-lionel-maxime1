import datetime

def generate_html_report(inputs, results):
    """
    Génère un rapport HTML stylisé des résultats de simulation.
    """
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extraction des données
    alpha = inputs.get('alpha', 'N/A')
    beta = inputs.get('beta', 'N/A')
    t_top = inputs.get('t_top', 'N/A')
    t_bottom = inputs.get('t_bottom', 'N/A')
    
    t_h1 = results.get('T_at_h1', 0) if results else 0
    q1_h1 = results.get('dQ1_h1', 0) if results else 0
    
    status_color = "#10b981" if t_h1 < 1100 else "#ef4444"
    status_text = "SÛR" if t_h1 < 1100 else "CRITIQUE"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ border-bottom: 2px solid #3b82f6; padding-bottom: 10px; margin-bottom: 20px; }}
            .title {{ color: #1e3a5f; font-size: 24px; font-weight: bold; }}
            .subtitle {{ color: #64748b; font-size: 14px; }}
            .section {{ margin-bottom: 30px; background: #f8fafc; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; }}
            .section-title {{ color: #3b82f6; font-size: 18px; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .item {{ margin-bottom: 10px; }}
            .label {{ font-weight: bold; color: #475569; }}
            .value {{ color: #1e293b; font-family: monospace; font-size: 1.1em; }}
            .status-box {{ text-align: center; padding: 15px; background: {status_color}; color: white; border-radius: 8px; font-weight: bold; font-size: 20px; margin-top: 20px; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">Rapport d'Analyse TBC</div>
            <div class="subtitle">Barrière Thermique Aéronautique — Projet ESTACA/ONERA</div>
            <div class="subtitle">Généré le : {date_str}</div>
        </div>

        <div class="status-box">
            STATUT GLOBAL : {status_text}
        </div>

        <div class="section">
            <div class="section-title">1. Paramètres d'Entrée</div>
            <div class="grid">
                <div class="item"><span class="label">Épaisseur (α) :</span> <span class="value">{alpha}</span></div>
                <div class="item"><span class="label">Anisotropie (β) :</span> <span class="value">{beta}</span></div>
                <div class="item"><span class="label">T° Surface :</span> <span class="value">{t_top} °C</span></div>
                <div class="item"><span class="label">T° Base :</span> <span class="value">{t_bottom} °C</span></div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">2. Résultats Thermomécaniques</div>
            <div class="grid">
                <div class="item"><span class="label">T° Interface (h1) :</span> <span class="value">{t_h1:.2f} °C</span></div>
                <div class="item"><span class="label">Saut de Flux (Q1) :</span> <span class="value">{abs(q1_h1):.4f} W/m²</span></div>
            </div>
            <p style="font-size: 0.9em; color: #64748b; margin-top: 10px;">
                Note : La température à l'interface ne doit pas dépasser 1100°C pour garantir l'intégrité du BondCoat.
            </p>
        </div>

        <div class="section">
            <div class="section-title">3. Recommandations</div>
            <ul>
                <li>{"✅ Configuration valide." if t_h1 < 1100 else "🚨 ATTENTION : Risque de surchauffe. Augmentez l'épaisseur (Alpha) ou réduisez la température de surface."}</li>
                <li>{"✅ Flux homogène." if abs(q1_h1) < 0.5 else "⚠️ Hétérogénéité détectée : Vérifier les contraintes de cisaillement."}</li>
            </ul>
        </div>

        <div class="footer">
            Rapport généré automatiquement par TBC Analysis Dashboard.
        </div>
    </body>
    </html>
    """
    return html
