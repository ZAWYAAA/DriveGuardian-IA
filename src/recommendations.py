def format_percentage(ratio):
    """
    Transforme un ratio entre 0 et 1 en pourcentage arrondi.
    
    """
    return int(ratio * 100)


def describe_risk_level(risk_level):
    """
    Retourne une phrase courte en fonction du niveau de risque global.
    """
    if risk_level == "faible":
        return "Niveau de risque global : faible. Votre conduite est globalement maîtrisée."
    elif risk_level == "modere":
        return "Niveau de risque global : modéré. Certains comportements augmentent le risque et peuvent être améliorés."
    elif risk_level == "eleve":
        return "Niveau de risque global : élevé. Votre conduite présente des situations de risque fréquentes ou prolongées."
    else:
        return "Niveau de risque global : non déterminé (données insuffisantes)."


def describe_main_risk(main_risk):
    """
    Retourne une description de la principale source de risque.
    """
    if main_risk == "distance":
        return (
            "Votre principale source de risque est la distance de sécurité : "
            "vous roulez trop souvent trop près du véhicule qui vous précède, "
            "ce qui réduit fortement votre marge de réaction en cas de freinage brusque."
        )
    elif main_risk == "lane_position":
        return (
            "Votre principale source de risque est le maintien dans la voie : "
            "vous vous rapprochez fréquemment des lignes de votre voie, voire les dépassez, "
            "ce qui augmente le risque de sortie de voie ou de collision latérale."
        )
    else:
        return (
            "Aucune source de risque dominante n'a été détectée. "
            "Votre conduite semble relativement équilibrée sur ce trajet."
        )


def propose_next_objective(main_risk):
    """
    Propose un petit objectif concret pour le prochain trajet.
    """
    if main_risk == "distance":
        return (
            "Objectif pour le prochain trajet : maintenir une distance de sécurité confortable, "
            "en particulier à vitesse élevée (autoroute, voies rapides). "
            "Laissez un intervalle plus long avec le véhicule devant, surtout lors des phases de suivi prolongé."
        )
    elif main_risk == "lane_position":
        return (
            "Objectif pour le prochain trajet : travailler le maintien dans la voie. "
            "Concentrez-vous sur un positionnement stable au centre de votre voie, "
            "en corrigeant rapidement toute dérive vers les lignes."
        )
    else:
        return (
            "Objectif pour le prochain trajet : conserver ce niveau de vigilance, "
            "et continuer à anticiper les situations potentiellement dangereuses."
        )


def generate_report(risk_summary, context=None):
    """
    Génère un texte de bilan à partir du risk_summary.
    
    context peut être un dict optionnel, par exemple :
    {
        "trajet_nom": "Maison → ENSAM Rabat",
        "type_route": "autoroute",
        "conditions": "jour, temps sec"
    }
    """

    duration_sec = risk_summary.get("duration_sec", 0.0)
    ratio_distance_close = risk_summary.get("ratio_distance_close", 0.0)
    ratio_distance_very_close = risk_summary.get("ratio_distance_very_close", 0.0)
    ratio_lane_near_line = risk_summary.get("ratio_lane_near_line", 0.0)
    ratio_lane_out = risk_summary.get("ratio_lane_out", 0.0)
    ratio_distance_risque = risk_summary.get("ratio_distance_risque", 0.0)
    ratio_position_risque = risk_summary.get("ratio_position_risque", 0.0)
    nb_high_risk_events = risk_summary.get("nb_high_risk_events", 0)
    risk_level = risk_summary.get("risk_level", "inconnu")
    main_risk = risk_summary.get("main_risk", "none")

    # Conversion en minutes et secondes pour affichage
    duration_min = int(duration_sec // 60)
    duration_rem_sec = int(duration_sec % 60)

    # Pourcentages
    pct_distance_close = format_percentage(ratio_distance_close)
    pct_distance_very_close = format_percentage(ratio_distance_very_close)
    pct_lane_near = format_percentage(ratio_lane_near_line)
    pct_lane_out = format_percentage(ratio_lane_out)
    pct_distance_risque = format_percentage(ratio_distance_risque)
    pct_position_risque = format_percentage(ratio_position_risque)

    # Description du contexte (si fourni)
    context_lines = []
    if context is not None:
        trajet_nom = context.get("trajet_nom")
        type_route = context.get("type_route")
        conditions = context.get("conditions")

        context_lines.append("Contexte du trajet :")
        if trajet_nom:
            context_lines.append(f"- Trajet : {trajet_nom}")
        if type_route:
            context_lines.append(f"- Type de route dominant : {type_route}")
        if conditions:
            context_lines.append(f"- Conditions : {conditions}")
        context_lines.append("")  # ligne vide

    # Partie 1 : résumé global
    lines = []

    lines.append("===== Bilan DriveGuardian IA – Synthèse de trajet =====")
    lines.append("")

    # Contexte si disponible
    lines.extend(context_lines)

    lines.append("1) Durée et niveau global de risque")
    lines.append(
        f"- Durée approximative du trajet : {duration_min} min {duration_rem_sec} s"
    )
    lines.append(f"- {describe_risk_level(risk_level)}")
    lines.append("")

    # Partie 2 : Indicateurs détaillés
    lines.append("2) Indicateurs de conduite")
    lines.append(
        f"- Distance de sécurité : {pct_distance_risque}% du temps en situation jugée insuffisante "
        f"({pct_distance_close}% 'trop près' + {pct_distance_very_close}% 'très proche')."
    )
    lines.append(
        f"- Position dans la voie : {pct_position_risque}% du temps en position défavorable "
        f"({pct_lane_near}% proche d'une ligne + {pct_lane_out}% hors de la voie)."
    )
    lines.append(
        f"- Épisodes de risque élevé (distance très dangereuse prolongée) : {nb_high_risk_events}."
    )
    lines.append("")

    # Partie 3 : Principale source de risque
    lines.append("3) Analyse de la principale source de risque")
    lines.append(describe_main_risk(main_risk))
    lines.append("")

    # Partie 4 : Recommandation et objectif pour le prochain trajet
    lines.append("4) Recommandation personnalisée")
    lines.append(propose_next_objective(main_risk))
    lines.append("")

    # On peut éventuellement ajouter une phrase de conclusion
    lines.append(
        "Ce bilan a pour objectif de vous aider à prendre conscience de vos habitudes de conduite "
        "et de vous proposer de petites améliorations progressives, trajet après trajet."
    )

    # On assemble toutes les lignes en un seul texte
    report_text = "\n".join(lines)
    return report_text
