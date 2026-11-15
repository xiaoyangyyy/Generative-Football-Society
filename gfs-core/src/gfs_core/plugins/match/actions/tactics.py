def passing_bias(tactics):
    return {
        "short": 0.25 * tactics.get("possession_orientation", 0.5) + 0.20 * tactics.get("build_up_short", 0.5),
        "through": 0.35 * tactics.get("through_ball_bias", 0.5) + 0.20 * tactics.get("verticality", 0.5),
        "long": 0.35 * tactics.get("long_ball_bias", 0.5) + 0.15 * (1.0 - tactics.get("possession_orientation", 0.5)),
        "wall": 0.15 * tactics.get("possession_orientation", 0.5) * tactics.get("build_up_short", 0.5),
        "cross": 0.40 * tactics.get("cross_frequency", 0.5) * tactics.get("wing_focus", 0.5),
        "tempo_scale": 0.85 + 0.30 * tactics.get("tempo", 0.5),
    }
