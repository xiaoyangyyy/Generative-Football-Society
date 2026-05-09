import numpy as np


def _soft_goal_cap(goals, lam):
    """Tame extreme Poisson tails while preserving normal variance."""
    lam = float(max(0.05, lam))
    g = int(max(0, goals))
    # ~upper range of plausible single-team output vs lambda
    cap = int(max(5, np.ceil(lam + 3.2 * np.sqrt(lam + 0.15) + 1.0)))
    return min(g, cap)


def simulate_match_score(status_a, status_b, is_knockout=False):
    """
    Simulates a football match score using Poisson distribution based on team status scores.

    Formula:
    - Base lambda (expected goals) = status / 35.0 with relative-strength tilt
    - Knockout: slightly lower effective lambdas (caution / tighter margins)
    """
    shrink = 0.93 if is_knockout else 1.0
    lambda_a = (status_a / 35.0) * (status_a / max(1.0, status_b)) ** 0.5 * shrink
    lambda_b = (status_b / 35.0) * (status_b / max(1.0, status_a)) ** 0.5 * shrink

    if is_knockout:
        # Variance shrink: average two independent draws (same mean, lower tails)
        g1 = int(np.random.poisson(lambda_a))
        g2 = int(np.random.poisson(lambda_a))
        h1 = int(np.random.poisson(lambda_b))
        h2 = int(np.random.poisson(lambda_b))
        goals_a = int(round((g1 + g2) / 2.0))
        goals_b = int(round((h1 + h2) / 2.0))
    else:
        goals_a = int(np.random.poisson(lambda_a))
        goals_b = int(np.random.poisson(lambda_b))

    goals_a = _soft_goal_cap(goals_a, lambda_a)
    goals_b = _soft_goal_cap(goals_b, lambda_b)

    xg_a = round(float(lambda_a), 2)
    xg_b = round(float(lambda_b), 2)

    return int(goals_a), int(goals_b), xg_a, xg_b


def simulate_extra_time_score(status_a, status_b):
    """
    Simulates extra time (30') with reduced scoring intensity.
    Returns additional goals and ET xG estimate.
    """
    lambda_a = (status_a / 35.0) * (status_a / max(1.0, status_b)) ** 0.5 * 0.30
    lambda_b = (status_b / 35.0) * (status_b / max(1.0, status_a)) ** 0.5 * 0.30
    goals_a = _soft_goal_cap(int(np.random.poisson(lambda_a)), lambda_a)
    goals_b = _soft_goal_cap(int(np.random.poisson(lambda_b)), lambda_b)
    return int(goals_a), int(goals_b), round(float(lambda_a), 2), round(float(lambda_b), 2)


def score_xg_anomaly_note(team_a, team_b, ga, gb, xa, xb):
    """
    Post-hoc, human-readable alignment note when goals and xG diverge sharply.
    """
    parts = []
    for name, g, x in ((team_a, ga, xa), (team_b, gb, xb)):
        delta = float(g - x)
        if delta >= 2.0 or (g >= 4 and x <= 1.75):
            parts.append(f"{name} over-performed xG ({g} goals vs {x:.2f} xG; finishing/low-block collapse proxy)")
        elif delta <= -2.0 or (g == 0 and x >= 1.4):
            parts.append(f"{name} under-performed xG ({g} goals vs {x:.2f} xG; woodwork/keeper proxy)")
    tot_g = ga + gb
    tot_x = xa + xb
    if tot_x > 0.5 and abs(tot_g - tot_x) >= 3.2:
        parts.append(
            f"high aggregate swing vs xG ({tot_g} goals vs {tot_x:.2f} combined xG; transition/set-piece noise)"
        )
    if not parts:
        return None
    return "; ".join(parts[:4])


def finalize_stage_xg_context(team_a, team_b, ga, gb, xa, xb, went_to_penalties=False):
    """
    Always-on analytic line for the Final when anomaly thresholds stay silent —
    pushes [XG_CONTEXT] to read like finishing / luck without threshold spikes.
    """
    da, db = float(ga - xa), float(gb - xb)
    under = team_a if da <= db else team_b
    du = min(da, db)
    over = team_a if da >= db else team_b
    ov = max(da, db)
    parts = []
    if du <= -0.22:
        parts.append(f"{under} under-performed xG")
    elif du <= -0.06:
        parts.append(f"{under} narrowly under-shot xG")
    if ov >= 0.22:
        parts.append(f"{over} over-performed finishing vs xG")
    elif ov >= 0.06:
        parts.append(f"{over} leaned on finishing variance vs xG")
    tot_g = int(ga + gb)
    tot_x = float(xa + xb)
    if tot_x > 0.4 and abs(tot_g - tot_x) >= 2.35:
        parts.append(f"combined goals ({tot_g}) swung wide of combined xG ({tot_x:.2f})")
    if went_to_penalties:
        parts.append("survival through penalty variance")
    if not parts:
        parts.append(f"tight ledger vs xG ({ga}-{gb}; ~{xa:.2f}-{xb:.2f})")
    return "; ".join(parts[:5])


def simulate_penalty_shootout():
    """
    Simulates a standard penalty shootout.
    """
    # Simply simulate until one side wins
    pa, pb = 0, 0
    # First 5 rounds
    for _ in range(5):
        if np.random.random() > 0.25: pa += 1 # 75% conversion rate
        if np.random.random() > 0.25: pb += 1
    
    # Sudden death with mild tail damping to avoid frequent extreme lengths.
    sudden_round = 0
    while pa == pb:
        sudden_round += 1
        tail_penalty = min(0.20, 0.02 * max(0, sudden_round - 2))
        conversion = 0.75 - tail_penalty
        if np.random.random() < conversion:
            pa += 1
        if np.random.random() < conversion:
            pb += 1
        # Hard-stop guard for simulation readability.
        if sudden_round >= 8 and pa == pb:
            if np.random.random() < 0.5:
                pa += 1
            else:
                pb += 1
        
    return pa, pb
