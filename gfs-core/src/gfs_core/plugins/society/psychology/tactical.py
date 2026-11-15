import numpy as np

from gfs_core.contracts.micro_math import sigmoid


def step_in_match_pressure(state, dt: float) -> None:
    score_home = state.home.score - state.away.score
    for team, swing in ((state.home, score_home), (state.away, -score_home)):
        tactics = dict(
            team.coach.tactical_current or team.coach.tactical_base or {}
        )
        if not tactics:
            continue
        lose_weight = float(sigmoid(-0.85 * swing))
        win_weight = float(sigmoid(0.85 * swing))
        low_block = tactics.get("low_block", 0.5)
        risk = tactics.get("risk_budget", 0.5)
        high_press = tactics.get("high_press", 0.5)
        tactics["low_block"] = float(np.clip(
            low_block + 0.018 * dt * lose_weight * (1.0 - low_block), 0.0, 1.0,
        ))
        tactics["risk_budget"] = float(np.clip(
            risk - 0.012 * dt * lose_weight * risk, 0.0, 1.0,
        ))
        tactics["high_press"] = float(np.clip(
            high_press + 0.018 * dt * win_weight * (1.0 - high_press), 0.0, 1.0,
        ))
        tactics["risk_budget"] = float(np.clip(
            tactics["risk_budget"] + 0.010 * dt * win_weight * (1.0 - risk),
            0.0,
            1.0,
        ))
        team.coach.tactical_current = tactics
