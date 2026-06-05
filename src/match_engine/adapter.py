"""
Adapter: micro simulation ↔ legacy Poisson match contract.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.match_engine.goal_generator import lambdas_from_micro, simulate_match_score_from_micro
from src.match_engine.match_micro_runner import run_match_micro_simulation
from src.match_engine.micro_config import MicroMatchConfig

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def simulate_match_score_micro(
    home_agent: "SocietyAgent",
    away_agent: "SocietyAgent",
    *,
    eff_status_home: float,
    eff_status_away: float,
    referee: Optional[Dict[str, Any]] = None,
    stage_pressure: float = 0.3,
    drama_score: float = 0.5,
    internal_home: Optional[Dict] = None,
    internal_away: Optional[Dict] = None,
    is_knockout: bool = False,
    config: Optional[MicroMatchConfig] = None,
    seed: int = 42,
    use_micro_goals: bool = True,
) -> Tuple[int, int, float, float, Any]:
    """
    Run full micro stack; return (goals_h, goals_a, xg_h, xg_a, MicroMatchSummary).
    xG values are lambdas from micro integral blend.
    """
    cfg = config or MicroMatchConfig()
    cfg.use_micro_goals = use_micro_goals

    from src.memory_engine.macro_goal_dynamics import simulate_match_score_dynamics

    from src.memory_engine.macro_goal_dynamics import expected_match_xg

    xh, xa, _ = expected_match_xg(
        home_agent,
        away_agent,
        eff_status_home,
        eff_status_away,
        is_knockout=False,
        stage_pressure=stage_pressure,
    )

    summary = run_match_micro_simulation(
        home_agent,
        away_agent,
        goals_home=0,
        goals_away=0,
        xg_home=xh,
        xg_away=xa,
        referee=referee,
        stage_pressure=stage_pressure,
        drama_score=drama_score,
        internal_home=internal_home,
        internal_away=internal_away,
        config=cfg,
        seed=seed,
        writeback_agents=True,
    )

    lam_h, lam_a = lambdas_from_micro(
        summary.micro_xg_home,
        summary.micro_xg_away,
        eff_status_home,
        eff_status_away,
        cfg,
    )

    if use_micro_goals:
        gh = int(summary.goals_micro_home)
        ga = int(summary.goals_micro_away)
        xg_h, xg_a = float(summary.micro_xg_home), float(summary.micro_xg_away)
    else:
        from src.memory_engine.macro_goal_dynamics import simulate_match_score_dynamics

        gh, ga, xg_h, xg_a, _ = simulate_match_score_dynamics(
            home_agent,
            away_agent,
            eff_status_home,
            eff_status_away,
            is_knockout=is_knockout,
            stage_pressure=stage_pressure,
        )

    return int(gh), int(ga), float(xg_h), float(xg_a), summary
