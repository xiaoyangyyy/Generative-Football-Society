"""Bridge GFS SocietyAgent + referee dict → MatchAffectiveState."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.squad_factory import build_team_squad
from src.match_engine.tactical_catalog import normalize_formation_key
from src.match_engine.tactical_profile import apply_vector_to_team_coach, build_tactical_vector_for_agent
from src.match_engine.state import (
    AssistantRefereeState,
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
)

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def build_match_affective_state(
    home_agent: "SocietyAgent",
    away_agent: "SocietyAgent",
    referee: Optional[Dict[str, Any]] = None,
    *,
    home_team_id: Optional[str] = None,
    away_team_id: Optional[str] = None,
    stage_pressure: float = 0.3,
    neutral_venue: bool = False,
    eff_status_home: Optional[float] = None,
    eff_status_away: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    base_dir: str | Path | None = None,
) -> MatchAffectiveState:
    rng = rng or np.random.default_rng(42)
    home_id = home_team_id or home_agent.team_name
    away_id = away_team_id or away_agent.team_name

    from src.match_engine.squad_factory import init_away_positions

    home = build_team_squad(
        home_agent, rng, base_dir=base_dir, match_eff_status=eff_status_home,
    )
    away = build_team_squad(
        away_agent, rng, base_dir=base_dir, match_eff_status=eff_status_away,
    )
    home.team_id = home_id
    away.team_id = away_id
    home.attacks_high_x = True
    away.attacks_high_x = False
    init_away_positions(away)
    home.coach.team_id = home_id
    away.coach.team_id = away_id
    home_agent._formation_key = normalize_formation_key(getattr(home_agent, "formation", "4-3-3"))
    away_agent._formation_key = normalize_formation_key(getattr(away_agent, "formation", "4-3-3"))
    apply_vector_to_team_coach(home, build_tactical_vector_for_agent(home_agent))
    apply_vector_to_team_coach(away, build_tactical_vector_for_agent(away_agent))

    ref = referee or {}
    strict = float(ref.get("strictness", 0.55))
    bias = float(ref.get("bias_t1", 0.0))

    referee_state = RefereeAffectiveState(
        profile_name=str(ref.get("profile_name", "balanced")),
        strictness_base=strict,
        bias_home=bias,
        z_calm=float(np.arctanh(0.55)),
        controversy_integral=float(home_agent.referee_grievance + away_agent.referee_grievance) * 0.5,
        assistants=[
            AssistantRefereeState(side="left", offside_strictness=0.52, trust_with_center=0.62),
            AssistantRefereeState(side="right", offside_strictness=0.48, trust_with_center=0.58),
        ],
    )

    crowd = CrowdState(
        psi=0.0 if neutral_venue else 0.14,
        home_team_id=home_id,
        is_neutral_venue=neutral_venue,
    )
    if not neutral_venue:
        for p in home.players:
            p.fan_affinity = float(np.clip(p.fan_affinity + 0.20, 0.0, 1.0))
            p.morale_logit = float(p.morale_logit + 0.15)

    return MatchAffectiveState(
        home=home,
        away=away,
        referee=referee_state,
        crowd=crowd,
        stage_pressure=float(np.clip(stage_pressure, 0.0, 1.0)),
    )


def apply_affective_endstate_to_agents(
    home_agent: "SocietyAgent",
    away_agent: "SocietyAgent",
    state: MatchAffectiveState,
    summary_emotion_home: Dict[str, float],
    summary_emotion_away: Dict[str, float],
    blend: float = 0.35,
) -> None:
    """
    Write back slow-moving team fields (tactical drift + emotion blend) for next fixture.
    blend=0.35 means 35% micro / 65% macro retention per match.
    """
    blend = float(np.clip(blend, 0.0, 1.0))
    for agent, team, emo_mean in (
        (home_agent, state.home, summary_emotion_home),
        (away_agent, state.away, summary_emotion_away),
    ):
        cur = team.coach.tactical_current
        for k, v in cur.items():
            if k in agent.tactical_controls:
                old = float(agent.tactical_controls[k])
                agent.tactical_controls[k] = float(np.clip(old * (1 - blend) + v * blend, 0.0, 1.0))

        ep = dict(agent.emotion_profile)
        for k in ("pride", "anger", "fear", "determination"):
            if k in ep and k in emo_mean:
                ep[k] = float(np.clip(ep[k] * (1 - blend) + emo_mean[k] * blend, 0.0, 1.0))
        if "shame" in ep:
            ep["shame"] = float(np.clip(ep["shame"] * (1 - blend) + emo_mean.get("fear", 0.2) * 0.4 * blend, 0, 1))
        agent.emotion_profile = ep

        agent.referee_grievance = float(
            np.clip(
                agent.referee_grievance * (1 - blend) + state.referee.controversy_integral * 0.02 * blend,
                0.0,
                1.0,
            )
        )
