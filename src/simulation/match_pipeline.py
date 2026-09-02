"""Unified match flow: prep carryover → physics micro (spectate) → score → post feedback."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple, TYPE_CHECKING

import numpy as np
from src.simulation.runtime import environment_snapshot, env_bool

from src.memory_engine.macro_micro_fusion import resolve_unified_score
from src.simulation.cross_match_state import (
    apply_carryover_to_agent,
    apply_carryover_to_roster_for_agent,
    ingest_match_result,
    sync_carryover_from_roster,
)
from src.simulation.squad_registry import load_effective_roster

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def prepare_match_agents(
    home: "SocietyAgent",
    away: "SocietyAgent",
    base_dir: str,
    *,
    home_lineup: Optional[Mapping[str, Any]] = None,
    away_lineup: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Apply cross-match memory to agents + roster snapshots for micro."""
    rh = load_effective_roster(base_dir, home.team_name)
    ra = load_effective_roster(base_dir, away.team_name)
    if rh:
        sync_carryover_from_roster(home, rh)
    apply_carryover_to_agent(home)
    if rh:
        rh = apply_carryover_to_roster_for_agent(home, rh)
        if home_lineup is not None:
            from src.simulation.lineup import LineupSelection, apply_lineup_selection

            rh = apply_lineup_selection(rh, LineupSelection.from_payload(home_lineup))
        home._roster_carryover_snapshot = rh
    if ra:
        sync_carryover_from_roster(away, ra)
    apply_carryover_to_agent(away)
    if ra:
        ra = apply_carryover_to_roster_for_agent(away, ra)
        if away_lineup is not None:
            from src.simulation.lineup import LineupSelection, apply_lineup_selection

            ra = apply_lineup_selection(ra, LineupSelection.from_payload(away_lineup))
        away._roster_carryover_snapshot = ra
    return rh, ra


from src.simulation.score_path import (
    ScorePathMode,
    finalize_official_score_from_micro,
    physics_official_enabled,
    resolve_score_path_mode,
)


def micro_layer_enabled() -> bool:
    return env_bool(environment_snapshot(), "MATCH_MICRO", False)


def micro_physics_score_enabled() -> bool:
    """Official goals from micro physics when MATCH_MICRO=1 and path is physics_official."""
    return resolve_score_path_mode() == ScorePathMode.PHYSICS_OFFICIAL


def build_micro_match_config():
    from src.match_engine.micro_config import MicroMatchConfig

    cfg = MicroMatchConfig()
    cfg.use_micro_goals = micro_physics_score_enabled()
    return cfg


def run_physics_first_micro(
    home: "SocietyAgent",
    away: "SocietyAgent",
    *,
    xg_prior_home: float,
    xg_prior_away: float,
    eff_status_home: float,
    eff_status_away: float,
    referee: Dict[str, Any],
    stage_pressure: float,
    drama_score: float,
    internal_home: Dict,
    internal_away: Dict,
    seed: int,
    stage_name: str = "match",
    neutral_venue: bool = False,
    match_seconds: Optional[float] = None,
) -> Any:
    """
    Run full spatial micro match first (spectate). Score emerges from shot physics when
    use_micro_goals=True; xg_prior_* only shapes foul/shot calendar density, not goals.
    """
    from src.match_engine.match_micro_runner import run_match_micro_simulation

    cfg = build_micro_match_config()
    return run_match_micro_simulation(
        home,
        away,
        goals_home=0,
        goals_away=0,
        xg_home=xg_prior_home,
        xg_away=xg_prior_away,
        referee=referee,
        stage_pressure=stage_pressure,
        drama_score=drama_score,
        internal_home=internal_home,
        internal_away=internal_away,
        config=cfg,
        seed=seed,
        writeback_agents=True,
        eff_status_home=eff_status_home,
        eff_status_away=eff_status_away,
        stage_name=stage_name,
        neutral_venue=neutral_venue,
        match_seconds=match_seconds,
    )


def run_extra_time_micro(
    home: "SocietyAgent",
    away: "SocietyAgent",
    *,
    xg_prior_home: float,
    xg_prior_away: float,
    eff_status_home: float,
    eff_status_away: float,
    referee: Dict[str, Any],
    stage_pressure: float,
    drama_score: float,
    internal_home: Dict,
    internal_away: Dict,
    seed: int,
    stage_name: str = "match",
    neutral_venue: bool = False,
) -> Any:
    """30-minute extra time via full micro simulation (no macro ET shortcut)."""
    from src.memory_engine.macro_goal_dynamics import EXTRA_TIME_MINUTES

    et_sec = float(EXTRA_TIME_MINUTES) * 60.0
    et_xg_h = max(0.12, float(xg_prior_home) * 0.38)
    et_xg_a = max(0.12, float(xg_prior_away) * 0.38)
    return run_physics_first_micro(
        home,
        away,
        xg_prior_home=et_xg_h,
        xg_prior_away=et_xg_a,
        eff_status_home=eff_status_home,
        eff_status_away=eff_status_away,
        referee=referee,
        stage_pressure=min(1.0, stage_pressure + 0.12),
        drama_score=min(1.0, drama_score + 0.15),
        internal_home=internal_home,
        internal_away=internal_away,
        seed=seed + 777,
        stage_name=f"{stage_name}_AET",
        neutral_venue=neutral_venue,
        match_seconds=et_sec,
    )


def print_micro_match_logs(aff: Any, t1_name: str, t2_name: str, a1: "SocietyAgent", a2: "SocietyAgent") -> None:
    print(
        f"  [MICRO] poss={aff.possession_home:.0%} | passes {aff.passes_home}/{aff.passes_away} "
        f"cmp {aff.pass_completion_home:.0%}/{aff.pass_completion_away:.0%} | "
        f"through {aff.through_balls_home}/{aff.through_balls_away} | "
        f"curve {aff.curved_passes_home}/{aff.curved_passes_away} "
        f"outside {aff.outside_foot_passes_home}/{aff.outside_foot_passes_away} "
        f"ground {aff.ground_passes_home}/{aff.ground_passes_away} "
        f"int {aff.pass_intercepts_home}/{aff.pass_intercepts_away} "
        f"wall {aff.wall_combos_home}/{aff.wall_combos_away} | "
        f"fmt {getattr(a1, '_formation_key', '?')}/{getattr(a2, '_formation_key', '?')} "
        f"sty {a1.style_archetype}/{a2.style_archetype} | "
        f"μxG {aff.micro_xg_home:.2f}-{aff.micro_xg_away:.2f} | "
        f"shots {aff.shots_home}/{aff.shots_away} "
        f"(sched {getattr(aff, 'shots_scheduled_home', 0)}/{getattr(aff, 'shots_scheduled_away', 0)}) "
        f"SOT {aff.shots_on_target_home}/{aff.shots_on_target_away} | "
        f"goals {aff.goals_micro_home}-{aff.goals_micro_away} "
        f"(phys {getattr(aff, 'goals_physics_home', aff.goals_micro_home)}-"
        f"{getattr(aff, 'goals_physics_away', aff.goals_micro_away)}) | "
        f"curve/knuckle {aff.curved_shots}/{aff.knuckle_shots} | λ {aff.lambda_home:.2f}-{aff.lambda_away:.2f}"
    )
    if getattr(aff, "cognitive_plans", None):
        print(
            f"  [COGNITIVE] triggers={len(aff.cognitive_triggers)} "
            f"plans={len(aff.cognitive_plans)} tiers={aff.cognitive_tier_usage}"
        )
    print(
        f"  [AFFECTIVE] ψ={aff.final_psi:+.2f} | coach stress {aff.home_coach_stress:.2f}/{aff.away_coach_stress:.2f} "
        f"| ref strict {aff.ref_strictness_mean:.2f} | drift {aff.tactical_drift_home:.2f}/{aff.tactical_drift_away:.2f}"
    )
    eh = aff.home_emotion_mean
    print(
        f"  [AFFECTIVE-EMO] {t1_name} pride={eh.get('pride', 0):.2f} anger={eh.get('anger', 0):.2f} "
        f"fear={eh.get('fear', 0):.2f} det={eh.get('determination', 0):.2f}"
    )
    if aff.timeline_snippet:
        print(f"  [AFFECTIVE-EVT] {' | '.join(aff.timeline_snippet[:8])}")
    if getattr(aff, "xg_supplement_applied", False):
        meta = getattr(aff, "xg_supplement_meta", {}) or {}
        print(f"  [XG-SUPPLEMENT] physics {meta.get('physics')} μxG {meta.get('micro_xg')} → final {meta.get('final')}")
    if getattr(aff, "ball_log_path", ""):
        print(f"  [BALL-LOG] {aff.ball_log_path}")


def run_micro_layer(
    home: "SocietyAgent",
    away: "SocietyAgent",
    *,
    goals_home: int,
    goals_away: int,
    xg_home: float,
    xg_away: float,
    referee: Dict[str, Any],
    stage_pressure: float,
    drama_score: float,
    internal_home: Dict,
    internal_away: Dict,
    seed: int,
    stage_name: str = "match",
) -> Any:
    from src.match_engine.match_micro_runner import run_match_micro_simulation

    cfg = build_micro_match_config()
    return run_match_micro_simulation(
        home,
        away,
        goals_home=goals_home,
        goals_away=goals_away,
        xg_home=xg_home,
        xg_away=xg_away,
        referee=referee,
        stage_pressure=stage_pressure,
        drama_score=drama_score,
        internal_home=internal_home,
        internal_away=internal_away,
        config=cfg,
        seed=seed,
        writeback_agents=True,
        stage_name=stage_name,
    )


def extract_micro_player_stats(summary: Any, team_id: str) -> Dict[str, Dict[str, float]]:
    if summary is None:
        return {}
    ps = getattr(summary, "player_stats", None) or {}
    return dict(ps.get(team_id, {}))


def finalize_match_feedback(
    home: "SocietyAgent",
    away: "SocietyAgent",
    *,
    base_dir: str,
    result_home: str,
    result_away: str,
    score_diff_home: int,
    xg_home: float,
    xg_away: float,
    prof_score: float,
    social_chaos: float,
    stage_name: str,
    micro_summary: Any = None,
    transaction_id: str | None = None,
    fatigue_load_home: float = 1.0,
    fatigue_load_away: float = 1.0,
    rng_home: np.random.Generator | None = None,
    rng_away: np.random.Generator | None = None,
) -> None:
    rh = load_effective_roster(base_dir, home.team_name)
    ra = load_effective_roster(base_dir, away.team_name)
    ingest_match_result(
        home,
        roster=rh,
        result=result_home,
        score_diff=score_diff_home,
        xg_for=xg_home,
        xg_against=xg_away,
        prof_score=prof_score,
        social_chaos=social_chaos,
        stage_name=stage_name,
        micro_player_stats=extract_micro_player_stats(micro_summary, home.team_name),
        transaction_id=transaction_id,
        fatigue_load_multiplier=fatigue_load_home,
        rng=rng_home,
    )
    ingest_match_result(
        away,
        roster=ra,
        result=result_away,
        score_diff=-score_diff_home,
        xg_for=xg_away,
        xg_against=xg_home,
        prof_score=-prof_score,
        social_chaos=social_chaos,
        stage_name=stage_name,
        micro_player_stats=extract_micro_player_stats(micro_summary, away.team_name),
        transaction_id=transaction_id,
        fatigue_load_multiplier=fatigue_load_away,
        rng=rng_away,
    )
    if micro_summary is not None:
        plans = getattr(micro_summary, "cognitive_plans", None) or []
        home.ingest_micro_cognitive_memory(plans, home.team_name)
        away.ingest_micro_cognitive_memory(plans, away.team_name)
        _save_cognitive_match_log(base_dir, home, away, micro_summary, stage_name)

    if env_bool(environment_snapshot(), "SAVE_CARRYOVER", True):
        from src.simulation.cross_match_state import save_persistence

        save_persistence(base_dir, {home.team_name: home, away.team_name: away})


def _save_cognitive_match_log(
    base_dir: str,
    home: "SocietyAgent",
    away: "SocietyAgent",
    micro_summary: Any,
    stage_name: str,
) -> None:
    import json
    from pathlib import Path

    log_dir = Path(base_dir) / "data" / "persistence" / "cognitive_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = stage_name.replace(" ", "_")
    key = f"{home.team_name}_vs_{away.team_name}_{safe_stage}".replace(" ", "_")
    payload = {
        "home": home.team_name,
        "away": away.team_name,
        "stage": stage_name,
        "cognitive_triggers": getattr(micro_summary, "cognitive_triggers", []),
        "cognitive_plans": getattr(micro_summary, "cognitive_plans", []),
        "cognitive_tier_usage": getattr(micro_summary, "cognitive_tier_usage", {}),
        "world_model_online_calibration": getattr(
            micro_summary, "world_model_online_calibration", {},
        ),
        "world_model_decision_adoption": getattr(
            micro_summary, "world_model_decision_adoption", {},
        ),
        "world_model_action_adoption": getattr(
            micro_summary, "world_model_action_adoption", {},
        ),
    }
    path = log_dir / f"{key}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
