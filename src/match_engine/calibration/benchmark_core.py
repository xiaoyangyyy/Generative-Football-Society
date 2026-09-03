"""Shared micro-match benchmark simulation and row extraction."""

from __future__ import annotations

from typing import Any, Callable

from src.match_engine.calibration.ablation import (
    AblationSpec,
    apply_ablation,
    roster_status_mode,
    tactical_overrides_for,
)
from src.match_engine.match_micro_runner import run_match_micro_simulation
from src.match_engine.micro_config import MicroMatchConfig
from src.simulation.world_cup_runner import build_world_and_tournament

DEFAULT_FIXTURES = [
    ("Mexico", "South Korea"),
    ("Brazil", "Germany"),
    ("France", "England"),
    ("Argentina", "Netherlands"),
    ("Spain", "Morocco"),
    ("Portugal", "Uruguay"),
]


def row_from_summary(s) -> dict[str, Any]:
    passes = s.passes_home + s.passes_away
    completed = (s.pass_completion_home * s.passes_home) + (s.pass_completion_away * s.passes_away)
    shots = s.shots_home + s.shots_away
    micro_xg = s.micro_xg_home + s.micro_xg_away
    goals = s.goals_micro_home + s.goals_micro_away
    action_adoption = getattr(s, "world_model_action_adoption", {}) or {}
    return {
        "pass_completion": completed / max(1.0, passes),
        "pass_completion_home": float(s.pass_completion_home),
        "pass_completion_away": float(s.pass_completion_away),
        "interceptions_per_pass": (s.pass_intercepts_home + s.pass_intercepts_away) / max(1.0, passes),
        "passes_per_team_match": passes / 2.0,
        "passes_home": float(s.passes_home),
        "passes_away": float(s.passes_away),
        "shots_home": float(s.shots_home),
        "shots_away": float(s.shots_away),
        "goals_home": float(s.goals_micro_home),
        "goals_away": float(s.goals_micro_away),
        "micro_xg_home": float(s.micro_xg_home),
        "micro_xg_away": float(s.micro_xg_away),
        "through_share": (s.through_balls_home + s.through_balls_away) / max(1.0, passes),
        "long_pass_share": (s.long_passes_home + s.long_passes_away) / max(1.0, passes),
        "shots_per_team_match": shots / 2.0,
        "shots_on_target_rate": (s.shots_on_target_home + s.shots_on_target_away) / max(1.0, shots),
        "shots_on_target_home": float(s.shots_on_target_home),
        "shots_on_target_away": float(s.shots_on_target_away),
        "goals_per_team_match": goals / 2.0,
        "fouls_committed_per_team_match": (s.fouls_committed_home + s.fouls_committed_away) / 2.0,
        "yellow_cards_per_team_match": (s.yellow_cards_home + s.yellow_cards_away) / 2.0,
        "red_cards_per_team_match": (s.red_cards_home + s.red_cards_away) / 2.0,
        "possession_share": float(s.possession_home),
        "possession_home": float(s.possession_home),
        "possession_away": 1.0 - float(s.possession_home),
        "crosses_per_team_match": s.crosses_attempted / 2.0,
        "headers_per_team_match": float(s.headers_attempted) / 2.0,
        "tackles_per_team_match": (s.tackles_home + s.tackles_away) / 2.0,
        "micro_xg_per_team_match": micro_xg / 2.0,
        "micro_xg_difference_home": float(s.micro_xg_home - s.micro_xg_away),
        "goal_difference_home": float(
            s.goals_micro_home - s.goals_micro_away
        ),
        "goals_to_micro_xg_ratio": (goals / max(0.05, micro_xg)) if micro_xg > 0.01 else 0.0,
        "wm_action_opportunities": float(action_adoption.get("opportunities", 0)),
        "wm_action_influenced_opportunities": float(
            action_adoption.get("influenced_opportunities", 0)
        ),
        "wm_action_adoptions": float(action_adoption.get("adopted", 0)),
        "wm_action_attributable_adoptions": float(
            action_adoption.get("attributable_adoptions", 0)
        ),
        "wm_action_attribution_eligible_opportunities": float(
            action_adoption.get("attribution_eligible_opportunities", 0)
        ),
        "wm_action_counterfactual_changes": float(
            action_adoption.get("counterfactual_action_changes", 0)
        ),
        "wm_action_expected_counterfactual_changes": float(
            action_adoption.get("expected_counterfactual_action_changes", 0.0)
        ),
        "wm_action_counterfactual_change_rate": float(
            action_adoption.get("counterfactual_change_rate", 0.0)
        ),
        "wm_action_expected_counterfactual_change_rate": float(
            action_adoption.get("expected_counterfactual_change_rate", 0.0)
        ),
        "wm_action_adoption_rate": float(action_adoption.get("adoption_rate", 0.0)),
        "wm_action_mean_probability_shift": float(
            action_adoption.get("mean_recommended_probability_shift", 0.0)
        ),
        "wm_pass_target_opportunities": float(
            action_adoption.get("pass_target_opportunities", 0)
        ),
        "wm_pass_target_influenced_opportunities": float(
            action_adoption.get("pass_target_influenced_opportunities", 0)
        ),
        "wm_pass_target_changes": float(
            action_adoption.get("pass_target_changes", 0)
        ),
        "wm_pass_target_expected_changes": float(
            action_adoption.get("pass_target_expected_changes", 0.0)
        ),
        "wm_realized_policy_utility_count": float(
            (action_adoption.get("realized_policy_utility") or {}).get(
                "count", 0
            )
        ),
        "wm_realized_policy_utility_mean": float(
            (action_adoption.get("realized_policy_utility") or {}).get(
                "mean", 0.0
            )
        ),
        "wm_attributable_realized_policy_utility_count": float(
            (action_adoption.get("realized_policy_utility") or {}).get(
                "attributable_count", 0
            )
        ),
        "wm_attributable_realized_policy_utility_mean": float(
            (action_adoption.get("realized_policy_utility") or {}).get(
                "attributable_mean", 0.0
            )
        ),
        "wm_runtime_loaded": bool(
            (getattr(s, "world_model_runtime", {}) or {}).get(
                "loaded", False
            )
        ),
        "wm_checkpoint_signature": (
            (getattr(s, "world_model_runtime", {}) or {}).get(
                "checkpoint_signature"
            )
        ),
        "wm_control_scope": str(
            (getattr(s, "world_model_runtime", {}) or {}).get(
                "control_scope", "none"
            )
        ),
        "wm_outcome_aligned_policy": bool(
            (getattr(s, "world_model_runtime", {}) or {}).get(
                "outcome_aligned_policy", False
            )
        ),
    }


def team_level_rows(rows: list[dict]) -> list[dict[str, float]]:
    """Expand match rows to team-rows for joint correlation checks."""
    out: list[dict[str, float]] = []
    for r in rows:
        out.append(
            {
                "possession_team": float(r.get("possession_home", r.get("possession_share", 0.5))),
                "passes_team": float(r.get("passes_home", r.get("passes_per_team_match", 0.0))),
                "shots_team": float(r.get("shots_home", r.get("shots_per_team_match", 0.0))),
                "goals_team": float(r.get("goals_home", r.get("goals_per_team_match", 0.0))),
            }
        )
        out.append(
            {
                "possession_team": float(r.get("possession_away", 1.0 - float(r.get("possession_share", 0.5)))),
                "passes_team": float(r.get("passes_away", r.get("passes_per_team_match", 0.0))),
                "shots_team": float(r.get("shots_away", r.get("shots_per_team_match", 0.0))),
                "goals_team": float(r.get("goals_away", r.get("goals_per_team_match", 0.0))),
            }
        )
    return out


def run_micro_benchmark_rows(
    *,
    root: str,
    fixtures: list[tuple[str, str]] | None = None,
    samples: int = 3,
    seed_start: int = 42,
    match_seconds: float = 5400.0,
    spec: AblationSpec | None = None,
    cfg: MicroMatchConfig | None = None,
    tactical_override_home: dict[str, float] | None = None,
    tactical_override_away: dict[str, float] | None = None,
    completed_keys: set[tuple[str, int]] | None = None,
    row_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    fixtures = fixtures or DEFAULT_FIXTURES
    engine, _, _ = build_world_and_tournament(
        root,
        require_tactics=False,
        initialization_seed=seed_start,
        load_persistence_state=False,
    )
    tac_h, tac_a = (None, None)
    if spec is not None:
        cfg = apply_ablation(spec, cfg)
        tac_h, tac_a = tactical_overrides_for(spec)
    if tactical_override_home is not None:
        tac_h = tactical_override_home
    if tactical_override_away is not None:
        tac_a = tactical_override_away
    cfg = cfg or MicroMatchConfig()
    cfg.use_micro_goals = True

    from src.match_engine.calibration.narrative_isolation import assert_isolated_env

    if spec is not None:
        assert_isolated_env(spec.name)

    rows: list[dict[str, Any]] = []
    completed = completed_keys or set()
    for home, away in fixtures:
        if home not in engine.agents or away not in engine.agents:
            continue
        for i in range(samples):
            fixture_key = f"{home}_vs_{away}"
            if (fixture_key, i) in completed:
                continue
            from src.simulation.random_control import derive_seed

            seed = derive_seed(seed_start, "calibration", home, away, i)
            eff_h = 68.0
            eff_a = 62.0
            if roster_status_mode():
                eff_h = float(engine.agents[home].status_score)
                eff_a = float(engine.agents[away].status_score)
            summary = run_match_micro_simulation(
                engine.agents[home],
                engine.agents[away],
                goals_home=0,
                goals_away=0,
                xg_home=0.7,
                xg_away=0.8,
                eff_status_home=eff_h,
                eff_status_away=eff_a,
                seed=seed,
                config=cfg,
                writeback_agents=False,
                match_seconds=match_seconds,
                tactical_override_home=tac_h,
                tactical_override_away=tac_a,
            )
            row = row_from_summary(summary)
            row["fixture"] = fixture_key
            row["sample_index"] = i
            rows.append(row)
            if row_callback is not None:
                row_callback(dict(row))
    return rows
