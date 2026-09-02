"""Public application API for Generative Football Society.

This module is the stable integration surface for scripts, notebooks, and
external callers. The larger engine remains split across simulation,
match_engine, memory_engine, and data_engine.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.data_engine.loader import load_data
from src.memory_engine.global_exposure_gate import compute_exposure_gate
from src.memory_engine.sample_confidence import compute_sample_confidence
from src.memory_engine.status_score import compute_team_status
from src.simulation.random_control import named_rng, set_global_seed
from src.simulation.world_cup_runner import build_world_and_tournament
from src.simulation.runtime import SimulationConfig, build_run_manifest, write_manifest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Locate the external GFS workspace used by an installed code package."""
    configured = os.environ.get("GFS_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "data/releases/current.json").is_file()
            or (candidate / "data/raw").is_dir()
        ):
            return candidate
    return PROJECT_ROOT


def load_status_table(base_dir: str | Path | None = None):
    """Load raw data and compute the canonical team status table."""
    root = Path(base_dir) if base_dir is not None else project_root()
    data = load_data(str(root / "data" / "raw"))
    cleaned_df = clean_results(data["results"])
    normalized_df = normalize_identities(cleaned_df, data["former_names"])
    stats, team_matches = compute_team_status(
        normalized_df,
        data["shootouts"],
        current_year=2026,
    )
    stats = compute_exposure_gate(stats, team_matches)
    stats = compute_sample_confidence(stats)
    return stats, team_matches, data


def build_simulation(base_dir: str | Path | None = None, *, require_tactics: bool = False):
    """Build the world engine and tournament manager."""
    root = Path(base_dir) if base_dir is not None else project_root()
    return build_world_and_tournament(str(root), require_tactics=require_tactics)


def run_full_tournament(
    *,
    base_dir: str | Path | None = None,
    resume: bool = False,
    seed: int | None = None,
    require_tactics: bool = False,
):
    """Run the full tournament and return the tournament manager."""
    actual_seed = set_global_seed(seed)
    root = Path(base_dir) if base_dir is not None else project_root()
    config = SimulationConfig(seed=42 if actual_seed is None else actual_seed)
    raw = root / "data" / "raw"
    data_paths = [raw / name for name in ("results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv")]
    model_path = root / "data" / "world_model" / "latent_wm.pt"
    manifest = build_run_manifest(
        config, root, data_paths=data_paths,
        model_paths=[model_path] if model_path.is_file() else [],
    )
    write_manifest(root / "data" / "persistence" / "run_manifest.json", manifest)
    _, tournament, _ = build_simulation(root, require_tactics=require_tactics)
    tournament.run_full_tournament(resume=resume)
    return tournament


def run_counterfactual(
    run,
    baseline,
    treatment,
    *,
    seed: int = 42,
    samples: int = 32,
    intervention: str = "intervention",
):
    """Public paired-seed causal evaluation surface."""
    from src.simulation.counterfactual import evaluate_intervention

    return evaluate_intervention(
        run, baseline, treatment,
        root_seed=seed, samples=samples, intervention=intervention,
    )


def run_micro_match(
    home: str = "Brazil",
    away: str = "Argentina",
    *,
    base_dir: str | Path | None = None,
    seed: int = 42,
    fast: bool = False,
    config: Any | None = None,
    home_tactic: str | None = None,
    away_tactic: str | None = None,
    stage_name: str = "match",
    continuity: bool = False,
    continuity_id: str | None = None,
    home_rotation: str | None = None,
    away_rotation: str | None = None,
    home_lineup: Mapping[str, Any] | None = None,
    away_lineup: Mapping[str, Any] | None = None,
    home_in_match_plan: Mapping[str, Any] | None = None,
    away_in_match_plan: Mapping[str, Any] | None = None,
    home_fatigue_load_factor: float = 1.0,
    away_fatigue_load_factor: float = 1.0,
) -> Any:
    """Run one micro match; optionally settle durable cross-match state."""
    from src.match_engine.match_micro_runner import run_match_micro_simulation
    from src.match_engine.micro_config import MicroMatchConfig
    from src.memory_engine.poisson_simulator import simulate_match_score

    if (home_lineup is not None or away_lineup is not None) and not continuity:
        raise ValueError("frozen lineups require continuity-enabled roster state")
    if (home_in_match_plan is not None or away_in_match_plan is not None) and not continuity:
        raise ValueError("in-match manager plans require continuity-enabled season state")
    fatigue_factors = {
        "home": float(home_fatigue_load_factor),
        "away": float(away_fatigue_load_factor),
    }
    if any(
        not math.isfinite(value) or not 0.75 <= value <= 1.0
        for value in fatigue_factors.values()
    ):
        raise ValueError("fatigue load factors must be finite and between 0.75 and 1")
    if not continuity and any(value != 1.0 for value in fatigue_factors.values()):
        raise ValueError("fatigue load support requires continuity-enabled season state")

    set_global_seed(seed)
    root = Path(base_dir) if base_dir is not None else project_root()
    world, _, _ = build_simulation(root)
    home_agent = world.agents[home]
    away_agent = world.agents[away]
    roster_home = roster_away = None
    continuity_before: dict[str, Any] = {}
    if continuity:
        if continuity_id is not None and not str(continuity_id).strip():
            raise ValueError("continuity_id must be non-empty when provided")
        from src.simulation.cross_match_state import carryover_snapshot
        from src.simulation.match_pipeline import prepare_match_agents

        roster_home, roster_away = prepare_match_agents(
            home_agent, away_agent, str(root),
            home_lineup=home_lineup, away_lineup=away_lineup,
        )
        continuity_before = {
            "home": carryover_snapshot(home_agent),
            "away": carryover_snapshot(away_agent),
        }
    if home_tactic or away_tactic:
        from src.match_engine.tactical_profile import apply_locked_tactical_preset

        if home_tactic:
            apply_locked_tactical_preset(
                home_agent, home_tactic, source="studio_user_intervention",
            )
        if away_tactic:
            apply_locked_tactical_preset(
                away_agent, away_tactic, source="studio_user_intervention",
            )
    rotation_levels = {"strongest": 0.0, "balanced": 0.5, "rotate": 1.0}
    rotation_penalties = {"strongest": 0.0, "balanced": 1.25, "rotate": 3.0}
    for side, agent, rotation in (
        ("home", home_agent, home_rotation), ("away", away_agent, away_rotation),
    ):
        if rotation is not None and rotation not in rotation_levels:
            raise ValueError(f"unsupported {side} rotation")
        if rotation is not None:
            if not hasattr(agent, "tactical_controls") or not isinstance(
                agent.tactical_controls, dict
            ):
                agent.tactical_controls = {}
            agent.tactical_controls["rotation_aggressiveness"] = rotation_levels[rotation]
            if getattr(agent, "_tactical_vector", None) is not None:
                agent._tactical_vector["rotation_aggressiveness"] = rotation_levels[rotation]
    neutral_effects = {"status_delta": 0.0, "fatigue_load_multiplier": 1.0}
    tactical_home = (
        home_agent.tactical_effects()
        if hasattr(home_agent, "tactical_effects") else dict(neutral_effects)
    )
    tactical_away = (
        away_agent.tactical_effects()
        if hasattr(away_agent, "tactical_effects") else dict(neutral_effects)
    )
    settled_fatigue_load_home = (
        tactical_home["fatigue_load_multiplier"] * fatigue_factors["home"]
    )
    settled_fatigue_load_away = (
        tactical_away["fatigue_load_multiplier"] * fatigue_factors["away"]
    )
    base_status_home = float(
        home_agent.get_effective_status()
        if continuity and hasattr(home_agent, "get_effective_status")
        else home_agent.status_score
    )
    base_status_away = float(
        away_agent.get_effective_status()
        if continuity and hasattr(away_agent, "get_effective_status")
        else away_agent.status_score
    )
    effective_status_home = max(
        12.0, base_status_home + tactical_home["status_delta"]
        - rotation_penalties.get(home_rotation, 0.0),
    )
    effective_status_away = max(
        12.0, base_status_away + tactical_away["status_delta"]
        - rotation_penalties.get(away_rotation, 0.0),
    )
    goals_home, goals_away, xg_home, xg_away = simulate_match_score(
        effective_status_home,
        effective_status_away,
        rng=named_rng(seed, "micro_match", home, away, "macro_prior"),
    )
    cfg = config if config is not None else (MicroMatchConfig.fast_demo() if fast else MicroMatchConfig())
    internal_home = home_agent.simulate_internal_game(0.4)
    internal_away = away_agent.simulate_internal_game(0.4)
    summary = run_match_micro_simulation(
        home_agent,
        away_agent,
        goals_home=goals_home,
        goals_away=goals_away,
        xg_home=xg_home,
        xg_away=xg_away,
        referee={"strictness": 0.58, "bias_t1": 0.04},
        internal_home=internal_home,
        internal_away=internal_away,
        config=cfg,
        seed=seed,
        stage_name=stage_name,
        base_dir=root,
        eff_status_home=effective_status_home,
        eff_status_away=effective_status_away,
        in_match_plan_home=(dict(home_in_match_plan) if home_in_match_plan else None),
        in_match_plan_away=(dict(away_in_match_plan) if away_in_match_plan else None),
    )
    from src.simulation.lineup import lineup_summary

    manager_effects = {
        "home": {
            "rotation": home_rotation,
            "base_status": base_status_home,
            "effective_status": effective_status_home,
            "rotation_status_penalty": rotation_penalties.get(home_rotation, 0.0),
            "fatigue_load_multiplier": settled_fatigue_load_home,
            "club_fatigue_load_factor": fatigue_factors["home"],
            "lineup": lineup_summary(roster_home),
        },
        "away": {
            "rotation": away_rotation,
            "base_status": base_status_away,
            "effective_status": effective_status_away,
            "rotation_status_penalty": rotation_penalties.get(away_rotation, 0.0),
            "fatigue_load_multiplier": settled_fatigue_load_away,
            "club_fatigue_load_factor": fatigue_factors["away"],
            "lineup": lineup_summary(roster_away),
        },
        "claim_boundary": "gameplay effects only; no causal or real-world claim",
    }
    if hasattr(summary, "__dict__"):
        summary.manager_effects = manager_effects
    if continuity:
        from src.simulation.cross_match_state import carryover_snapshot
        from src.simulation.match_pipeline import finalize_match_feedback

        score_home = int(getattr(summary, "goals_micro_home", goals_home))
        score_away = int(getattr(summary, "goals_micro_away", goals_away))
        if score_home > score_away:
            result_home, result_away = "win", "loss"
        elif score_home < score_away:
            result_home, result_away = "loss", "win"
        else:
            result_home = result_away = "draw"
        settled_xg_home = float(getattr(summary, "micro_xg_home", xg_home))
        settled_xg_away = float(getattr(summary, "micro_xg_away", xg_away))
        finalize_match_feedback(
            home_agent,
            away_agent,
            base_dir=str(root),
            result_home=result_home,
            result_away=result_away,
            score_diff_home=score_home - score_away,
            xg_home=settled_xg_home,
            xg_away=settled_xg_away,
            prof_score=0.0,
            social_chaos=0.0,
            stage_name=stage_name,
            micro_summary=summary,
            transaction_id=continuity_id or stage_name,
            fatigue_load_home=settled_fatigue_load_home,
            fatigue_load_away=settled_fatigue_load_away,
            rng_home=named_rng(
                seed, "cross_match_settlement",
                continuity_id or stage_name, home,
            ),
            rng_away=named_rng(
                seed, "cross_match_settlement",
                continuity_id or stage_name, away,
            ),
        )
        summary.continuity_state = {
            "enabled": True,
            "stage_name": stage_name,
            "transaction_id": continuity_id or stage_name,
            "rosters_loaded": {"home": roster_home is not None, "away": roster_away is not None},
            "before": continuity_before,
            "after": {
                "home": carryover_snapshot(home_agent),
                "away": carryover_snapshot(away_agent),
            },
        }
    return summary


def run_monte_carlo(
    num_sims: int = 1000,
    *,
    base_dir: str | Path | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run the lightweight Poisson tournament Monte Carlo and return rankings."""
    from src.memory_engine.full_tournament_simulator import GROUPS_2026, simulate_full_tournament

    actual_seed = set_global_seed(seed)
    stats, _, _ = load_status_table(base_dir)
    champion_counts: Counter[str] = Counter()
    root_seed = 42 if actual_seed is None else actual_seed
    for index in range(int(num_sims)):
        champion_counts[simulate_full_tournament(stats, verbose=False, rng=named_rng(root_seed, "monte_carlo", index))] += 1

    teams = sorted({team for group in GROUPS_2026.values() for team in group})
    rows = []
    for team in teams:
        wins = champion_counts.get(team, 0)
        score = float(stats.loc[team]["final_status_score"]) if team in stats.index else 30.0
        rows.append(
            {
                "team": team,
                "wins": wins,
                "probability": wins / max(1, int(num_sims)),
                "status_score": score,
            }
        )
    return sorted(rows, key=lambda row: row["probability"], reverse=True)
