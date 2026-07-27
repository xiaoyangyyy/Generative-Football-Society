"""Public application API for Generative Football Society.

This module is the stable integration surface for scripts, notebooks, and
external callers. The larger engine remains split across simulation,
match_engine, memory_engine, and data_engine.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

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
) -> Any:
    """Run one micro match and return a MicroMatchSummary."""
    from src.match_engine.match_micro_runner import run_match_micro_simulation
    from src.match_engine.micro_config import MicroMatchConfig
    from src.memory_engine.poisson_simulator import simulate_match_score

    actual_seed = set_global_seed(seed)
    root = Path(base_dir) if base_dir is not None else project_root()
    world, _, _ = build_simulation(root)
    home_agent = world.agents[home]
    away_agent = world.agents[away]
    goals_home, goals_away, xg_home, xg_away = simulate_match_score(
        home_agent.status_score,
        away_agent.status_score,
        rng=named_rng(seed, "micro_match", home, away, "macro_prior"),
    )
    cfg = config if config is not None else (MicroMatchConfig.fast_demo() if fast else MicroMatchConfig())
    internal_home = home_agent.simulate_internal_game(0.4)
    internal_away = away_agent.simulate_internal_game(0.4)
    return run_match_micro_simulation(
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
    )


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
