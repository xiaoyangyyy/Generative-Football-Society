import json
import math
import os

from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.coach_loader import (
    attach_coaches_to_agents,
    default_coaches_path,
    load_coaches_json,
    merge_coach_into_tactical_map,
)
from src.data_engine.identity_normalizer import normalize_identities
from src.memory_engine.status_score import compute_team_status
from src.simulation.engine import WorldEngine
from src.simulation.runtime import environment_snapshot, env_int
from src.simulation.tournament_2026 import TournamentManager


def _attach_team_dynamics_from_rosters(base_dir: str, agents: dict) -> None:
    from src.simulation.squad_registry import load_effective_roster
    from src.memory_engine.macro_goal_dynamics import TEAM_STATE_KEYS

    for team_name, agent in agents.items():
        roster = load_effective_roster(base_dir, team_name)
        dynamics = roster.get("team_dynamics") if roster else None
        if not isinstance(dynamics, dict) or not all(
            key in dynamics
            and isinstance(dynamics[key], (int, float))
            and not isinstance(dynamics[key], bool)
            and math.isfinite(float(dynamics[key]))
            for key in TEAM_STATE_KEYS
        ):
            continue
        agent.team_dynamics = {
            key: float(dynamics[key]) for key in TEAM_STATE_KEYS
        }


def build_world_and_tournament(
    base_dir, require_tactics=False, load_coaches=True,
    initialization_seed=None,
    run_identity_sha256=None,
    load_persistence_state=True,
):
    raw_dir = os.path.join(base_dir, "data", "raw")
    data = load_data(raw_dir)
    cleaned_df = clean_results(data["results"])
    normalized_df = normalize_identities(cleaned_df, data["former_names"])
    stats, _ = compute_team_status(normalized_df, data["shootouts"], current_year=2026)

    tactical_map = {}
    tactics_path = os.path.join(base_dir, "data", "tactics_final_en.json")
    if os.path.exists(tactics_path):
        with open(tactics_path, "r", encoding="utf-8") as f:
            tactical_map = json.load(f)
    elif require_tactics:
        raise FileNotFoundError(f"Required tactics file not found: {tactics_path}")

    coaches_path = default_coaches_path(base_dir)
    coaches = {}
    if load_coaches and os.path.exists(coaches_path):
        coaches = load_coaches_json(coaches_path)
        tactical_map = merge_coach_into_tactical_map(tactical_map, coaches)

    root_seed = (
        int(initialization_seed)
        if initialization_seed is not None
        else env_int(environment_snapshot(), "GFS_SEED", 42)
    )
    engine = WorldEngine(
        stats, tactical_map=tactical_map,
        root_seed=root_seed,
    )
    if coaches:
        n = attach_coaches_to_agents(engine.agents, coaches)
        print(f"Loaded {n} real coach profiles from {coaches_path}")

    _attach_team_dynamics_from_rosters(base_dir, engine.agents)

    from src.simulation.cross_match_state import load_persistence

    if load_persistence_state:
        load_persistence(base_dir, engine.agents)

    tournament = TournamentManager(
        engine, base_dir=base_dir,
        run_identity_sha256=run_identity_sha256,
    )
    return engine, tournament, tactical_map
