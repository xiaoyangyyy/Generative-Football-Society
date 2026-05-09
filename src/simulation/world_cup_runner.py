import json
import os

from src.data_engine.loader import load_data
from src.data_engine.cleaner import clean_results
from src.data_engine.identity_normalizer import normalize_identities
from src.memory_engine.status_score import compute_team_status
from src.simulation.engine import WorldEngine
from src.simulation.tournament_2026 import TournamentManager


def build_world_and_tournament(base_dir, require_tactics=False):
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

    engine = WorldEngine(stats, tactical_map=tactical_map)
    tournament = TournamentManager(engine)
    return engine, tournament, tactical_map
