"""Persist / resume full tournament progress (standings, bracket, phase)."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

CHECKPOINT_VERSION = 1
DEFAULT_PATH = os.path.join("data", "persistence", "tournament_checkpoint.json")


def checkpoint_path(base_dir: str) -> str:
    return os.path.join(base_dir, DEFAULT_PATH.replace("/", os.sep))


def save_checkpoint(
    base_dir: str,
    *,
    standings: Dict[str, Any],
    qualified_teams: List[str],
    phase: str,
    group_schedule_progress: Dict[str, Any],
    ko_round: Optional[str],
    ko_fixture_index: int,
    r32_fixtures: List[Tuple[str, str]],
    completed_matches: List[str],
    final_result: Dict[str, Any],
    match_index: int,
    match_results: Optional[Dict[str, str]] = None,
    post_group_reflection_done: bool = False,
) -> str:
    path = checkpoint_path(base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "phase": phase,
        "standings": standings,
        "qualified_teams": qualified_teams,
        "group_schedule_progress": group_schedule_progress,
        "ko_round": ko_round,
        "ko_fixture_index": ko_fixture_index,
        "r32_fixtures": [{"home": a, "away": b} for a, b in r32_fixtures],
        "completed_matches": completed_matches,
        "final_result": final_result,
        "match_index": match_index,
        "match_results": match_results or {},
        "post_group_reflection_done": post_group_reflection_done,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_checkpoint(base_dir: str) -> Optional[Dict[str, Any]]:
    path = checkpoint_path(base_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_checkpoint(base_dir: str) -> None:
    path = checkpoint_path(base_dir)
    if os.path.isfile(path):
        os.remove(path)


def restore_r32_fixtures(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    out = []
    for row in data.get("r32_fixtures", []):
        out.append((row["home"], row["away"]))
    return out
