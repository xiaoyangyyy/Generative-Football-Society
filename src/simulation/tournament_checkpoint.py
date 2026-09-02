"""Persist / resume full tournament progress (standings, bracket, phase)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

CHECKPOINT_VERSION = 2
RANDOM_WORLD_CONTRACT = "identity_scoped_rng_v1"
DEFAULT_PATH = os.path.join("data", "persistence", "tournament_checkpoint.json")


def checkpoint_path(base_dir: str) -> str:
    return os.path.join(base_dir, DEFAULT_PATH.replace("/", os.sep))


def _content_sha256(payload: Dict[str, Any]) -> str:
    content = {
        key: value for key, value in payload.items()
        if key != "content_sha256"
    }
    encoded = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_root_seed(payload: Dict[str, Any]) -> int:
    random_world = payload.get("random_world")
    if not isinstance(random_world, dict) or set(random_world) != {
        "contract", "root_seed",
    }:
        raise ValueError("Invalid tournament checkpoint random-world identity")
    if random_world.get("contract") != RANDOM_WORLD_CONTRACT:
        raise ValueError("Unsupported tournament checkpoint random-world contract")
    root_seed = random_world.get("root_seed")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise ValueError("Invalid tournament checkpoint root seed")
    return root_seed


def _validate_payload(payload: Dict[str, Any]) -> None:
    required = {
        "phase", "standings", "qualified_teams", "group_schedule_progress",
        "ko_round", "ko_fixture_index", "r32_fixtures",
        "completed_matches", "final_result", "match_index", "match_results",
        "post_group_reflection_done", "random_world", "content_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"Invalid tournament checkpoint; missing: {', '.join(missing)}"
        )
    expected = str(payload.get("content_sha256") or "")
    if len(expected) != 64 or not hmac.compare_digest(
        expected, _content_sha256(payload),
    ):
        raise ValueError("Tournament checkpoint content integrity mismatch")
    checkpoint_root_seed(payload)
    if payload["phase"] not in {"group", "post_group", "knockout", "complete"}:
        raise ValueError("Invalid tournament checkpoint phase")
    for key in ("ko_fixture_index", "match_index"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid tournament checkpoint {key}")
    if not isinstance(payload["standings"], dict):
        raise ValueError("Invalid tournament checkpoint standings")
    if not isinstance(payload["group_schedule_progress"], dict):
        raise ValueError("Invalid tournament checkpoint group progress")
    if not isinstance(payload["final_result"], dict):
        raise ValueError("Invalid tournament checkpoint final result")
    if not isinstance(payload["match_results"], dict):
        raise ValueError("Invalid tournament checkpoint match results")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in payload["match_results"].items()
    ):
        raise ValueError("Invalid tournament checkpoint match result entries")
    if payload["ko_round"] is not None and not isinstance(payload["ko_round"], str):
        raise ValueError("Invalid tournament checkpoint knockout round")
    if not isinstance(payload["post_group_reflection_done"], bool):
        raise ValueError("Invalid tournament checkpoint reflection state")
    qualified = payload["qualified_teams"]
    completed = payload["completed_matches"]
    if (
        not isinstance(qualified, list) or not all(isinstance(v, str) for v in qualified)
        or len(qualified) != len(set(qualified))
    ):
        raise ValueError("Invalid tournament checkpoint qualified teams")
    if (
        not isinstance(completed, list) or not all(isinstance(v, str) for v in completed)
        or len(completed) != len(set(completed))
    ):
        raise ValueError("Invalid tournament checkpoint completed matches")
    fixtures = payload["r32_fixtures"]
    if not isinstance(fixtures, list) or any(
        not isinstance(row, dict)
        or set(row) != {"home", "away"}
        or not all(isinstance(row[key], str) and row[key] for key in ("home", "away"))
        or row["home"] == row["away"]
        for row in fixtures
    ):
        raise ValueError("Invalid tournament checkpoint R32 fixtures")


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
    root_seed: int,
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
        "random_world": {
            "contract": RANDOM_WORLD_CONTRACT,
            "root_seed": int(root_seed),
        },
    }
    payload["content_sha256"] = _content_sha256(payload)
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".tournament-", suffix=".json.tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def load_checkpoint(base_dir: str) -> Optional[Dict[str, Any]]:
    path = checkpoint_path(base_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid tournament checkpoint document")
    if payload.get("version") == 1:
        raise ValueError(
            "Legacy tournament checkpoint V1 lacks random-world identity; "
            "start a fresh tournament instead of resuming it"
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported tournament checkpoint version: {payload.get('version')!r}"
        )
    _validate_payload(payload)
    return payload


def clear_checkpoint(base_dir: str) -> None:
    path = checkpoint_path(base_dir)
    if os.path.isfile(path):
        os.remove(path)


def restore_r32_fixtures(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    out = []
    for row in data.get("r32_fixtures", []):
        out.append((row["home"], row["away"]))
    return out
