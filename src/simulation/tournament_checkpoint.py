"""Persist / resume full tournament progress (standings, bracket, phase)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.simulation.runtime import environment_snapshot, env_bool
from src.simulation.world_state import validate_world_state


CHECKPOINT_VERSION = 4
RANDOM_WORLD_CONTRACT = "identity_scoped_rng_v1"
DEFAULT_PATH = os.path.join("data", "persistence", "tournament_checkpoint.json")
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
STATE_ARTIFACT_PATHS = (
    "data/persistence/squad_carryover.json",
    "data/persistence/world_model_fusion.jsonl",
    "data/persistence/tactical_counterfactuals.jsonl",
)


def checkpoint_path(base_dir: str) -> str:
    return os.path.join(base_dir, DEFAULT_PATH.replace("/", os.sep))


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(path.rglob("*")):
        if candidate.is_symlink():
            raise ValueError("Tournament state directories cannot contain symlinks")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(_file_sha256(str(candidate)).encode("ascii"))
    return digest.hexdigest()


def _cognitive_cache_artifact(base_dir: str) -> tuple[str, Path] | None:
    values = environment_snapshot()
    if not env_bool(values, "MATCH_COGNITIVE", False):
        return None
    configured = str(values.get("MATCH_COGNITIVE_CACHE", "")).strip()
    root = Path(base_dir).resolve()
    path = Path(configured).resolve() if configured else root / "data/cache/cognitive"
    try:
        key = path.relative_to(root).as_posix()
    except ValueError:
        key = "external/MATCH_COGNITIVE_CACHE"
    return key, path


def capture_state_artifacts(base_dir: str) -> Dict[str, str]:
    root = os.path.abspath(base_dir)
    artifacts: Dict[str, str] = {}
    for relative in STATE_ARTIFACT_PATHS:
        path = os.path.join(root, *relative.split("/"))
        if os.path.isfile(path):
            artifacts[relative] = _file_sha256(path)
    cache = _cognitive_cache_artifact(base_dir)
    if cache is not None:
        key, path = cache
        if path.is_dir():
            artifacts[key] = _directory_sha256(path)
    return artifacts


def verify_state_artifacts(base_dir: str, payload: Dict[str, Any]) -> None:
    expected = payload.get("state_artifacts")
    if not isinstance(expected, dict) or any(
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for path, digest in expected.items()
    ):
        raise ValueError("Invalid tournament checkpoint state artifacts")
    if expected != capture_state_artifacts(base_dir):
        raise ValueError("Tournament checkpoint external state integrity mismatch")


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


def checkpoint_run_identity(payload: Dict[str, Any]) -> str:
    identity = payload.get("run_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64 or any(
        char not in "0123456789abcdef" for char in identity
    ):
        raise ValueError("Invalid tournament checkpoint run identity")
    return identity


def new_reflection_journal() -> Dict[str, Any]:
    return {"schema_version": 1, "receipts": {}, "applied": []}


def validate_reflection_journal(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "receipts", "applied",
    } or value.get("schema_version") != 1:
        raise ValueError("Invalid tournament reflection journal")
    receipts = value["receipts"]
    applied = value["applied"]
    if not isinstance(receipts, dict) or len(receipts) > 256 or any(
        not isinstance(operation_id, str)
        or not operation_id
        or len(operation_id) > 512
        or not isinstance(receipt, dict)
        or set(receipt) != {"agent", "payload"}
        or not isinstance(receipt["agent"], str)
        or not receipt["agent"]
        or len(receipt["agent"]) > 128
        or not isinstance(receipt["payload"], dict)
        for operation_id, receipt in receipts.items()
    ):
        raise ValueError("Invalid tournament reflection receipts")
    try:
        payload_sizes = [
            len(json.dumps(
                receipt["payload"], ensure_ascii=False, allow_nan=False,
            ).encode("utf-8"))
            for receipt in receipts.values()
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid tournament reflection receipt payload") from exc
    if any(size > 256 * 1024 for size in payload_sizes):
        raise ValueError("Tournament reflection receipt exceeds 256 KiB")
    if sum(payload_sizes) > 16 * 1024 * 1024:
        raise ValueError("Tournament reflection journal exceeds 16 MiB")
    if (
        not isinstance(applied, list)
        or not all(isinstance(item, str) and item for item in applied)
        or len(applied) != len(set(applied))
        or any(item not in receipts for item in applied)
    ):
        raise ValueError("Invalid tournament applied reflections")
    return value


def validate_reflection_world_consistency(
    journal: Dict[str, Any], world_state: Dict[str, Any],
) -> None:
    receipts = journal["receipts"]
    applied = set(journal["applied"])
    agents = world_state["agents"]
    audit_owners: Dict[str, str] = {}
    for agent_name, record in agents.items():
        audits = record["state"].get("llm_reflection_audit", [])
        if not isinstance(audits, list):
            raise ValueError("Invalid Agent reflection audit snapshot")
        for audit in audits:
            if not isinstance(audit, dict):
                continue
            operation_id = audit.get("operation_id")
            if operation_id is None:
                continue
            if not isinstance(operation_id, str) or not operation_id:
                raise ValueError("Invalid Agent reflection operation identity")
            if operation_id in audit_owners:
                raise ValueError("Duplicate Agent reflection operation identity")
            audit_owners[operation_id] = agent_name
    for operation_id, receipt in receipts.items():
        agent_name = receipt["agent"]
        if agent_name not in agents:
            raise ValueError("Reflection receipt references an unknown Agent")
        audited = audit_owners.get(operation_id)
        if operation_id in applied:
            if audited != agent_name:
                raise ValueError("Applied reflection is missing from Agent state")
        elif audited is not None:
            raise ValueError("Unapplied reflection already mutated Agent state")
    if any(operation_id not in receipts for operation_id in audit_owners):
        raise ValueError("Agent reflection operation lacks a durable receipt")


def _validate_payload(payload: Dict[str, Any]) -> None:
    required = {
        "phase", "standings", "qualified_teams", "group_schedule_progress",
        "ko_round", "ko_fixture_index", "r32_fixtures",
        "completed_matches", "final_result", "match_index", "match_results",
        "post_group_reflection_done", "random_world", "content_sha256",
        "run_identity_sha256",
        "state_artifacts",
        "world_state",
        "reflection_journal",
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
    checkpoint_run_identity(payload)
    world_state = validate_world_state(payload["world_state"])
    reflection_journal = validate_reflection_journal(payload["reflection_journal"])
    validate_reflection_world_consistency(reflection_journal, world_state)
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
    run_identity_sha256: str,
    world_state: Dict[str, Any],
    reflection_journal: Dict[str, Any],
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
        "run_identity_sha256": run_identity_sha256,
        "state_artifacts": capture_state_artifacts(base_dir),
        "world_state": validate_world_state(world_state),
        "reflection_journal": validate_reflection_journal(reflection_journal),
    }
    validate_reflection_world_consistency(
        payload["reflection_journal"], payload["world_state"],
    )
    checkpoint_run_identity(payload)
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
    if os.path.getsize(path) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Tournament checkpoint exceeds 64 MiB")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid tournament checkpoint document")
    if payload.get("version") == 1:
        raise ValueError(
            "Legacy tournament checkpoint V1 lacks random-world identity; "
            "start a fresh tournament instead of resuming it"
        )
    if payload.get("version") == 2:
        raise ValueError(
            "Tournament checkpoint V2 lacks full run identity; start a fresh "
            "tournament instead of resuming it"
        )
    if payload.get("version") == 3:
        raise ValueError(
            "Tournament checkpoint V3 lacks dynamic world state; start a fresh "
            "tournament instead of resuming it"
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported tournament checkpoint version: {payload.get('version')!r}"
        )
    _validate_payload(payload)
    verify_state_artifacts(base_dir, payload)
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
