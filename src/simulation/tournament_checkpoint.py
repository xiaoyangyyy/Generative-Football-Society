"""Persist / resume full tournament progress (standings, bracket, phase)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.locking import FileLease
from src.simulation.runtime import environment_snapshot, env_bool
from src.simulation.world_state import validate_world_state


CHECKPOINT_VERSION = 5
RANDOM_WORLD_CONTRACT = "identity_scoped_rng_v1"
DEFAULT_PATH = os.path.join("data", "persistence", "tournament_checkpoint.json")
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
STATE_SNAPSHOT_VERSION = 1
MAX_STATE_SNAPSHOT_BYTES = 32 * 1024 * 1024
MAX_STATE_SNAPSHOT_FILES = 10_000
RECOVERY_PATH = os.path.join(
    "data", "persistence", "tournament_state_recovery.json",
)
LAST_RECOVERY_PATH = os.path.join(
    "data", "persistence", "tournament_state_recovery_last.json",
)
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


def _state_artifact_targets(base_dir: str) -> Dict[str, tuple[Path, str, bool]]:
    root = Path(base_dir).resolve()
    targets = {
        relative: (
            root.joinpath(*relative.split("/")), "file", True,
        )
        for relative in STATE_ARTIFACT_PATHS
    }
    cache = _cognitive_cache_artifact(base_dir)
    if cache is not None:
        key, path = cache
        targets[key] = (path, "directory", key != "external/MATCH_COGNITIVE_CACHE")
    return targets


def capture_state_artifacts(base_dir: str) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    for key, (path, kind, _internal) in _state_artifact_targets(base_dir).items():
        if path.is_symlink():
            raise ValueError("Tournament state artifacts cannot be symlinks")
        if not path.exists():
            continue
        if kind == "file":
            if not path.is_file():
                raise ValueError("Tournament state file has an invalid type")
            artifacts[key] = _file_sha256(str(path))
        else:
            if not path.is_dir():
                raise ValueError("Tournament state directory has an invalid type")
            artifacts[key] = _directory_sha256(path)
    return artifacts


def verify_state_artifacts(base_dir: str, payload: Dict[str, Any]) -> None:
    expected = payload.get("state_artifacts")
    _validate_state_artifact_digests(expected)
    if expected != capture_state_artifacts(base_dir):
        raise ValueError("Tournament checkpoint external state integrity mismatch")


def _validate_state_artifact_digests(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for path, digest in value.items()
    ):
        raise ValueError("Invalid tournament checkpoint state artifacts")
    return value


def _snapshot_file(path: Path, relative: str) -> tuple[dict[str, str], int]:
    if path.stat().st_size > MAX_STATE_SNAPSHOT_BYTES:
        raise ValueError("Tournament state snapshot exceeds safety limits")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return {
        "path": relative,
        "sha256": digest,
        "content_b64": base64.b64encode(content).decode("ascii"),
    }, len(content)


def capture_state_snapshot(
    base_dir: str, artifacts: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Embed a bounded rollback image of every causal external artifact."""
    expected = _validate_state_artifact_digests(
        artifacts if artifacts is not None else capture_state_artifacts(base_dir)
    )
    targets = _state_artifact_targets(base_dir)
    snapshot: Dict[str, Any] = {
        "schema_version": STATE_SNAPSHOT_VERSION,
        "artifacts": {},
    }
    total_bytes = 0
    file_count = 0
    for key in sorted(expected):
        if key not in targets:
            raise ValueError("Tournament state snapshot target is unavailable")
        path, kind, _internal = targets[key]
        files: list[dict[str, str]] = []
        if kind == "file":
            if not path.is_file() or path.is_symlink():
                raise ValueError("Tournament state file changed during snapshot")
            item, size = _snapshot_file(path, "")
            files.append(item)
            total_bytes += size
            file_count += 1
        else:
            if not path.is_dir() or path.is_symlink():
                raise ValueError("Tournament state directory changed during snapshot")
            for candidate in sorted(path.rglob("*")):
                if candidate.is_symlink():
                    raise ValueError(
                        "Tournament state directories cannot contain symlinks"
                    )
                if not candidate.is_file():
                    continue
                relative = candidate.relative_to(path).as_posix()
                item, size = _snapshot_file(candidate, relative)
                files.append(item)
                total_bytes += size
                file_count += 1
        if (
            total_bytes > MAX_STATE_SNAPSHOT_BYTES
            or file_count > MAX_STATE_SNAPSHOT_FILES
        ):
            raise ValueError("Tournament state snapshot exceeds safety limits")
        snapshot["artifacts"][key] = {"kind": kind, "files": files}
    validate_state_snapshot(snapshot, expected)
    return snapshot


def _safe_snapshot_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid tournament state snapshot path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Invalid tournament state snapshot path")
    return path.as_posix()


def _decode_snapshot_content(item: Dict[str, Any]) -> bytes:
    if not isinstance(item, dict) or set(item) != {
        "path", "sha256", "content_b64",
    }:
        raise ValueError("Invalid tournament state snapshot file")
    digest = item["sha256"]
    encoded = item["content_b64"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or not isinstance(encoded, str)
    ):
        raise ValueError("Invalid tournament state snapshot file identity")
    try:
        content = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("Invalid tournament state snapshot encoding") from exc
    if not hmac.compare_digest(hashlib.sha256(content).hexdigest(), digest):
        raise ValueError("Tournament state snapshot file integrity mismatch")
    return content


def validate_state_snapshot(
    value: Any, expected_artifacts: Dict[str, str],
) -> Dict[str, Any]:
    expected = _validate_state_artifact_digests(expected_artifacts)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "artifacts"}
        or value.get("schema_version") != STATE_SNAPSHOT_VERSION
        or not isinstance(value["artifacts"], dict)
        or set(value["artifacts"]) != set(expected)
    ):
        raise ValueError("Invalid tournament state snapshot")
    total_bytes = 0
    file_count = 0
    for key, record in value["artifacts"].items():
        if (
            not isinstance(record, dict)
            or set(record) != {"kind", "files"}
            or record["kind"] not in {"file", "directory"}
            or not isinstance(record["files"], list)
        ):
            raise ValueError("Invalid tournament state snapshot artifact")
        rows = record["files"]
        if record["kind"] == "file" and (
            len(rows) != 1
            or not isinstance(rows[0], dict)
            or rows[0].get("path") != ""
        ):
            raise ValueError("Invalid tournament state file snapshot")
        seen: set[str] = set()
        directory_digest = hashlib.sha256()
        for item in rows:
            relative = item.get("path") if isinstance(item, dict) else None
            if record["kind"] == "directory":
                relative = _safe_snapshot_relative(relative)
            elif relative != "":
                raise ValueError("Invalid tournament state file snapshot path")
            if relative in seen:
                raise ValueError("Duplicate tournament state snapshot path")
            seen.add(relative)
            content = _decode_snapshot_content(item)
            total_bytes += len(content)
            file_count += 1
            if record["kind"] == "directory":
                encoded_path = relative.encode("utf-8")
                directory_digest.update(len(encoded_path).to_bytes(4, "big"))
                directory_digest.update(encoded_path)
                directory_digest.update(item["sha256"].encode("ascii"))
        observed = (
            rows[0]["sha256"]
            if record["kind"] == "file"
            else directory_digest.hexdigest()
        )
        if not hmac.compare_digest(observed, expected[key]):
            raise ValueError("Tournament state snapshot artifact integrity mismatch")
        if (
            total_bytes > MAX_STATE_SNAPSHOT_BYTES
            or file_count > MAX_STATE_SNAPSHOT_FILES
        ):
            raise ValueError("Tournament state snapshot exceeds safety limits")
    return value


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Tournament recovery journal exceeds 64 MiB")
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _internal_restore_target(root: Path, target: Path) -> Path:
    resolved = target.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Automatic recovery cannot modify an external state directory"
        ) from exc
    if not relative.parts:
        raise ValueError("Automatic recovery target cannot be the project root")
    return resolved


def _restore_file_target(path: Path, record: Dict[str, Any] | None) -> None:
    if record is None:
        if path.exists():
            path.unlink()
        return
    content = _decode_snapshot_content(record["files"][0])
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.restore-", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _restore_directory_target(
    root: Path, path: Path, record: Dict[str, Any] | None,
) -> None:
    if record is None:
        if path.exists():
            checked = _internal_restore_target(root, path)
            shutil.rmtree(checked)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{path.name}.restore-", dir=str(path.parent),
    ))
    try:
        for item in record["files"]:
            relative = _safe_snapshot_relative(item["path"])
            destination = staging.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as stream:
                stream.write(_decode_snapshot_content(item))
                stream.flush()
                os.fsync(stream.fileno())
        if path.exists():
            checked = _internal_restore_target(root, path)
            shutil.rmtree(checked)
        os.replace(staging, path)
    finally:
        if staging.exists():
            checked_staging = _internal_restore_target(root, staging)
            shutil.rmtree(checked_staging)


def restore_state_artifacts(base_dir: str, payload: Dict[str, Any]) -> bool:
    """Idempotently roll internal causal files back after identity verification."""
    expected = _validate_state_artifact_digests(payload.get("state_artifacts"))
    desired = validate_state_snapshot(payload.get("state_snapshot"), expected)
    checkpoint_identity = str(payload.get("content_sha256") or "")
    if (
        len(checkpoint_identity) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_identity)
    ):
        raise ValueError("Invalid tournament checkpoint content identity")
    root = Path(base_dir).resolve()
    recovery_path = root.joinpath(*RECOVERY_PATH.split(os.sep))
    last_recovery_path = root.joinpath(*LAST_RECOVERY_PATH.split(os.sep))
    lock_path = recovery_path.with_suffix(".lock")
    with FileLease(lock_path, timeout=5.0):
        if recovery_path.is_symlink() or (
            recovery_path.exists() and not recovery_path.is_file()
        ):
            raise ValueError("Tournament recovery journal must be a regular file")
        if last_recovery_path.is_symlink() or (
            last_recovery_path.exists() and not last_recovery_path.is_file()
        ):
            raise ValueError(
                "Tournament last-recovery record must be a regular file"
            )
        recovery = None
        if recovery_path.is_file():
            if recovery_path.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise ValueError("Tournament recovery journal exceeds 64 MiB")
            try:
                recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid tournament state recovery journal") from exc
            if (
                not isinstance(recovery, dict)
                or set(recovery) != {
                    "schema_version", "checkpoint_content_sha256",
                    "original_state_artifacts", "original_state_snapshot",
                }
                or recovery.get("schema_version") != 1
                or recovery.get("checkpoint_content_sha256")
                != checkpoint_identity
            ):
                raise ValueError("Conflicting tournament state recovery journal")
            validate_state_snapshot(
                recovery["original_state_snapshot"],
                recovery["original_state_artifacts"],
            )

        current = capture_state_artifacts(base_dir)
        targets = _state_artifact_targets(base_dir)
        if any(
            key not in targets
            or desired["artifacts"][key]["kind"] != targets[key][1]
            for key in expected
        ):
            raise ValueError("Tournament state snapshot target type mismatch")
        mismatches = {
            key for key in set(current) | set(expected)
            if current.get(key) != expected.get(key)
        }
        unavailable = mismatches - set(targets)
        external = {
            key for key in mismatches
            if key in targets and not targets[key][2]
        }
        if unavailable:
            raise ValueError("Tournament recovery target is unavailable")
        if external:
            raise ValueError(
                "Automatic recovery cannot modify an external cognitive cache"
            )
        if not mismatches:
            if recovery is not None:
                os.replace(recovery_path, last_recovery_path)
                return True
            return False

        if recovery is None:
            original = capture_state_snapshot(base_dir, current)
            recovery = {
                "schema_version": 1,
                "checkpoint_content_sha256": checkpoint_identity,
                "original_state_artifacts": current,
                "original_state_snapshot": original,
            }
            _atomic_json(recovery_path, recovery)

        # All paths and snapshot bytes were preflighted before the first mutation.
        for key, (target, kind, internal) in targets.items():
            if not internal:
                continue
            target = _internal_restore_target(root, target)
            record = desired["artifacts"].get(key)
            if kind == "file":
                _restore_file_target(target, record)
            else:
                _restore_directory_target(root, target, record)
        if capture_state_artifacts(base_dir) != expected:
            raise ValueError("Tournament external state recovery did not converge")
        os.replace(recovery_path, last_recovery_path)
        return True


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
        "state_snapshot",
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
    validate_state_snapshot(payload["state_snapshot"], payload["state_artifacts"])
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
    state_artifacts = capture_state_artifacts(base_dir)
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
        "state_artifacts": state_artifacts,
        "state_snapshot": capture_state_snapshot(base_dir, state_artifacts),
        "world_state": validate_world_state(world_state),
        "reflection_journal": validate_reflection_journal(reflection_journal),
    }
    validate_reflection_world_consistency(
        payload["reflection_journal"], payload["world_state"],
    )
    checkpoint_run_identity(payload)
    payload["content_sha256"] = _content_sha256(payload)
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Tournament checkpoint exceeds 64 MiB")
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix=".tournament-", suffix=".json.tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def load_checkpoint(
    base_dir: str, *, verify_external_state: bool = True,
) -> Optional[Dict[str, Any]]:
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
    if payload.get("version") == 4:
        raise ValueError(
            "Tournament checkpoint V4 lacks recoverable external state; start "
            "a fresh tournament instead of resuming it"
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported tournament checkpoint version: {payload.get('version')!r}"
        )
    _validate_payload(payload)
    if verify_external_state:
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
