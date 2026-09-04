"""Durable all-or-nothing boundary for one tournament match."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.infrastructure.locking import FileLease
from src.simulation.tournament_checkpoint import (
    MAX_CHECKPOINT_BYTES,
    checkpoint_run_identity,
    load_checkpoint,
    require_internal_match_transaction_targets,
    restore_r32_fixtures,
    restore_state_artifacts,
    validate_reflection_journal,
    verify_state_artifacts,
)
from src.simulation.world_state import preflight_world_state, snapshot_world_state


TRANSACTION_LOCK_PATH = (
    "data", "persistence", "tournament_match_transaction.lock",
)


class TournamentMatchRollbackError(RuntimeError):
    """The match failed and its required rollback could not be proven."""

    def __init__(
        self, original_error: BaseException, rollback_error: BaseException,
    ) -> None:
        self.original_error_type = type(original_error).__name__
        self.rollback_error_type = type(rollback_error).__name__
        super().__init__(
            "Tournament match rollback failed "
            f"(original={self.original_error_type}, "
            f"rollback={self.rollback_error_type})"
        )


def _match_identity(stage: str, home: str, away: str) -> str:
    values = (stage, home, away)
    if any(
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 160
        for value in values
    ):
        raise ValueError("Tournament match transaction identity is invalid")
    return f"{stage}:{home}:{away}"


def _project_owned_path(base_dir: str, *parts: str) -> Path:
    root = Path(base_dir).resolve()
    path = root.joinpath(*parts).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Tournament transaction path escapes project root") from exc
    if not relative.parts:
        raise ValueError("Tournament transaction path cannot be project root")
    return path


def _receipt_path(base_dir: str, match_identity: str) -> Path:
    directory = _project_owned_path(
        base_dir, "data", "persistence", "tournament_match_rollbacks",
    )
    if directory.is_symlink():
        raise ValueError("Tournament rollback receipt directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(match_identity.encode("utf-8")).hexdigest()[:24]
    path = directory / f"{digest}.json"
    if path.is_symlink():
        raise ValueError("Tournament rollback receipt cannot be a symlink")
    return path


def _write_rollback_receipt(
    base_dir: str,
    *,
    match_identity: str,
    checkpoint_identity: str,
    original_error: BaseException,
    rollback_error: BaseException | None,
) -> None:
    path = _receipt_path(base_dir, match_identity)
    payload = {
        "schema_version": 1,
        "match_identity": match_identity,
        "pre_match_checkpoint_sha256": checkpoint_identity,
        "rollback_verified": rollback_error is None,
        "original_error_type": type(original_error).__name__,
        "rollback_error_type": (
            None if rollback_error is None else type(rollback_error).__name__
        ),
        "contains_error_message": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Tournament rollback receipt exceeds safety limits")
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


def _verify_match_commit(
    manager: Any,
    *,
    match_key: str,
    match_index_before: int,
    completed_before: list[str],
    results_before: dict[str, str],
) -> None:
    expected_completed = [*completed_before, match_key]
    expected_result_keys = {*results_before, match_key}
    if (
        manager.match_index != match_index_before + 1
        or manager.completed_matches != expected_completed
        or set(manager.match_results) != expected_result_keys
        or any(
            manager.match_results.get(key) != value
            for key, value in results_before.items()
        )
        or not isinstance(manager.match_results.get(match_key), str)
        or not manager.match_results[match_key]
    ):
        raise RuntimeError(
            "Tournament match returned without one in-memory result commit"
        )
    durable = load_checkpoint(manager.base_dir)
    if durable is None:
        raise RuntimeError("Tournament match committed without a checkpoint")
    if (
        int(durable["match_index"]) != manager.match_index
        or durable["completed_matches"] != manager.completed_matches
        or durable["match_results"] != manager.match_results
        or match_key not in durable["completed_matches"]
    ):
        raise RuntimeError(
            "Tournament match memory and checkpoint commit disagree"
        )


def _verify_rollback(manager: Any, checkpoint: dict[str, Any]) -> None:
    progress_matches = (
        manager.standings == checkpoint["standings"]
        and manager.qualified_teams == checkpoint["qualified_teams"]
        and manager.phase == checkpoint["phase"]
        and manager.completed_matches == checkpoint["completed_matches"]
        and manager.match_results == checkpoint["match_results"]
        and manager.ko_round == checkpoint["ko_round"]
        and manager.ko_fixture_index == checkpoint["ko_fixture_index"]
        and manager.r32_fixtures == restore_r32_fixtures(checkpoint)
        and manager.final_result == checkpoint["final_result"]
        and manager.match_index == checkpoint["match_index"]
        and manager.post_group_reflection_done
        == checkpoint["post_group_reflection_done"]
        and manager.reflection_journal == checkpoint["reflection_journal"]
    )
    if not progress_matches or snapshot_world_state(manager) != checkpoint["world_state"]:
        raise RuntimeError("Tournament in-memory rollback did not converge")
    verify_state_artifacts(manager.base_dir, checkpoint)
    durable = load_checkpoint(manager.base_dir)
    if durable != checkpoint:
        raise RuntimeError("Tournament durable checkpoint rollback did not converge")


@contextmanager
def _locked_tournament_match_transaction(
    manager: Any,
    *,
    stage: str,
    home: str,
    away: str,
) -> Iterator[None]:
    """Checkpoint before a match; verify commit or restore every causal surface."""
    identity = _match_identity(stage, home, away)
    match_key = manager._match_key(stage, home, away)
    active = getattr(manager, "_active_match_transaction", None)
    if active is not None:
        raise RuntimeError("Nested tournament match transactions are forbidden")
    if match_key in manager.completed_matches:
        raise ValueError("Tournament match is already committed")
    require_internal_match_transaction_targets(manager.base_dir)
    manager._save_checkpoint()
    checkpoint = load_checkpoint(manager.base_dir)
    if checkpoint is None:
        raise RuntimeError("Tournament pre-match checkpoint was not persisted")
    checkpoint_identity = str(checkpoint["content_sha256"])
    if checkpoint_run_identity(checkpoint) != manager.run_identity_sha256:
        raise RuntimeError("Tournament pre-match checkpoint identity mismatch")
    match_index_before = int(manager.match_index)
    completed_before = list(manager.completed_matches)
    results_before = dict(manager.match_results)
    manager._active_match_transaction = identity
    try:
        yield
        _verify_match_commit(
            manager,
            match_key=match_key,
            match_index_before=match_index_before,
            completed_before=completed_before,
            results_before=results_before,
        )
    except BaseException as original_error:
        rollback_error = None
        try:
            preflight_world_state(manager, checkpoint["world_state"])
            validate_reflection_journal(checkpoint["reflection_journal"])
            restore_state_artifacts(manager.base_dir, checkpoint)
            manager._restore_from_checkpoint(checkpoint)
            manager._save_checkpoint()
            _verify_rollback(manager, checkpoint)
        except BaseException as error:
            rollback_error = error
        try:
            _write_rollback_receipt(
                manager.base_dir,
                match_identity=identity,
                checkpoint_identity=checkpoint_identity,
                original_error=original_error,
                rollback_error=rollback_error,
            )
        except BaseException as receipt_error:
            if rollback_error is None:
                rollback_error = receipt_error
        if rollback_error is not None:
            raise TournamentMatchRollbackError(
                original_error, rollback_error,
            ) from original_error
        raise
    finally:
        if getattr(manager, "_active_match_transaction", None) == identity:
            del manager._active_match_transaction


@contextmanager
def tournament_match_transaction(
    manager: Any,
    *,
    stage: str,
    home: str,
    away: str,
) -> Iterator[None]:
    """Serialize and execute one rollback-complete tournament match."""
    if getattr(manager, "_active_match_transaction", None) is not None:
        raise RuntimeError("Nested tournament match transactions are forbidden")
    lock_path = _project_owned_path(manager.base_dir, *TRANSACTION_LOCK_PATH)
    with FileLease(lock_path, timeout=0.0):
        with _locked_tournament_match_transaction(
            manager, stage=stage, home=home, away=away,
        ):
            yield
