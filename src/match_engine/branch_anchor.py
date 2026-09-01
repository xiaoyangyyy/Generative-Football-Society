"""Deterministic replay identity for a selectable in-match policy fork."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return round(value, 12)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def capture_branch_anchor(
    state: Any,
    rng: np.random.Generator,
    *,
    requested_sec: float,
    actual_sec: float,
    tick: int,
    event_cursor: int,
    passing: Any,
    shots: Any,
    aerial: Any,
    player_tracker: Any,
    continuous_clock: Any,
    subtick_queue: Any,
    manager_runtimes: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the pre-intervention replay state; this is not a process snapshot."""
    payload = _canonical({
        "state": state,
        "rng_state": rng.bit_generator.state,
        "tick": tick,
        "event_cursor": event_cursor,
        "passing_stats": getattr(passing, "stats", {}),
        "shot_stats": getattr(shots, "stats", {}),
        "aerial_stats": getattr(aerial, "stats", {}),
        "player_stats": player_tracker.export_by_team(state),
        "substitutions": getattr(player_tracker, "substitutions", []),
        "continuous_clock": (
            continuous_clock.diagnostics()
            if continuous_clock is not None else {"enabled": False}
        ),
        "subtick_reception_queue": (
            subtick_queue.diagnostics()
            if subtick_queue is not None else {"enabled": False}
        ),
        "manager_runtimes": {
            side: runtime.diagnostics() for side, runtime in manager_runtimes.items()
        },
    })
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "available": True,
        "kind": "deterministic_replay_pre_intervention_anchor",
        "requested_sec": float(requested_sec),
        "actual_sec": float(actual_sec),
        "engine_state_clock_sec": float(state.clock_seconds),
        "tick": int(tick),
        "state_identity": hashlib.sha256(encoded).hexdigest(),
        "identity_algorithm": "sha256_canonical_json",
        "identity_scope": (
            "observable match state, RNG, event cursor and mutable engine diagnostics"
        ),
        "resume_capability": "deterministic_replay_only",
    }
