"""Action encoding for world-model transitions."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import numpy as np

from src.match_engine.world_model.schema import (
    HORIZON_INDEX,
    HORIZON_SCALE_SECONDS,
    PASS_OUTCOME_INDEX,
    SHOT_GOAL_INDEX,
    SHOT_ON_TARGET_INDEX,
)

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState

ACTION_DIM = 18
_KIND_MAP = {"short": 0, "through": 1, "long": 2, "wall": 3}
# [pass, shot, hold, cross, intercept, tackle]
_TYPE_PASS = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
_TYPE_SHOT = np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)
_TYPE_HOLD = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)
_TYPE_CROSS = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
_TYPE_INTERCEPT = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
_TYPE_TACKLE = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
ACTION_TYPE_NAMES = ("pass", "shot", "hold", "cross", "intercept", "tackle")


def zero_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


def decode_action_kind(action: np.ndarray) -> str:
    """Decode the executed high-level action without reading outcome labels."""
    vector = np.asarray(action, dtype=float).reshape(-1)
    if vector.shape[0] < 6 or not np.isfinite(vector[:6]).all():
        return "other"
    if float(np.max(vector[:6])) <= 0.0:
        return "other"
    return ACTION_TYPE_NAMES[int(np.argmax(vector[:6]))]


def decode_action_kinds(actions: np.ndarray) -> np.ndarray:
    """Vectorized strict action decoding; zero/invalid rows remain ``other``."""
    matrix = np.asarray(actions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 6:
        raise ValueError("actions must be a two-dimensional matrix with six types")
    output = np.full(len(matrix), "other", dtype="U9")
    finite = np.isfinite(matrix[:, :6]).all(axis=1)
    positive = np.max(
        np.where(np.isfinite(matrix[:, :6]), matrix[:, :6], -np.inf),
        axis=1,
    ) > 0.0
    valid = finite & positive
    names = np.asarray(ACTION_TYPE_NAMES, dtype="U9")
    output[valid] = names[np.argmax(matrix[valid, :6], axis=1)]
    return output


def _kind_onehot(kind: str) -> np.ndarray:
    out = np.zeros(4, dtype=np.float32)
    idx = _KIND_MAP.get(kind, 0)
    out[idx] = 1.0
    return out


def _receiver_slot(
    state: "MatchAffectiveState",
    carrier: "PlayerAffectiveState",
    receiver: "PlayerAffectiveState",
) -> float:
    team = state.team(carrier.team_id)
    mates = [p for p in team.players if p.on_pitch and p.player_id != carrier.player_id]
    mates.sort(key=lambda p: p.position[0])
    ids = [p.player_id for p in mates]
    if receiver.player_id not in ids:
        return 0.5
    return float(ids.index(receiver.player_id) / max(1, len(ids) - 1))


def encode_pass_candidate(
    state: "MatchAffectiveState",
    carrier: "PlayerAffectiveState",
    receiver: "PlayerAffectiveState",
    kind: str,
    target: np.ndarray,
    *,
    success_p: float = 0.5,
    horizon_s: float = 10.0,
) -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[0:6] = _TYPE_PASS
    a[6] = float(np.clip(target[0], 0, 1))
    a[7] = float(np.clip(target[1], 0, 1))
    a[8:12] = _kind_onehot(kind)
    a[12] = _receiver_slot(state, carrier, receiver)
    a[13] = float(np.clip(np.nan_to_num(success_p, nan=0.5, posinf=1.0, neginf=0.0), 0, 1))
    a[HORIZON_INDEX] = float(np.clip(horizon_s / HORIZON_SCALE_SECONDS, 0, 1))
    return a


def encode_shot_action(
    position: np.ndarray,
    *,
    shot_kind: str = "power",
    xg: float = 0.1,
    horizon_s: float = 10.0,
) -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[0:6] = _TYPE_SHOT
    a[6] = float(np.clip(position[0], 0, 1))
    a[7] = float(np.clip(position[1], 0, 1))
    kind_idx = {"power": 0, "curved": 1, "knuckle": 2, "header": 3}.get(shot_kind, 0)
    a[8 + kind_idx] = 1.0
    a[13] = float(np.clip(xg, 0, 1))
    a[HORIZON_INDEX] = float(np.clip(horizon_s / HORIZON_SCALE_SECONDS, 0, 1))
    return a


def encode_intercept_action(land_xy: np.ndarray, *, success: float = 1.0) -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[0:6] = _TYPE_INTERCEPT
    a[6] = float(np.clip(land_xy[0], 0, 1))
    a[7] = float(np.clip(land_xy[1], 0, 1))
    a[13] = float(np.clip(success, 0, 1))
    return a


def encode_high_level_action(
    action: str,
    target: np.ndarray | None = None,
    *,
    horizon_s: float = 10.0,
) -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    if action == "shot":
        a[0:6] = _TYPE_SHOT
    elif action == "cross":
        a[0:6] = _TYPE_CROSS
    elif action == "hold":
        a[0:6] = _TYPE_HOLD
    elif action == "intercept":
        a[0:6] = _TYPE_INTERCEPT
    elif action == "tackle":
        a[0:6] = _TYPE_TACKLE
    else:
        a[0:6] = _TYPE_PASS
    if target is not None:
        a[6] = float(np.clip(target[0], 0, 1))
        a[7] = float(np.clip(target[1], 0, 1))
    a[HORIZON_INDEX] = float(np.clip(horizon_s / HORIZON_SCALE_SECONDS, 0, 1))
    return a


def encode_from_ball_log_event(event: Dict[str, Any]) -> np.ndarray:
    """Build action vector from a ball_path_log jsonl event."""
    et = event.get("type", "")
    if et == "shot":
        xy = np.array(event.get("from_xy", [0.5, 0.5]), dtype=float)
        action = encode_shot_action(
            xy,
            shot_kind=str(event.get("kind", "power")),
            xg=float(event.get("xg", 0.1)),
        )
        outcome = str(event.get("outcome", "")).upper()
        action[SHOT_GOAL_INDEX] = 1.0 if outcome == "GOAL" else 0.0
        action[SHOT_ON_TARGET_INDEX] = 1.0 if outcome in {"GOAL", "SAVED", "ON_TARGET"} else 0.0
        return action
    if et == "pass":
        land = event.get("land_xy", event.get("to_xy", [0.5, 0.5]))
        tgt = np.array(land, dtype=float)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        outcome = str(event.get("outcome", ""))
        if outcome == "INTERCEPTED":
            a[0:6] = _TYPE_INTERCEPT
        else:
            a[0:6] = _TYPE_PASS
        a[6] = float(np.clip(tgt[0], 0, 1))
        a[7] = float(np.clip(tgt[1], 0, 1))
        a[8:12] = _kind_onehot(str(event.get("kind", "short")))
        a[13] = float(event.get("p_success", 0.5))
        if outcome == "COMPLETE":
            a[PASS_OUTCOME_INDEX] = 1.0
        elif outcome == "INTERCEPTED":
            a[PASS_OUTCOME_INDEX] = 0.0
        return a
    if et == "cross":
        land = np.array(
            event.get("land_xy", [0.5, 0.5]), dtype=float,
        )
        return encode_high_level_action("cross", target=land)
    if et == "intercept":
        xy = np.array(event.get("land_xy", [0.5, 0.5]), dtype=float)
        return encode_intercept_action(xy)
    return zero_action()
