"""
Optional per-action ball-path log for spectator-style match replay.

Enable with MATCH_BALL_LOG=1. Writes human-readable .txt and machine .jsonl under
outputs/ball_log/ after each micro match.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState


def ball_path_log_enabled() -> bool:
    return os.environ.get("MATCH_BALL_LOG", "").strip().lower() in ("1", "true", "yes")


def ball_path_log_txt_enabled() -> bool:
    """Human .txt log is optional; jsonl is the default analysis format."""
    if os.environ.get("MATCH_BALL_LOG_TXT", "").strip().lower() in ("0", "false", "no"):
        return False
    if os.environ.get("MATCH_BALL_LOG_TXT", "").strip().lower() in ("1", "true", "yes"):
        return True
    return False


def ball_log_wm_snapshot_enabled() -> bool:
    if os.environ.get("MATCH_WM_SNAPSHOT_IN_BALL_LOG", "").strip().lower() in ("0", "false", "no"):
        return False
    if os.environ.get("MATCH_WM_SNAPSHOT_IN_BALL_LOG", "").strip().lower() in ("1", "true", "yes"):
        return True
    return False


def ball_path_log_max_entries() -> int:
    raw = os.environ.get("MATCH_BALL_LOG_MAX", "800")
    try:
        return max(50, int(raw))
    except ValueError:
        return 800


def ball_path_log_dir(base_dir: str) -> Path:
    custom = os.environ.get("MATCH_BALL_LOG_DIR", "").strip()
    if custom:
        return Path(custom)
    return Path(base_dir) / "outputs" / "ball_log"


def _clock_label(sec: float) -> str:
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def _xy(pair: np.ndarray) -> str:
    return f"({float(pair[0]):.3f},{float(pair[1]):.3f})"


def _player_label(p: "PlayerAffectiveState") -> str:
    return f"{p.name} [{p.role}]"


def _safe_slug(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text.strip()).strip("_")[:80]


def record_pass(
    state: "MatchAffectiveState",
    *,
    carrier: "PlayerAffectiveState",
    receiver: "PlayerAffectiveState",
    kind: str,
    target: np.ndarray,
    landed: np.ndarray,
    completed: bool,
    success_p: float,
    press: float,
    lane: float,
    intercepted: bool,
    interceptor: Optional["PlayerAffectiveState"] = None,
    omega: float = 0.0,
    outside_foot: float = 0.0,
    ground_weight: float = 0.0,
    time_of_flight: float = 0.0,
    target_miss: float = 0.0,
    lateral_dev: float = 0.0,
    wall_combo: bool = False,
    obs_pre: Optional[np.ndarray] = None,
    obs_post: Optional[np.ndarray] = None,
) -> None:
    if not ball_path_log_enabled():
        return
    entries: List[Dict[str, Any]] = getattr(state, "ball_path_log", None) or []
    if len(entries) >= ball_path_log_max_entries():
        return
    dist = float(np.linalg.norm(landed - carrier.position))
    outcome = "COMPLETE" if completed else ("INTERCEPTED" if intercepted else "INCOMPLETE")
    entry = {
        "type": "pass",
        "t_sec": float(state.clock_seconds),
        "clock": _clock_label(state.clock_seconds),
        "team": carrier.team_id,
        "from_id": carrier.player_id,
        "from": _player_label(carrier),
        "to_id": receiver.player_id,
        "to": _player_label(receiver),
        "kind": kind,
        "from_xy": [float(carrier.position[0]), float(carrier.position[1])],
        "to_xy": [float(target[0]), float(target[1])],
        "land_xy": [float(landed[0]), float(landed[1])],
        "dist_norm": round(dist, 4),
        "press": round(float(press), 3),
        "lane": round(float(lane), 3),
        "p_success": round(float(success_p), 3),
        "outcome": outcome,
        "omega": round(float(omega), 3),
        "outside_foot": round(float(outside_foot), 3),
        "ground": round(float(ground_weight), 3),
        "tof_s": round(float(time_of_flight), 3),
        "target_miss": round(float(target_miss), 4),
        "lateral_dev": round(float(lateral_dev), 4),
        "wall_combo": bool(wall_combo),
    }
    if interceptor is not None:
        entry["interceptor"] = _player_label(interceptor)
        entry["interceptor_id"] = interceptor.player_id
    entry["intercepted"] = bool(intercepted)
    if ball_log_wm_snapshot_enabled():
        if obs_pre is not None:
            entry["obs"] = np.asarray(obs_pre, dtype=np.float32).tolist()
        if obs_post is not None:
            entry["next_obs"] = np.asarray(obs_post, dtype=np.float32).tolist()
    entries.append(entry)
    state.ball_path_log = entries


def record_shot(
    state: "MatchAffectiveState",
    *,
    carrier: "PlayerAffectiveState",
    gk: Optional["PlayerAffectiveState"],
    kind: str,
    xg: float,
    goal: bool,
    on_target: bool,
    saved: bool,
    dist: float,
    traj_peak: float = 0.0,
    traj_tof: float = 0.0,
    in_goal: bool = False,
    obs_pre: Optional[np.ndarray] = None,
    obs_post: Optional[np.ndarray] = None,
) -> None:
    if not ball_path_log_enabled():
        return
    entries: List[Dict[str, Any]] = getattr(state, "ball_path_log", None) or []
    if len(entries) >= ball_path_log_max_entries():
        return
    if goal:
        outcome = "GOAL"
    elif saved:
        outcome = "SAVED"
    elif on_target:
        outcome = "ON_TARGET"
    else:
        outcome = "OFF_TARGET"
    entry = {
        "type": "shot",
        "t_sec": float(state.clock_seconds),
        "clock": _clock_label(state.clock_seconds),
        "team": carrier.team_id,
        "shooter_id": carrier.player_id,
        "shooter": _player_label(carrier),
        "from_xy": [float(carrier.position[0]), float(carrier.position[1])],
        "kind": kind,
        "dist_norm": round(float(dist), 4),
        "xg": round(float(xg), 3),
        "outcome": outcome,
        "peak_m": round(float(traj_peak), 3),
        "tof_s": round(float(traj_tof), 3),
        "in_goal_mouth": bool(in_goal),
    }
    if gk is not None:
        entry["gk"] = _player_label(gk)
        entry["gk_id"] = gk.player_id
    if ball_log_wm_snapshot_enabled():
        if obs_pre is not None:
            entry["obs"] = np.asarray(obs_pre, dtype=np.float32).tolist()
        if obs_post is not None:
            entry["next_obs"] = np.asarray(obs_post, dtype=np.float32).tolist()
    entries.append(entry)
    state.ball_path_log = entries


def format_entry_line(entry: Dict[str, Any]) -> str:
    clk = entry.get("clock", "?")
    if entry.get("type") == "pass":
        arrow = "→" if entry.get("outcome") == "COMPLETE" else "⇢"
        line = (
            f"{clk} PASS {entry.get('kind', '?')} {entry.get('team', '')} "
            f"{entry.get('from', '?')} {arrow} {entry.get('to', '?')} | "
            f"{_xy(np.array(entry['from_xy']))}→{_xy(np.array(entry['land_xy']))} "
            f"d={entry.get('dist_norm', 0):.3f} lane={entry.get('lane', 0):.2f} press={entry.get('press', 0):.2f} "
            f"p={entry.get('p_success', 0):.2f} ω={entry.get('omega', 0):.1f} "
            f"{'ground' if entry.get('ground', 0) > 0.5 else 'aerial'} | {entry.get('outcome', '')}"
        )
        if entry.get("interceptor"):
            line += f" | cut by {entry['interceptor']}"
        if entry.get("wall_combo"):
            line += " | wall-return"
        return line
    if entry.get("type") == "shot":
        gk = entry.get("gk", "")
        gk_part = f" vs {gk}" if gk else ""
        return (
            f"{clk} SHOT {entry.get('kind', '?')} {entry.get('team', '')} "
            f"{entry.get('shooter', '?')}{gk_part} | {_xy(np.array(entry['from_xy']))} "
            f"d={entry.get('dist_norm', 0):.3f} xG={entry.get('xg', 0):.2f} "
            f"h={entry.get('peak_m', 0):.2f}m tof={entry.get('tof_s', 0):.2f}s | **{entry.get('outcome', '')}**"
        )
    return f"{clk} {entry.get('type', 'event')} {entry}"


def persist_ball_path_log(
    state: "MatchAffectiveState",
    *,
    home_team: str,
    away_team: str,
    stage_name: str,
    base_dir: str,
    score_home: int,
    score_away: int,
) -> str:
    """Write .jsonl (always when enabled); optional .txt. Returns primary path (jsonl)."""
    if not ball_path_log_enabled():
        return ""
    entries: List[Dict[str, Any]] = list(getattr(state, "ball_path_log", None) or [])
    if not entries:
        return ""

    out_dir = ball_path_log_dir(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{_safe_slug(home_team)}_vs_{_safe_slug(away_team)}_{_safe_slug(stage_name)}"
    txt_path = out_dir / f"{slug}.txt"
    jsonl_path = out_dir / f"{slug}.jsonl"

    meta = {
        "home": home_team,
        "away": away_team,
        "stage": stage_name,
        "score": f"{score_home}-{score_away}",
        "n_events": len(entries),
        "truncated": len(entries) >= ball_path_log_max_entries(),
    }
    with jsonl_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for e in entries:
            fp.write(json.dumps(e, ensure_ascii=False) + "\n")

    if ball_path_log_txt_enabled():
        lines = [
            f"# {home_team} {score_home}-{score_away} {away_team} | {stage_name}",
            f"# events={len(entries)} | MATCH_BALL_LOG=1",
            "",
        ]
        lines.extend(format_entry_line(e) for e in entries)
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(txt_path)

    return str(jsonl_path)
