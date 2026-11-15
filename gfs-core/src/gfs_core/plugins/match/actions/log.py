from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


class BallLogService:
    def __init__(
        self,
        *,
        enabled: bool = False,
        max_entries: int = 800,
        output_dir: str | Path | None = None,
        text: bool = False,
    ) -> None:
        self.enabled = enabled
        self.max_entries = max(50, max_entries)
        self.output_dir = Path(output_dir) if output_dir else None
        self.text = text

    def record_shot(self, state, **values) -> None:
        if not self.enabled or len(state.ball_path_log) >= self.max_entries:
            return
        carrier = values["carrier"]
        trajectory = values["trajectory"]
        if values["goal"]:
            outcome = "GOAL"
        elif values["saved"]:
            outcome = "SAVED"
        elif values["on_target"]:
            outcome = "ON_TARGET"
        else:
            outcome = "OFF_TARGET"
        entry = {
            "type": "shot",
            "t_sec": float(state.clock_seconds),
            "clock": _clock(state.clock_seconds),
            "team": carrier.team_id,
            "shooter_id": carrier.player_id,
            "shooter": _player(carrier),
            "from_xy": _position(carrier.position),
            "kind": values["kind"],
            "dist_norm": round(float(values["dist"]), 4),
            "xg": round(float(values["xg"]), 3),
            "outcome": outcome,
            "peak_m": round(float(trajectory.peak_height), 3),
            "tof_s": round(float(trajectory.time_of_flight), 3),
            "in_goal_mouth": bool(trajectory.in_goal_mouth),
        }
        goalkeeper = values.get("goalkeeper")
        if goalkeeper is not None:
            entry["gk"] = _player(goalkeeper)
            entry["gk_id"] = goalkeeper.player_id
        state.ball_path_log.append(entry)

    def record_cross(self, state, **values) -> None:
        if not self.enabled or len(state.ball_path_log) >= self.max_entries:
            return
        carrier = values["carrier"]
        trajectory = values["trajectory"]
        contact = str(values.get("contact") or "loose")
        entry = {
            "type": "cross",
            "t_sec": float(state.clock_seconds),
            "clock": _clock(state.clock_seconds),
            "team": carrier.team_id,
            "crosser_id": carrier.player_id,
            "crosser": _player(carrier),
            "from_xy": _position(carrier.position),
            "land_xy": _position(values["landed"]),
            "kind": "aerial",
            "contact": contact[:40],
            "xg_added": round(float(values["xg_added"]), 4),
            "goal": bool(values["goal"]),
            "outcome": "GOAL" if values["goal"] else contact.upper(),
            "peak_m": round(float(trajectory.peak_height), 3),
            "tof_s": round(float(trajectory.time_of_flight), 3),
        }
        winner = values.get("winner")
        if winner is not None:
            entry["winner_id"] = winner.player_id
            entry["winner"] = _player(winner)
        state.ball_path_log.append(entry)

    def record_pass(self, state, **values) -> None:
        if not self.enabled or len(state.ball_path_log) >= self.max_entries:
            return
        carrier = values["carrier"]
        receiver = values["receiver"]
        landed = values["landed"]
        outcome = (
            "COMPLETE" if values["completed"]
            else "INTERCEPTED" if values["intercepted"] else "INCOMPLETE"
        )
        entry = {
            "type": "pass",
            "t_sec": float(state.clock_seconds),
            "clock": _clock(state.clock_seconds),
            "team": carrier.team_id,
            "from_id": carrier.player_id,
            "from": _player(carrier),
            "to_id": receiver.player_id,
            "to": _player(receiver),
            "kind": values["kind"],
            "from_xy": _position(carrier.position),
            "to_xy": _position(values["target"]),
            "land_xy": _position(landed),
            "dist_norm": round(float(np.linalg.norm(landed - carrier.position)), 4),
            "press": round(float(values["press"]), 3),
            "lane": round(float(values["lane"]), 3),
            "p_success": round(float(values["success_p"]), 3),
            "outcome": outcome,
            "omega": round(float(values.get("omega", 0.0)), 3),
            "outside_foot": round(float(values.get("outside_foot", 0.0)), 3),
            "ground": round(float(values.get("ground_weight", 0.0)), 3),
            "tof_s": round(float(values.get("time_of_flight", 0.0)), 3),
            "target_miss": round(float(values.get("target_miss", 0.0)), 4),
            "lateral_dev": round(float(values.get("lateral_dev", 0.0)), 4),
            "wall_combo": bool(values.get("wall_combo", False)),
            "intercepted": bool(values["intercepted"]),
        }
        interceptor = values.get("interceptor")
        if interceptor is not None:
            entry["interceptor"] = _player(interceptor)
            entry["interceptor_id"] = interceptor.player_id
        state.ball_path_log.append(entry)

    def persist(self, state, *, home: str, away: str, stage: str) -> str:
        if not self.enabled or not state.ball_path_log or self.output_dir is None:
            return ""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        slug = f"{_slug(home)}_vs_{_slug(away)}_{_slug(stage)}"
        path = self.output_dir / f"{slug}.jsonl"
        meta = {
            "home": home,
            "away": away,
            "stage": stage,
            "score": f"{home_score(state)}-{away_score(state)}",
            "n_events": len(state.ball_path_log),
            "truncated": len(state.ball_path_log) >= self.max_entries,
        }
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
            for entry in state.ball_path_log:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return str(path)


def _clock(seconds):
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _player(player):
    return f"{player.name} [{player.role}]"


def _position(position):
    return [float(position[0]), float(position[1])]


def _slug(value):
    return re.sub(r"[^\w\-]+", "_", value.strip()).strip("_")[:80]


def home_score(state):
    return state.home.score


def away_score(state):
    return state.away.score
