"""Build WM training rows from outputs/ball_log/*.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.match_engine.world_model.action_codec import ACTION_DIM, encode_from_ball_log_event
from src.match_engine.world_model.observation import OBS_DIM, synthesize_obs_from_ball_event
from src.match_engine.world_model.schema import HORIZON_INDEX, HORIZON_SCALE_SECONDS


def _load_jsonl_events(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta: Dict[str, Any] = {}
    events: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if i == 0 and "meta" in row:
                meta = row["meta"]
                continue
            if row.get("type") in ("pass", "shot", "cross", "intercept"):
                events.append(row)
    return meta, events


def transitions_from_ball_log_file(path: Path) -> List[Dict[str, Any]]:
    meta, events = _load_jsonl_events(path)
    if len(events) < 2:
        if len(events) == 1:
            ev = events[0]
            obs = np.array(ev["obs"], dtype=np.float32) if ev.get("obs") else synthesize_obs_from_ball_event(ev, meta=meta)
            act = encode_from_ball_log_event(ev)
            nxt = synthesize_obs_from_ball_event(ev, meta=meta)
            nxt[200:202] = np.array(ev.get("land_xy", ev.get("from_xy", [0.5, 0.5])), dtype=np.float32)
            return [{"obs": obs, "action": act, "next_obs": nxt, "meta": {"source": path.name, "event": ev.get("type")}}]
        return []

    home = meta.get("home", "")
    out: List[Dict[str, Any]] = []
    for i in range(len(events) - 1):
        ev, ev_next = events[i], events[i + 1]
        att = ev.get("team") == home if home else True
        if ev.get("obs"):
            obs = np.array(ev["obs"], dtype=np.float32)
        else:
            obs = synthesize_obs_from_ball_event(ev, meta=meta, attacking_home=att)
        if ev.get("next_obs"):
            nxt = np.array(ev["next_obs"], dtype=np.float32)
        elif ev_next.get("obs"):
            nxt = np.array(ev_next["obs"], dtype=np.float32)
        else:
            nxt = synthesize_obs_from_ball_event(ev_next, meta=meta, attacking_home=ev_next.get("team") == home if home else att)
        act = encode_from_ball_log_event(ev)
        delta_t = max(0.0, float(ev_next.get("t_sec", 0.0)) - float(ev.get("t_sec", 0.0)))
        act[HORIZON_INDEX] = float(np.clip(delta_t / HORIZON_SCALE_SECONDS, 0.0, 1.0))
        out.append(
            {
                "obs": obs.astype(np.float32),
                "action": act.astype(np.float32),
                "next_obs": nxt.astype(np.float32),
                "meta": {
                    "source": path.name,
                    "t_sec": ev.get("t_sec"),
                    "type": ev.get("type"),
                    "outcome": ev.get("outcome"),
                },
            }
        )
    return out


def load_ball_log_dataset(
    ball_log_dir: str,
    *,
    max_files: int = 500,
    return_groups: bool = False,
):
    root = Path(ball_log_dir)
    if not root.is_dir():
        empty = (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, ACTION_DIM), dtype=np.float32),
            np.zeros((0, OBS_DIM), dtype=np.float32),
        )
        return (*empty, np.asarray([], dtype=str)) if return_groups else empty
    obs_list, act_list, nxt_list, groups = [], [], [], []
    files = sorted(root.glob("*.jsonl"))[:max_files]
    for fp in files:
        for row in transitions_from_ball_log_file(fp):
            obs_list.append(row["obs"])
            act_list.append(row["action"])
            nxt_list.append(row["next_obs"])
            groups.append(fp.stem)
    if not obs_list:
        empty = (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, ACTION_DIM), dtype=np.float32),
            np.zeros((0, OBS_DIM), dtype=np.float32),
        )
        return (*empty, np.asarray([], dtype=str)) if return_groups else empty
    result = (
        np.stack(obs_list).astype(np.float32),
        np.stack(act_list).astype(np.float32),
        np.stack(nxt_list).astype(np.float32),
    )
    if return_groups:
        return (*result, np.asarray(groups, dtype=str))
    return result
