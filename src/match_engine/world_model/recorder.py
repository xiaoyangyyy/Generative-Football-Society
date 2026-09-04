"""Record (obs, action, next_obs) transitions for world-model training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.match_engine.world_model.action_codec import ACTION_DIM
from src.match_engine.world_model.config import default_trace_dir
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.schema import PASS_OUTCOME_INDEX, SHOT_GOAL_INDEX


def _pad_vector(vec, dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    if arr.shape[0] == dim:
        return arr
    out = np.zeros(dim, dtype=np.float32)
    out[: min(dim, arr.shape[0])] = arr[: min(dim, arr.shape[0])]
    return out


class TransitionRecorder:
    def __init__(self, trace_dir: str, *, match_slug: str = "match"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"{match_slug}.jsonl"
        self._fp = open(self.path, "a", encoding="utf-8")
        self.count = 0
        self.action_counts = {"pass": 0, "shot": 0, "hold": 0, "cross": 0, "other": 0}
        self.pass_successes = 0
        self.shot_goals = 0

    @classmethod
    def for_match(cls, base_dir: str, home: str, away: str, stage: str) -> "TransitionRecorder":
        slug = f"{home}_vs_{away}_{stage}".replace(" ", "_")
        return cls(default_trace_dir(base_dir), match_slug=slug)

    def write(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        action = _pad_vector(action, ACTION_DIM)
        labels = ("pass", "shot", "hold", "cross")
        kind = next((labels[index] for index in range(4) if action[index] > 0.5), "other")
        self.action_counts[kind] += 1
        if kind == "pass":
            self.pass_successes += int(action[PASS_OUTCOME_INDEX] > 0.5)
        if kind == "shot":
            self.shot_goals += int(action[SHOT_GOAL_INDEX] > 0.5)
        row = {
            "obs": obs.tolist(),
            "action": action.tolist(),
            "next_obs": next_obs.tolist(),
            "meta": meta or {},
        }
        self._fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self) -> str:
        self._fp.close()
        return str(self.path)


def load_trace_batches(
    trace_dir: str,
    *,
    max_files: int = 200,
    max_rows_per_file: int = 5000,
    return_groups: bool = False,
    allowed_files: set[str] | None = None,
):
    """Load all jsonl traces into (obs, action, next_obs) arrays."""
    root = Path(trace_dir)
    if not root.is_dir():
        empty = (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, ACTION_DIM), dtype=np.float32),
            np.zeros((0, OBS_DIM), dtype=np.float32),
        )
        return (*empty, np.asarray([], dtype=str)) if return_groups else empty
    obs_list, act_list, nxt_list, groups = [], [], [], []
    files = [
        fp for fp in sorted(root.glob("*.jsonl"))
        if not fp.name.startswith("_") and (allowed_files is None or fp.name in allowed_files)
    ][:max_files]
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_rows_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                obs_list.append(_pad_vector(row["obs"], OBS_DIM))
                act_list.append(_pad_vector(row["action"], ACTION_DIM))
                nxt_list.append(_pad_vector(row["next_obs"], OBS_DIM))
                groups.append(fp.stem)
    if not obs_list:
        empty = (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, ACTION_DIM), dtype=np.float32),
            np.zeros((0, OBS_DIM), dtype=np.float32),
        )
        return (*empty, np.asarray([], dtype=str)) if return_groups else empty
    result = (
        np.array(obs_list, dtype=np.float32),
        np.array(act_list, dtype=np.float32),
        np.array(nxt_list, dtype=np.float32),
    )
    if return_groups:
        return (*result, np.asarray(groups, dtype=str))
    return result
