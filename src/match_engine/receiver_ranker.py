"""Compact real-tracking prior for receiver choice, separate from pass success."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = ("distance", "forward_progress", "lateral_change", "receiver_defender_distance", "lane_clearance")


@dataclass(frozen=True)
class ReceiverRanker:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    version: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "ReceiverRanker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1 or tuple(payload.get("features", ())) != FEATURES:
            raise ValueError("Unsupported receiver-ranker schema")
        vectors = [np.asarray(payload[name], dtype=float) for name in ("mean", "scale", "weights")]
        if any(vector.shape != (len(FEATURES),) or not np.isfinite(vector).all() for vector in vectors):
            raise ValueError("Invalid receiver-ranker parameters")
        return cls(vectors[0], np.maximum(vectors[1], 1e-8), vectors[2], float(payload["bias"]))

    def score(self, features: np.ndarray | list[float]) -> float:
        values = np.asarray(features, dtype=float)
        if values.shape != (len(FEATURES),) or not np.isfinite(values).all():
            raise ValueError("Receiver features must be five finite values")
        return float(((values - self.mean) / self.scale) @ self.weights + self.bias)


def default_receiver_ranker_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "data" / "world_model" / "receiver_ranker.json"
