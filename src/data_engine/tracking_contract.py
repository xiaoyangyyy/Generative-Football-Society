"""Provider-neutral tracking rows, quality checks, and grouped evaluation splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class TrackingFrame:
    provider: str
    competition: str
    match_id: str
    period: int
    frame_id: int
    timestamp_s: float
    home: Mapping[str, np.ndarray]
    away: Mapping[str, np.ndarray]
    ball: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.provider or not self.match_id:
            raise ValueError("provider and match_id are required")
        if self.period < 1 or self.frame_id < 0 or self.timestamp_s < 0:
            raise ValueError("invalid tracking coordinates")
        for positions in (self.home, self.away):
            for value in positions.values():
                point = np.asarray(value, dtype=float)
                if point.shape not in {(2,), (4,)} or not np.isfinite(point).all():
                    raise ValueError("player state must be finite [x,y] or [x,y,vx,vy]")


def grouped_leave_one_domain_out(
    rows: Iterable[Mapping[str, object]],
    *,
    domain: str = "provider",
) -> list[tuple[set[str], set[str]]]:
    """Return train/test match-id folds with entire domains held out."""
    records = list(rows)
    values = sorted({str(row[domain]) for row in records})
    folds = []
    for held_out in values:
        test = {str(row["match_id"]) for row in records if str(row[domain]) == held_out}
        train = {str(row["match_id"]) for row in records if str(row[domain]) != held_out}
        if train and test:
            folds.append((train, test))
    return folds


def tracking_quality(frames: Iterable[TrackingFrame]) -> dict[str, object]:
    values = list(frames)
    providers = sorted({frame.provider for frame in values})
    matches = sorted({frame.match_id for frame in values})
    complete = [
        len(frame.home) >= 7 and len(frame.away) >= 7
        for frame in values
    ]
    timestamps = {}
    for frame in values:
        timestamps.setdefault(frame.match_id, []).append(frame.timestamp_s)
    monotonic = all(
        np.all(np.diff(sorted(series)) > 0)
        for series in timestamps.values()
        if len(series) > 1
    )
    return {
        "providers": providers,
        "matches": len(matches),
        "frames": len(values),
        "complete_frame_rate": float(np.mean(complete)) if complete else 0.0,
        "timestamps_monotonic": bool(monotonic),
        "cross_provider_ready": len(providers) >= 2 and len(matches) >= 4,
    }


def perturb_player_state(
    state: np.ndarray,
    rng: np.random.Generator,
    *,
    position_sigma: float = 0.002,
    velocity_sigma: float = 0.01,
    missing_probability: float = 0.0,
) -> np.ndarray | None:
    """Deterministic-under-seed robustness perturbation for tracking evaluation."""
    if rng.random() < np.clip(missing_probability, 0.0, 1.0):
        return None
    value = np.asarray(state, dtype=float).copy()
    value[:2] += rng.normal(0.0, max(0.0, position_sigma), size=2)
    if len(value) >= 4:
        value[2:4] += rng.normal(0.0, max(0.0, velocity_sigma), size=2)
    return value
