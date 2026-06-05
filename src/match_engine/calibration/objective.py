"""Continuous calibration objective from observable contract evaluation."""

from __future__ import annotations

import math
from typing import Any


def calibration_loss(evaluation: dict[str, Any], *, fail_penalty: float = 25.0) -> float:
    """
    Differentiable-style loss: sum of squared z deviations + penalty per failed metric.
    Lower is better; 0 when all z=0 and nothing fails.
    """
    loss = 0.0
    for _key, row in evaluation.get("hard_metrics", {}).items():
        if row.get("skipped"):
            continue
        z = float(row.get("z_abs", 0.0))
        loss += z * z
        if not row.get("pass", True):
            loss += fail_penalty
    for _key, row in evaluation.get("soft_constraints", {}).items():
        if row.get("skipped"):
            continue
        z = row.get("z_abs")
        if z is None:
            continue
        z = float(z)
        loss += z * z
        if not row.get("pass", True):
            loss += fail_penalty
    return float(loss)


def calibration_score(evaluation: dict[str, Any]) -> float:
    """Higher is better — geometric mean of per-metric scores."""
    scores: list[float] = []
    for bucket in ("hard_metrics", "soft_constraints"):
        for row in evaluation.get(bucket, {}).values():
            if row.get("skipped"):
                continue
            s = row.get("score")
            if s is not None:
                scores.append(float(s))
    if not scores:
        return 0.0
    log_mean = sum(math.log(max(1e-12, s)) for s in scores) / len(scores)
    return float(math.exp(log_mean))
