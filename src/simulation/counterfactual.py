"""Matched-seed counterfactual evaluation for simulation interventions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable

import numpy as np

from src.simulation.random_control import derive_seed


@dataclass(frozen=True)
class CounterfactualReport:
    intervention: str
    samples: int
    baseline_mean: float
    treatment_mean: float
    average_treatment_effect: float
    standard_error: float
    ci_low: float
    ci_high: float
    paired_deltas: tuple[float, ...]

    @property
    def directionally_supported(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def evaluate_intervention(
    run: Callable[[int, Any], float],
    baseline: Any,
    treatment: Any,
    *,
    root_seed: int = 42,
    samples: int = 32,
    intervention: str = "intervention",
    confidence: float = 0.95,
) -> CounterfactualReport:
    if samples < 2:
        raise ValueError("Counterfactual evaluation requires at least two paired samples")
    base_values, treatment_values = [], []
    for index in range(samples):
        seed = derive_seed(root_seed, "counterfactual", intervention, index)
        base_values.append(float(run(seed, baseline)))
        treatment_values.append(float(run(seed, treatment)))
    base = np.asarray(base_values, dtype=float)
    treated = np.asarray(treatment_values, dtype=float)
    if not np.isfinite(base).all() or not np.isfinite(treated).all():
        raise ValueError("Counterfactual outcomes must be finite")
    delta = treated - base
    ate = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(samples))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return CounterfactualReport(
        intervention=intervention,
        samples=samples,
        baseline_mean=float(base.mean()),
        treatment_mean=float(treated.mean()),
        average_treatment_effect=ate,
        standard_error=se,
        ci_low=ate - z * se,
        ci_high=ate + z * se,
        paired_deltas=tuple(float(value) for value in delta),
    )
