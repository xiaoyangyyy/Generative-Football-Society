"""Calibrated epistemic/aleatoric uncertainty contracts for world-model use."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def bounded_uncertainty(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return float(np.clip(number, 0.0, 1.0))


def compose_uncertainty(epistemic: Any, aleatoric: Any) -> float:
    """Union-like composition that is monotone and remains in [0, 1]."""
    epi = bounded_uncertainty(epistemic)
    alea = bounded_uncertainty(aleatoric)
    return float(1.0 - (1.0 - epi) * (1.0 - alea))


def compound_uncertainty(value: Any, steps: int) -> float:
    uncertainty = bounded_uncertainty(value)
    return float(1.0 - (1.0 - uncertainty) ** max(1, int(steps)))


def ensemble_uncertainty_decomposition(
    pass_probabilities: np.ndarray,
    shot_probabilities: np.ndarray,
    progress_samples: np.ndarray,
    *,
    progress_aleatoric: float = 0.0,
) -> dict[str, Any]:
    """Apply total-variance semantics to bootstrapped outcome heads.

    Head disagreement estimates reducible epistemic uncertainty. Bernoulli
    variance within each head estimates irreducible event randomness. Progress
    aleatoric uncertainty is accepted only from held-out residual evidence.
    """
    passes = np.clip(np.asarray(pass_probabilities, dtype=float), 0.0, 1.0)
    shots = np.clip(np.asarray(shot_probabilities, dtype=float), 0.0, 1.0)
    progress = np.asarray(progress_samples, dtype=float)
    if not len(passes) or not len(shots) or not len(progress):
        raise ValueError("ensemble samples are required")
    if not all(np.isfinite(values).all() for values in (passes, shots, progress)):
        raise ValueError("ensemble samples must be finite")
    components = {
        "pass_epistemic": float(np.clip(2.0 * np.std(passes), 0.0, 1.0)),
        "shot_epistemic": float(np.clip(2.0 * np.std(shots), 0.0, 1.0)),
        "progress_epistemic": float(np.clip(
            np.std(progress) / 0.25, 0.0, 1.0,
        )),
        "pass_aleatoric": float(np.clip(
            2.0 * np.sqrt(np.mean(passes * (1.0 - passes))), 0.0, 1.0,
        )),
        "shot_aleatoric": float(np.clip(
            2.0 * np.sqrt(np.mean(shots * (1.0 - shots))), 0.0, 1.0,
        )),
        "progress_aleatoric": bounded_uncertainty(
            progress_aleatoric, default=0.0,
        ),
    }
    epistemic = float(np.clip(
        0.35 * components["pass_epistemic"]
        + 0.35 * components["shot_epistemic"]
        + 0.30 * components["progress_epistemic"],
        0.0, 1.0,
    ))
    # Event randomness is intentionally capped below one: it should price risk,
    # but must not close a planner gate that is governed by epistemic quality.
    aleatoric = float(np.clip(
        0.20 * components["pass_aleatoric"]
        + 0.20 * components["shot_aleatoric"]
        + 0.10 * components["progress_aleatoric"],
        0.0, 0.50,
    ))
    return {
        "version": 1,
        "source": "bootstrap_head_total_variance",
        "epistemic_uncertainty": epistemic,
        "aleatoric_uncertainty": aleatoric,
        "total_uncertainty": compose_uncertainty(epistemic, aleatoric),
        "components": components,
    }


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def uncertainty_decomposition_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    """Audit decomposition fields against realized policy-utility errors."""
    epistemic: list[float] = []
    aleatoric: list[float] = []
    total: list[float] = []
    errors: list[float] = []
    identity_error: list[float] = []
    legacy_predictions = 0
    for cluster in record_clusters:
        for record in cluster:
            outcomes = (
                record.get("multi_horizon_regime_outcomes")
                or record.get("multi_horizon_outcomes")
                or {}
            )
            for outcome in outcomes.values():
                prediction = outcome.get("world_model_prediction") or {}
                if not {
                    "epistemic_uncertainty", "aleatoric_uncertainty",
                }.issubset(prediction):
                    legacy_predictions += 1
                    continue
                try:
                    predicted_value = prediction.get("raw_policy_utility")
                    if predicted_value is None:
                        predicted_value = prediction["policy_utility"]
                    predicted = float(predicted_value)
                    actual = float(outcome["policy_utility"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(predicted) or not math.isfinite(actual):
                    continue
                epi = bounded_uncertainty(prediction["epistemic_uncertainty"])
                alea = bounded_uncertainty(prediction["aleatoric_uncertainty"])
                combined = compose_uncertainty(epi, alea)
                reported = bounded_uncertainty(
                    prediction.get("uncertainty", combined),
                )
                epistemic.append(epi)
                aleatoric.append(alea)
                total.append(reported)
                errors.append(abs(actual - predicted))
                identity_error.append(abs(reported - combined))
    samples = len(errors)
    return {
        "version": 1,
        "samples": samples,
        "legacy_predictions": legacy_predictions,
        "decomposition_available": samples > 0,
        "mean_epistemic_uncertainty": (
            float(np.mean(epistemic)) if samples else 0.0
        ),
        "mean_aleatoric_uncertainty": (
            float(np.mean(aleatoric)) if samples else 0.0
        ),
        "mean_total_uncertainty": float(np.mean(total)) if samples else 0.0,
        "mean_composition_identity_error": (
            float(np.mean(identity_error)) if samples else 0.0
        ),
        "epistemic_error_correlation": _correlation(epistemic, errors),
        "aleatoric_error_correlation": _correlation(aleatoric, errors),
        "total_error_correlation": _correlation(total, errors),
        "interpretation": (
            "Head disagreement is a reducible-uncertainty proxy; aleatoric "
            "event variance prices risk and is excluded from acquisition value."
        ),
    }
