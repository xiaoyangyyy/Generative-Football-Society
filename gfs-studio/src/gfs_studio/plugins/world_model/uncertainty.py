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


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def compose_uncertainty(epistemic: Any, aleatoric: Any) -> float:
    """Union-like composition that is monotone and remains in [0, 1]."""
    epi = bounded_uncertainty(epistemic)
    alea = bounded_uncertainty(aleatoric)
    return float(1.0 - (1.0 - epi) * (1.0 - alea))


def compound_uncertainty(value: Any, steps: int) -> float:
    uncertainty = bounded_uncertainty(value)
    return float(1.0 - (1.0 - uncertainty) ** max(1, int(steps)))


def transition_ensemble_uncertainty(
    state_samples: np.ndarray | None,
    *,
    trained: bool = True,
) -> dict[str, Any]:
    """Measure decoded trajectory disagreement without inventing it for v6."""
    if state_samples is None:
        values = np.zeros((0, 0), dtype=float)
    else:
        values = np.asarray(state_samples, dtype=float)
    if not trained or values.ndim < 2 or values.shape[0] < 2:
        return {
            "trained": bool(trained),
            "members": int(values.shape[0]) if values.ndim >= 1 else 0,
            "state_rms_disagreement": 0.0,
            "ball_rms_disagreement": 0.0,
            "player_rms_disagreement": 0.0,
            "epistemic_uncertainty": 0.0,
        }
    if not np.isfinite(values).all():
        raise ValueError("transition ensemble samples must be finite")
    flattened = values.reshape(values.shape[0], -1, values.shape[-1])
    member_std = np.std(flattened, axis=0)
    state_rms = float(np.sqrt(np.mean(np.square(member_std))))
    width = values.shape[-1]

    def block_rms(start: int, stop: int) -> float:
        if width < stop:
            return state_rms
        block = member_std[..., start:stop]
        return float(np.sqrt(np.mean(np.square(block))))

    ball_rms = block_rms(200, 204)
    player_rms = block_rms(210, 298)
    state_component = float(np.clip(state_rms / 0.08, 0.0, 1.0))
    ball_component = float(np.clip(ball_rms / 0.05, 0.0, 1.0))
    player_component = float(np.clip(player_rms / 0.05, 0.0, 1.0))
    epistemic = float(np.clip(
        0.45 * state_component
        + 0.30 * ball_component
        + 0.25 * player_component,
        0.0,
        1.0,
    ))
    return {
        "trained": True,
        "members": int(values.shape[0]),
        "state_rms_disagreement": state_rms,
        "ball_rms_disagreement": ball_rms,
        "player_rms_disagreement": player_rms,
        "epistemic_uncertainty": epistemic,
    }


def ensemble_uncertainty_decomposition(
    pass_probabilities: np.ndarray,
    shot_probabilities: np.ndarray,
    progress_samples: np.ndarray,
    *,
    progress_aleatoric: float = 0.0,
    transition_state_samples: np.ndarray | None = None,
    transition_ensemble_trained: bool = True,
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
    transition = transition_ensemble_uncertainty(
        transition_state_samples,
        trained=transition_ensemble_trained,
    )

    def hierarchical_epistemic(
        values: np.ndarray, scale: float,
    ) -> tuple[float, float]:
        members = transition["members"]
        if (
            transition["trained"]
            and members >= 2
            and values.ndim >= 2
            and values.shape[0] == members
        ):
            shaped = values.reshape(members, values.shape[1], -1)
            within = float(np.sqrt(np.mean(np.var(
                shaped, axis=1,
            ))))
            between = float(np.sqrt(np.mean(np.var(
                np.mean(shaped, axis=1), axis=0,
            ))))
            return (
                float(np.clip(within / scale, 0.0, 1.0)),
                float(np.clip(between / scale, 0.0, 1.0)),
            )
        return float(np.clip(np.std(values) / scale, 0.0, 1.0)), 0.0

    pass_head_epistemic, pass_transition_epistemic = (
        hierarchical_epistemic(passes, 0.50)
    )
    shot_head_epistemic, shot_transition_epistemic = (
        hierarchical_epistemic(shots, 0.50)
    )
    progress_head_epistemic, progress_transition_epistemic = (
        hierarchical_epistemic(progress, 0.25)
    )
    components = {
        "pass_epistemic": pass_head_epistemic,
        "shot_epistemic": shot_head_epistemic,
        "progress_epistemic": progress_head_epistemic,
        "pass_transition_epistemic": pass_transition_epistemic,
        "shot_transition_epistemic": shot_transition_epistemic,
        "progress_transition_epistemic": progress_transition_epistemic,
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
    outcome_epistemic = float(np.clip(
        0.35 * components["pass_epistemic"]
        + 0.35 * components["shot_epistemic"]
        + 0.30 * components["progress_epistemic"],
        0.0, 1.0,
    ))
    transition_output_epistemic = float(np.clip(
        0.35 * pass_transition_epistemic
        + 0.35 * shot_transition_epistemic
        + 0.30 * progress_transition_epistemic,
        0.0,
        1.0,
    ))
    transition_epistemic = max(
        transition["epistemic_uncertainty"],
        transition_output_epistemic,
    )
    components["outcome_epistemic"] = outcome_epistemic
    components["transition_output_epistemic"] = transition_output_epistemic
    components["transition_epistemic"] = transition_epistemic
    components["transition_state_rms_disagreement"] = transition[
        "state_rms_disagreement"
    ]
    components["transition_ball_rms_disagreement"] = transition[
        "ball_rms_disagreement"
    ]
    components["transition_player_rms_disagreement"] = transition[
        "player_rms_disagreement"
    ]
    components["transition_ensemble_trained"] = transition["trained"]
    components["transition_ensemble_members"] = transition["members"]
    epistemic = compose_uncertainty(
        outcome_epistemic, transition_epistemic,
    )
    # Event randomness is intentionally capped below one: it should price risk,
    # but must not close a planner gate that is governed by epistemic quality.
    aleatoric = float(np.clip(
        0.20 * components["pass_aleatoric"]
        + 0.20 * components["shot_aleatoric"]
        + 0.10 * components["progress_aleatoric"],
        0.0, 0.50,
    ))
    return {
        "version": 2,
        "source": (
            "bootstrap_transition_and_outcome_total_variance"
            if transition["trained"] and transition["members"] >= 2
            else "bootstrap_outcome_total_variance_legacy_transition"
        ),
        "epistemic_uncertainty": epistemic,
        "aleatoric_uncertainty": aleatoric,
        "total_uncertainty": compose_uncertainty(epistemic, aleatoric),
        "components": components,
        "transition_ensemble": transition,
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
    transition_epistemic: list[float] = []
    transition_errors: list[float] = []
    transition_contract_samples = 0
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
                components = prediction.get("uncertainty_components") or {}
                trained_value = components.get(
                    "transition_ensemble_trained", False,
                )
                transition_trained = (
                    trained_value is True or trained_value == 1
                )
                transition_members = _nonnegative_int(
                    components.get("transition_ensemble_members", 0)
                )
                if transition_trained and transition_members >= 2:
                    transition_contract_samples += 1
                    transition_epistemic.append(bounded_uncertainty(
                        components.get("transition_epistemic", 0.0),
                        default=0.0,
                    ))
                    transition_errors.append(abs(actual - predicted))
    samples = len(errors)
    return {
        "version": 2,
        "samples": samples,
        "legacy_predictions": legacy_predictions,
        "transition_ensemble_predictions": transition_contract_samples,
        "transition_ensemble_available": transition_contract_samples > 0,
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
        "mean_transition_epistemic_uncertainty": (
            float(np.mean(transition_epistemic))
            if transition_epistemic else 0.0
        ),
        "transition_epistemic_error_correlation": _correlation(
            transition_epistemic, transition_errors,
        ),
        "interpretation": (
            "Head disagreement is a reducible-uncertainty proxy; aleatoric "
            "event variance prices risk and is excluded from acquisition value."
        ),
    }
