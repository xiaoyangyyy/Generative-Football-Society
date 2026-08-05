"""Proper-scoring training and grouped validation for semantic event heads."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.world_model.state_scales import (
    FALSIFIABLE_SEMANTIC_EVENTS,
    PREDICTIVE_MECHANISM_PATHS,
)


def bootstrap_semantic_event_loss(
    logits,
    targets,
    bootstrap_weights,
):
    """Member-specific proper BCE objective without prevalence reweighting."""
    import torch

    if (
        logits.ndim != 3
        or targets.ndim != 2
        or logits.shape[1:] != targets.shape
        or bootstrap_weights.shape != logits.shape[:2]
    ):
        raise ValueError("invalid semantic event bootstrap shapes")
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets.unsqueeze(0).expand_as(logits),
        reduction="none",
    ).mean(dim=2)
    member_losses = (
        (raw * bootstrap_weights).sum(dim=1)
        / bootstrap_weights.sum(dim=1).clamp_min(1.0)
    )
    return member_losses.mean(), member_losses


def bootstrap_semantic_path_loss(logits, event_targets, bootstrap_weights):
    """Autoregressive joint NLL with member-specific bootstrap routing."""
    import torch

    if (
        logits.ndim != 4 or logits.shape[2:] != (
            len(PREDICTIVE_MECHANISM_PATHS), 7,
        )
        or event_targets.ndim != 2
        or event_targets.shape != (
            logits.shape[1], len(FALSIFIABLE_SEMANTIC_EVENTS),
        )
        or bootstrap_weights.shape != logits.shape[:2]
    ):
        raise ValueError("invalid semantic path bootstrap shapes")
    event_indices = {
        event: index for index, event in enumerate(FALSIFIABLE_SEMANTIC_EVENTS)
    }
    path_losses = []
    for path_index, path in enumerate(PREDICTIVE_MECHANISM_PATHS):
        targets = [
            event_targets[:, event_indices[event]].long() for event in path
        ]
        a, b, c = targets
        path_logits = logits[:, :, path_index, :]
        selected = torch.stack([
            path_logits[:, :, 0],
            torch.gather(
                path_logits[:, :, 1:3], 2,
                a.view(1, -1, 1).expand(logits.shape[0], -1, -1),
            ).squeeze(2),
            torch.gather(
                path_logits[:, :, 3:7], 2,
                (2 * a + b).view(1, -1, 1).expand(
                    logits.shape[0], -1, -1,
                ),
            ).squeeze(2),
        ], dim=2)
        target = torch.stack([a, b, c], dim=1).to(logits.dtype)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            selected, target.unsqueeze(0).expand_as(selected), reduction="none",
        ).mean(dim=2)
        path_losses.append(raw)
    raw_paths = torch.stack(path_losses, dim=2).mean(dim=2)
    member_losses = (
        (raw_paths * bootstrap_weights).sum(dim=1)
        / bootstrap_weights.sum(dim=1).clamp_min(1.0)
    )
    return member_losses.mean(), member_losses


def semantic_path_cell_rates(event_targets: Any) -> np.ndarray:
    """Training-only Dirichlet-smoothed cell rates for each declared path."""
    labels = np.asarray(event_targets, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != len(FALSIFIABLE_SEMANTIC_EVENTS):
        raise ValueError("invalid semantic path target shape")
    indices = {
        event: index for index, event in enumerate(FALSIFIABLE_SEMANTIC_EVENTS)
    }
    rates = []
    for path in PREDICTIVE_MECHANISM_PATHS:
        selected = labels[:, [indices[event] for event in path]].astype(int)
        cells = 4 * selected[:, 0] + 2 * selected[:, 1] + selected[:, 2]
        counts = np.bincount(cells, minlength=8).astype(np.float64)
        rates.append((counts + 0.5) / (len(cells) + 4.0))
    return np.asarray(rates, dtype=np.float32)


def _group_equal_mean(values: np.ndarray, groups: np.ndarray) -> float:
    return float(np.mean([
        np.mean(values[groups == group]) for group in np.unique(groups)
    ])) if len(values) else 0.0


def _calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    bins: int = 8,
) -> float:
    match_errors = []
    for group in np.unique(groups):
        selected = groups == group
        p, y = probabilities[selected], targets[selected]
        error = 0.0
        for index in range(bins):
            low, high = index / bins, (index + 1) / bins
            mask = (p >= low) & ((p < high) | ((index == bins - 1) & (p <= high)))
            if np.any(mask):
                error += float(np.mean(mask)) * abs(
                    float(np.mean(p[mask])) - float(np.mean(y[mask]))
                )
        match_errors.append(error)
    return float(np.mean(match_errors)) if match_errors else 0.0


def semantic_event_validation(
    member_probabilities: Any,
    targets: Any,
    projection_probabilities: Any,
    training_base_rates: Any,
    groups: Any,
) -> dict[str, Any]:
    """Compare learned probabilities with both prevalence and geometry baselines."""
    members = np.asarray(member_probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    projection = np.asarray(projection_probabilities, dtype=np.float64)
    base_rates = np.asarray(training_base_rates, dtype=np.float64).reshape(-1)
    group_values = np.asarray(groups).reshape(-1)
    if members.ndim != 3:
        raise ValueError("invalid semantic event validation shapes")
    expected = (members.shape[1], len(FALSIFIABLE_SEMANTIC_EVENTS))
    if (
        labels.shape != expected
        or projection.shape != expected
        or base_rates.shape != (expected[1],)
        or group_values.shape != (expected[0],)
    ):
        raise ValueError("invalid semantic event validation shapes")
    if not all(np.isfinite(value).all() for value in (
        members, labels, projection, base_rates,
    )):
        raise ValueError("semantic event validation values must be finite")
    learned = np.mean(members, axis=0)
    event_reports = {}
    for index, name in enumerate(FALSIFIABLE_SEMANTIC_EVENTS):
        target = labels[:, index]
        probability = learned[:, index]
        member_errors = (members[:, :, index] - target[None, :]) ** 2
        learned_error = (probability - target) ** 2
        projection_error = (projection[:, index] - target) ** 2
        base_error = (base_rates[index] - target) ** 2
        learned_brier = _group_equal_mean(learned_error, group_values)
        projection_brier = _group_equal_mean(projection_error, group_values)
        base_brier = _group_equal_mean(base_error, group_values)
        member_mean_brier = float(np.mean([
            _group_equal_mean(member_error, group_values)
            for member_error in member_errors
        ]))
        best_baseline = min(projection_brier, base_brier)
        disagreement = np.std(members[:, :, index], axis=0)
        absolute_error = np.abs(probability - target)
        correlation = (
            float(np.corrcoef(disagreement, absolute_error)[0, 1])
            if np.std(disagreement) > 1e-12
            and np.std(absolute_error) > 1e-12 else 0.0
        )
        positives = int(np.sum(target >= 0.5))
        negatives = int(len(target) - positives)
        skill = float(
            1.0 - learned_brier / max(best_baseline, 1e-12)
        )
        event_reports[name] = {
            "samples": int(len(target)),
            "groups": int(len(np.unique(group_values))),
            "positives": positives,
            "negatives": negatives,
            "training_base_rate": float(base_rates[index]),
            "learned_brier": learned_brier,
            "projection_brier": projection_brier,
            "training_rate_baseline_brier": base_brier,
            "best_baseline_brier": best_baseline,
            "skill_vs_best_baseline": skill,
            "member_mean_brier": member_mean_brier,
            "ensemble_gain_vs_member_mean": float(
                member_mean_brier - learned_brier
            ),
            "disagreement_error_correlation": correlation,
            "calibration_error": _calibration_error(
                probability, target, group_values,
            ),
            "group_equal_weighting": True,
        }
    return {
        "version": 1,
        "proper_scoring_rule": "binary_cross_entropy_training_brier_validation",
        "events": event_reports,
    }


def semantic_path_validation(
    member_probabilities: Any,
    event_targets: Any,
    projection_probabilities: Any,
    training_cell_rates: Any,
    groups: Any,
) -> dict[str, Any]:
    """Validate learned eight-cell joints against projection and train rates."""
    members = np.asarray(member_probabilities, dtype=np.float64)
    labels = np.asarray(event_targets, dtype=np.float64)
    projection = np.asarray(projection_probabilities, dtype=np.float64)
    rates = np.asarray(training_cell_rates, dtype=np.float64)
    group_values = np.asarray(groups).reshape(-1)
    expected = (labels.shape[0], len(PREDICTIVE_MECHANISM_PATHS), 8)
    if (
        members.ndim != 4
        or members.shape[1:] != expected
        or labels.shape != (
            members.shape[1], len(FALSIFIABLE_SEMANTIC_EVENTS),
        )
        or projection.shape != expected
        or rates.shape != (len(PREDICTIVE_MECHANISM_PATHS), 8)
        or group_values.shape != (members.shape[1],)
    ):
        raise ValueError("invalid semantic path validation shapes")
    if not all(np.isfinite(value).all() for value in (
        members, labels, projection, rates,
    )):
        raise ValueError("semantic path validation values must be finite")
    if (
        np.max(np.abs(members.sum(axis=3) - 1.0)) > 1e-5
        or np.max(np.abs(projection.sum(axis=2) - 1.0)) > 1e-5
        or np.max(np.abs(rates.sum(axis=1) - 1.0)) > 1e-5
    ):
        raise ValueError("semantic path probabilities must normalize")
    event_indices = {
        event: index for index, event in enumerate(FALSIFIABLE_SEMANTIC_EVENTS)
    }
    learned = members.mean(axis=0)
    reports = {}
    for path_index, path in enumerate(PREDICTIVE_MECHANISM_PATHS):
        indices = [event_indices[event] for event in path]
        cells = (
            4 * labels[:, indices[0]].astype(int)
            + 2 * labels[:, indices[1]].astype(int)
            + labels[:, indices[2]].astype(int)
        )
        one_hot = np.eye(8, dtype=np.float64)[cells]
        learned_brier_rows = np.sum(
            (learned[:, path_index, :] - one_hot) ** 2, axis=1,
        )
        projection_brier_rows = np.sum(
            (projection[:, path_index, :] - one_hot) ** 2, axis=1,
        )
        base_brier_rows = np.sum(
            (rates[path_index] - one_hot) ** 2, axis=1,
        )
        learned_brier = _group_equal_mean(
            learned_brier_rows, group_values,
        )
        projection_brier = _group_equal_mean(
            projection_brier_rows, group_values,
        )
        base_brier = _group_equal_mean(base_brier_rows, group_values)
        best_baseline = min(projection_brier, base_brier)
        member_briers = [
            _group_equal_mean(np.sum(
                (member[:, path_index, :] - one_hot) ** 2, axis=1,
            ), group_values)
            for member in members
        ]
        positives = int(np.sum(cells == 7))
        reports["->".join(path)] = {
            "samples": int(len(cells)),
            "groups": int(len(np.unique(group_values))),
            "occupied_cells": int(len(np.unique(cells))),
            "completion_positives": positives,
            "completion_negatives": int(len(cells) - positives),
            "learned_brier": learned_brier,
            "projection_brier": projection_brier,
            "training_rate_baseline_brier": base_brier,
            "best_baseline_brier": best_baseline,
            "skill_vs_best_baseline": float(
                1.0 - learned_brier / max(best_baseline, 1e-12)
            ),
            "member_mean_brier": float(np.mean(member_briers)),
            "ensemble_gain_vs_member_mean": float(
                np.mean(member_briers) - learned_brier
            ),
            "mean_log_loss": _group_equal_mean(
                -np.log(np.clip(
                    learned[np.arange(len(cells)), path_index, cells],
                    1e-12, 1.0,
                )), group_values,
            ),
            "group_equal_weighting": True,
        }
    return {
        "version": 1,
        "proper_scoring_rule": "autoregressive_joint_nll_categorical_brier_validation",
        "paths": reports,
    }
