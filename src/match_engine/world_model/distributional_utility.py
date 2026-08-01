"""Member-level policy utility distributions and auditable action frontiers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.observation import OBS_DIM


DISTRIBUTIONAL_UTILITY_VERSION = 1
DISTRIBUTIONAL_CRITERIA = (
    "mean_utility",
    "lower_tail_cvar_25",
    "upside_probability",
)


def member_policy_utility_distribution(
    current_observation: Any,
    future_member_observations: Any,
    progress_member_samples: Any,
    *,
    attacking_home: bool,
    rollout_steps: int,
    ensemble_trained: bool,
) -> dict[str, Any]:
    """Compute the simulator-declared utility for every dynamics member."""
    current = np.asarray(current_observation, dtype=np.float64).reshape(-1)
    future = np.asarray(future_member_observations, dtype=np.float64)
    if future.ndim == 3 and future.shape[1] == 1:
        future = future[:, 0, :]
    if current.shape != (OBS_DIM,) or (
        future.ndim != 2 or future.shape[1] != OBS_DIM
    ):
        raise ValueError("distribution states must align as [members, 307]")
    members = len(future)
    progress_heads = np.asarray(progress_member_samples, dtype=np.float64)
    if progress_heads.ndim < 1 or progress_heads.shape[0] != members:
        raise ValueError("progress samples must preserve transition member axis")
    progress_heads = progress_heads.reshape(members, -1)
    current = np.clip(
        np.nan_to_num(current, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0,
    )
    future = np.clip(
        np.nan_to_num(future, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0,
    )
    progress_heads = np.nan_to_num(
        progress_heads, nan=0.0, posinf=1.0, neginf=-1.0,
    )
    direction = 1.0 if attacking_home else -1.0
    spatial_progress = np.clip(
        direction * (future[:, 200] - current[200]), -1.0, 1.0,
    )
    current_goal_diff = direction * 5.0 * (
        current[204] - current[205]
    )
    goal_delta = np.clip(
        direction * 5.0 * (future[:, 204] - future[:, 205])
        - current_goal_diff,
        -2.0,
        2.0,
    )
    possession_home = np.clip(future[:, 209], 0.0, 1.0)
    retention = possession_home if attacking_home else 1.0 - possession_home
    retention_edge = 2.0 * retention - 1.0
    steps = max(1, int(rollout_steps))
    xg_net = np.clip(np.mean(progress_heads, axis=1) * steps, -1.0, 1.0)
    utilities = (
        goal_delta
        + 0.35 * xg_net
        + 0.15 * spatial_progress
        + 0.05 * retention_edge
    )
    utilities = np.nan_to_num(
        utilities, nan=0.0, posinf=2.5, neginf=-2.5,
    )
    available = bool(ensemble_trained and members >= 2)
    if not available:
        return {
            "version": DISTRIBUTIONAL_UTILITY_VERSION,
            "available": False,
            "reason": "trained_transition_ensemble_required",
            "ensemble_members": members,
            "member_values": [],
            "counts_trusted": False,
            "member_identity_preserved": False,
            "observed": False,
            "causal_interpretation": False,
        }
    ordered = np.sort(utilities)
    tail_count = max(1, int(np.ceil(0.25 * members)))
    quantiles = np.quantile(
        utilities, [0.10, 0.25, 0.50, 0.75, 0.90], method="linear",
    )
    downside_count = int(np.sum(utilities < 0.0))
    upside_count = int(np.sum(utilities > 0.0))
    return {
        "version": DISTRIBUTIONAL_UTILITY_VERSION,
        "available": True,
        "reason": "member_level_declared_policy_utility",
        "ensemble_members": members,
        "member_values": list(map(float, utilities)),
        "mean_utility": float(np.mean(utilities)),
        "utility_std": float(np.std(utilities)),
        "minimum_utility": float(ordered[0]),
        "maximum_utility": float(ordered[-1]),
        "lower_tail_cvar_25": float(np.mean(ordered[:tail_count])),
        "upper_tail_mean_25": float(np.mean(ordered[-tail_count:])),
        "quantiles": {
            key: float(value) for key, value in zip(
                ("q10", "q25", "q50", "q75", "q90"), quantiles,
            )
        },
        "downside_probability": float(
            (downside_count + 0.5) / (members + 1.0)
        ),
        "upside_probability": float(
            (upside_count + 0.5) / (members + 1.0)
        ),
        "downside_member_count": downside_count,
        "upside_member_count": upside_count,
        "tail_member_count": tail_count,
        "counts_trusted": True,
        "member_identity_preserved": True,
        "utility_definition": (
            "goal_diff_delta + 0.35*xg_net_delta + 0.15*progress "
            "+ 0.05*retention_edge"
        ),
        "probability_source": "jeffreys_smoothed_member_frequency",
        "observed": False,
        "causal_interpretation": False,
    }


def build_distributional_action_frontiers(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return per-horizon nondominated actions and criterion leaders."""
    candidates = list(candidates)
    horizons = sorted({
        str(horizon)
        for candidate in candidates
        for horizon in (candidate.get("multi_horizon_predictions") or {})
    })
    reports: dict[str, Any] = {}
    for horizon in horizons:
        rows = []
        for candidate in candidates:
            prediction = (candidate.get("multi_horizon_predictions") or {}).get(
                horizon
            ) or {}
            distribution = prediction.get("distributional_policy_utility") or {}
            if not (
                distribution.get("available")
                and distribution.get("counts_trusted")
                and distribution.get("member_identity_preserved")
            ):
                continue
            rows.append({
                "action": str(candidate.get("action", "")),
                **{
                    criterion: float(distribution[criterion])
                    for criterion in DISTRIBUTIONAL_CRITERIA
                },
                "utility_std": float(distribution["utility_std"]),
            })
        if not rows:
            reports[horizon] = {
                "available": False,
                "reason": "trained_member_distributions_unavailable",
                "actions": [],
            }
            continue
        nondominated = []
        for row in rows:
            dominated = any(
                other is not row
                and all(
                    other[criterion] >= row[criterion] - 1e-12
                    for criterion in DISTRIBUTIONAL_CRITERIA
                )
                and any(
                    other[criterion] > row[criterion] + 1e-12
                    for criterion in DISTRIBUTIONAL_CRITERIA
                )
                for other in rows
            )
            if not dominated:
                nondominated.append(row["action"])
        leaders = {}
        for criterion in DISTRIBUTIONAL_CRITERIA:
            maximum = max(row[criterion] for row in rows)
            leaders[criterion] = sorted(
                row["action"] for row in rows
                if abs(row[criterion] - maximum) <= 1e-12
            )
        reports[horizon] = {
            "available": True,
            "actions": rows,
            "pareto_actions": sorted(nondominated),
            "criterion_leaders": leaders,
            "criteria": list(DISTRIBUTIONAL_CRITERIA),
            "causal_interpretation": False,
        }
    return {
        "version": DISTRIBUTIONAL_UTILITY_VERSION,
        "horizons": reports,
        "authority_active": False,
        "policy_mutated": False,
        "causal_interpretation": False,
    }


def align_distribution_location(
    distribution: dict[str, Any],
    target_mean: float,
    *,
    source: str,
    max_absolute_shift: float = 0.50,
) -> dict[str, Any]:
    """Align distribution location to a calibrated point without changing spread."""
    output = dict(distribution)
    if not output.get("available"):
        return output
    try:
        target = float(target_mean)
        current_mean = float(output["mean_utility"])
        values = np.asarray(output["member_values"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            **output,
            "available": False,
            "reason": "distribution_location_contract_invalid",
            "counts_trusted": False,
        }
    shift = target - current_mean
    if (
        not np.isfinite(target) or not np.isfinite(shift)
        or values.ndim != 1 or len(values) < 2
        or not np.all(np.isfinite(values))
        or abs(shift) > max(0.0, float(max_absolute_shift))
    ):
        return {
            **output,
            "available": False,
            "reason": "distribution_location_shift_gate_closed",
            "counts_trusted": False,
            "requested_location_shift": float(shift) if np.isfinite(shift) else 0.0,
        }
    shifted = values + shift
    ordered = np.sort(shifted)
    members = len(shifted)
    tail_count = min(members, max(1, int(output.get(
        "tail_member_count", np.ceil(0.25 * members),
    ))))
    quantiles = np.quantile(
        shifted, [0.10, 0.25, 0.50, 0.75, 0.90], method="linear",
    )
    downside_count = int(np.sum(shifted < 0.0))
    upside_count = int(np.sum(shifted > 0.0))
    output.update({
        "member_values": list(map(float, shifted)),
        "mean_utility": target,
        "minimum_utility": float(ordered[0]),
        "maximum_utility": float(ordered[-1]),
        "lower_tail_cvar_25": float(np.mean(ordered[:tail_count])),
        "upper_tail_mean_25": float(np.mean(ordered[-tail_count:])),
        "quantiles": dict(zip(
            ("q10", "q25", "q50", "q75", "q90"), map(float, quantiles),
        )),
        "downside_probability": float(
            (downside_count + 0.5) / (members + 1.0)
        ),
        "upside_probability": float(
            (upside_count + 0.5) / (members + 1.0)
        ),
        "downside_member_count": downside_count,
        "upside_member_count": upside_count,
        "location_alignment_source": str(source),
        "location_alignment_shift": float(shift),
        "spread_preserved_by_location_alignment": True,
    })
    return output

