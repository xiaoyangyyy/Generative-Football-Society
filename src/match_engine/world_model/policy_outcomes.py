"""Multi-horizon simulator outcomes for randomized policy opportunities."""

from __future__ import annotations

import math
from typing import Any, Iterable

from src.match_engine.world_model.policy_experiment import policy_horizon_key


def normalize_outcome_horizons(values: Iterable[Any]) -> tuple[float, ...]:
    normalized = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0.0:
            normalized.append(number)
    return tuple(sorted(set([0.0, *normalized])))[:6]


def censor_overlapping_policy_outcomes(
    records: list[dict[str, Any]],
    *,
    team_id: str,
    t_sec: float,
) -> None:
    """Censor only the isolated-effect view; regime outcomes keep running."""
    for previous in records:
        if (
            str(previous["team_id"]) == str(team_id)
            and previous.get("policy_opportunity_observed")
            and previous.get("outcome_baseline") is not None
            and previous.get("outcome_censored_t_sec") is None
        ):
            previous["outcome_censored_t_sec"] = float(t_sec)
            previous["outcome_censor_reason"] = (
                "subsequent_same_team_coach_decision"
            )


def capture_policy_outcome_baseline(
    state,
    *,
    team_id: str,
) -> dict[str, float]:
    """Capture simulator-native quantities available before action sampling."""
    attacking_home = str(team_id) == str(state.home.team_id)
    team = state.home if attacking_home else state.away
    opponent = state.away if attacking_home else state.home
    return {
        "ball_x": float(state.ball.position[0]),
        "xg_for": float(
            state.micro_xg_home if attacking_home else state.micro_xg_away
        ),
        "xg_against": float(
            state.micro_xg_away if attacking_home else state.micro_xg_home
        ),
        "goal_diff": float(team.score - opponent.score),
    }


def observe_policy_intervention_outcomes(
    state,
    *,
    t_sec: float,
) -> None:
    """Attach every due isolated and policy-regime outcome horizon."""
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    for record in records:
        baseline = record.get("outcome_baseline")
        if (
            not record.get("policy_opportunity_observed")
            or baseline is None
            or float(record.get("intervention_t_sec") or 0.0) > now
        ):
            continue
        team_id = str(record["team_id"])
        attacking_home = team_id == str(state.home.team_id)
        team = state.home if attacking_home else state.away
        opponent = state.away if attacking_home else state.home
        direction = 1.0 if attacking_home else -1.0
        anchor = float(record.get("outcome_anchor_t_sec") or 0.0)
        isolated = record.setdefault("multi_horizon_outcomes", {})
        regime = record.setdefault("multi_horizon_regime_outcomes", {})
        censor_t = record.get("outcome_censored_t_sec")
        xg_for = float(
            state.micro_xg_home if attacking_home else state.micro_xg_away
        )
        xg_against = float(
            state.micro_xg_away if attacking_home else state.micro_xg_home
        )
        progress = direction * (
            float(state.ball.position[0]) - float(baseline["ball_x"])
        )
        xg_net = (
            xg_for - float(baseline["xg_for"])
            - xg_against + float(baseline["xg_against"])
        )
        goal_delta = (
            float(team.score - opponent.score) - float(baseline["goal_diff"])
        )
        retained = str(state.ball.possession_team_id) == team_id
        retention_edge = 1.0 if retained else -1.0
        utility = (
            goal_delta
            + 0.35 * xg_net
            + 0.15 * progress
            + 0.05 * retention_edge
        )
        for horizon_s in record.get("outcome_horizons_s", [0.0]):
            horizon_s = max(0.0, float(horizon_s))
            key = policy_horizon_key(horizon_s)
            due_t = anchor + horizon_s
            if key in regime or now + 1e-9 < due_t:
                continue
            outcome = {
                "horizon_s": horizon_s,
                "elapsed_s": max(0.0, now - anchor),
                "observed_t_sec": now,
                "progress": progress,
                "retained_possession": retained,
                "xg_net_delta": xg_net,
                "goal_diff_delta": goal_delta,
                "policy_utility": utility,
            }
            prediction = (
                record.get("world_model_outcome_predictions") or {}
            ).get(key)
            if isinstance(prediction, dict):
                predicted_utility = prediction.get("policy_utility")
                if predicted_utility is not None:
                    predicted_utility = float(predicted_utility)
                    uncertainty = max(
                        0.05, float(prediction.get("uncertainty", 1.0)),
                    )
                    residual = utility - predicted_utility
                    outcome["world_model_prediction"] = prediction
                    outcome["prediction_residual"] = residual
                    outcome["standardized_prediction_residual"] = (
                        residual / uncertainty
                    )
            regime[key] = outcome
            if censor_t is None or due_t <= float(censor_t) + 1e-9:
                isolated[key] = outcome
            if key == "transition":
                record["short_horizon_outcome"] = outcome
