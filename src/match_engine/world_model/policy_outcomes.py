"""Multi-horizon simulator outcomes for randomized policy opportunities."""

from __future__ import annotations

import math
from typing import Any, Iterable

from src.match_engine.world_model.policy_experiment import (
    policy_horizon_key,
    policy_horizon_seconds,
)


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
        if str(previous["team_id"]) != str(team_id):
            continue
        evaluation = _event_option_evaluation(previous)
        if (
            previous.get("event_option_resolved_t_sec") is not None
            and previous.get("event_option_next_action") is None
            and not previous.get("event_option_followup_expired")
        ):
            previous["event_option_followup_expired"] = True
            previous["event_option_followup_superseded_t_sec"] = float(t_sec)
            if evaluation is not None:
                evaluation["continuation_followup_expired"] = True
                evaluation["continuation_calibration_eligible"] = False
                evaluation["continuation_calibration_reason"] = (
                    "superseded_by_newer_same_team_coach_decision"
                )
        elif (
            previous.get("event_option_followup_anchor_t_sec") is not None
            and evaluation is not None
            and not evaluation.get("continuation_calibration_observed")
        ):
            previous["event_option_followup_censored_t_sec"] = float(t_sec)
            evaluation["continuation_calibration_eligible"] = False
            evaluation["continuation_calibration_reason"] = (
                "followup_value_censored_by_newer_same_team_coach_decision"
            )
        if (
            previous.get("policy_opportunity_observed")
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
    oriented_x = (
        float(state.ball.position[0])
        if attacking_home else 1.0 - float(state.ball.position[0])
    )
    zone = (
        "defensive" if oriented_x < 0.34
        else "middle" if oriented_x < 0.67
        else "final_third"
    )
    goal_diff = float(team.score - opponent.score)
    score_state = (
        "leading" if goal_diff > 0
        else "trailing" if goal_diff < 0
        else "level"
    )
    clock_seconds = float(getattr(state, "clock_seconds", 0.0))
    match_phase = (
        "early" if clock_seconds < 30.0 * 60.0
        else "middle" if clock_seconds < 70.0 * 60.0
        else "late"
    )
    return {
        "ball_x": float(state.ball.position[0]),
        "xg_for": float(
            state.micro_xg_home if attacking_home else state.micro_xg_away
        ),
        "xg_against": float(
            state.micro_xg_away if attacking_home else state.micro_xg_home
        ),
        "goal_diff": goal_diff,
        "opponent_team_id": str(opponent.team_id),
        "zone": zone,
        "score_state": score_state,
        "match_phase": match_phase,
        "clock_seconds": clock_seconds,
    }


def _realized_policy_utility(
    state,
    *,
    team_id: str,
    baseline: dict[str, Any],
    now: float,
    anchor: float,
) -> dict[str, Any]:
    attacking_home = str(team_id) == str(state.home.team_id)
    team = state.home if attacking_home else state.away
    opponent = state.away if attacking_home else state.home
    direction = 1.0 if attacking_home else -1.0
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
    retained = str(state.ball.possession_team_id) == str(team_id)
    utility = (
        goal_delta
        + 0.35 * xg_net
        + 0.15 * progress
        + 0.05 * (1.0 if retained else -1.0)
    )
    return {
        "elapsed_s": max(0.0, now - anchor),
        "observed_t_sec": now,
        "progress": progress,
        "retained_possession": retained,
        "xg_net_delta": xg_net,
        "goal_diff_delta": goal_delta,
        "policy_utility": utility,
    }


def _observe_interval_risk(
    record: dict[str, Any],
    outcome: dict[str, Any],
    baseline: dict[str, Any],
    *,
    attacking_home: bool,
    now: float,
    anchor: float,
) -> dict[str, Any] | None:
    context = record.get("llm_risk_certificate_context") or {}
    constraint = context.get("constraint") or {}
    if (
        not context.get("accepted")
        or constraint.get("risk_scope") != "within_horizon"
        or str(constraint.get("selected_action", "")).lower()
        != str(record.get("intervention_actual_action", "")).lower()
    ):
        return None
    try:
        horizon_s = policy_horizon_seconds(str(constraint["horizon"]))
    except (KeyError, TypeError, ValueError):
        return None
    due = anchor + horizon_s
    monitor = record.setdefault("llm_risk_interval_monitor", {
        "risk_scope": "within_horizon",
        "downside_event": str(constraint.get("downside_event", "")),
        "horizon_s": horizon_s,
        "downside_observed": False,
        "success_observed": bool(
            str(constraint.get("downside_event", ""))
            == "fail_enter_final_third"
            and (
                float(baseline["ball_x"])
                if attacking_home else 1.0 - float(baseline["ball_x"])
            ) >= 0.67
        ),
        "observations": 0,
        "first_observed_t_sec": None,
        "last_observed_t_sec": None,
        "max_gap_s": 0.0,
        "complete": False,
        "closed": False,
    })
    if monitor.get("closed") or now + 1e-9 < anchor:
        return monitor
    from src.match_engine.world_model.risk_certificate import (
        observed_downside_event,
    )

    observed = observed_downside_event(
        str(constraint.get("downside_event", "")),
        outcome,
        baseline,
        attacking_home=attacking_home,
    )
    previous = monitor.get("last_observed_t_sec")
    if previous is None:
        monitor["first_observed_t_sec"] = now
        gap = max(0.0, now - anchor)
    else:
        gap = max(0.0, now - float(previous))
    monitor["max_gap_s"] = max(float(monitor["max_gap_s"]), gap)
    monitor["last_observed_t_sec"] = now
    monitor["observations"] = int(monitor["observations"]) + 1
    if str(constraint.get("downside_event", "")) == "fail_enter_final_third":
        monitor["success_observed"] = bool(
            monitor["success_observed"] or not observed
        )
    else:
        monitor["downside_observed"] = bool(
            monitor["downside_observed"] or observed
        )
    if now + 1e-9 >= due:
        if str(constraint.get("downside_event", "")) == (
            "fail_enter_final_third"
        ):
            monitor["downside_observed"] = not bool(
                monitor["success_observed"]
            )
        end_gap = max(0.0, due - now)
        monitor["max_gap_s"] = max(float(monitor["max_gap_s"]), end_gap)
        tolerance = max(5.0, 0.10 * horizon_s)
        start_delay = max(
            0.0, float(monitor["first_observed_t_sec"]) - anchor,
        )
        monitor["complete"] = bool(
            int(monitor["observations"]) >= 2
            and start_delay <= tolerance
            and float(monitor["max_gap_s"]) <= tolerance
        )
        monitor["closed"] = True
        monitor["coverage_tolerance_s"] = tolerance
    return monitor


def _event_option_evaluation(record: dict[str, Any]) -> dict[str, Any] | None:
    for outcome in (
        record.get("multi_horizon_regime_outcomes") or {}
    ).values():
        if not isinstance(outcome, dict):
            continue
        evaluation = outcome.get("llm_event_option_evaluation")
        if isinstance(evaluation, dict):
            return evaluation
    return None


def _observe_event_option_followup(state, record: dict[str, Any], now: float) -> None:
    evaluation = _event_option_evaluation(record)
    if evaluation is None or evaluation.get("continuation_calibration_observed"):
        return
    resolved = record.get("event_option_resolved_t_sec")
    if (
        resolved is not None
        and record.get("event_option_next_action") is None
        and now > float(resolved) + 120.0
    ):
        record["event_option_followup_expired"] = True
        evaluation["continuation_followup_expired"] = True
        evaluation["continuation_calibration_eligible"] = False
        evaluation["continuation_calibration_reason"] = (
            "no_same_team_action_within_followup_window"
        )
        return
    if not evaluation.get("continuation_calibration_eligible"):
        return
    anchor = record.get("event_option_followup_anchor_t_sec")
    due = record.get("event_option_followup_due_t_sec")
    baseline = record.get("event_option_followup_baseline")
    if anchor is None or due is None or not isinstance(baseline, dict):
        return
    if now + 1e-9 < float(due):
        return
    try:
        predicted = float(evaluation["expected_continuation_policy_utility"])
        realized = _realized_policy_utility(
            state,
            team_id=str(record["team_id"]),
            baseline=baseline,
            now=now,
            anchor=float(anchor),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        evaluation["continuation_calibration_eligible"] = False
        evaluation["continuation_calibration_reason"] = (
            "malformed_followup_calibration_state"
        )
        return
    observed = float(realized["policy_utility"])
    if not math.isfinite(predicted) or not math.isfinite(observed):
        evaluation["continuation_calibration_eligible"] = False
        evaluation["continuation_calibration_reason"] = (
            "non_finite_followup_calibration_value"
        )
        return
    residual = observed - predicted
    evaluation["continuation_calibration_observed"] = True
    evaluation["continuation_calibration_reason"] = (
        "matching_branch_value_scored_on_realized_followup"
    )
    evaluation["realized_continuation_outcome"] = realized
    evaluation["continuation_policy_utility_residual"] = residual
    evaluation["continuation_policy_utility_absolute_error"] = abs(residual)
    evaluation["continuation_policy_utility_squared_error"] = residual ** 2


def observe_policy_intervention_outcomes(
    state,
    *,
    t_sec: float,
) -> None:
    """Attach every due isolated and policy-regime outcome horizon."""
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    for record in records:
        _observe_event_option_followup(state, record, now)
        baseline = record.get("outcome_baseline")
        if (
            not record.get("policy_opportunity_observed")
            or baseline is None
            or float(record.get("intervention_t_sec") or 0.0) > now
        ):
            continue
        team_id = str(record["team_id"])
        attacking_home = team_id == str(state.home.team_id)
        anchor = float(record.get("outcome_anchor_t_sec") or 0.0)
        isolated = record.setdefault("multi_horizon_outcomes", {})
        regime = record.setdefault("multi_horizon_regime_outcomes", {})
        censor_t = record.get("outcome_censored_t_sec")
        realized = _realized_policy_utility(
            state,
            team_id=team_id,
            baseline=baseline,
            now=now,
            anchor=anchor,
        )
        progress = float(realized["progress"])
        xg_net = float(realized["xg_net_delta"])
        goal_delta = float(realized["goal_diff_delta"])
        retained = bool(realized["retained_possession"])
        utility = float(realized["policy_utility"])
        interval_monitor = _observe_interval_risk(
            record,
            realized,
            baseline,
            attacking_home=attacking_home,
            now=now,
            anchor=anchor,
        )
        for horizon_s in record.get("outcome_horizons_s", [0.0]):
            horizon_s = max(0.0, float(horizon_s))
            key = policy_horizon_key(horizon_s)
            due_t = anchor + horizon_s
            if key in regime or now + 1e-9 < due_t:
                continue
            outcome = {
                "horizon_s": horizon_s,
                **realized,
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
            event_context = record.get("llm_semantic_event_context") or {}
            event_hypothesis = event_context.get("hypothesis") or {}
            event_action_realized = (
                str(event_hypothesis.get("action", "")).lower()
                == str(record.get("intervention_actual_action", "")).lower()
            )
            if (
                str(event_hypothesis.get("horizon", "")) == key
                and event_action_realized
            ):
                from src.match_engine.world_model.llm_event_hypothesis import (
                    score_llm_event_hypothesis,
                )

                event_score = score_llm_event_hypothesis(
                    event_context,
                    outcome,
                    baseline,
                    attacking_home=attacking_home,
                    checkpoint_signature=str(record.get(
                        "checkpoint_signature", "runtime_unspecified",
                    )),
                    environment_signature=str(record.get(
                        "environment_signature", "environment_unspecified",
                    )),
                )
                if event_score is not None:
                    outcome["llm_semantic_event_evaluation"] = event_score
            risk_context = record.get("llm_risk_certificate_context") or {}
            risk_constraint = risk_context.get("constraint") or {}
            risk_horizon = str(risk_constraint.get("horizon", ""))
            risk_horizon_key = (
                policy_horizon_key(0.0)
                if risk_horizon == "transition" else risk_horizon
            )
            risk_action_realized = (
                str(risk_constraint.get("selected_action", "")).lower()
                == str(record.get("intervention_actual_action", "")).lower()
            )
            if risk_horizon_key == key and risk_action_realized:
                from src.match_engine.world_model.risk_certificate import (
                    score_llm_risk_certificate,
                )

                risk_score = score_llm_risk_certificate(
                    risk_context,
                    outcome,
                    baseline,
                    attacking_home=attacking_home,
                    checkpoint_signature=str(record.get(
                        "checkpoint_signature", "runtime_unspecified",
                    )),
                    environment_signature=str(record.get(
                        "environment_signature", "environment_unspecified",
                    )),
                    interval_monitor=interval_monitor,
                )
                if risk_score is not None:
                    outcome["llm_risk_certificate_evaluation"] = risk_score
            distributional_context = (
                record.get("llm_distributional_claim_context") or {}
            )
            distributional_claim = (
                distributional_context.get("claim") or {}
            )
            distributional_action_realized = (
                str(distributional_claim.get("selected_action", "")).lower()
                == str(record.get("intervention_actual_action", "")).lower()
            )
            if (
                str(distributional_claim.get("horizon", "")) == key
                and distributional_action_realized
            ):
                from src.match_engine.world_model.distributional_claim import (
                    score_distributional_claim,
                )

                distributional_score = score_distributional_claim(
                    distributional_context,
                    outcome,
                    checkpoint_signature=str(record.get(
                        "checkpoint_signature", "runtime_unspecified",
                    )),
                    environment_signature=str(record.get(
                        "environment_signature", "environment_unspecified",
                    )),
                )
                if distributional_score is not None:
                    outcome["llm_distributional_claim_evaluation"] = (
                        distributional_score
                    )
            preference_context = (
                record.get("llm_risk_preference_context") or {}
            )
            preference = preference_context.get("preference") or {}
            preference_action_realized = (
                str(preference.get("selected_action", "")).lower()
                == str(record.get("intervention_actual_action", "")).lower()
            )
            if (
                key in (preference.get("horizon_weights") or {})
                and preference_action_realized
            ):
                from src.match_engine.world_model import (
                    risk_preference_evaluation,
                )

                score_preference = (
                    risk_preference_evaluation.score_llm_risk_preference
                )
                preference_score = score_preference(
                    preference_context,
                    outcome,
                    horizon=key,
                    checkpoint_signature=str(record.get(
                        "checkpoint_signature", "runtime_unspecified",
                    )),
                    environment_signature=str(record.get(
                        "environment_signature", "environment_unspecified",
                    )),
                )
                if preference_score is not None:
                    outcome["llm_risk_preference_evaluation"] = (
                        preference_score
                    )
            option_context = record.get("llm_event_option_context") or {}
            option = option_context.get("option") or {}
            option_first_action_realized = (
                str(option.get("first_action", "")).lower()
                == str(record.get("intervention_actual_action", "")).lower()
            )
            if (
                option_context.get("accepted")
                and str(option.get("horizon", "")) == key
                and option_first_action_realized
            ):
                from src.match_engine.world_model.llm_event_hypothesis import (
                    observed_semantic_event,
                )

                option_event_observed = observed_semantic_event(
                    str(option.get("event", "")),
                    outcome,
                    baseline,
                    attacking_home=attacking_home,
                )
                expected_action = str(
                    option.get(
                        "on_occurrence"
                        if option_event_observed else "on_absence",
                        "none",
                    )
                )
                option_evaluation = {
                    "version": int(option_context.get("version", 1)),
                    "option": dict(option),
                    "event_observed": bool(option_event_observed),
                    "expected_continuation_action": expected_action,
                    "next_action_observed": False,
                    "expected_action_matched": None,
                    "conditional_gain_vs_best_fixed": float(
                        option_context.get(
                            "conditional_gain_vs_best_fixed", 0.0,
                        )
                    ),
                    "continuation_value_calibratable": bool(
                        option_context.get(
                            "continuation_value_calibratable", False,
                        )
                    ),
                    "continuation_prediction_horizon_s": float(
                        option_context.get(
                            "continuation_prediction_horizon_s", 10.0,
                        )
                    ),
                    "expected_continuation_policy_utility": (
                        (option_context.get(
                            "conditional_policy_utility_predictions"
                        ) or {}).get(
                            "on_occurrence"
                            if option_event_observed else "on_absence"
                        )
                    ),
                    "continuation_calibration_eligible": False,
                    "continuation_calibration_observed": False,
                    "member_evaluations": int(option_context.get(
                        "member_evaluations", 0,
                    )),
                    "member_evaluation_budget": int(option_context.get(
                        "member_evaluation_budget", 0,
                    )),
                    "shadow_only": True,
                    "authority_active": False,
                    "policy_mutated": False,
                    "can_execute_future_action": False,
                    "causal_interpretation": False,
                    "option_signature": str(option_context.get(
                        "option_signature", "",
                    )),
                    "checkpoint_signature": str(record.get(
                        "checkpoint_signature", "",
                    )),
                    "environment_signature": str(record.get(
                        "environment_signature", "",
                    )),
                    "resolved_t_sec": now,
                }
                outcome["llm_event_option_evaluation"] = option_evaluation
                record["event_option_resolved_t_sec"] = now
                record["event_option_expected_action"] = expected_action
            regime[key] = outcome
            if censor_t is None or due_t <= float(censor_t) + 1e-9:
                isolated[key] = outcome
            if key == "transition":
                record["short_horizon_outcome"] = outcome
        from src.match_engine.world_model.temporal_calibration import (
            attach_temporal_path_evaluation,
        )

        attach_temporal_path_evaluation(record)
