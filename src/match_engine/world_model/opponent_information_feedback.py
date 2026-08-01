"""Validated closed-loop feedback for resolved opponent information queries."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.opponent_contract import TACTICAL_FEATURES
from src.match_engine.world_model.opponent_information_query import (
    opponent_information_query_audit_is_valid,
)
from src.match_engine.world_model.opponent_information_query_outcomes import (
    opponent_information_query_score_is_valid,
)


OPPONENT_INFORMATION_FEEDBACK_VERSION = 1


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "opponent-information-feedback:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def _resolution(
    record: dict[str, Any],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, Any] | None:
    if (
        str(record.get("checkpoint_signature")) != checkpoint_signature
        or str(record.get("environment_signature")) != environment_signature
    ):
        return None
    audit = record.get("llm_opponent_information_query_context") or {}
    if not opponent_information_query_audit_is_valid(audit):
        return None
    query = audit["query"]
    if str(record.get("intervention_actual_action", "")).lower() != query[
        "selected_action"
    ]:
        return None
    outcome = ((record.get("multi_horizon_regime_outcomes") or {}).get(
        query["horizon"]
    ) or {})
    score = outcome.get("llm_opponent_information_query_evaluation") or {}
    if (
        not opponent_information_query_score_is_valid(score, audit)
        or not score.get("issuance_evaluation_provenance_compatible")
    ):
        return None
    selected = audit["selected_query_report"]
    feature = query["feature"]
    event = bool(score["observed_high_events"][feature])
    forecast = float(selected["forecast_high_rate"])
    raw_forecast = float(selected["raw_forecast_high_rate"])
    signed_surprise = float(event) - forecast
    brier = float(score["selected_feature_brier_score"])
    regret = float(score["observed_branch_policy_regret"])
    rank = int(score["selected_query_model_rank"])
    issued_t_sec = float(record.get("created_t_sec", 0.0))
    observed_t_sec = float(outcome.get("observed_t_sec", 0.0))
    if (
        not math.isfinite(issued_t_sec)
        or not math.isfinite(observed_t_sec)
        or observed_t_sec + 1e-9 < issued_t_sec
    ):
        return None
    flags = []
    if brier >= 0.25:
        flags.append("forecast_surprise")
    if rank > 1 or float(score["selected_query_objective_efficiency"]) < 0.8:
        flags.append("query_selection_inefficient")
    if regret > 0.05 or not score["observed_branch_action_aligned"]:
        flags.append("contingent_action_misaligned")
    return {
        "decision_id": str(record.get("decision_id", "")),
        "issued_t_sec": issued_t_sec,
        "observed_t_sec": observed_t_sec,
        "selected_action": query["selected_action"],
        "feature": feature,
        "horizon": query["horizon"],
        "purpose": query["purpose"],
        "forecast_high_rate": forecast,
        "raw_forecast_high_rate": raw_forecast,
        "calibration_applied": bool(selected["calibration_applied"]),
        "observed_feature_value": float(score["observed_features"][feature]),
        "observed_high": event,
        "signed_probability_surprise": signed_surprise,
        "brier_score": brier,
        "raw_brier_score": float(score["raw_selected_feature_brier_score"]),
        "resolved_branch": score["resolved_observation_branch"],
        "proposed_continuation_action": score[
            "proposed_continuation_action"
        ],
        "model_best_continuation_action": score[
            "model_best_continuation_action"
        ],
        "observed_branch_policy_regret": regret,
        "observed_branch_action_aligned": bool(
            score["observed_branch_action_aligned"]
        ),
        "query_model_rank": rank,
        "query_objective_efficiency": float(
            score["selected_query_objective_efficiency"]
        ),
        "purpose_supported": bool(score["purpose_supported"]),
        "attention_flags": flags,
        "branch_action_executed": False,
        "counterfactual_outcome_observed": False,
        "hidden_opponent_intent_observed": False,
        "causal_interpretation": False,
    }


def build_opponent_information_feedback(
    records: Iterable[dict[str, Any]],
    *,
    team_id: str,
    checkpoint_signature: str,
    environment_signature: str,
    as_of_t_sec: float,
    recent_limit: int = 4,
) -> dict[str, Any]:
    """Build a compact, model-verified feedback ledger for the next LLM turn."""
    valid = []
    malformed = incompatible = pending = ineligible = future_excluded = 0
    as_of = float(as_of_t_sec)
    if not math.isfinite(as_of):
        as_of = 0.0
    as_of = max(0.0, as_of)
    for record in records:
        if str(record.get("team_id")) != str(team_id):
            continue
        context = record.get("llm_opponent_information_query_context") or {}
        if not context:
            continue
        if not opponent_information_query_audit_is_valid(context):
            malformed += 1
            continue
        query = context["query"]
        if str(record.get("intervention_actual_action", "")).lower() != query[
            "selected_action"
        ]:
            if record.get("intervention_actual_action") is None:
                pending += 1
            else:
                ineligible += 1
            continue
        outcome = ((record.get("multi_horizon_regime_outcomes") or {}).get(
            query["horizon"]
        ) or {})
        if not outcome.get("llm_opponent_information_query_evaluation"):
            pending += 1
            continue
        if (
            str(record.get("checkpoint_signature"))
            != str(checkpoint_signature)
            or str(record.get("environment_signature"))
            != str(environment_signature)
        ):
            incompatible += 1
            continue
        row = _resolution(
            record,
            checkpoint_signature=str(checkpoint_signature),
            environment_signature=str(environment_signature),
        )
        if row is None:
            malformed += 1
        elif (
            row["observed_t_sec"] > as_of + 1e-9
            or row["issued_t_sec"] > as_of + 1e-9
        ):
            future_excluded += 1
        else:
            valid.append(row)
    valid.sort(key=lambda row: (row["observed_t_sec"], row["issued_t_sec"]))
    limit = min(8, max(1, int(recent_limit)))
    recent = valid[-limit:]
    feature_profiles = {}
    for feature in TACTICAL_FEATURES:
        rows = [row for row in valid if row["feature"] == feature]
        if not rows:
            continue
        feature_profiles[feature] = {
            "resolved_queries": len(rows),
            "mean_brier_score": float(np.mean([
                row["brier_score"] for row in rows
            ])),
            "mean_raw_brier_score": float(np.mean([
                row["raw_brier_score"] for row in rows
            ])),
            "mean_signed_probability_surprise": float(np.mean([
                row["signed_probability_surprise"] for row in rows
            ])),
            "mean_query_objective_efficiency": float(np.mean([
                row["query_objective_efficiency"] for row in rows
            ])),
            "mean_observed_branch_policy_regret": float(np.mean([
                row["observed_branch_policy_regret"] for row in rows
            ])),
        }
    payload = {
        "version": OPPONENT_INFORMATION_FEEDBACK_VERSION,
        "available": bool(recent),
        "reason": "resolved_query_feedback" if recent else (
            "no_compatible_resolved_queries"
        ),
        "team_id": str(team_id),
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "resolved_queries": len(valid),
        "pending_queries": pending,
        "ineligible_action_queries": ineligible,
        "incompatible_scope_queries": incompatible,
        "malformed_queries": malformed,
        "future_dated_queries_excluded": future_excluded,
        "as_of_t_sec": as_of,
        "recent_resolutions": recent,
        "feature_profiles": feature_profiles,
        "interpretation": (
            "Use observed tactical controls to revise future questions and "
            "conditional reasoning. Branch actions were shadow proposals, not "
            "executed counterfactuals; no hidden intent or causal effect was "
            "observed."
        ),
        "can_change_current_action": False,
        "can_update_world_model_weights": False,
        "causal_interpretation": False,
    }
    return {**payload, "feedback_digest": _digest(payload)}


def opponent_information_feedback_is_valid(feedback: Any) -> bool:
    if not isinstance(feedback, dict):
        return False
    payload = dict(feedback)
    digest = payload.pop("feedback_digest", None)
    try:
        expected = _digest(payload)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        digest == expected
        and payload.get("version") == OPPONENT_INFORMATION_FEEDBACK_VERSION
        and payload.get("can_change_current_action") is False
        and payload.get("can_update_world_model_weights") is False
        and payload.get("causal_interpretation") is False
        and all(
            row.get("branch_action_executed") is False
            and row.get("counterfactual_outcome_observed") is False
            and row.get("hidden_opponent_intent_observed") is False
            and row.get("causal_interpretation") is False
            for row in payload.get("recent_resolutions") or []
        )
    )


def opponent_information_feedback_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    contexts = available = malformed = exposures = temporal_leaks = 0
    scope_mismatches = unsafe_claims = 0
    clustered_coverage = []
    for records in record_clusters:
        rows = list(records)
        eligible = valid_available = 0
        for record in rows:
            feedback = record.get("opponent_information_feedback_context") or {}
            if not feedback:
                continue
            contexts += 1
            if not opponent_information_feedback_is_valid(feedback):
                malformed += 1
                continue
            if (
                str(feedback.get("team_id")) != str(record.get("team_id"))
                or str(feedback.get("checkpoint_signature"))
                != str(record.get("checkpoint_signature"))
                or str(feedback.get("environment_signature"))
                != str(record.get("environment_signature"))
            ):
                scope_mismatches += 1
                continue
            eligible += 1
            resolutions = feedback.get("recent_resolutions") or []
            exposures += len(resolutions)
            if feedback.get("available"):
                available += 1
                valid_available += 1
            created = float(record.get("created_t_sec", 0.0))
            temporal_leaks += sum(
                float(row.get("observed_t_sec", 0.0)) > created + 1e-9
                for row in resolutions
            )
            unsafe_claims += sum(
                bool(row.get("branch_action_executed"))
                or bool(row.get("counterfactual_outcome_observed"))
                or bool(row.get("hidden_opponent_intent_observed"))
                or bool(row.get("causal_interpretation"))
                for row in resolutions
            )
        if eligible:
            clustered_coverage.append(valid_available / eligible)
    return {
        "version": OPPONENT_INFORMATION_FEEDBACK_VERSION,
        "feedback_contexts": contexts,
        "available_feedback_contexts": available,
        "resolution_exposures": exposures,
        "malformed_feedback_contexts": malformed,
        "scope_mismatches": scope_mismatches,
        "future_information_leaks": temporal_leaks,
        "unsafe_counterfactual_or_causal_claims": unsafe_claims,
        "match_clustered_available_feedback_rate": float(np.mean(
            clustered_coverage
        )) if clustered_coverage else 0.0,
        "all_feedback_temporally_prospective": temporal_leaks == 0,
        "all_feedback_scope_compatible": scope_mismatches == 0,
        "all_feedback_non_controlling_non_causal": unsafe_claims == 0,
    }
