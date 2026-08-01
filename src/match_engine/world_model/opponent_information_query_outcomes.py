"""Observe and score outcomes for frozen opponent information queries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_contract import TACTICAL_FEATURES
from src.match_engine.world_model.opponent_information_query import (
    OPPONENT_INFORMATION_QUERY_VERSION,
    opponent_information_query_audit_is_valid,
)


def observed_opponent_tactical_features(
    state,
    *,
    observer_team_id: str,
) -> dict[str, float] | None:
    if str(observer_team_id) == str(state.home.team_id):
        opponent = state.away
    elif str(observer_team_id) == str(state.away.team_id):
        opponent = state.home
    else:
        return None
    controls = opponent.coach.tactical_current or opponent.coach.tactical_base
    if not isinstance(controls, dict):
        return None
    return {
        feature: float(np.clip(float(controls.get(feature, 0.5)), 0.0, 1.0))
        for feature in TACTICAL_FEATURES
    }


def score_opponent_information_query(
    audit: dict[str, Any],
    observed_features: dict[str, Any],
    *,
    horizon: str,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, Any] | None:
    if (
        not opponent_information_query_audit_is_valid(audit)
        or str(horizon) != audit["query"]["horizon"]
    ):
        return None
    try:
        observed = {
            feature: float(observed_features[feature])
            for feature in TACTICAL_FEATURES
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0
               for value in observed.values()):
        return None
    reports = {
        row["feature"]: row for row in audit["query_leaderboard"]
    }
    events = {feature: observed[feature] >= 0.5
              for feature in TACTICAL_FEATURES}
    brier = {
        feature: (
            float(reports[feature]["forecast_high_rate"])
            - float(events[feature])
        ) ** 2
        for feature in TACTICAL_FEATURES
    }
    selected_feature = audit["query"]["feature"]
    resolved_high = bool(events[selected_feature])
    branch = "high" if resolved_high else "low"
    contingent = audit["contingent_policy"]
    return {
        "version": OPPONENT_INFORMATION_QUERY_VERSION,
        "audit_digest": audit["audit_digest"],
        "query_signature": audit["query_signature"],
        "query": dict(audit["query"]),
        "observed_features": observed,
        "observed_high_events": events,
        "feature_brier_scores": brier,
        "selected_feature_brier_score": float(brier[selected_feature]),
        "all_feature_mean_brier_score": float(np.mean(list(brier.values()))),
        "selected_query_model_rank": int(audit["selected_query_model_rank"]),
        "selected_query_objective_efficiency": float(
            audit["selected_query_objective_efficiency"]
        ),
        "purpose_supported": bool(audit["purpose_supported"]),
        "resolved_observation_branch": branch,
        "proposed_continuation_action": contingent[f"action_if_{branch}"],
        "model_best_continuation_action": contingent[
            f"model_best_action_if_{branch}"
        ],
        "observed_branch_policy_regret": float(
            contingent[f"regret_if_{branch}"]
        ),
        "observed_branch_action_aligned": bool(
            contingent[f"{branch}_branch_action_aligned"]
        ),
        "contingent_policy_model_checked_consistent": bool(
            contingent["model_checked_consistent"]
        ),
        "expected_contingent_policy_regret": float(
            contingent["expected_policy_regret"]
        ),
        "paired_same_action_horizon": True,
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "issuance_evaluation_provenance_compatible": bool(
            str(checkpoint_signature)
            == str(audit["model_evidence"]["checkpoint_signature"])
            and str(environment_signature)
            == str(audit["model_evidence"]["environment_signature"])
        ),
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "causal_interpretation": False,
        "hidden_opponent_intent_observed": False,
    }


def opponent_information_query_score_is_valid(
    score: Any, audit: dict[str, Any],
) -> bool:
    if not isinstance(score, dict):
        return False
    expected = score_opponent_information_query(
        audit,
        score.get("observed_features") or {},
        horizon=str((score.get("query") or {}).get("horizon", "")),
        checkpoint_signature=str(score.get("checkpoint_signature", "")),
        environment_signature=str(score.get("environment_signature", "")),
    )
    return bool(expected is not None and score == expected)
