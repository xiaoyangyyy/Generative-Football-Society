"""Score feedback-grounded query adaptation on the next realized query."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.match_engine.world_model.opponent_information_adaptation import (
    OPPONENT_INFORMATION_ADAPTATION_VERSION,
    opponent_information_adaptation_audit_is_valid,
)
from src.match_engine.world_model.opponent_information_query_outcomes import (
    opponent_information_query_score_is_valid,
)


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "opponent-information-adaptation-score:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def _query_score_fingerprint(score: dict[str, Any]) -> str:
    encoded = json.dumps(
        score, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "opponent-information-query-score:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def score_opponent_information_adaptation(
    adaptation_audit: dict[str, Any],
    query_audit: dict[str, Any],
    feedback: dict[str, Any],
    query_score: dict[str, Any],
) -> dict[str, Any] | None:
    if (
        not opponent_information_adaptation_audit_is_valid(
            adaptation_audit, query_audit, feedback,
        )
        or not opponent_information_query_score_is_valid(
            query_score, query_audit,
        )
    ):
        return None
    prior = adaptation_audit["prior_resolution"]
    kind = adaptation_audit["adaptation"]["adaptation_kind"]
    brier_delta = float(prior["brier_score"]) - float(
        query_score["selected_feature_brier_score"]
    )
    efficiency_delta = float(
        query_score["selected_query_objective_efficiency"]
    ) - float(prior["query_objective_efficiency"])
    regret_delta = float(prior["observed_branch_policy_regret"]) - float(
        query_score["observed_branch_policy_regret"]
    )
    if kind == "repair_contingent_policy":
        target_metric = "observed_branch_policy_regret_reduction"
        target_improvement = regret_delta
    elif (
        kind == "change_query_feature"
        and "query_selection_inefficient" in prior["attention_flags"]
    ):
        target_metric = "query_objective_efficiency_gain"
        target_improvement = efficiency_delta
    elif kind in {"change_query_feature", "change_query_horizon"}:
        target_metric = "selected_feature_brier_reduction"
        target_improvement = brier_delta
    else:
        target_metric = "retained_query_non_degradation"
        target_improvement = min(
            brier_delta, efficiency_delta, regret_delta,
        )
    successful = bool(
        adaptation_audit["model_checked_consistent"]
        and (
            target_improvement >= -1e-12
            if kind == "retain_validated_query"
            else target_improvement > 1e-6
        )
    )
    payload = {
        "version": OPPONENT_INFORMATION_ADAPTATION_VERSION,
        "adaptation_audit_digest": adaptation_audit["audit_digest"],
        "current_query_audit_digest": query_audit["audit_digest"],
        "current_query_score_fingerprint": _query_score_fingerprint(
            query_score
        ),
        "feedback_digest": feedback["feedback_digest"],
        "adaptation_kind": kind,
        "model_checked_consistent": bool(
            adaptation_audit["model_checked_consistent"]
        ),
        "prior_brier_score": float(prior["brier_score"]),
        "current_brier_score": float(
            query_score["selected_feature_brier_score"]
        ),
        "brier_reduction": brier_delta,
        "prior_query_objective_efficiency": float(
            prior["query_objective_efficiency"]
        ),
        "current_query_objective_efficiency": float(
            query_score["selected_query_objective_efficiency"]
        ),
        "query_objective_efficiency_gain": efficiency_delta,
        "prior_observed_branch_policy_regret": float(
            prior["observed_branch_policy_regret"]
        ),
        "current_observed_branch_policy_regret": float(
            query_score["observed_branch_policy_regret"]
        ),
        "observed_branch_policy_regret_reduction": regret_delta,
        "adaptation_target_metric": target_metric,
        "adaptation_target_improvement": float(target_improvement),
        "adaptation_improved": successful,
        "paired_prior_and_current_observational_queries": True,
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "causal_interpretation": False,
        "counterfactual_outcome_observed": False,
    }
    return {**payload, "score_digest": _digest(payload)}


def opponent_information_adaptation_score_is_valid(
    score: Any,
    adaptation_audit: dict[str, Any],
    query_audit: dict[str, Any],
    feedback: dict[str, Any],
    query_score: dict[str, Any],
) -> bool:
    if not isinstance(score, dict):
        return False
    expected = score_opponent_information_adaptation(
        adaptation_audit, query_audit, feedback, query_score,
    )
    return bool(expected is not None and score == expected)
