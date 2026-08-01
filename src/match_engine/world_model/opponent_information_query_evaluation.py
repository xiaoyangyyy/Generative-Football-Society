"""Cross-match scoring for LLM-selected opponent information queries."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.opponent_information_query import (
    OPPONENT_INFORMATION_QUERY_VERSION,
    opponent_information_query_audit_is_valid,
)
from src.match_engine.world_model.opponent_information_query_outcomes import (
    opponent_information_query_score_is_valid,
)


def opponent_information_query_diagnostics(
    match_logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: list[list[dict[str, Any]]] = []
    accepted = eligible = missing = 0
    malformed_contexts = malformed_scores = 0
    for payload in match_logs:
        match_rows = []
        records = (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        )
        for record in records:
            context = record.get(
                "llm_opponent_information_query_context"
            ) or {}
            if context and not opponent_information_query_audit_is_valid(
                context
            ):
                malformed_contexts += 1
                continue
            if not context.get("accepted"):
                continue
            accepted += 1
            query = context["query"]
            if query["selected_action"] != str(record.get(
                "intervention_actual_action", "",
            )).lower():
                continue
            outcome = (
                record.get("multi_horizon_regime_outcomes") or {}
            ).get(query["horizon"])
            if not isinstance(outcome, dict):
                continue
            eligible += 1
            score = outcome.get(
                "llm_opponent_information_query_evaluation"
            )
            if not isinstance(score, dict):
                missing += 1
            elif opponent_information_query_score_is_valid(score, context):
                match_rows.append(score)
            else:
                malformed_scores += 1
        if match_rows:
            grouped.append(match_rows)
    valid = [row for rows in grouped for row in rows]

    def clustered(key: str) -> float:
        values = [
            float(np.mean([float(row[key]) for row in rows]))
            for rows in grouped
        ]
        return float(np.mean(values)) if values else 0.0

    def clustered_rate(predicate) -> float:
        values = [
            float(np.mean([bool(predicate(row)) for row in rows]))
            for rows in grouped
        ]
        return float(np.mean(values)) if values else 0.0

    scopes = sorted({(
        str(row.get("checkpoint_signature", "")),
        str(row.get("environment_signature", "")),
    ) for row in valid})
    unspecified = {
        "", "runtime_unspecified", "environment_unspecified",
        "opponent-information-query-unspecified",
    }
    signatures = sorted({str(row.get("query_signature", "")) for row in valid})
    provenance_compatible = bool(
        len(scopes) == 1
        and all(value not in unspecified for value in scopes[0])
        and len(signatures) == 1 and signatures[0] not in unspecified
        and all(
            row.get("issuance_evaluation_provenance_compatible")
            for row in valid
        )
    )
    return {
        "version": OPPONENT_INFORMATION_QUERY_VERSION,
        "evaluation_kind": "paired_opponent_tactical_query_brier",
        "accepted_queries": accepted,
        "eligible_realized_queries": eligible,
        "realized_query_scores": len(valid),
        "matches": len(grouped),
        "missing_scores": missing,
        "malformed_query_contexts": malformed_contexts,
        "malformed_query_scores": malformed_scores,
        "match_clustered_selected_feature_brier": clustered(
            "selected_feature_brier_score"
        ),
        "match_clustered_all_feature_mean_brier": clustered(
            "all_feature_mean_brier_score"
        ),
        "match_clustered_raw_selected_feature_brier": clustered(
            "raw_selected_feature_brier_score"
        ),
        "match_clustered_raw_all_feature_mean_brier": clustered(
            "raw_all_feature_mean_brier_score"
        ),
        "match_clustered_calibration_skill_vs_raw": (
            clustered("raw_all_feature_mean_brier_score")
            - clustered("all_feature_mean_brier_score")
        ),
        "calibrated_query_scores": sum(
            bool(row.get("any_query_calibration_applied")) for row in valid
        ),
        "calibration_no_worse_than_raw": bool(
            not any(row.get("any_query_calibration_applied") for row in valid)
            or clustered("all_feature_mean_brier_score")
            <= clustered("raw_all_feature_mean_brier_score") + 1e-12
        ),
        "uninformative_half_probability_brier": 0.25,
        "match_clustered_model_top_query_rate": clustered_rate(
            lambda row: int(row["selected_query_model_rank"]) == 1
        ),
        "match_clustered_query_objective_efficiency": clustered(
            "selected_query_objective_efficiency"
        ),
        "match_clustered_supported_query_purpose_rate": clustered_rate(
            lambda row: row["purpose_supported"]
        ),
        "match_clustered_contingent_policy_consistency_rate": clustered_rate(
            lambda row: row["contingent_policy_model_checked_consistent"]
        ),
        "match_clustered_observed_branch_action_alignment_rate": clustered_rate(
            lambda row: row["observed_branch_action_aligned"]
        ),
        "match_clustered_observed_branch_policy_regret": clustered(
            "observed_branch_policy_regret"
        ),
        "match_clustered_expected_contingent_policy_regret": clustered(
            "expected_contingent_policy_regret"
        ),
        "all_paired_same_action_horizon": all(
            row.get("paired_same_action_horizon") for row in valid
        ),
        "all_shadow_only": all(row.get("shadow_only") for row in valid),
        "all_non_controlling": all(
            not row.get("authority_active") and not row.get("policy_mutated")
            for row in valid
        ),
        "all_non_causal": all(
            not row.get("causal_interpretation") for row in valid
        ),
        "all_hidden_intent_claims_disabled": all(
            not row.get("hidden_opponent_intent_observed") for row in valid
        ),
        "provenance_compatible": provenance_compatible,
        "model_policy_scopes": [list(scope) for scope in scopes],
        "query_signatures": signatures,
    }
