"""Cross-match diagnostics for feedback-grounded query adaptation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.opponent_information_adaptation import (
    OPPONENT_INFORMATION_ADAPTATION_VERSION,
    opponent_information_adaptation_audit_is_valid,
)
from src.match_engine.world_model.opponent_information_adaptation_outcomes import (
    opponent_information_adaptation_score_is_valid,
)
from src.match_engine.world_model.opponent_information_feedback import (
    opponent_information_feedback_is_valid,
)


def opponent_information_adaptation_diagnostics(
    match_logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: list[list[dict[str, Any]]] = []
    accepted = consistent = eligible = missing = 0
    malformed_audits = malformed_scores = 0
    signatures = set()
    consistency_rates = []
    feedback_opportunities = adaptation_declarations = 0
    adaptation_coverage_rates = []
    for payload in match_logs:
        match_rows = []
        match_accepted = match_consistent = 0
        match_opportunities = match_declarations = 0
        records = ((payload.get("world_model_decision_adoption") or {}).get(
            "records"
        ) or [])
        for record in records:
            query_audit = record.get(
                "llm_opponent_information_query_context"
            ) or {}
            feedback = record.get("opponent_information_feedback_context") or {}
            scope_and_time_compatible = bool(
                str(feedback.get("team_id")) == str(record.get("team_id"))
                and str(feedback.get("checkpoint_signature"))
                == str(record.get("checkpoint_signature"))
                and str(feedback.get("environment_signature"))
                == str(record.get("environment_signature"))
                and float(feedback.get("as_of_t_sec", 0.0))
                <= float(record.get("created_t_sec", 0.0)) + 1e-9
            )
            if (
                feedback.get("available")
                and opponent_information_feedback_is_valid(feedback)
                and scope_and_time_compatible
            ):
                feedback_opportunities += 1
                match_opportunities += 1
            audit = record.get("llm_opponent_information_adaptation_context") or {}
            if not audit:
                continue
            adaptation_declarations += 1
            match_declarations += 1
            if not opponent_information_adaptation_audit_is_valid(
                audit, query_audit, feedback,
            ) or not scope_and_time_compatible:
                malformed_audits += 1
                continue
            accepted += 1
            consistent += bool(audit["model_checked_consistent"])
            match_accepted += 1
            match_consistent += bool(audit["model_checked_consistent"])
            signatures.add(str(audit.get("adaptation_signature", "")))
            query = query_audit["query"]
            if query["selected_action"] != str(record.get(
                "intervention_actual_action", "",
            )).lower():
                continue
            outcome = ((record.get("multi_horizon_regime_outcomes") or {}).get(
                query["horizon"]
            ) or {})
            query_score = outcome.get(
                "llm_opponent_information_query_evaluation"
            ) or {}
            score = outcome.get(
                "llm_opponent_information_adaptation_evaluation"
            )
            eligible += 1
            if not isinstance(score, dict):
                missing += 1
            elif opponent_information_adaptation_score_is_valid(
                score, audit, query_audit, feedback, query_score,
            ):
                match_rows.append(score)
            else:
                malformed_scores += 1
        if match_rows:
            grouped.append(match_rows)
        if match_accepted:
            consistency_rates.append(match_consistent / match_accepted)
        if match_opportunities:
            adaptation_coverage_rates.append(
                min(1.0, match_declarations / match_opportunities)
            )
    valid = [row for rows in grouped for row in rows]

    def clustered(key: str) -> float:
        values = [
            float(np.mean([float(row[key]) for row in rows]))
            for rows in grouped
        ]
        return float(np.mean(values)) if values else 0.0

    def clustered_rate(key: str) -> float:
        values = [
            float(np.mean([bool(row[key]) for row in rows]))
            for rows in grouped
        ]
        return float(np.mean(values)) if values else 0.0

    unspecified = {"", "opponent-information-adaptation-unspecified"}
    return {
        "version": OPPONENT_INFORMATION_ADAPTATION_VERSION,
        "evaluation_kind": "paired_prior_and_current_query_adaptation",
        "accepted_adaptations": accepted,
        "feedback_adaptation_opportunities": feedback_opportunities,
        "adaptation_declarations": adaptation_declarations,
        "model_checked_consistent_adaptations": consistent,
        "eligible_realized_adaptations": eligible,
        "realized_adaptation_scores": len(valid),
        "matches": len(grouped),
        "missing_scores": missing,
        "malformed_adaptation_audits": malformed_audits,
        "malformed_adaptation_scores": malformed_scores,
        "match_clustered_model_consistency_rate": (
            float(np.mean(consistency_rates)) if consistency_rates else 0.0
        ),
        "match_clustered_feedback_adaptation_rate": (
            float(np.mean(adaptation_coverage_rates))
            if adaptation_coverage_rates else 0.0
        ),
        "match_clustered_adaptation_improvement_rate": clustered_rate(
            "adaptation_improved"
        ),
        "match_clustered_target_improvement": clustered(
            "adaptation_target_improvement"
        ),
        "match_clustered_brier_reduction": clustered("brier_reduction"),
        "match_clustered_query_efficiency_gain": clustered(
            "query_objective_efficiency_gain"
        ),
        "match_clustered_branch_regret_reduction": clustered(
            "observed_branch_policy_regret_reduction"
        ),
        "all_paired_prior_and_current_observational_queries": all(
            row.get("paired_prior_and_current_observational_queries")
            for row in valid
        ),
        "all_shadow_only": all(row.get("shadow_only") for row in valid),
        "all_non_controlling": all(
            not row.get("authority_active") and not row.get("policy_mutated")
            for row in valid
        ),
        "all_non_causal": all(
            not row.get("causal_interpretation")
            and not row.get("counterfactual_outcome_observed")
            for row in valid
        ),
        "adaptation_signatures": sorted(signatures),
        "provenance_compatible": bool(
            len(signatures) == 1
            and next(iter(signatures), "") not in unspecified
        ),
    }
