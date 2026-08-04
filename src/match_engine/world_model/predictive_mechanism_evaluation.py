"""Cross-match diagnostics for predictive mechanism tournaments."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.predictive_mechanism import (
    PREDICTIVE_MECHANISM_VERSION,
    predictive_mechanism_audit_is_valid,
    predictive_mechanism_evaluation_is_valid,
)


def predictive_mechanism_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    accepted = executed = scored = malformed_audits = malformed_scores = 0
    unsafe = matches = 0
    log_ratios = []
    brier_skills = []
    hypotheses = set()
    edges = set()
    for cluster in record_clusters:
        match_scores = 0
        for record in cluster:
            audit = record.get("llm_predictive_mechanism_context") or {}
            if not audit:
                continue
            if not predictive_mechanism_audit_is_valid(audit):
                malformed_audits += 1
                continue
            accepted += 1
            hypothesis = audit["hypothesis"]
            hypotheses.add(str(hypothesis["hypothesis_id"]))
            edges.add((
                str(hypothesis["driver_event"]),
                str(hypothesis["outcome_event"]),
            ))
            unsafe += int(
                audit.get("selected_after_action_freeze") is not True
                or audit.get("can_change_current_action") is not False
                or audit.get("can_change_tactical_controls") is not False
                or audit.get("can_schedule_future_action") is not False
                or audit.get("can_update_world_model") is not False
                or audit.get("causal_interpretation") is not False
            )
            action_executed = bool(
                str(record.get("intervention_actual_action", "")).lower()
                == str(hypothesis["action"])
            )
            executed += int(action_executed)
            if not action_executed:
                continue
            outcome = (
                record.get("multi_horizon_regime_outcomes") or {}
            ).get(str(hypothesis["horizon"])) or {}
            evaluation = outcome.get(
                "llm_predictive_mechanism_evaluation"
            ) or {}
            if not evaluation:
                continue
            if not predictive_mechanism_evaluation_is_valid(
                evaluation, audit,
            ):
                malformed_scores += 1
                continue
            try:
                log_ratio = float(evaluation[
                    "categorical_log_likelihood_ratio_vs_independence"
                ])
                brier_skill = float(evaluation[
                    "brier_skill_vs_independence"
                ])
            except (KeyError, TypeError, ValueError, OverflowError):
                malformed_scores += 1
                continue
            if not np.isfinite(log_ratio) or not np.isfinite(brier_skill):
                malformed_scores += 1
                continue
            scored += 1
            match_scores += 1
            log_ratios.append(log_ratio)
            brier_skills.append(brier_skill)
        matches += int(match_scores > 0)
    return {
        "version": PREDICTIVE_MECHANISM_VERSION,
        "evaluation_kind": "member_grounded_predictive_mechanism_tournament",
        "accepted_hypothesis_audits": accepted,
        "executed_hypothesis_actions": executed,
        "scored_joint_outcomes": scored,
        "distinct_hypotheses": len(hypotheses),
        "distinct_mechanism_edges": len(edges),
        "matches": matches,
        "malformed_hypothesis_audits": malformed_audits,
        "malformed_joint_scores": malformed_scores,
        "mean_log_likelihood_ratio_vs_independence": (
            float(np.mean(log_ratios)) if log_ratios else 0.0
        ),
        "mean_brier_skill_vs_independence": (
            float(np.mean(brier_skills)) if brier_skills else 0.0
        ),
        "positive_likelihood_evidence_rate": (
            float(np.mean(np.asarray(log_ratios) > 0.0))
            if log_ratios else 0.0
        ),
        "unsafe_action_tactical_or_learning_authority_claims": unsafe,
        "all_hypotheses_post_action_association_only": bool(
            accepted > 0 and unsafe == 0
        ),
        "interpretation": (
            "Mechanisms are predictive associations from aligned ensemble "
            "members and realized joint events, not identified causal paths."
        ),
    }
