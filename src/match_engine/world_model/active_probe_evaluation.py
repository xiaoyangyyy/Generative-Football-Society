"""Cross-match diagnostics for pre-registered active world-model probes."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.active_probe import (
    ACTIVE_PROBE_VERSION,
    active_probe_audit_is_valid,
    active_probe_evaluation_is_valid,
)
from src.match_engine.world_model.active_probe_portfolio import (
    active_probe_portfolio_audit_is_valid,
)


def active_probe_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    accepted = malformed = executed = scored = unsafe = 0
    malformed_evaluations = 0
    falsification_signals = 0
    brier_skills = []
    log_ratios = []
    matches = 0
    for cluster in record_clusters:
        match_scored = 0
        for record in cluster:
            audit_rows = []
            single_audit = record.get("llm_active_probe_context") or {}
            if single_audit:
                audit_rows.append((
                    single_audit, "llm_active_probe_evaluation",
                ))
            portfolio_audit = record.get(
                "llm_active_probe_portfolio_context"
            ) or {}
            if portfolio_audit:
                if not active_probe_portfolio_audit_is_valid(portfolio_audit):
                    malformed += 1
                else:
                    audit_rows.extend((
                        audit, "llm_active_probe_portfolio_evaluation"
                    ) for audit in portfolio_audit["probe_audits"])
            for audit, evaluation_key in audit_rows:
                if not active_probe_audit_is_valid(audit):
                    malformed += 1
                    continue
                accepted += 1
                unsafe += int(
                    audit.get("selected_after_action_freeze") is not True
                    or audit.get("can_change_current_action") is not False
                    or audit.get("can_change_tactical_controls") is not False
                    or audit.get("can_schedule_future_action") is not False
                    or audit.get("causal_interpretation") is not False
                )
                probe = audit["probe"]
                action_executed = bool(
                    str(record.get("intervention_actual_action", ""))
                    == str(probe["action"])
                )
                executed += int(action_executed)
                if not action_executed:
                    continue
                outcome = (
                    record.get("multi_horizon_regime_outcomes") or {}
                ).get(str(probe["horizon"])) or {}
                evaluation = outcome.get(evaluation_key) or {}
                if not evaluation:
                    continue
                if not active_probe_evaluation_is_valid(evaluation, audit):
                    malformed_evaluations += 1
                    continue
                try:
                    skill = float(evaluation["brier_skill_vs_null"])
                    log_ratio = float(
                        evaluation["log_likelihood_ratio_vs_null"]
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if not all(math.isfinite(value) for value in (
                    skill, log_ratio,
                )):
                    continue
                scored += 1
                match_scored += 1
                brier_skills.append(skill)
                log_ratios.append(log_ratio)
                falsification_signals += int(bool(
                    evaluation.get("falsification_signal")
                ))
        matches += int(match_scored > 0)
    return {
        "version": ACTIVE_PROBE_VERSION,
        "evaluation_kind": "pre_registered_active_probe_cross_match",
        "accepted_probe_audits": accepted,
        "malformed_probe_audits": malformed,
        "malformed_probe_evaluations": malformed_evaluations,
        "executed_probe_actions": executed,
        "scored_probe_outcomes": scored,
        "matches": matches,
        "falsification_signals": falsification_signals,
        "mean_brier_skill_vs_exploit_null": (
            float(np.mean(brier_skills)) if brier_skills else 0.0
        ),
        "mean_log_likelihood_ratio_vs_exploit_null": (
            float(np.mean(log_ratios)) if log_ratios else 0.0
        ),
        "positive_brier_skill_rate": (
            sum(value > 0.0 for value in brier_skills)
            / max(1, len(brier_skills))
        ),
        "unsafe_action_or_tactical_authority_claims": unsafe,
        "all_probes_post_action_non_controlling": bool(
            accepted > 0 and unsafe == 0
        ),
        "interpretation": (
            "Probe outcomes compare a pre-registered conditional forecast "
            "with the exploit forecast as a predictive null; this is not a "
            "causal treatment-effect estimate."
        ),
    }
