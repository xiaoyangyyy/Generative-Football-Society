"""Cross-match diagnostics for bounded active-probe portfolios."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.active_probe import (
    active_probe_evaluation_is_valid,
)
from src.match_engine.world_model.active_probe_portfolio import (
    ACTIVE_PROBE_PORTFOLIO_VERSION,
    active_probe_portfolio_audit_is_valid,
)


def active_probe_portfolio_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    accepted = malformed = executed = completed = 0
    multi_horizon = scored_probes = unsafe = malformed_scores = 0
    completion_rates = []
    joint_values = []
    matches = 0
    for cluster in record_clusters:
        match_completed = 0
        for record in cluster:
            audit = record.get("llm_active_probe_portfolio_context") or {}
            if not audit:
                continue
            if not active_probe_portfolio_audit_is_valid(audit):
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
            portfolio = audit["portfolio"]
            action_executed = bool(
                str(record.get("intervention_actual_action", ""))
                == str(portfolio["action"])
            )
            executed += int(action_executed)
            if not action_executed:
                continue
            scored = 0
            outcomes = record.get("multi_horizon_regime_outcomes") or {}
            for probe_audit in audit["probe_audits"]:
                probe = probe_audit["probe"]
                outcome = outcomes.get(str(probe["horizon"])) or {}
                evaluation = outcome.get(
                    "llm_active_probe_portfolio_evaluation"
                ) or {}
                if not evaluation:
                    continue
                if not active_probe_evaluation_is_valid(
                    evaluation, probe_audit,
                ):
                    malformed_scores += 1
                    continue
                scored += 1
                scored_probes += 1
            expected = int(portfolio["observation_count"])
            rate = scored / max(1, expected)
            completion_rates.append(rate)
            is_complete = scored == expected
            completed += int(is_complete)
            match_completed += int(is_complete)
            multi_horizon += int(is_complete and expected >= 2)
            if is_complete:
                joint_values.append(float(
                    portfolio["joint_information_value"]
                ))
        matches += int(match_completed > 0)
    return {
        "version": ACTIVE_PROBE_PORTFOLIO_VERSION,
        "evaluation_kind": "bounded_multi_horizon_active_probe_portfolio",
        "accepted_portfolio_audits": accepted,
        "malformed_portfolio_audits": malformed,
        "executed_portfolio_actions": executed,
        "completed_portfolios": completed,
        "completed_multi_horizon_portfolios": multi_horizon,
        "scored_portfolio_probes": scored_probes,
        "malformed_portfolio_scores": malformed_scores,
        "matches": matches,
        "mean_observation_completion_rate": (
            float(np.mean(completion_rates)) if completion_rates else 0.0
        ),
        "mean_completed_joint_information_value": (
            float(np.mean(joint_values)) if joint_values else 0.0
        ),
        "unsafe_action_or_tactical_authority_claims": unsafe,
        "all_portfolios_single_action_non_controlling": bool(
            accepted > 0 and unsafe == 0
        ),
        "interpretation": (
            "Multiple horizons observe one realized exploration action; "
            "redundancy-adjusted information is prospective and horizons "
            "are not treated as independent causal interventions."
        ),
    }
