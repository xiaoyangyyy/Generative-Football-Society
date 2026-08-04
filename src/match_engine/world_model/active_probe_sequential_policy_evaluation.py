"""Cross-match diagnostics for machine-owned active-probe stopping."""

from __future__ import annotations

from typing import Any, Iterable

from src.match_engine.world_model.active_probe_sequential_policy import (
    ACTIVE_PROBE_SEQUENTIAL_POLICY_VERSION,
    active_probe_sequential_policy_audit_is_valid,
)


def _boundary_compliant(option: dict[str, Any]) -> bool:
    evidence = option.get("evidence") or []
    statuses = {row.get("sequential_status") for row in evidence}
    if option.get("decision") == "continue":
        return bool(evidence) and statuses <= {"start", "continue"}
    reason = option.get("reason")
    if reason == "conflicting_horizon_boundaries":
        return {"stop_supported", "stop_falsified"} <= statuses
    if reason in {
        "stop_supported", "stop_falsified",
        "stop_inconclusive_maximum_matches",
    }:
        return reason in statuses
    return bool(statuses) and statuses <= {
        "stop_supported", "stop_falsified",
        "stop_inconclusive_maximum_matches",
    }


def active_probe_sequential_policy_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    accepted = malformed = continues = stops = conflicts = violations = 0
    unsafe = matches = 0
    for cluster in record_clusters:
        match_audits = 0
        for record in cluster:
            audit = record.get(
                "llm_active_probe_sequential_policy_context"
            ) or {}
            if not audit:
                continue
            if not active_probe_sequential_policy_audit_is_valid(audit):
                malformed += 1
                continue
            accepted += 1
            match_audits += 1
            option = audit["policy_option"]
            decision = option["decision"]
            continues += int(decision == "continue")
            stops += int(decision == "stop")
            conflicts += int(
                option["reason"] == "conflicting_horizon_boundaries"
            )
            violations += int(not _boundary_compliant(option))
            unsafe += int(
                audit.get("selected_after_action_freeze") is not True
                or audit.get("can_change_current_action") is not False
                or audit.get("can_change_tactical_controls") is not False
                or audit.get("can_schedule_future_action") is not False
                or audit.get("causal_interpretation") is not False
            )
        matches += int(match_audits > 0)
    return {
        "version": ACTIVE_PROBE_SEQUENTIAL_POLICY_VERSION,
        "evaluation_kind": "machine_owned_active_probe_sequential_policy",
        "accepted_policy_audits": accepted,
        "continue_decisions": continues,
        "machine_stop_decisions": stops,
        "conflict_review_stops": conflicts,
        "matches": matches,
        "malformed_policy_audits": malformed,
        "boundary_compliance_violations": violations,
        "unsafe_action_or_tactical_authority_claims": unsafe,
        "all_decisions_post_action_non_controlling": bool(
            accepted > 0 and unsafe == 0
        ),
        "interpretation": (
            "Stopping uses match-clustered raw pre-registered predictive "
            "likelihood evidence. It is not a causal claim, and optional "
            "stopping guarantees require the forecast protocol assumptions."
        ),
    }
