"""Machine-owned sequential stopping for pre-registered active probes.

The LLM may explain or choose among exact engine-authorized options. It cannot
move evidence boundaries, reopen stopped probe families, or alter execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from src.match_engine.world_model.active_probe import active_probe_design_is_valid
from src.match_engine.world_model.active_probe_memory import (
    SEQUENTIAL_LOG_EVIDENCE_BOUNDARY,
    SEQUENTIAL_MAXIMUM_MATCHES,
)
from src.match_engine.world_model.active_probe_portfolio import (
    active_probe_portfolio_audit_is_valid,
    active_probe_portfolio_design_is_valid,
    evaluate_llm_active_probe_portfolio,
)


ACTIVE_PROBE_SEQUENTIAL_POLICY_VERSION = 1
_CONTINUING = {"start", "continue"}
_TERMINAL = {
    "stop_supported",
    "stop_falsified",
    "stop_inconclusive_maximum_matches",
}


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _evidence(probe: dict[str, Any]) -> dict[str, Any]:
    memory = probe["discovery_memory"]
    return {
        "probe_id": str(probe["probe_id"]),
        "horizon": str(probe["horizon"]),
        "matches": int(memory["profile_matches"]),
        "cumulative_log_likelihood_ratio": float(
            memory["cumulative_log_likelihood_ratio"]
        ),
        "sequential_status": str(memory["sequential_status"]),
    }


def _option(
    *, decision: str, reason: str, evidence: list[dict[str, Any]],
    portfolio: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "version": ACTIVE_PROBE_SEQUENTIAL_POLICY_VERSION,
        "decision": decision,
        "reason": reason,
        "portfolio_id": (
            str(portfolio["portfolio_id"]) if portfolio else "none"
        ),
        "probe_ids": list(portfolio["probe_ids"]) if portfolio else [],
        "evidence": evidence,
        "positive_log_evidence_boundary": SEQUENTIAL_LOG_EVIDENCE_BOUNDARY,
        "negative_log_evidence_boundary": -SEQUENTIAL_LOG_EVIDENCE_BOUNDARY,
        "maximum_matches": SEQUENTIAL_MAXIMUM_MATCHES,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "policy_option_id": _digest(payload, "active-probe-sequential-option:"),
    }


def build_active_probe_sequential_policy_design(
    packet: dict[str, Any],
) -> dict[str, Any]:
    probe_design = packet.get("active_probe_design") or {}
    portfolio_design = packet.get("active_probe_portfolio_design") or {}
    probes = {
        probe["probe_id"]: probe for probe in probe_design.get("options") or []
    }
    valid_sources = bool(
        active_probe_design_is_valid(probe_design)
        and active_probe_portfolio_design_is_valid(
            portfolio_design, probe_design,
        )
    )
    evidence = [_evidence(probe) for probe in probes.values()] if valid_sources else []
    statuses = {row["sequential_status"] for row in evidence}
    conflict = bool({"stop_supported", "stop_falsified"} <= statuses)
    options: list[dict[str, Any]] = []
    if conflict:
        options.append(_option(
            decision="stop", reason="conflicting_horizon_boundaries",
            evidence=evidence,
        ))
    elif valid_sources:
        for portfolio in portfolio_design.get("options") or []:
            selected = [probes[probe_id] for probe_id in portfolio["probe_ids"]]
            if all(
                probe["discovery_memory"]["sequential_status"] in _CONTINUING
                for probe in selected
            ):
                options.append(_option(
                    decision="continue", reason="within_evidence_and_match_bounds",
                    evidence=[_evidence(probe) for probe in selected],
                    portfolio=portfolio,
                ))
        if not options and evidence:
            terminal = sorted(statuses & _TERMINAL)
            reason = (
                terminal[0] if len(terminal) == 1
                else "all_probe_families_terminal"
            )
            options.append(_option(
                decision="stop", reason=reason, evidence=evidence,
            ))
    options.sort(key=lambda row: (
        row["decision"] != "continue",
        -next((
            float(portfolio["joint_information_value"])
            for portfolio in portfolio_design.get("options") or []
            if portfolio["portfolio_id"] == row["portfolio_id"]
        ), 0.0),
        row["policy_option_id"],
    ))
    payload = {
        "version": ACTIVE_PROBE_SEQUENTIAL_POLICY_VERSION,
        "available": bool(options),
        "reason": (
            "conflict_review_required" if conflict else
            "sequential_options_available" if options else
            "active_probe_sources_unavailable"
        ),
        "team_id": str(packet.get("team_id", "")),
        "checkpoint_signature": str(packet.get("checkpoint_signature", "")),
        "environment_signature": str(packet.get("environment_signature", "")),
        "source_probe_design_digest": str(probe_design.get("design_digest", "")),
        "source_portfolio_design_digest": str(
            portfolio_design.get("design_digest", "")
        ),
        "options": options,
        "recommended_policy_option_id": (
            options[0]["policy_option_id"] if options else "none"
        ),
        "conflict_review_required": conflict,
        "evidence_basis": "raw_pre_registered_bernoulli_likelihood_ratio",
        "selection_scope": "post_action_engine_authorized_option_only",
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "design_digest": _digest(payload, "active-probe-sequential-design:"),
    }


def active_probe_sequential_policy_design_is_valid(
    design: Any, probe_design: dict[str, Any] | None = None,
    portfolio_design: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(design, dict):
        return False
    payload = dict(design)
    digest = payload.pop("design_digest", None)
    options = design.get("options") or []
    try:
        basic = bool(
            digest == _digest(payload, "active-probe-sequential-design:")
            and design.get("version") == ACTIVE_PROBE_SEQUENTIAL_POLICY_VERSION
            and bool(design.get("available")) == bool(options)
            and design.get("recommended_policy_option_id")
            == (options[0]["policy_option_id"] if options else "none")
            and len({row["policy_option_id"] for row in options}) == len(options)
            and all(
                row["policy_option_id"] == _digest(
                    {k: v for k, v in row.items() if k != "policy_option_id"},
                    "active-probe-sequential-option:",
                )
                and row["decision"] in {"continue", "stop"}
                and row["can_change_current_action"] is False
                and row["can_change_tactical_controls"] is False
                and row["can_schedule_future_action"] is False
                and row["causal_interpretation"] is False
                for row in options
            )
            and design.get("evidence_basis")
            == "raw_pre_registered_bernoulli_likelihood_ratio"
            and design.get("can_change_current_action") is False
            and design.get("can_change_tactical_controls") is False
            and design.get("can_schedule_future_action") is False
            and design.get("causal_interpretation") is False
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not basic:
        return False
    if probe_design is None or portfolio_design is None:
        return True
    rebuilt = build_active_probe_sequential_policy_design({
        "team_id": design.get("team_id"),
        "checkpoint_signature": design.get("checkpoint_signature"),
        "environment_signature": design.get("environment_signature"),
        "active_probe_design": probe_design,
        "active_probe_portfolio_design": portfolio_design,
    })
    return rebuilt == design


def validate_llm_active_probe_sequential_policy(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    option_id = str(raw.get("policy_option_id", ""))[:96]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not option_id or not math.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
        return None
    return {
        "policy_option_id": option_id,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def evaluate_llm_active_probe_sequential_policy(
    packet: dict[str, Any], raw: Any, *, selected_action: str,
    decision_mode: str, selected_after_action_freeze: bool,
) -> dict[str, Any]:
    selection = validate_llm_active_probe_sequential_policy(raw)
    design = packet.get("active_probe_sequential_policy_design") or {}
    base = {
        "version": ACTIVE_PROBE_SEQUENTIAL_POLICY_VERSION,
        "accepted": False,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    if selection is None:
        return {**base, "reason": "missing_or_invalid_sequential_policy"}
    if not selected_after_action_freeze or not active_probe_sequential_policy_design_is_valid(
        design, packet.get("active_probe_design") or {},
        packet.get("active_probe_portfolio_design") or {},
    ):
        return {**base, "reason": "post_action_valid_sequential_policy_required"}
    option = next((row for row in design["options"] if row["policy_option_id"] == selection["policy_option_id"]), None)
    if option is None:
        return {**base, "reason": "unknown_sequential_policy_option"}
    portfolio_audit = None
    if option["decision"] == "continue":
        portfolio_audit = evaluate_llm_active_probe_portfolio(
            packet, {"portfolio_id": option["portfolio_id"],
                     "confidence": selection["confidence"],
                     "rationale": selection["rationale"]},
            selected_action=selected_action, decision_mode=decision_mode,
            selected_after_action_freeze=True,
        )
        if not active_probe_portfolio_audit_is_valid(portfolio_audit):
            return {**base, "reason": "authorized_portfolio_execution_failed"}
    payload = {
        **base, "accepted": True,
        "reason": "machine_owned_sequential_policy_applied",
        "selection": selection, "policy_option": option,
        "portfolio_audit": portfolio_audit,
        "source_design_digest": design["design_digest"],
        "team_id": design["team_id"],
        "checkpoint_signature": design["checkpoint_signature"],
        "environment_signature": design["environment_signature"],
        "selected_after_action_freeze": True,
    }
    return {**payload, "audit_digest": _digest(payload, "active-probe-sequential-audit:")}


def active_probe_sequential_policy_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    selection = validate_llm_active_probe_sequential_policy(audit.get("selection"))
    option = audit.get("policy_option") or {}
    portfolio_audit = audit.get("portfolio_audit")
    return bool(
        digest == _digest(payload, "active-probe-sequential-audit:")
        and selection is not None
        and selection["policy_option_id"] == option.get("policy_option_id")
        and ((option.get("decision") == "continue" and active_probe_portfolio_audit_is_valid(portfolio_audit))
             or (option.get("decision") == "stop" and portfolio_audit is None))
        and audit.get("selected_after_action_freeze") is True
        and audit.get("can_change_current_action") is False
        and audit.get("can_change_tactical_controls") is False
        and audit.get("can_schedule_future_action") is False
        and audit.get("causal_interpretation") is False
    )
