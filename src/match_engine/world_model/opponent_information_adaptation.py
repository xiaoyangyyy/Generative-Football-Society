"""Model-check how an LLM adapts after resolved information-query feedback."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.match_engine.world_model.opponent_information_feedback import (
    opponent_information_feedback_is_valid,
)
from src.match_engine.world_model.opponent_information_query import (
    opponent_information_query_audit_is_valid,
)


OPPONENT_INFORMATION_ADAPTATION_VERSION = 1
ADAPTATION_KINDS = {
    "change_query_feature",
    "change_query_horizon",
    "repair_contingent_policy",
    "retain_validated_query",
}


def llm_opponent_information_adaptation_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": OPPONENT_INFORMATION_ADAPTATION_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "feedback_grounded_opponent_information_adaptation",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-opponent-information-adaptation:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_opponent_information_adaptation(
    raw: Any,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    prior = str(raw.get("prior_decision_id", ""))[:160]
    kind = str(raw.get("adaptation_kind", "")).strip().lower()
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not prior or kind not in ADAPTATION_KINDS or not 0.5 <= confidence <= 1.0:
        return None
    return {
        "prior_decision_id": prior,
        "adaptation_kind": kind,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "opponent-information-adaptation-audit:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def _supported_kinds(prior: dict[str, Any]) -> list[str]:
    flags = set(prior.get("attention_flags") or [])
    supported = set()
    if "query_selection_inefficient" in flags:
        supported.add("change_query_feature")
    if "forecast_surprise" in flags:
        supported.update({"change_query_feature", "change_query_horizon"})
    if "contingent_action_misaligned" in flags:
        supported.add("repair_contingent_policy")
    if not flags:
        supported.add("retain_validated_query")
    return sorted(supported)


def _audit_from_evidence(
    adaptation: dict[str, Any],
    feedback: dict[str, Any],
    query_audit: dict[str, Any],
    *,
    adaptation_signature: str,
) -> dict[str, Any] | None:
    if (
        not opponent_information_feedback_is_valid(feedback)
        or not feedback.get("available")
        or not opponent_information_query_audit_is_valid(query_audit)
    ):
        return None
    prior = next((
        row for row in feedback.get("recent_resolutions") or []
        if str(row.get("decision_id")) == adaptation["prior_decision_id"]
    ), None)
    if prior is None:
        return None
    current_query = query_audit["query"]
    current_policy = query_audit["contingent_policy"]
    kind = adaptation["adaptation_kind"]
    supported = _supported_kinds(prior)
    structural_change = False
    if kind == "change_query_feature":
        structural_change = bool(
            current_query["feature"] != prior["feature"]
            and current_query["horizon"] == prior["horizon"]
            and current_query["purpose"] == prior["purpose"]
        )
    elif kind == "change_query_horizon":
        structural_change = bool(
            current_query["horizon"] != prior["horizon"]
            and current_query["feature"] == prior["feature"]
            and current_query["purpose"] == prior["purpose"]
        )
    elif kind == "repair_contingent_policy":
        branch = prior["resolved_branch"]
        structural_change = bool(
            current_query["feature"] == prior["feature"]
            and current_query["horizon"] == prior["horizon"]
            and current_query["purpose"] == prior["purpose"]
            and current_query[f"action_if_{branch}"]
            != prior["proposed_continuation_action"]
            and current_policy["model_checked_consistent"]
        )
    elif kind == "retain_validated_query":
        resolved_action = current_query[
            f"action_if_{prior['resolved_branch']}"
        ]
        structural_change = bool(
            current_query["feature"] == prior["feature"]
            and current_query["horizon"] == prior["horizon"]
            and current_query["purpose"] == prior["purpose"]
            and resolved_action == prior.get("proposed_continuation_action")
            and current_policy["model_checked_consistent"]
        )
    evidence_supported = kind in supported
    model_checked_consistent = bool(evidence_supported and structural_change)
    prior_snapshot = {
        key: prior[key] for key in (
            "decision_id", "feature", "horizon", "purpose", "resolved_branch",
            "brier_score", "query_objective_efficiency",
            "observed_branch_policy_regret", "proposed_continuation_action",
            "model_best_continuation_action", "attention_flags",
        )
    }
    payload = {
        "version": OPPONENT_INFORMATION_ADAPTATION_VERSION,
        "accepted": True,
        "reason": "model_checked_feedback_adaptation",
        "adaptation": adaptation,
        "feedback_digest": feedback["feedback_digest"],
        "prior_resolution": prior_snapshot,
        "current_query_audit_digest": query_audit["audit_digest"],
        "current_query": dict(current_query),
        "supported_adaptation_kinds": supported,
        "evidence_supported": evidence_supported,
        "structural_change_verified": structural_change,
        "model_checked_consistent": model_checked_consistent,
        "adaptation_signature": str(adaptation_signature),
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "can_change_current_action": False,
        "can_update_world_model_weights": False,
        "causal_interpretation": False,
    }
    return {**payload, "audit_digest": _digest(payload)}


def evaluate_llm_opponent_information_adaptation(
    packet: dict[str, Any],
    raw_adaptation: Any,
    query_audit: dict[str, Any],
    *,
    adaptation_signature: str = "opponent-information-adaptation-unspecified",
) -> dict[str, Any]:
    adaptation = validate_llm_opponent_information_adaptation(raw_adaptation)
    base = {
        "version": OPPONENT_INFORMATION_ADAPTATION_VERSION,
        "accepted": False,
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "can_change_current_action": False,
        "can_update_world_model_weights": False,
        "causal_interpretation": False,
        "adaptation_signature": str(adaptation_signature),
    }
    if adaptation is None:
        return {**base, "reason": "missing_or_invalid_feedback_adaptation"}
    feedback = packet.get("opponent_information_feedback") or {}
    audit = _audit_from_evidence(
        adaptation, feedback, query_audit,
        adaptation_signature=adaptation_signature,
    )
    return audit or {**base, "reason": "compatible_feedback_and_query_required"}


def opponent_information_adaptation_audit_is_valid(
    audit: Any,
    query_audit: dict[str, Any],
    feedback: dict[str, Any],
) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    adaptation = validate_llm_opponent_information_adaptation(
        audit.get("adaptation")
    )
    if adaptation is None:
        return False
    expected = _audit_from_evidence(
        adaptation, feedback, query_audit,
        adaptation_signature=str(audit.get("adaptation_signature", "")),
    )
    return bool(expected is not None and audit == expected)
