"""Live contract for falsifiable LLM critiques of world-model forecasts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS


LLM_CRITIC_VERSION = 1
CRITIQUE_DIRECTIONS = ("underestimate", "overestimate", "neutral")
CRITIQUE_EVIDENCE_FEATURES = frozenset({
    "zone",
    "score_state",
    "match_phase",
    "opponent_belief",
    "opponent_response",
    "trajectory_rollout",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
    "tactical_shape",
})
_HORIZON_PATTERN = re.compile(r"^(transition|\d+(?:\.\d+)?s)$")


def llm_critic_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": LLM_CRITIC_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "falsifiable_world_model_policy_utility_residual",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-critic:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def validate_llm_world_model_critique(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action", "")).strip().lower()
    horizon = str(raw.get("horizon", "")).strip().lower()
    direction = str(raw.get("direction", "")).strip().lower()
    if (
        action not in RESPONSE_ACTIONS
        or not _HORIZON_PATTERN.fullmatch(horizon)
        or direction not in CRITIQUE_DIRECTIONS
    ):
        return None
    try:
        confidence = float(np.clip(float(raw.get("confidence", 0.0)), 0.0, 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = raw.get("evidence_features") or []
    if not isinstance(evidence, list):
        evidence = []
    evidence = list(dict.fromkeys(
        str(item).strip().lower()
        for item in evidence
        if str(item).strip().lower() in CRITIQUE_EVIDENCE_FEATURES
    ))[:5]
    if not evidence:
        return None
    return {
        "action": action,
        "horizon": horizon,
        "direction": direction,
        "confidence": confidence,
        "evidence_features": evidence,
        "rationale": str(raw.get("rationale", ""))[:280],
    }


def _proposal(direction: str, confidence: float) -> float:
    sign = 1.0 if direction == "underestimate" else (
        -1.0 if direction == "overestimate" else 0.0
    )
    return float(sign * 0.08 * np.clip(confidence, 0.0, 1.0))


def apply_llm_world_model_critique(
    packet: dict[str, Any],
    raw_critique: Any,
    memory: Any = None,
    *,
    selected_action: str,
    critic_signature: str = "critic-unspecified",
) -> dict[str, Any]:
    critique = validate_llm_world_model_critique(raw_critique)
    if critique is None:
        return {
            "version": LLM_CRITIC_VERSION,
            "accepted": False,
            "authority_active": False,
            "reason": "missing_or_invalid_world_model_critique",
        }
    if critique["action"] != str(selected_action).lower():
        return {
            "version": LLM_CRITIC_VERSION,
            "accepted": False,
            "authority_active": False,
            "reason": "critique_action_must_equal_selected_action",
            "critique": critique,
        }
    candidate = next((
        item for item in packet.get("candidates") or []
        if str(item.get("action")) == critique["action"]
    ), None)
    predictions = (
        candidate.get("multi_horizon_predictions") if candidate else None
    ) or {}
    if critique["horizon"] not in predictions:
        return {
            "version": LLM_CRITIC_VERSION,
            "accepted": False,
            "authority_active": False,
            "reason": "critique_horizon_not_evaluated",
            "critique": critique,
            "available_horizons": sorted(predictions),
        }
    decision_context = packet.get("decision_context") or {}
    available_evidence = {
        feature for feature in critique["evidence_features"]
        if (
            feature in {"zone", "score_state", "match_phase"}
            and decision_context.get(feature) is not None
        ) or (
            feature == "opponent_belief" and packet.get("opponent_belief")
        ) or (
            feature == "opponent_response"
            and candidate.get("opponent_response_prediction")
        ) or (
            feature == "trajectory_rollout"
            and (packet.get("second_order_game") or {}).get(
                "trajectory_rollout"
            )
        ) or (
            feature in {"epistemic_uncertainty", "aleatoric_uncertainty"}
            and candidate.get(feature) is not None
        ) or (
            feature == "tactical_shape"
            and (packet.get("opponent_belief") or {}).get(
                "observed_feature_vector"
            )
        )
    }
    if not available_evidence:
        return {
            "version": LLM_CRITIC_VERSION,
            "accepted": False,
            "authority_active": False,
            "reason": "critique_evidence_not_present_in_packet",
            "critique": critique,
        }
    proposed = _proposal(critique["direction"], critique["confidence"])
    pre_critic_recommendation = str(packet.get("recommended_action", "none"))
    memory_compatible = bool(
        memory is not None
        and str(memory.checkpoint_signature)
        == str(packet.get("checkpoint_signature", "runtime_unspecified"))
        and str(memory.environment_signature)
        == str(packet.get("environment_signature", "environment_unspecified"))
        and str(memory.critic_signature) == str(critic_signature)
    )
    authority = (
        memory.authority(critique["action"], critique["horizon"])
        if memory_compatible else {
            "version": LLM_CRITIC_VERSION,
            "active": False,
            "scale": 0.0,
            "trust": 0.0,
            "reason": (
                "critic_memory_incompatible"
                if memory is not None else "critic_memory_unavailable"
            ),
        }
    )
    applied = float(np.clip(
        proposed
        * float(authority.get("scale", 0.0))
        * float(authority.get("trust", 0.0)),
        -0.04,
        0.04,
    ))
    base_value = float(candidate.get(
        "pre_llm_critic_risk_adjusted_value",
        candidate.get("risk_adjusted_value", -1.0),
    ))
    candidate["pre_llm_critic_risk_adjusted_value"] = base_value
    candidate["llm_semantic_critic_proposal"] = proposed
    candidate["llm_semantic_critic_adjustment"] = applied
    candidate["llm_semantic_critic_bridge_factor"] = float(np.clip(
        1.0 - 0.50 * max(0.0, -applied) / 0.04,
        0.50,
        1.0,
    ))
    candidate["llm_semantic_critic_information"] = float(np.clip(
        abs(proposed) / 0.08, 0.0, 1.0,
    ))
    candidate["risk_adjusted_value"] = base_value + applied
    eligible = [
        item for item in packet.get("candidates") or []
        if float(item.get("effective_confidence", 0.0)) > 0.0
    ]
    best = max(
        eligible,
        key=lambda item: float(item.get("risk_adjusted_value", -1.0)),
        default=None,
    )
    if best is not None:
        packet["recommended_action"] = str(best["action"])
    audit = {
        "version": LLM_CRITIC_VERSION,
        "accepted": True,
        "authority_active": bool(authority.get("active")),
        "correction_applied": abs(applied) > 1e-12,
        "reason": (
            "validated_semantic_residual_correction"
            if authority.get("active") else "shadow_critique_for_validation"
        ),
        "critique": critique,
        "grounded_evidence_features": sorted(available_evidence),
        "proposed_policy_utility_correction": proposed,
        "applied_ranking_correction": applied,
        "validated_policy_utility_correction": applied,
        "bridge_safety_factor": candidate[
            "llm_semantic_critic_bridge_factor"
        ],
        "authority": authority,
        "memory_compatible": memory_compatible,
        "critic_signature": str(critic_signature),
        "world_model_prediction_mutated": False,
        "can_update_world_model": False,
        "can_update_critic_memory": False,
        "causal_interpretation": False,
        "pre_critic_recommended_action": pre_critic_recommendation,
        "post_critic_recommended_action": str(packet.get(
            "recommended_action", "none",
        )),
    }
    packet["llm_world_model_critique_audit"] = audit
    return audit
