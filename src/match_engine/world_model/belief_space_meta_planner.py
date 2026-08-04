"""Route bounded future world-model compute from resolved epistemic answers."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_contract import (
    OPPONENT_HYPOTHESES,
    normalized_entropy,
)
from src.match_engine.world_model.opponent_information_feedback import (
    opponent_information_feedback_is_valid,
)
from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS


BELIEF_SPACE_META_PLANNER_VERSION = 1


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _option(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "option_id": _digest(payload, "belief-space-compute-option:"),
    }


def _route_is_valid(route: Any) -> bool:
    if not isinstance(route, dict) or not route:
        return False
    payload = dict(route)
    digest = payload.pop("route_digest", None)
    return bool(
        digest == _digest(payload, "belief-space-compute-route:")
        and 1 <= len(route.get("target_first_actions") or []) <= 2
        and set(route.get("target_first_actions") or []) <= set(
            RESPONSE_ACTIONS
        )
        and 1 <= len(route.get("target_continuation_actions") or []) <= 2
        and set(route.get("target_continuation_actions") or []) <= set(
            RESPONSE_ACTIONS
        )
        and set(route.get("hypothesis_priority") or [])
        == set(OPPONENT_HYPOTHESES)
        and route.get("routing_scope")
        == "future_shadow_world_model_compute_only"
        and route.get("can_change_current_action") is False
        and route.get("causal_interpretation") is False
    )


def _compute_option_is_valid(option: Any) -> bool:
    if not isinstance(option, dict):
        return False
    payload = dict(option)
    option_id = payload.pop("option_id", None)
    try:
        budget = int(option["branch_evaluation_budget"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    mode = option.get("mode")
    route = option.get("route") or {}
    return bool(
        option_id == _digest(payload, "belief-space-compute-option:")
        and mode in {
            "answer_conditioned_route", "broad_route", "stop_extra_compute",
        }
        and (
            _route_is_valid(route)
            if mode == "answer_conditioned_route" else not route
        )
        and 16 <= budget <= 112
        and option.get("can_change_current_action") is False
    )


def _latest_answer(feedback: dict[str, Any]) -> dict[str, Any] | None:
    if not opponent_information_feedback_is_valid(feedback):
        return None
    rows = list(feedback.get("recent_resolutions") or [])
    return rows[-1] if rows else None


def _answer_route(
    candidates: list[dict[str, Any]], answer_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    answer = answer_row.get("resolved_answer") or {}
    posterior_raw = answer.get("posterior_after_answer") or {}
    try:
        posterior = np.asarray([
            float(posterior_raw[name]) for name in OPPONENT_HYPOTHESES
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError):
        return {}, {}
    if (
        not np.all(np.isfinite(posterior))
        or np.any(posterior < 0.0)
        or abs(float(np.sum(posterior)) - 1.0) > 1e-6
    ):
        return {}, {}
    action_rows = []
    for candidate in candidates:
        action = str(candidate.get("action", ""))
        values = candidate.get("opponent_hypothesis_values") or {}
        try:
            vector = np.asarray([
                float(values[name]) for name in OPPONENT_HYPOTHESES
            ], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not np.all(np.isfinite(vector)):
            continue
        mean = float(np.dot(posterior, vector))
        std = float(np.sqrt(np.dot(posterior, np.square(vector - mean))))
        action_rows.append({
            "action": action,
            "posterior_expected_value": mean,
            "posterior_value_std": std,
            "robust_value": mean - 0.25 * std,
        })
    action_rows.sort(key=lambda row: (
        -row["robust_value"], RESPONSE_ACTIONS.index(row["action"]),
    ))
    if not action_rows:
        return {}, {}
    targets = [row["action"] for row in action_rows[:2]]
    hypothesis_priority = sorted(
        OPPONENT_HYPOTHESES,
        key=lambda name: (
            -float(posterior_raw[name]), OPPONENT_HYPOTHESES.index(name),
        ),
    )
    margin = (
        action_rows[0]["robust_value"] - action_rows[1]["robust_value"]
        if len(action_rows) > 1 else 1.0
    )
    entropy = normalized_entropy(posterior)
    route_payload = {
        "version": BELIEF_SPACE_META_PLANNER_VERSION,
        "source_policy_audit_digest": str((
            answer_row.get("question_policy") or {}
        ).get("policy_audit_digest", "")),
        "source_decision_id": str(answer_row.get("decision_id", "")),
        "source_answer_feature": str(answer.get("feature", "")),
        "source_answer_branch": str(answer.get("resolved_branch", "")),
        "target_first_actions": targets,
        "target_continuation_actions": targets,
        "hypothesis_priority": hypothesis_priority,
        "posterior_normalized_entropy": float(entropy),
        "top_action_robust_margin": float(max(0.0, margin)),
        "action_values": action_rows,
        "routing_scope": "future_shadow_world_model_compute_only",
        "can_change_current_action": False,
        "causal_interpretation": False,
    }
    route = {
        **route_payload,
        "route_digest": _digest(
            route_payload, "belief-space-compute-route:"
        ),
    }
    return route, {
        "posterior_normalized_entropy": float(entropy),
        "top_action_robust_margin": float(max(0.0, margin)),
    }


def belief_space_compute_preference_is_valid(preference: Any) -> bool:
    if not isinstance(preference, dict):
        return False
    payload = dict(preference)
    digest = payload.pop("preference_digest", None)
    option = payload.get("selected_option") or {}
    try:
        created = float(payload["created_t_sec"])
        expires = float(payload["expires_t_sec"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        digest == _digest(payload, "belief-space-compute-preference:")
        and payload.get("version") == BELIEF_SPACE_META_PLANNER_VERSION
        and _compute_option_is_valid(option)
        and math.isfinite(created) and math.isfinite(expires)
        and expires > created
        and bool(payload.get("answer_evidence_digest"))
        and payload.get("can_change_current_action") is False
        and payload.get("can_change_tactical_controls") is False
        and payload.get("causal_interpretation") is False
    )


def build_belief_space_meta_plan(
    packet: dict[str, Any],
    *,
    default_branch_budget: int,
    stored_preference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = list(packet.get("candidates") or [])
    feedback = packet.get("opponent_information_feedback") or {}
    as_of = float(
        feedback.get("as_of_t_sec", 0.0)
        or (packet.get("decision_context") or {}).get("clock_seconds", 0.0)
        or 0.0
    )
    route, answer_metrics = _answer_route(
        candidates, _latest_answer(feedback) or {},
    )
    pair_floor = max(16, len(candidates) * len(candidates))
    broad_budget = max(pair_floor, min(112, int(default_branch_budget)))
    options = []
    if route:
        options.append(_option({
            "mode": "answer_conditioned_route",
            "branch_evaluation_budget": broad_budget,
            "route": route,
            "expected_compute_role": (
                "preserve one-hypothesis broad coverage, then concentrate "
                "remaining branches on answer-relevant actions and beliefs"
            ),
            "can_change_current_action": False,
        }))
    options.extend([
        _option({
            "mode": "broad_route",
            "branch_evaluation_budget": broad_budget,
            "route": {},
            "expected_compute_role": "uniform broad belief-space coverage",
            "can_change_current_action": False,
        }),
        _option({
            "mode": "stop_extra_compute",
            "branch_evaluation_budget": pair_floor,
            "route": {},
            "expected_compute_role": (
                "minimum one-hypothesis coverage for every action pair"
            ),
            "can_change_current_action": False,
        }),
    ])
    mode = "broad_route"
    if route:
        entropy = answer_metrics["posterior_normalized_entropy"]
        margin = answer_metrics["top_action_robust_margin"]
        mode = (
            "stop_extra_compute"
            if entropy <= 0.25 and margin >= 0.10
            else "answer_conditioned_route"
        )
    recommended = next(option for option in options if option["mode"] == mode)
    answer_evidence_digest = str(
        route.get("route_digest", "no_resolved_answer")
    )
    applied = recommended
    preference_status = "model_recommended_current_route"
    preference = dict(stored_preference or {})
    if belief_space_compute_preference_is_valid(preference):
        scope_ok = bool(
            str(preference.get("team_id")) == str(packet.get("team_id"))
            and str(preference.get("checkpoint_signature"))
            == str(packet.get("checkpoint_signature"))
            and str(preference.get("environment_signature"))
            == str(packet.get("environment_signature"))
            and float(preference["created_t_sec"]) <= as_of + 1e-9
            and float(preference["expires_t_sec"]) >= as_of - 1e-9
        )
        if scope_ok:
            preferred = preference["selected_option"]
            if preference.get(
                "answer_evidence_digest"
            ) == answer_evidence_digest:
                applied = preferred
                preference_status = "prior_llm_compute_preference_applied"
            else:
                preference_status = "stale_answer_route_rejected"
        else:
            preference_status = "incompatible_or_expired_preference_rejected"
    payload = {
        "version": BELIEF_SPACE_META_PLANNER_VERSION,
        "available": bool(options),
        "as_of_t_sec": as_of,
        "checkpoint_signature": str(packet.get("checkpoint_signature", "")),
        "environment_signature": str(packet.get("environment_signature", "")),
        "team_id": str(packet.get("team_id", "")),
        "answer_conditioned_route_available": bool(route),
        "answer_evidence_digest": answer_evidence_digest,
        "answer_metrics": answer_metrics,
        "options": options,
        "recommended_option_id": recommended["option_id"],
        "applied_option": applied,
        "preference_status": preference_status,
        "selection_scope": "next_decision_shadow_compute_only",
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "plan_digest": _digest(payload, "belief-space-meta-plan:"),
    }


def belief_space_meta_plan_is_valid(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    payload = dict(plan)
    digest = payload.pop("plan_digest", None)
    options = payload.get("options") or []
    if (
        digest != _digest(payload, "belief-space-meta-plan:")
        or payload.get("version") != BELIEF_SPACE_META_PLANNER_VERSION
        or not options
        or not payload.get("answer_evidence_digest")
        or payload.get("recommended_option_id")
        not in {row.get("option_id") for row in options}
        or payload.get("applied_option", {}).get("mode")
        not in {"answer_conditioned_route", "broad_route", "stop_extra_compute"}
        or not _compute_option_is_valid(payload.get("applied_option"))
        or payload.get("selection_scope")
        != "next_decision_shadow_compute_only"
        or payload.get("can_change_current_action") is not False
        or payload.get("can_change_tactical_controls") is not False
        or payload.get("can_schedule_future_action") is not False
        or payload.get("causal_interpretation") is not False
    ):
        return False
    for option in options:
        if not _compute_option_is_valid(option):
            return False
    return True


def validate_llm_belief_space_meta_plan(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    option_id = str(raw.get("option_id", ""))[:96]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not option_id or not math.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
        return None
    return {
        "option_id": option_id,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def evaluate_llm_belief_space_meta_plan(
    packet: dict[str, Any], raw: Any, *, selected_after_action_freeze: bool,
) -> dict[str, Any]:
    selection = validate_llm_belief_space_meta_plan(raw)
    plan = packet.get("belief_space_meta_plan") or {}
    base = {
        "version": BELIEF_SPACE_META_PLANNER_VERSION,
        "accepted": False,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    if selection is None:
        return {**base, "reason": "missing_or_invalid_meta_plan_selection"}
    if not selected_after_action_freeze or not belief_space_meta_plan_is_valid(plan):
        return {**base, "reason": "post_action_valid_meta_plan_required"}
    option = next((
        row for row in plan["options"]
        if row["option_id"] == selection["option_id"]
    ), None)
    if option is None:
        return {**base, "reason": "unknown_belief_space_compute_option"}
    preference_payload = {
        "version": BELIEF_SPACE_META_PLANNER_VERSION,
        "team_id": plan["team_id"],
        "checkpoint_signature": plan["checkpoint_signature"],
        "environment_signature": plan["environment_signature"],
        "created_t_sec": float(plan["as_of_t_sec"]),
        "expires_t_sec": float(plan["as_of_t_sec"]) + 300.0,
        "source_plan_digest": plan["plan_digest"],
        "answer_evidence_digest": plan["answer_evidence_digest"],
        "selected_option": option,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "causal_interpretation": False,
    }
    preference = {
        **preference_payload,
        "preference_digest": _digest(
            preference_payload, "belief-space-compute-preference:"
        ),
    }
    payload = {
        "version": BELIEF_SPACE_META_PLANNER_VERSION,
        "accepted": True,
        "reason": "next_decision_shadow_compute_preference_accepted",
        "selection": selection,
        "source_plan_digest": plan["plan_digest"],
        "recommended_option_id": plan["recommended_option_id"],
        "selected_option": option,
        "matched_model_recommendation": bool(
            option["option_id"] == plan["recommended_option_id"]
        ),
        "compute_preference": preference,
        "selected_after_action_freeze": True,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "audit_digest": _digest(payload, "belief-space-meta-plan-audit:"),
    }


def belief_space_meta_plan_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    selection = validate_llm_belief_space_meta_plan(payload.get("selection"))
    option = payload.get("selected_option") or {}
    preference = payload.get("compute_preference") or {}
    return bool(
        digest == _digest(payload, "belief-space-meta-plan-audit:")
        and selection is not None
        and selection["option_id"] == option.get("option_id")
        and belief_space_compute_preference_is_valid(preference)
        and preference.get("selected_option") == option
        and preference.get("source_plan_digest")
        == payload.get("source_plan_digest")
        and payload.get("matched_model_recommendation") == bool(
            option.get("option_id") == payload.get("recommended_option_id")
        )
        and payload.get("selected_after_action_freeze") is True
        and payload.get("can_change_current_action") is False
        and payload.get("can_change_tactical_controls") is False
        and payload.get("can_schedule_future_action") is False
        and payload.get("causal_interpretation") is False
    )
