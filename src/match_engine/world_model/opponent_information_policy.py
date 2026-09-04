"""Finite-horizon, model-owned policy for asking or stopping tactical queries."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from src.match_engine.world_model.opponent_information_feedback import (
    opponent_information_feedback_is_valid,
)
from src.match_engine.world_model.opponent_information_query import (
    evaluate_llm_opponent_information_query,
    opponent_information_query_audit_is_valid,
    opponent_information_question_menu_is_valid,
)
from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS
from src.match_engine.world_model.policy_experiment import policy_horizon_seconds


OPPONENT_INFORMATION_POLICY_VERSION = 1
MAXIMUM_QUESTIONS_PER_EPISODE = 2
QUESTION_COST = 0.015


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _proposal_value(proposal: dict[str, Any]) -> float:
    horizon_s = float(policy_horizon_seconds(str(proposal["horizon"])))
    return float(
        float(proposal["expected_normalized_information_gain"])
        + min(
            0.15,
            3.0 * float(
                proposal["expected_decision_value_of_information"]
            ),
        )
        - 0.00005 * horizon_s
        - 0.01 * max(0, int(proposal["model_rank"]) - 1)
    )


def _new_episode_id(
    *, packet: dict[str, Any], action: str, as_of_t_sec: float,
) -> str:
    return _digest({
        "checkpoint_signature": str(packet.get("checkpoint_signature", "")),
        "environment_signature": str(packet.get("environment_signature", "")),
        "team_id": str(packet.get("team_id", "")),
        "selected_action": str(action),
        "anchor_t_sec": float(as_of_t_sec),
    }, "opponent-information-episode:")


def _prior_episode(
    feedback: dict[str, Any], *, action: str, as_of_t_sec: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not opponent_information_feedback_is_valid(feedback):
        return None, []
    rows = [
        row for row in feedback.get("recent_resolutions") or []
        if row.get("selected_action") == action
        and isinstance(row.get("question_policy"), dict)
        and float(row.get("observed_t_sec", 0.0)) <= as_of_t_sec + 1e-9
    ]
    if not rows:
        return None, []
    rows.sort(key=lambda row: (
        float(row.get("observed_t_sec", 0.0)),
        float(row.get("issued_t_sec", 0.0)),
    ))
    last = rows[-1]
    if as_of_t_sec - float(last.get("observed_t_sec", 0.0)) > 300.0:
        return None, []
    episode_id = str((last.get("question_policy") or {}).get(
        "episode_id", "",
    ))
    episode_rows = [
        row for row in rows
        if str((row.get("question_policy") or {}).get("episode_id", ""))
        == episode_id
    ]
    return last, episode_rows


def build_opponent_information_cognitive_policy(
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Plan at most two questions and make stopping an explicit decision."""
    menu = packet.get("opponent_information_question_menu") or {}
    feedback = packet.get("opponent_information_feedback") or {}
    as_of_t_sec = float(
        (feedback.get("as_of_t_sec") if isinstance(feedback, dict) else 0.0)
        or (packet.get("decision_context") or {}).get("clock_seconds", 0.0)
        or 0.0
    )
    recommendations = {}
    proposals = list(menu.get("proposals") or [])
    if opponent_information_question_menu_is_valid(menu, packet):
        for action in RESPONSE_ACTIONS:
            action_proposals = [
                row for row in proposals if row["selected_action"] == action
            ]
            last, episode_rows = _prior_episode(
                feedback, action=action, as_of_t_sec=as_of_t_sec,
            )
            if last is None:
                episode_id = _new_episode_id(
                    packet=packet, action=action, as_of_t_sec=as_of_t_sec,
                )
                round_index = 1
                used_features: set[str] = set()
                posterior_entropy = 1.0
            else:
                prior_policy = last["question_policy"]
                episode_id = str(prior_policy["episode_id"])
                round_index = int(prior_policy["round_index"]) + 1
                used_features = {
                    str(row.get("feature", "")) for row in episode_rows
                }
                posterior_entropy = float((
                    last.get("resolved_answer") or {}
                ).get("posterior_normalized_entropy", 1.0))
            diverse = [
                row for row in action_proposals
                if row["feature"] not in used_features
            ]
            candidates = diverse or action_proposals
            ranked = sorted(candidates, key=lambda row: (
                -_proposal_value(row),
                int(row["model_rank"]),
                str(row["proposal_id"]),
            ))
            best = ranked[0] if ranked else None
            marginal_value = _proposal_value(best) if best else 0.0
            stop_reason = "continue_information_acquisition"
            decision = "ask"
            if round_index > MAXIMUM_QUESTIONS_PER_EPISODE:
                decision = "stop"
                stop_reason = "episode_question_budget_exhausted"
            elif last is not None and posterior_entropy <= 0.25:
                decision = "stop"
                stop_reason = "posterior_entropy_below_stop_threshold"
            elif best is None:
                decision = "stop"
                stop_reason = "no_compatible_observable_question"
            elif marginal_value <= QUESTION_COST:
                decision = "stop"
                stop_reason = "marginal_information_value_below_query_cost"
            recommendations[action] = {
                "selected_action": action,
                "episode_id": episode_id,
                "round_index": round_index,
                "questions_already_resolved": len(episode_rows),
                "maximum_questions": MAXIMUM_QUESTIONS_PER_EPISODE,
                "remaining_question_budget": max(
                    0, MAXIMUM_QUESTIONS_PER_EPISODE - round_index + 1,
                ),
                "used_features": sorted(used_features),
                "posterior_normalized_entropy": posterior_entropy,
                "recommended_decision": decision,
                "recommended_proposal_id": (
                    str(best["proposal_id"]) if decision == "ask" and best
                    else ""
                ),
                "best_marginal_question_value": marginal_value,
                "question_cost": QUESTION_COST,
                "stop_reason": stop_reason,
            }
    payload = {
        "version": OPPONENT_INFORMATION_POLICY_VERSION,
        "available": bool(recommendations),
        "reason": (
            "finite_horizon_question_policy"
            if recommendations else "valid_question_menu_required"
        ),
        "as_of_t_sec": as_of_t_sec,
        "question_menu_digest": str(menu.get("menu_digest", "")),
        "maximum_questions_per_episode": MAXIMUM_QUESTIONS_PER_EPISODE,
        "question_cost": QUESTION_COST,
        "recommendations_by_action": recommendations,
        "policy_scope": "post_action_shadow_information_acquisition",
        "can_change_current_action": False,
        "can_schedule_future_action": False,
        "can_directly_mutate_live_belief": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "policy_digest": _digest(
            payload, "opponent-information-cognitive-policy:"
        ),
    }


def opponent_information_cognitive_policy_self_is_valid(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    payload = dict(policy)
    digest = payload.pop("policy_digest", None)
    try:
        as_of = float(payload["as_of_t_sec"])
        question_cost = float(payload["question_cost"])
        recommendations = payload["recommendations_by_action"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if (
        digest != _digest(payload, "opponent-information-cognitive-policy:")
        or payload.get("version") != OPPONENT_INFORMATION_POLICY_VERSION
        or not math.isfinite(as_of)
        or abs(question_cost - QUESTION_COST) > 1e-12
        or set(recommendations) != set(RESPONSE_ACTIONS)
        or payload.get("policy_scope")
        != "post_action_shadow_information_acquisition"
        or payload.get("can_change_current_action") is not False
        or payload.get("can_schedule_future_action") is not False
        or payload.get("can_directly_mutate_live_belief") is not False
        or payload.get("causal_interpretation") is not False
    ):
        return False
    for action, row in recommendations.items():
        try:
            round_index = int(row["round_index"])
            remaining = int(row["remaining_question_budget"])
            marginal = float(row["best_marginal_question_value"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        decision = row.get("recommended_decision")
        if (
            row.get("selected_action") != action
            or decision not in {"ask", "stop"}
            or not row.get("episode_id")
            or round_index < 1
            or remaining < 0
            or not math.isfinite(marginal)
            or (decision == "ask") != bool(row.get("recommended_proposal_id"))
        ):
            return False
    return bool(policy.get("available"))


def opponent_information_cognitive_policy_is_valid(
    policy: Any, packet: dict[str, Any],
) -> bool:
    try:
        return (
            opponent_information_cognitive_policy_self_is_valid(policy)
            and policy == build_opponent_information_cognitive_policy(packet)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def validate_llm_opponent_information_policy(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    decision = str(raw.get("decision", "")).strip().lower()
    proposal_id = str(raw.get("proposal_id", ""))[:96]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        decision not in {"ask", "stop"}
        or (decision == "ask") != bool(proposal_id)
        or not math.isfinite(confidence)
        or not 0.5 <= confidence <= 1.0
    ):
        return None
    return {
        "decision": decision,
        "proposal_id": proposal_id,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def _audit_digest(payload: dict[str, Any]) -> str:
    return _digest(payload, "opponent-information-policy-audit:")


def evaluate_llm_opponent_information_policy(
    packet: dict[str, Any],
    raw_policy: Any,
    *,
    selected_action: str,
    query_signature: str,
    selected_after_action_freeze: bool,
) -> dict[str, Any]:
    choice = validate_llm_opponent_information_policy(raw_policy)
    cognitive_policy = packet.get("opponent_information_cognitive_policy") or {}
    menu = packet.get("opponent_information_question_menu") or {}
    base = {
        "version": OPPONENT_INFORMATION_POLICY_VERSION,
        "accepted": False,
        "reason": "missing_or_invalid_information_policy",
        "can_change_current_action": False,
        "can_schedule_future_action": False,
        "can_directly_mutate_live_belief": False,
        "causal_interpretation": False,
    }
    if choice is None:
        return base
    if (
        not selected_after_action_freeze
        or not opponent_information_cognitive_policy_is_valid(
            cognitive_policy, packet,
        )
        or not opponent_information_question_menu_is_valid(menu, packet)
    ):
        return {**base, "reason": "post_action_valid_policy_evidence_required"}
    action = str(selected_action).lower()
    recommendation = (
        cognitive_policy["recommendations_by_action"].get(action) or {}
    )
    if not recommendation:
        return {**base, "reason": "selected_action_policy_unavailable"}
    selected_proposal = None
    query_audit: dict[str, Any] = {}
    if choice["decision"] == "ask":
        selected_proposal = next((
            row for row in menu["proposals"]
            if row["proposal_id"] == choice["proposal_id"]
        ), None)
        if (
            selected_proposal is None
            or selected_proposal["selected_action"] != action
        ):
            return {**base, "reason": "action_scoped_question_proposal_required"}
        query = {
            key: selected_proposal[key] for key in (
                "proposal_id", "selected_action", "feature", "horizon",
                "purpose", "action_if_high", "action_if_low",
            )
        } | {
            "confidence": choice["confidence"],
            "rationale": choice["rationale"],
        }
        query_audit = evaluate_llm_opponent_information_query(
            packet,
            query,
            selected_action=action,
            query_signature=query_signature,
            selected_after_action_freeze=True,
        )
        if not opponent_information_query_audit_is_valid(query_audit):
            return {**base, "reason": "canonical_question_evaluation_failed"}
    payload = {
        "version": OPPONENT_INFORMATION_POLICY_VERSION,
        "accepted": True,
        "reason": "finite_horizon_information_policy_audited",
        "choice": choice,
        "selected_action": action,
        "cognitive_policy": cognitive_policy,
        "question_menu_digest": menu["menu_digest"],
        "recommendation": recommendation,
        "selected_proposal": dict(selected_proposal or {}),
        "query_audit": query_audit,
        "decision_matches_model": (
            choice["decision"] == recommendation["recommended_decision"]
        ),
        "proposal_matches_model": bool(
            choice["decision"] == "stop"
            or choice["proposal_id"]
            == recommendation["recommended_proposal_id"]
        ),
        "model_checked_consistent": bool(
            choice["decision"] == recommendation["recommended_decision"]
            and (
                choice["decision"] == "stop"
                or choice["proposal_id"]
                == recommendation["recommended_proposal_id"]
            )
        ),
        "selected_after_action_freeze": True,
        "policy_scope": "post_action_shadow_information_acquisition",
        "can_change_current_action": False,
        "can_schedule_future_action": False,
        "can_directly_mutate_live_belief": False,
        "causal_interpretation": False,
    }
    return {**payload, "audit_digest": _audit_digest(payload)}


def opponent_information_policy_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    choice = validate_llm_opponent_information_policy(payload.get("choice"))
    cognitive = payload.get("cognitive_policy") or {}
    query_audit = payload.get("query_audit") or {}
    recommendation = payload.get("recommendation") or {}
    selected = payload.get("selected_proposal") or {}
    return bool(
        digest == _audit_digest(payload)
        and choice is not None
        and opponent_information_cognitive_policy_self_is_valid(cognitive)
        and recommendation == (
            cognitive["recommendations_by_action"].get(
                payload.get("selected_action")
            ) or {}
        )
        and payload.get("question_menu_digest")
        == cognitive.get("question_menu_digest")
        and payload.get("decision_matches_model") == (
            choice["decision"] == recommendation.get("recommended_decision")
        )
        and payload.get("proposal_matches_model") == bool(
            choice["decision"] == "stop"
            or choice["proposal_id"]
            == recommendation.get("recommended_proposal_id")
        )
        and payload.get("model_checked_consistent") == bool(
            choice["decision"] == recommendation.get("recommended_decision")
            and (
                choice["decision"] == "stop"
                or choice["proposal_id"]
                == recommendation.get("recommended_proposal_id")
            )
        )
        and (
            choice["decision"] == "stop"
            and not selected and not query_audit
            or choice["decision"] == "ask"
            and selected.get("proposal_id") == choice["proposal_id"]
            and opponent_information_query_audit_is_valid(query_audit)
            and selected == (
                (query_audit.get("question_selection") or {}).get(
                    "selected_proposal"
                ) or {}
            )
            and (query_audit.get("query") or {}).get("proposal_id")
            == choice["proposal_id"]
        )
        and payload.get("selected_after_action_freeze") is True
        and payload.get("policy_scope")
        == "post_action_shadow_information_acquisition"
        and payload.get("can_change_current_action") is False
        and payload.get("can_schedule_future_action") is False
        and payload.get("can_directly_mutate_live_belief") is False
        and payload.get("causal_interpretation") is False
    )


def opponent_information_policy_diagnostics(record_clusters) -> dict[str, Any]:
    accepted = malformed = eligible = consistent = 0
    ask_decisions = stop_decisions = realized_answers = 0
    budget_violations = redundant_second_questions = 0
    unsafe = 0
    match_rates = []
    episodes: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for match_index, cluster in enumerate(record_clusters):
        rows = list(cluster)
        match_total = match_consistent = 0
        for record in rows:
            audit = record.get(
                "llm_opponent_information_policy_context"
            ) or {}
            if not audit:
                continue
            if not opponent_information_policy_audit_is_valid(audit):
                malformed += 1
                continue
            accepted += 1
            action_realized = (
                str(record.get("intervention_actual_action", "")).lower()
                == str(audit.get("selected_action", "")).lower()
            )
            if not action_realized:
                continue
            eligible += 1
            match_total += 1
            is_consistent = bool(audit.get("model_checked_consistent"))
            consistent += int(is_consistent)
            match_consistent += int(is_consistent)
            choice = audit["choice"]
            recommendation = audit["recommendation"]
            episode_key = (
                match_index, str(recommendation["episode_id"]),
            )
            episodes.setdefault(episode_key, []).append(audit)
            budget_violations += int(
                choice["decision"] == "ask"
                and int(recommendation["round_index"])
                > int(recommendation["maximum_questions"])
            )
            unsafe += int(
                audit.get("selected_after_action_freeze") is not True
                or audit.get("can_change_current_action") is not False
                or audit.get("can_schedule_future_action") is not False
                or audit.get("can_directly_mutate_live_belief") is not False
                or audit.get("causal_interpretation") is not False
            )
            if choice["decision"] == "stop":
                stop_decisions += 1
                continue
            ask_decisions += 1
            query_audit = audit["query_audit"]
            query = query_audit["query"]
            outcome = ((record.get(
                "multi_horizon_regime_outcomes"
            ) or {}).get(query["horizon"]) or {})
            score = outcome.get(
                "llm_opponent_information_query_evaluation"
            ) or {}
            from src.match_engine.world_model.opponent_information_query_outcomes import (
                opponent_information_query_score_is_valid,
            )

            realized_answers += int(
                opponent_information_query_score_is_valid(
                    score, query_audit,
                )
            )
        if match_total:
            match_rates.append(match_consistent / match_total)
    multi_round = completed_stop = 0
    for rows in episodes.values():
        asks = [
            row for row in rows if row["choice"]["decision"] == "ask"
        ]
        stops = [
            row for row in rows if row["choice"]["decision"] == "stop"
        ]
        rounds = sorted({
            int(row["recommendation"]["round_index"]) for row in asks
        })
        if len(rounds) >= 2:
            multi_round += 1
            features = [
                str((row["query_audit"]["query"])["feature"])
                for row in sorted(
                    asks,
                    key=lambda item: int(
                        item["recommendation"]["round_index"]
                    ),
                )[:2]
            ]
            redundant_second_questions += int(
                len(features) == 2 and features[0] == features[1]
            )
        if stops and len(rounds) >= 2:
            completed_stop += 1
    return {
        "version": OPPONENT_INFORMATION_POLICY_VERSION,
        "evaluation_kind": "finite_horizon_opponent_information_policy",
        "accepted_policy_audits": accepted,
        "eligible_realized_policy_decisions": eligible,
        "matches": len(match_rates),
        "ask_decisions": ask_decisions,
        "stop_decisions": stop_decisions,
        "realized_policy_answers": realized_answers,
        "multi_round_episodes": multi_round,
        "completed_stop_episodes": completed_stop,
        "malformed_policy_audits": malformed,
        "question_budget_violations": budget_violations,
        "redundant_second_questions": redundant_second_questions,
        "match_clustered_model_consistency_rate": (
            sum(match_rates) / len(match_rates) if match_rates else 0.0
        ),
        "overall_model_consistency_rate": (
            consistent / eligible if eligible else 0.0
        ),
        "unsafe_policy_authority_claims": unsafe,
        "all_policy_decisions_post_action_shadow_only": bool(
            eligible > 0 and unsafe == 0
        ),
    }
