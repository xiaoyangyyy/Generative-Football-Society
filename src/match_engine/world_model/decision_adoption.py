"""Audit whether an LLM-selected world-model action appears after the decision."""

from __future__ import annotations

import math
from typing import Any


TRACKED_ACTIONS = {"hold", "pass", "cross", "shot"}


def register_coach_action_decision(
    state,
    *,
    team_id: str,
    trigger_kind: str,
    llm_selected_action: str,
    world_model_recommended_action: str,
    recommendation_confidence: float,
    horizon_s: float,
    intervention_strength: float = 0.0,
    intervention_enabled: bool = False,
) -> dict[str, Any] | None:
    """Register a prospective decision; never count an already-executed action."""
    selected = str(llm_selected_action).lower()
    if selected not in TRACKED_ACTIONS:
        return None
    records = getattr(state, "_wm_coach_decision_adoption", None)
    if records is None:
        records = []
        state._wm_coach_decision_adoption = records
    created = float(getattr(state, "clock_seconds", 0.0))
    horizon = max(0.1, float(horizon_s))
    for previous in records:
        if (
            not previous["resolved"]
            and str(previous["team_id"]) == str(team_id)
        ):
            previous["resolved"] = True
            previous["adopted"] = False
            previous["resolved_t_sec"] = created
            previous["resolution"] = "superseded_by_newer_coach_decision"
    strength = float(intervention_strength)
    if not math.isfinite(strength):
        strength = 0.0
    strength = min(0.5, max(0.0, strength))
    record = {
        "decision_id": f"{team_id}:{created:.3f}:{len(records)}",
        "team_id": str(team_id),
        "trigger_kind": str(trigger_kind),
        "created_t_sec": created,
        "expires_t_sec": created + horizon,
        "llm_selected_action": selected,
        "world_model_recommended_action": str(
            world_model_recommended_action
        ).lower(),
        "recommendation_confidence": float(recommendation_confidence),
        "intervention_enabled": bool(intervention_enabled and strength > 0.0),
        "intervention_strength": strength,
        "intervention_applied": False,
        "intervention_t_sec": None,
        "intervention_actual_action": None,
        "observed_actions": [],
        "resolved": False,
        "adopted": None,
        "resolved_t_sec": None,
        "resolution": "pending",
    }
    records.append(record)
    return record


def pending_policy_action_bias(
    state,
    *,
    team_id: str,
    feasible_actions: set[str],
    t_sec: float,
) -> dict[str, Any] | None:
    """Return the newest eligible one-shot logit bias without consuming it."""
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    for record in reversed(records):
        if record["resolved"] or record["intervention_applied"]:
            continue
        if not record.get("intervention_enabled", False):
            continue
        if str(record["team_id"]) != str(team_id):
            continue
        if now <= float(record["created_t_sec"]):
            continue
        if now > float(record["expires_t_sec"]):
            record["resolved"] = True
            record["adopted"] = False
            record["resolved_t_sec"] = now
            record["resolution"] = "horizon_expired"
            continue
        selected = str(record["llm_selected_action"])
        if selected not in feasible_actions:
            continue
        return {
            "decision_id": record["decision_id"],
            "action": selected,
            "logit_bias": float(record["intervention_strength"]),
        }
    return None


def record_policy_intervention_result(
    state,
    *,
    decision_id: str,
    actual_action: str,
    t_sec: float,
) -> None:
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    for record in records:
        if record["decision_id"] != decision_id or record["resolved"]:
            continue
        actual = str(actual_action).lower()
        record["intervention_applied"] = True
        record["intervention_t_sec"] = float(t_sec)
        record["intervention_actual_action"] = actual
        record["observed_actions"].append({
            "t_sec": float(t_sec), "action": actual,
        })
        record["resolved"] = True
        record["adopted"] = actual == record["llm_selected_action"]
        record["resolved_t_sec"] = float(t_sec)
        record["resolution"] = (
            "matching_action_after_bounded_bias"
            if record["adopted"] else "different_action_after_bounded_bias"
        )
        return


def observe_executed_action(
    state,
    *,
    team_id: str,
    action_kind: str,
    t_sec: float,
) -> None:
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    action = str(action_kind).lower()
    for record in records:
        if record["resolved"]:
            continue
        # Decisions are registered after the current tick's action; strict
        # ordering prevents that earlier action from being counted as adoption.
        if now <= float(record["created_t_sec"]):
            continue
        if now > float(record["expires_t_sec"]):
            record["resolved"] = True
            record["adopted"] = False
            record["resolved_t_sec"] = now
            record["resolution"] = "horizon_expired"
            continue
        if str(record["team_id"]) != str(team_id):
            continue
        record["observed_actions"].append({"t_sec": now, "action": action})
        if action == record["llm_selected_action"]:
            record["resolved"] = True
            record["adopted"] = True
            record["resolved_t_sec"] = now
            record["resolution"] = "matching_action_observed"


def finalize_decision_adoption(state, *, t_sec: float) -> None:
    records = getattr(state, "_wm_coach_decision_adoption", None) or []
    now = float(t_sec)
    for record in records:
        if not record["resolved"]:
            record["resolved"] = True
            record["adopted"] = False
            record["resolved_t_sec"] = now
            record["resolution"] = "match_ended"


def decision_adoption_diagnostics(state) -> dict[str, Any]:
    records = list(getattr(state, "_wm_coach_decision_adoption", None) or [])
    resolved = [record for record in records if record["resolved"]]
    adopted = [record for record in resolved if record["adopted"]]
    interventions = [
        record for record in resolved if record.get("intervention_applied")
    ]
    intervention_adopted = [
        record for record in interventions if record["adopted"]
    ]
    return {
        "version": 1,
        "registered": len(records),
        "resolved": len(resolved),
        "adopted": len(adopted),
        "pending": len(records) - len(resolved),
        "adoption_rate": len(adopted) / max(1, len(resolved)),
        "interventions_applied": len(interventions),
        "intervention_adopted": len(intervention_adopted),
        "intervention_adoption_rate": (
            len(intervention_adopted) / max(1, len(interventions))
        ),
        "causal_ordering": "decision_then_future_team_action",
        "records": records,
        "interpretation": (
            "A bounded intervention changes one action logit but does not force "
            "the sampled action; non-intervention adoption remains associative "
            "and does not prove causal influence."
        ),
    }
