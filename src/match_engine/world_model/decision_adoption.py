"""Audit whether an LLM-selected world-model action appears after the decision."""

from __future__ import annotations

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
        "observed_actions": [],
        "resolved": False,
        "adopted": None,
        "resolved_t_sec": None,
        "resolution": "pending",
    }
    records.append(record)
    return record


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
    return {
        "version": 1,
        "registered": len(records),
        "resolved": len(resolved),
        "adopted": len(adopted),
        "pending": len(records) - len(resolved),
        "adoption_rate": len(adopted) / max(1, len(resolved)),
        "causal_ordering": "decision_then_future_team_action",
        "records": records,
        "interpretation": (
            "Adoption means a matching high-level team action occurred within "
            "the decision horizon; it does not prove the coach plan caused it."
        ),
    }
