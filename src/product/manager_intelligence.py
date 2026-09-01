"""Evidence-bounded manager briefing and debrief read models."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from src.match_engine.tactical_catalog import TACTICAL_KEYS
from src.product.world_model_action_execution import (
    project_world_model_action_execution,
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _strict_tactical_vector(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(TACTICAL_KEYS):
        raise ValueError("tactical execution vector coverage mismatch")
    result = {}
    for key in TACTICAL_KEYS:
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("tactical execution vector value is invalid")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("tactical execution vector value is invalid")
        result[key] = round(number, 6)
    return result


def _tactical_binding(
    management: Mapping[str, Any], *, side: str, team: str,
    expected_tactic: str,
) -> dict[str, Any]:
    execution = management.get("tactical_execution")
    if not isinstance(execution, Mapping) or not execution:
        return {
            "schema_version": 1, "available": False,
            "reason": "legacy_report_without_tactical_binding",
        }
    row = execution.get(side)
    if not isinstance(row, Mapping) or set(row) != {
        "schema_version", "team", "applied_tactic", "native_archetype",
        "binding_kind",
        "source", "preset_locked", "initial_vector", "final_vector",
        "changed_controls", "final_delta_l1",
    }:
        raise ValueError("tactical execution binding structure mismatch")
    initial = _strict_tactical_vector(row.get("initial_vector"))
    final = _strict_tactical_vector(row.get("final_vector"))
    changed = [
        key for key in TACTICAL_KEYS
        if abs(final[key] - initial[key]) > 1e-6
    ]
    delta = round(sum(abs(final[key] - initial[key]) for key in TACTICAL_KEYS), 6)
    native_binding = expected_tactic == "team_identity"
    expected_kind = "native_team_vector" if native_binding else "locked_preset"
    expected_source = (
        "agent_team_identity" if native_binding else "studio_user_intervention"
    )
    if (
        row.get("schema_version") != 1
        or row.get("team") != team
        or row.get("applied_tactic") != expected_tactic
        or not isinstance(row.get("native_archetype"), str)
        or not row.get("native_archetype")
        or len(row.get("native_archetype")) > 64
        or row.get("binding_kind") != expected_kind
        or row.get("source") != expected_source
        or row.get("preset_locked") is not (not native_binding)
        or row.get("changed_controls") != changed
        or row.get("final_delta_l1") != delta
    ):
        raise ValueError("tactical execution binding identity mismatch")
    payload = {
        "schema_version": 1, "available": True,
        "team": team, "side": side,
        "applied_tactic": expected_tactic,
        "native_archetype": row.get("native_archetype"),
        "binding_kind": expected_kind,
        "source": expected_source,
        "preset_locked": not native_binding,
        "initial_vector": initial,
        "initial_vector_identity": _identity(initial),
        "final_vector": final,
        "changed_controls": changed,
        "final_delta_l1": delta,
        "claim_boundary": (
            "direct engine input and final tactical state; changes during the match "
            "are execution facts, not estimates of score or causal effectiveness"
        ),
    }
    payload["binding_identity"] = _identity(payload)
    return payload


def _condition(catalog: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = catalog.get("team_condition") if isinstance(catalog, Mapping) else None
    raw = raw if isinstance(raw, Mapping) else {}
    return {
        "fatigue": min(1.0, max(0.0, _finite(raw.get("fatigue")))),
        "morale": min(1.0, max(0.0, _finite(raw.get("morale"), 0.55))),
        "media_pressure": min(1.0, max(0.0, _finite(raw.get("media_pressure")))),
        "injured_players": max(0, int(_finite(raw.get("injured_players")))),
        "suspended_players": max(0, int(_finite(raw.get("suspended_players")))),
        "unavailable_players": max(0, int(_finite(raw.get("unavailable_players")))),
        "matches_settled": max(0, int(_finite(raw.get("matches_settled")))),
        "source": str(raw.get("source") or "unavailable")[:64],
    }


def _squad_facts(catalog: Mapping[str, Any] | None) -> dict[str, Any]:
    catalog = catalog if isinstance(catalog, Mapping) else {}
    raw_players = catalog.get("players")
    raw_players = raw_players if isinstance(raw_players, (list, tuple)) else []
    players = [
        player for player in raw_players[:80]
        if isinstance(player, Mapping)
    ]
    selectable = [player for player in players if bool(player.get("selectable"))]
    return {
        "available": bool(catalog.get("available")),
        "formation": str(catalog.get("formation") or "unknown")[:32],
        "registered_players": len(players),
        "selectable_players": len(selectable),
        "high_load_players": sum(
            _finite(player.get("minutes_ema")) >= 72.0 for player in selectable
        ),
        "mean_selectable_form": (
            round(sum(_finite(player.get("form_ema"), 0.55) for player in selectable) / len(selectable), 4)
            if selectable else None
        ),
        "condition": _condition(catalog),
    }


def build_prematch_intelligence(
    command: Mapping[str, Any],
    manager_squad: Mapping[str, Any] | None,
    opponent_squad: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build deterministic decision prompts without predicting an outcome."""
    if not isinstance(command, Mapping) or not command.get("enabled"):
        return {"schema_version": 1, "available": False, "reason": "manager_mode_unavailable"}
    fixture = command.get("current_fixture") or command.get("next_fixture")
    if not isinstance(fixture, Mapping):
        return {"schema_version": 1, "available": False, "reason": "no_pending_fixture"}

    manager = _squad_facts(manager_squad)
    opponent = _squad_facts(opponent_squad)
    condition = manager["condition"]
    signals: list[dict[str, Any]] = []

    def add_signal(
        signal_id: str, severity: str, title: str, evidence: str, prompt: str,
    ) -> None:
        signals.append({
            "id": signal_id, "severity": severity, "title": title,
            "evidence": evidence, "decision_prompt": prompt,
        })

    fatigue = condition["fatigue"]
    if fatigue >= 0.65:
        add_signal(
            "high_fatigue", "high", "High accumulated fatigue",
            f"persisted team fatigue is {fatigue:.2f}",
            "Review balanced or rotate selection; rotation reduces the engine's load component but has an explicit status cost.",
        )
    elif fatigue >= 0.40:
        add_signal(
            "moderate_fatigue", "moderate", "Meaningful accumulated fatigue",
            f"persisted team fatigue is {fatigue:.2f}",
            "Compare strongest and balanced lineups before freezing the decision.",
        )

    unavailable = condition["unavailable_players"]
    if unavailable:
        add_signal(
            "player_unavailability", "high" if unavailable >= 3 else "moderate",
            "Player availability constraint",
            f"{condition['injured_players']} injured and {condition['suspended_players']} suspended",
            "Verify role coverage and planned substitutions against the currently selectable squad.",
        )
    if manager["available"] and manager["selectable_players"] < 14:
        add_signal(
            "thin_match_squad", "high", "Thin selectable squad",
            f"only {manager['selectable_players']} players are currently selectable",
            "Prefer a contingency plan that does not depend on unavailable bench roles.",
        )
    if condition["morale"] < 0.45:
        add_signal(
            "low_morale", "moderate", "Low simulated squad morale",
            f"persisted squad morale is {condition['morale']:.2f}",
            "Treat this as a readiness warning; the current evidence does not identify a superior tactic.",
        )
    if condition["media_pressure"] >= 0.60:
        add_signal(
            "media_pressure", "moderate", "Elevated simulated media pressure",
            f"persisted media pressure is {condition['media_pressure']:.2f}",
            "Use explicit score-state contingencies so the match plan remains auditable under pressure.",
        )

    decision = fixture.get("decision")
    if isinstance(decision, Mapping):
        plan = decision.get("in_match_plan")
        raw_instructions = plan.get("instructions") if isinstance(plan, Mapping) else None
        instructions = (
            list(raw_instructions[:5])
            if isinstance(raw_instructions, (list, tuple)) else []
        )
        covered = sorted({
            str(item.get("condition")) for item in instructions
            if isinstance(item, Mapping) and item.get("condition")
        })
        if not instructions:
            add_signal(
                "no_contingency_plan", "informational", "No in-match contingency frozen",
                "the current manager decision has no in-match instruction",
                "Optionally add bounded score-state rules before the fixture starts.",
            )
    else:
        covered = []

    if not signals:
        add_signal(
            "no_material_condition_warning", "informational",
            "No material condition warning",
            "configured fatigue, availability, morale and pressure thresholds are clear",
            "Choose a plan for gameplay intent; this briefing does not predict the result.",
        )

    manager_persisted = condition["source"] == "persisted_squad_carryover"
    opponent_persisted = (
        opponent["condition"]["source"] == "persisted_squad_carryover"
    )
    coverage = (
        "direct_persisted_state"
        if manager_persisted and opponent_persisted and manager["available"] and opponent["available"]
        else "mixed_persisted_and_baseline"
        if (manager_persisted or opponent_persisted) and manager["available"] and opponent["available"]
        else "roster_plus_default_baseline" if manager["available"] and opponent["available"]
        else "partial_team_state"
    )
    return {
        "schema_version": 1,
        "available": True,
        "fixture_id": str(fixture.get("fixture_id") or "")[:96],
        "evidence_coverage": coverage,
        "manager": manager,
        "opponent": opponent,
        "table_context": {
            "manager_position": fixture.get("manager_position"),
            "manager_points": fixture.get("manager_points"),
            "opponent_position": fixture.get("opponent_position"),
            "opponent_points": fixture.get("opponent_points"),
        },
        "contingency_conditions": covered,
        "signals": signals[:8],
        "rotation_tradeoff": {
            "strongest": {"status_penalty": 0.0, "rotation_level": 0.0},
            "balanced": {"status_penalty": 1.25, "rotation_level": 0.5},
            "rotate": {"status_penalty": 3.0, "rotation_level": 1.0},
            "boundary": "engine mechanics, not an estimate of match outcome",
        },
        "claim_boundary": (
            "decision support from simulated persisted state and roster facts only; "
            "no tactic ranking, win probability or real-world recommendation"
        ),
    }


def build_postmatch_debrief(
    report: Mapping[str, Any], *, manager_team: str, expected_match_id: str,
) -> dict[str, Any]:
    """Extract direct execution facts while refusing single-match causality."""
    if not isinstance(report, Mapping) or str(report.get("match_id")) != expected_match_id:
        return {"schema_version": 1, "available": False, "reason": "report_identity_mismatch"}
    fixture = report.get("fixture")
    if not isinstance(fixture, Mapping):
        return {"schema_version": 1, "available": False, "reason": "fixture_evidence_missing"}
    if manager_team == fixture.get("home"):
        side = "home"
    elif manager_team == fixture.get("away"):
        side = "away"
    else:
        return {"schema_version": 1, "available": False, "reason": "manager_team_identity_mismatch"}
    layers = report.get("layers")
    management = layers.get("management") if isinstance(layers, Mapping) else None
    if not isinstance(management, Mapping):
        return {"schema_version": 1, "available": False, "reason": "manager_decision_evidence_missing"}
    decision = management.get("decision")
    if not isinstance(decision, Mapping) or decision.get("team") != manager_team:
        return {"schema_version": 1, "available": False, "reason": "manager_decision_evidence_missing"}
    raw_support = management.get("club_support")
    support_view = {"available": False, "reason": "legacy_report_without_club_support"}
    expected_support_factor = 1.0
    if raw_support is not None:
        if not isinstance(raw_support, Mapping) or raw_support.get("team") != manager_team:
            return {"schema_version": 1, "available": False, "reason": "club_support_identity_mismatch"}
        try:
            from src.product.season import ClubResourcePlan, club_resource_effects

            support_plan = ClubResourcePlan.from_payload(raw_support.get("plan") or {})
            support_effects = club_resource_effects(support_plan)
        except (TypeError, ValueError):
            return {"schema_version": 1, "available": False, "reason": "club_support_evidence_invalid"}
        if raw_support.get("effects") != support_effects:
            return {"schema_version": 1, "available": False, "reason": "club_support_evidence_invalid"}
        expected_support_factor = float(support_effects["fatigue_load_factor"])
        support_view = {
            "available": True,
            "plan": {
                "recovery": support_plan.recovery,
                "medical": support_plan.medical,
                "sports_science": support_plan.sports_science,
            },
            "effects": support_effects,
        }
    effects = management.get("effects") or {}
    effect = effects.get(side) if isinstance(effects, Mapping) else None
    effect = effect if isinstance(effect, Mapping) else {}
    observed_support_factor = _finite(
        effect.get("club_fatigue_load_factor"), 1.0,
    )
    if raw_support is not None and observed_support_factor != expected_support_factor:
        return {"schema_version": 1, "available": False, "reason": "club_support_runtime_mismatch"}
    runtime_map = management.get("in_match") or {}
    runtime = runtime_map.get(side) if isinstance(runtime_map, Mapping) else None
    runtime = runtime if isinstance(runtime, Mapping) else {}
    raw_outcomes = runtime.get("outcomes")
    raw_outcomes = raw_outcomes if isinstance(raw_outcomes, (list, tuple)) else []
    outcomes = [
        item for item in raw_outcomes[:5]
        if isinstance(item, Mapping)
    ]
    counts = {
        status: sum(str(item.get("status")) == status for item in outcomes)
        for status in ("applied", "skipped", "failed")
    }
    result = report.get("result")
    result = result if isinstance(result, Mapping) else {}
    score = result.get("score")
    score = score if isinstance(score, Mapping) else {}
    lineup = decision.get("lineup")
    lineup = lineup if isinstance(lineup, Mapping) else {}
    base = _finite(effect.get("base_status"))
    effective = _finite(effect.get("effective_status"), base)
    try:
        tactical_binding = _tactical_binding(
            management, side=side, team=manager_team,
            expected_tactic=str(decision.get("tactic") or "team_identity"),
        )
    except ValueError:
        return {
            "schema_version": 1, "available": False,
            "reason": "tactical_execution_binding_mismatch",
        }
    try:
        world_model_action_execution = project_world_model_action_execution(
            report,
            manager_team=manager_team,
            expected_match_id=expected_match_id,
        )
    except ValueError:
        world_model_action_execution = {
            "schema_version": 1,
            "available": False,
            "reason": "world_model_action_evidence_invalid",
        }
    return {
        "schema_version": 1,
        "available": True,
        "match_id": expected_match_id,
        "team": manager_team,
        "side": side,
        "decision": {
            "tactic": str(decision.get("tactic") or "team_identity")[:64],
            "rotation": str(decision.get("rotation") or "balanced")[:32],
            "lineup_source": str(lineup.get("source") or "team_level")[:32],
        },
        "direct_engine_effects": {
            "base_status": round(base, 6),
            "effective_status": round(effective, 6),
            "combined_status_delta": round(effective - base, 6),
            "rotation_status_penalty": round(_finite(effect.get("rotation_status_penalty")), 6),
            "fatigue_load_multiplier": round(_finite(effect.get("fatigue_load_multiplier"), 1.0), 6),
            "club_fatigue_load_factor": round(observed_support_factor, 6),
        },
        "club_support": support_view,
        "in_match_execution": {
            "evaluated": len(outcomes), **counts,
            "outcomes": [
                {
                    "minute": max(0, int(_finite(item.get("scheduled_minute")))),
                    "condition": str(item.get("condition") or "always")[:32],
                    "status": str(item.get("status") or "unknown")[:32],
                    "reason": str(item.get("reason") or "")[:160],
                }
                for item in outcomes
            ],
        },
        "tactical_binding": tactical_binding,
        "world_model_action_execution": world_model_action_execution,
        "observed_result": {
            "score": {
                "home": max(0, int(_finite(score.get("home")))),
                "away": max(0, int(_finite(score.get("away")))),
            },
            "descriptive_only": True,
        },
        "evidence_grade": "direct_runtime_execution",
        "causal_outcome_attribution": False,
        "claim_boundary": (
            "engine effects, instruction execution and bounded world-model action "
            "records are direct runtime facts; one observed result cannot establish "
            "that the decision or action policy caused the outcome"
        ),
    }
