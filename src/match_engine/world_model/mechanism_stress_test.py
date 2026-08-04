"""Bounded post-action context stress tests for predictive mechanisms."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.context_interventions import (
    CONTEXT_FACTORS,
    neutralize_context,
)
from src.match_engine.world_model.predictive_mechanism import (
    predictive_mechanism_design_is_valid,
)
from src.match_engine.world_model.state_scales import predictive_mechanism_edges


MECHANISM_STRESS_TEST_VERSION = 1
MAXIMUM_STRESSED_MECHANISMS = 2
MAXIMUM_MODEL_CALLS = 10
MAXIMUM_STRESS_OPTIONS = 8
_CELLS = (
    "driver_false_outcome_false", "driver_false_outcome_true",
    "driver_true_outcome_false", "driver_true_outcome_true",
)


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _joint(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != set(_CELLS):
        return None
    try:
        result = {key: float(value[key]) for key in _CELLS}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        not all(math.isfinite(number) and 0.0 < number < 1.0
                for number in result.values())
        or abs(sum(result.values()) - 1.0) > 1e-9
    ):
        return None
    return result


def _classification(score: float, flipped: bool) -> str:
    if flipped or score > 0.20:
        return "fragile"
    if score > 0.05:
        return "context_sensitive"
    return "robust_within_tested_neutralization"


def _stress_option(
    source: dict[str, Any], *, factor: str, masked_indices: list[int],
    intervention_magnitude: float, neutralized_edge: dict[str, Any],
) -> dict[str, Any] | None:
    observed = _joint(source.get("joint_probabilities"))
    neutralized = _joint(neutralized_edge.get("joint_probabilities"))
    if observed is None or neutralized is None:
        return None
    try:
        observed_lift = float(source["conditional_lift"])
        neutralized_lift = float(neutralized_edge["conditional_lift"])
        members = int(neutralized_edge["ensemble_members"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (
        observed_lift, neutralized_lift, intervention_magnitude,
    )) or members < 2:
        return None
    total_variation = 0.5 * sum(
        abs(observed[key] - neutralized[key]) for key in _CELLS
    )
    lift_shift = neutralized_lift - observed_lift
    relation_flipped = bool(
        observed_lift * neutralized_lift < 0.0
        and abs(observed_lift) >= 0.05
        and abs(neutralized_lift) >= 0.05
    )
    fragility = float(min(1.0, max(
        total_variation, abs(lift_shift) / 2.0,
        1.0 if relation_flipped else 0.0,
    )))
    payload = {
        "version": MECHANISM_STRESS_TEST_VERSION,
        "source_hypothesis_id": str(source["hypothesis_id"]),
        "action": str(source["action"]),
        "horizon": str(source["horizon"]),
        "driver_event": str(source["driver_event"]),
        "outcome_event": str(source["outcome_event"]),
        "observed_relationship": str(source["relationship"]),
        "context_factor": str(factor),
        "masked_feature_indices": list(map(int, masked_indices)),
        "intervention_magnitude": float(intervention_magnitude),
        "observed_context_joint_probabilities": observed,
        "neutralized_context_joint_probabilities": neutralized,
        "observed_context_conditional_lift": observed_lift,
        "neutralized_context_conditional_lift": neutralized_lift,
        "conditional_lift_shift": lift_shift,
        "joint_total_variation": float(total_variation),
        "relationship_flipped": relation_flipped,
        "fragility_score": fragility,
        "classification": _classification(fragility, relation_flipped),
        "neutralized_ensemble_members": members,
        "stress_scope": "single_schema_grounded_neutralization",
        "selected_after_action_freeze": True,
        "shadow_only": True,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "stress_test_id": _digest(payload, "mechanism-stress-option:"),
    }


def _option_is_valid(option: Any) -> bool:
    if not isinstance(option, dict):
        return False
    payload = dict(option)
    stress_test_id = payload.pop("stress_test_id", None)
    observed = _joint(option.get("observed_context_joint_probabilities"))
    neutralized = _joint(
        option.get("neutralized_context_joint_probabilities")
    )
    try:
        observed_lift = float(option["observed_context_conditional_lift"])
        neutralized_lift = float(option[
            "neutralized_context_conditional_lift"
        ])
        lift_shift = float(option["conditional_lift_shift"])
        total_variation = float(option["joint_total_variation"])
        fragility = float(option["fragility_score"])
        magnitude = float(option["intervention_magnitude"])
        members = int(option["neutralized_ensemble_members"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if observed is None or neutralized is None:
        return False
    expected_tv = 0.5 * sum(
        abs(observed[key] - neutralized[key]) for key in _CELLS
    )
    expected_flip = bool(
        observed_lift * neutralized_lift < 0.0
        and abs(observed_lift) >= 0.05
        and abs(neutralized_lift) >= 0.05
    )
    expected_fragility = min(1.0, max(
        expected_tv, abs(neutralized_lift - observed_lift) / 2.0,
        1.0 if expected_flip else 0.0,
    ))
    return bool(
        stress_test_id == _digest(payload, "mechanism-stress-option:")
        and option.get("action") in {"hold", "pass", "cross", "shot"}
        and bool(option.get("source_hypothesis_id"))
        and bool(option.get("horizon"))
        and option.get("driver_event") != option.get("outcome_event")
        and option.get("context_factor") in CONTEXT_FACTORS
        and bool(option.get("masked_feature_indices"))
        and math.isfinite(magnitude) and magnitude > 0.0
        and all(math.isfinite(value) for value in (
            observed_lift, neutralized_lift, lift_shift,
            total_variation, fragility,
        ))
        and abs(lift_shift - (neutralized_lift - observed_lift)) <= 1e-9
        and abs(total_variation - expected_tv) <= 1e-9
        and option.get("relationship_flipped") == expected_flip
        and abs(fragility - expected_fragility) <= 1e-9
        and option.get("classification")
        == _classification(expected_fragility, expected_flip)
        and members >= 2
        and option.get("stress_scope")
        == "single_schema_grounded_neutralization"
        and option.get("selected_after_action_freeze") is True
        and option.get("shadow_only") is True
        and option.get("can_change_current_action") is False
        and option.get("can_change_tactical_controls") is False
        and option.get("can_schedule_future_action") is False
        and option.get("can_update_world_model") is False
        and option.get("causal_interpretation") is False
    )


def _planning_allowed(runtime: Any, rollout_steps: int) -> bool:
    if rollout_steps == 1:
        return True
    provider = getattr(runtime, "two_step_planning_gate", None)
    if not callable(provider):
        return False
    try:
        gate = provider() or {}
        return bool(gate.get("active") and float(gate.get("authority", 0.0)) > 0.0)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False


def build_mechanism_stress_test_design(
    runtime: Any, state: Any, packet: dict[str, Any], *, team_id: str,
    selected_action: str, maximum_model_calls: int = MAXIMUM_MODEL_CALLS,
) -> dict[str, Any]:
    mechanism_design = packet.get("predictive_mechanism_design") or {}
    action = str(selected_action).lower()
    source_options = [
        row for row in mechanism_design.get("options") or []
        if str(row.get("action")) == action
    ][:MAXIMUM_STRESSED_MECHANISMS]
    candidate = next((
        row for row in packet.get("candidates") or []
        if str(row.get("action")) == action
    ), None)
    budget = max(0, min(MAXIMUM_MODEL_CALLS, int(maximum_model_calls)))
    options = []
    calls = 0
    if (
        runtime is not None and candidate is not None
        and predictive_mechanism_design_is_valid(mechanism_design)
    ):
        attacking_home = str(team_id) == str(state.home.team_id)
        try:
            observation = np.asarray(runtime.encode_state(
                state, attacking_home=attacking_home,
            ), dtype=np.float32)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            observation = np.asarray([], dtype=np.float32)
        for source in source_options:
            prediction = (
                candidate.get("multi_horizon_predictions") or {}
            ).get(str(source["horizon"])) or {}
            rollout_steps = int(prediction.get("rollout_steps", 1))
            horizon_s = float(prediction.get(
                "horizon_s", packet.get("horizon_s", 10.0),
            ))
            if (
                observation.ndim != 1
                or not _planning_allowed(runtime, rollout_steps)
            ):
                continue
            action_vector = encode_high_level_action(
                action, target=np.asarray(candidate["target"], dtype=np.float32),
                horizon_s=horizon_s / max(1, rollout_steps),
            )
            action_vector[13] = float(candidate.get("physics_prior", 0.5))
            for factor in CONTEXT_FACTORS:
                if calls >= budget:
                    break
                try:
                    neutral, indices, magnitude = neutralize_context(
                        observation, factor=factor,
                        attacking_home=attacking_home,
                    )
                    if magnitude < 1e-6:
                        continue
                    confidence = float(runtime.planner_confidence(
                        neutral, kind="shot" if action == "shot" else "pass",
                    ))
                    if not math.isfinite(confidence) or confidence <= 0.0:
                        continue
                    output = runtime.imagine(
                        neutral, action_vector, steps=rollout_steps,
                        carry_hidden=False,
                        quality_kind="shot" if action == "shot" else "pass",
                    )
                    calls += 1
                    samples = (output.uncertainty_samples or {}).get(
                        "transition_states"
                    )
                    trained = bool((output.uncertainty_samples or {}).get(
                        "transition_ensemble_trained", False,
                    ))
                    edges = predictive_mechanism_edges(
                        neutral, samples, ensemble_trained=trained,
                    )
                    edge = edges.get(
                        f"{source['driver_event']}->{source['outcome_event']}"
                    ) or {}
                    if not edge.get("available"):
                        continue
                    option = _stress_option(
                        source, factor=factor, masked_indices=indices,
                        intervention_magnitude=magnitude,
                        neutralized_edge=edge,
                    )
                except (
                    AttributeError, KeyError, RuntimeError, TypeError,
                    ValueError, OverflowError,
                ):
                    option = None
                if option is not None:
                    options.append(option)
    options.sort(key=lambda row: (
        -float(row["fragility_score"]),
        -float(row["joint_total_variation"]),
        str(row["context_factor"]), str(row["stress_test_id"]),
    ))
    options = options[:MAXIMUM_STRESS_OPTIONS]
    payload = {
        "version": MECHANISM_STRESS_TEST_VERSION,
        "available": bool(options),
        "reason": (
            "bounded_context_mechanism_stress_tests_available" if options
            else "no_valid_post_action_mechanism_stress_test"
        ),
        "team_id": str(team_id), "selected_action": action,
        "checkpoint_signature": str(packet.get("checkpoint_signature", "")),
        "environment_signature": str(packet.get("environment_signature", "")),
        "source_mechanism_design_digest": str(mechanism_design.get(
            "design_digest", "",
        )),
        "options": options,
        "recommended_stress_test_id": (
            options[0]["stress_test_id"] if options else "none"
        ),
        "stressed_mechanisms": len(source_options),
        "maximum_stressed_mechanisms": MAXIMUM_STRESSED_MECHANISMS,
        "model_calls": calls, "model_call_budget": budget,
        "selection_scope": "post_action_explanation_only",
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "design_digest": _digest(payload, "mechanism-stress-design:"),
    }


def mechanism_stress_test_design_is_valid(design: Any) -> bool:
    if not isinstance(design, dict):
        return False
    payload = dict(design)
    digest = payload.pop("design_digest", None)
    options = design.get("options") or []
    try:
        calls = int(design["model_calls"])
        budget = int(design["model_call_budget"])
        stressed = int(design["stressed_mechanisms"])
        maximum_stressed = int(design["maximum_stressed_mechanisms"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        digest == _digest(payload, "mechanism-stress-design:")
        and design.get("version") == MECHANISM_STRESS_TEST_VERSION
        and bool(design.get("available")) == bool(options)
        and all(_option_is_valid(row) for row in options)
        and len(options) <= MAXIMUM_STRESS_OPTIONS
        and len({row["stress_test_id"] for row in options}) == len(options)
        and all(row["action"] == design.get("selected_action") for row in options)
        and design.get("recommended_stress_test_id")
        == (options[0]["stress_test_id"] if options else "none")
        and 0 <= calls <= budget <= MAXIMUM_MODEL_CALLS
        and stressed <= maximum_stressed == MAXIMUM_STRESSED_MECHANISMS
        and design.get("selection_scope") == "post_action_explanation_only"
        and design.get("can_change_current_action") is False
        and design.get("can_change_tactical_controls") is False
        and design.get("can_schedule_future_action") is False
        and design.get("can_update_world_model") is False
        and design.get("causal_interpretation") is False
    )


def validate_llm_mechanism_stress_test(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    stress_test_id = str(raw.get("stress_test_id", ""))[:96]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not stress_test_id or not math.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
        return None
    return {
        "stress_test_id": stress_test_id, "confidence": confidence,
        "failure_condition": str(raw.get("failure_condition", ""))[:280],
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def evaluate_llm_mechanism_stress_test(
    packet: dict[str, Any], raw: Any, *, selected_action: str,
    selected_after_action_freeze: bool,
) -> dict[str, Any]:
    selection = validate_llm_mechanism_stress_test(raw)
    design = packet.get("mechanism_stress_test_design") or {}
    base = {
        "version": MECHANISM_STRESS_TEST_VERSION, "accepted": False,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False, "causal_interpretation": False,
    }
    if selection is None:
        return {**base, "reason": "missing_or_invalid_stress_test_selection"}
    if not selected_after_action_freeze or not mechanism_stress_test_design_is_valid(design):
        return {**base, "reason": "post_action_valid_stress_design_required"}
    option = next((
        row for row in design["options"]
        if row["stress_test_id"] == selection["stress_test_id"]
    ), None)
    if option is None:
        return {**base, "reason": "unknown_mechanism_stress_test"}
    if str(option["action"]) != str(selected_action).lower():
        return {**base, "reason": "frozen_action_stress_test_mismatch"}
    payload = {
        **base, "accepted": True,
        "reason": "engine_computed_mechanism_stress_test_articulated",
        "selection": selection, "stress_test": option,
        "source_design_digest": design["design_digest"],
        "team_id": design["team_id"],
        "checkpoint_signature": design["checkpoint_signature"],
        "environment_signature": design["environment_signature"],
        "selected_after_action_freeze": True, "shadow_only": True,
    }
    return {
        **payload,
        "audit_digest": _digest(payload, "mechanism-stress-audit:"),
    }


def mechanism_stress_test_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    selection = validate_llm_mechanism_stress_test(audit.get("selection"))
    option = audit.get("stress_test") or {}
    return bool(
        digest == _digest(payload, "mechanism-stress-audit:")
        and selection is not None
        and selection["stress_test_id"] == option.get("stress_test_id")
        and _option_is_valid(option)
        and audit.get("selected_after_action_freeze") is True
        and audit.get("shadow_only") is True
        and audit.get("can_change_current_action") is False
        and audit.get("can_change_tactical_controls") is False
        and audit.get("can_schedule_future_action") is False
        and audit.get("can_update_world_model") is False
        and audit.get("causal_interpretation") is False
    )
