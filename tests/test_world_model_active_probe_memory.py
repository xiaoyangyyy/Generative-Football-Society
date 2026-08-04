"""Held-out probe discoveries calibrate only future probe forecasts."""

import copy
import math

import pytest

from src.match_engine.world_model.active_probe import (
    active_probe_design_is_valid,
    build_active_probe_design,
    evaluate_llm_active_probe,
    score_active_probe,
)
from src.match_engine.world_model.active_probe_memory import (
    active_probe_discovery_memory_diagnostics,
    compile_active_probe_discovery_memory,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)


def _packet(*, environment="env-a"):
    return {
        "team_id": "Home",
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": environment,
        "decision_context": {"clock_seconds": 100.0},
        "active_learning": {
            "eligible": True,
            "exploit_action": "shot",
            "exploration_action": "pass",
            "estimated_regret": 0.03,
            "constraints": {"max_regret": 0.08},
        },
        "candidates": [
            {
                "action": "shot",
                "multi_horizon_predictions": {
                    "transition": {"retention_probability": 0.10},
                },
            },
            {
                "action": "pass",
                "multi_horizon_predictions": {
                    "transition": {"retention_probability": 0.30},
                },
            },
        ],
    }


def _log(observed: bool, index: int):
    packet = _packet()
    design = build_active_probe_design(packet)
    packet["active_probe_design"] = design
    option = design["options"][0]
    audit = evaluate_llm_active_probe(
        packet,
        {"probe_id": option["probe_id"], "confidence": 0.8},
        selected_action="pass",
        decision_mode="explore",
        selected_after_action_freeze=True,
    )
    evaluation = score_active_probe(
        audit,
        {"retained_possession": observed},
        horizon="transition",
        realized_action="pass",
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    record = {
        "decision_id": f"Home:{index}",
        "team_id": "Home",
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a",
        "llm_active_probe_context": audit,
        "intervention_actual_action": "pass",
        "multi_horizon_regime_outcomes": {
            "transition": {"llm_active_probe_evaluation": evaluation},
        },
    }
    return {"world_model_decision_adoption": {"records": [record]}}


def test_held_out_discovery_memory_calibrates_without_mutating_raw_forecast():
    logs = [_log(True, index) for index in range(8)]
    memory = compile_active_probe_discovery_memory(
        logs,
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profile(
        action="pass",
        null_action="shot",
        horizon="transition",
        endpoint="retained_possession",
    )

    assert profile is not None and profile.active
    assert profile.training_matches == 4
    assert profile.validation_matches == 4
    assert profile.validation_skill >= 0.02
    assert 0.0 < profile.authority <= 0.35
    assert profile.cumulative_log_likelihood_ratio == pytest.approx(
        8.0 * math.log(0.30 / 0.10)
    )
    assert profile.sequential_status == "stop_supported"

    packet = _packet()
    design = build_active_probe_design(packet, discovery_memory=memory)
    assert active_probe_design_is_valid(design)
    option = design["options"][0]
    assert option["raw_alternative_probability"] == 0.30
    assert option["alternative_probability"] > 0.30
    assert option["discovery_memory"]["active"]
    assert not option["discovery_memory"]["can_update_world_model"]

    incompatible = build_active_probe_design(
        _packet(environment="env-b"), discovery_memory=memory,
    )["options"][0]
    assert incompatible["alternative_probability"] == 0.30
    assert not incompatible["discovery_memory"]["active"]
    assert incompatible["discovery_memory"]["status"] == (
        "incompatible_checkpoint_or_environment"
    )


def test_observed_rate_shift_quarantines_discovery_instead_of_reusing_it():
    logs = [
        _log(index >= 4, index) for index in range(8)
    ]
    memory = compile_active_probe_discovery_memory(
        logs,
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profile(
        action="pass",
        null_action="shot",
        horizon="transition",
        endpoint="retained_possession",
    )
    assert profile is not None
    assert profile.status == "quarantined_drift"
    assert profile.observed_rate_shift == 1.0
    assert profile.authority == 0.0

    class BrokenMemory:
        def calibration(self, **kwargs):
            raise ValueError("malformed external memory")

    fallback = build_active_probe_design(
        _packet(), discovery_memory=BrokenMemory(),
    )
    assert active_probe_design_is_valid(fallback)
    assert not fallback["options"][0]["discovery_memory"]["active"]


def test_discovery_memory_has_an_independent_strict_online_gate():
    logs = [_log(True, index) for index in range(8)]
    diagnostics = active_probe_discovery_memory_diagnostics(logs)
    assert diagnostics["active_profiles"] == 1
    assert diagnostics["active_profile_matches"] == 8
    assert diagnostics["all_authority_bounded"]
    report = aggregate_online_calibration(
        logs,
        min_transitions=0,
        require_active_probe_discovery_memory=True,
    )
    assert report["version"] == 44
    assert report["active_probe_discovery_memory_ready"]
    assert report["gates"]["active_probe_discovery_memory"]

    tampered = copy.deepcopy(logs)
    evaluation = tampered[0]["world_model_decision_adoption"]["records"][0][
        "multi_horizon_regime_outcomes"
    ]["transition"]["llm_active_probe_evaluation"]
    evaluation["raw_alternative_probability"] = 0.99
    rejected = active_probe_discovery_memory_diagnostics(tampered)
    assert rejected["malformed_source_rows"] == 1


def test_sequential_memory_stops_inconclusive_at_twenty_matches():
    logs = [
        _log(index < 4, index) for index in range(20)
    ]
    memory = compile_active_probe_discovery_memory(
        logs, checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profile(
        action="pass", null_action="shot", horizon="transition",
        endpoint="retained_possession",
    )
    assert profile is not None
    assert abs(profile.cumulative_log_likelihood_ratio) < math.log(20.0)
    assert profile.sequential_status == "stop_inconclusive_maximum_matches"
