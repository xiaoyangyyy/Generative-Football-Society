"""Machine-owned stopping tests for active-probe portfolios."""

import copy

from src.match_engine.cognitive.schemas import validate_coach_plan
from src.match_engine.world_model.active_probe import build_active_probe_design
from src.match_engine.world_model.active_probe_memory import (
    ActiveProbeDiscoveryMemory,
    ActiveProbeDiscoveryProfile,
)
from src.match_engine.world_model.active_probe_portfolio import (
    build_active_probe_portfolio_design,
)
from src.match_engine.world_model.active_probe_sequential_policy import (
    active_probe_sequential_policy_audit_is_valid,
    active_probe_sequential_policy_design_is_valid,
    build_active_probe_sequential_policy_design,
    evaluate_llm_active_probe_sequential_policy,
)
from src.match_engine.world_model.active_probe_sequential_policy_evaluation import (
    active_probe_sequential_policy_diagnostics,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)


def _packet(memory=None):
    packet = {
        "team_id": "Home",
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a",
        "decision_context": {"clock_seconds": 100.0},
        "active_learning": {
            "eligible": True, "exploit_action": "shot",
            "exploration_action": "pass", "estimated_regret": 0.03,
            "constraints": {"max_regret": 0.08},
        },
        "candidates": [
            {"action": "shot", "multi_horizon_predictions": {
                "transition": {"retention_probability": 0.20},
                "60s": {"retention_probability": 0.20},
            }},
            {"action": "pass", "multi_horizon_predictions": {
                "transition": {"retention_probability": 0.80},
                "60s": {"retention_probability": 0.80},
            }},
        ],
    }
    packet["active_probe_design"] = build_active_probe_design(
        packet, discovery_memory=memory,
    )
    packet["active_probe_portfolio_design"] = (
        build_active_probe_portfolio_design(packet)
    )
    packet["active_probe_sequential_policy_design"] = (
        build_active_probe_sequential_policy_design(packet)
    )
    return packet


def _profile(horizon, cumulative, sequential_status):
    return ActiveProbeDiscoveryProfile(
        action="pass", null_action="shot", horizon=horizon,
        endpoint="retained_possession", matches=2, rows=2,
        training_matches=1, validation_matches=1, calibration_offset=0.0,
        raw_validation_brier=0.0, corrected_validation_brier=0.0,
        validation_skill=0.0, train_observed_rate=0.5,
        validation_observed_rate=0.5, observed_rate_shift=0.0,
        status="insufficient_history", authority=0.0,
        cumulative_log_likelihood_ratio=cumulative,
        mean_log_likelihood_ratio=cumulative / 2.0,
        sequential_status=sequential_status,
    )


def test_continue_option_expands_only_its_exact_authorized_portfolio():
    packet = _packet()
    design = packet["active_probe_sequential_policy_design"]
    assert active_probe_sequential_policy_design_is_valid(
        design, packet["active_probe_design"],
        packet["active_probe_portfolio_design"],
    )
    assert all(row["decision"] == "continue" for row in design["options"])
    option = design["options"][0]
    validated = validate_coach_plan({
        "world_model_active_probe_sequential_policy": {
            "policy_option_id": option["policy_option_id"],
            "confidence": 0.8, "rationale": "continue exact portfolio",
        }
    })
    audit = evaluate_llm_active_probe_sequential_policy(
        packet, validated["world_model_active_probe_sequential_policy"],
        selected_action="pass", decision_mode="explore",
        selected_after_action_freeze=True,
    )
    assert active_probe_sequential_policy_audit_is_valid(audit)
    assert audit["policy_option"]["portfolio_id"] == (
        audit["portfolio_audit"]["portfolio"]["portfolio_id"]
    )

    tampered = copy.deepcopy(design)
    tampered["options"][0]["maximum_matches"] = 200
    assert not active_probe_sequential_policy_design_is_valid(tampered)


def test_opposite_horizon_boundaries_force_machine_conflict_stop():
    profiles = {
        ("pass", "shot", "transition", "retained_possession"):
            _profile("transition", 3.2, "stop_supported"),
        ("pass", "shot", "60s", "retained_possession"):
            _profile("60s", -3.2, "stop_falsified"),
    }
    memory = ActiveProbeDiscoveryMemory(
        checkpoint_signature="checkpoint-a", environment_signature="env-a",
        profiles=profiles, source_logs=2, compatible_matches=2,
    )
    packet = _packet(memory)
    design = packet["active_probe_sequential_policy_design"]
    assert design["conflict_review_required"]
    assert len(design["options"]) == 1
    option = design["options"][0]
    assert option["decision"] == "stop"
    assert option["reason"] == "conflicting_horizon_boundaries"
    audit = evaluate_llm_active_probe_sequential_policy(
        packet, {"policy_option_id": option["policy_option_id"],
                 "confidence": 0.9, "rationale": "review conflict"},
        selected_action="shot", decision_mode="exploit",
        selected_after_action_freeze=True,
    )
    assert active_probe_sequential_policy_audit_is_valid(audit)
    assert audit["portfolio_audit"] is None


def test_sequential_policy_has_an_independent_strict_online_gate():
    continue_packet = _packet()
    continue_option = continue_packet[
        "active_probe_sequential_policy_design"
    ]["options"][0]
    continue_audit = evaluate_llm_active_probe_sequential_policy(
        continue_packet,
        {"policy_option_id": continue_option["policy_option_id"],
         "confidence": 0.8},
        selected_action="pass", decision_mode="explore",
        selected_after_action_freeze=True,
    )
    profiles = {
        ("pass", "shot", "transition", "retained_possession"):
            _profile("transition", 3.2, "stop_supported"),
        ("pass", "shot", "60s", "retained_possession"):
            _profile("60s", -3.2, "stop_falsified"),
    }
    stop_packet = _packet(ActiveProbeDiscoveryMemory(
        checkpoint_signature="checkpoint-a", environment_signature="env-a",
        profiles=profiles, source_logs=2, compatible_matches=2,
    ))
    stop_option = stop_packet[
        "active_probe_sequential_policy_design"
    ]["options"][0]
    stop_audit = evaluate_llm_active_probe_sequential_policy(
        stop_packet,
        {"policy_option_id": stop_option["policy_option_id"],
         "confidence": 0.8},
        selected_action="shot", decision_mode="exploit",
        selected_after_action_freeze=True,
    )
    clusters = [
        [{"llm_active_probe_sequential_policy_context": audit}]
        for audit in (continue_audit, continue_audit, stop_audit, stop_audit)
    ]
    diagnostics = active_probe_sequential_policy_diagnostics(clusters)
    assert diagnostics["continue_decisions"] == 2
    assert diagnostics["machine_stop_decisions"] == 2
    assert diagnostics["boundary_compliance_violations"] == 0
    logs = [
        {"world_model_decision_adoption": {"records": records}}
        for records in clusters
    ]
    report = aggregate_online_calibration(
        logs, min_transitions=0,
        require_active_probe_sequential_policy=True,
    )
    assert report["version"] == 45
    assert report["active_probe_sequential_policy_ready"]
    assert report["gates"]["active_probe_sequential_policy"]
