"""Joint-trajectory predictive mechanism contracts."""

import copy
from types import SimpleNamespace

import numpy as np

from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.predictive_mechanism import (
    build_predictive_mechanism_design,
    evaluate_llm_predictive_mechanism,
    predictive_mechanism_audit_is_valid,
    predictive_mechanism_design_is_valid,
    predictive_mechanism_evaluation_is_valid,
    score_predictive_mechanism,
)
from src.match_engine.world_model.predictive_mechanism_evaluation import (
    predictive_mechanism_diagnostics,
)
from src.match_engine.world_model.predictive_mechanism_memory import (
    compile_predictive_mechanism_memory,
    predictive_mechanism_memory_diagnostics,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)
from src.match_engine.world_model.decision_adoption import (
    record_policy_intervention_result,
    register_coach_action_decision,
)
from src.match_engine.world_model.policy_outcomes import (
    capture_policy_outcome_baseline,
    observe_policy_intervention_outcomes,
)
from src.match_engine.world_model.state_scales import (
    predictive_mechanism_edges,
)


def _packet():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    current[200] = 0.50
    current[-1] = 1.0
    future = np.repeat(current[None, :], 8, axis=0)
    future[:4, 209] = 1.0
    future[:4, 200] = 0.70
    future[4:, 209] = 0.0
    future[4:, 200] = 0.50
    return {
        "team_id": "Home", "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a",
        "candidates": [{
            "action": "pass", "multi_horizon_predictions": {
                "60s": {"state_scales": {
                    "predictive_mechanism_edges": predictive_mechanism_edges(
                        current, future, ensemble_trained=True,
                    ),
                }},
            },
        }],
    }


def test_llm_can_only_articulate_an_exact_member_grounded_mechanism():
    packet = _packet()
    design = build_predictive_mechanism_design(packet)
    packet["predictive_mechanism_design"] = design
    assert predictive_mechanism_design_is_valid(design)
    hypothesis = next(
        row for row in design["options"]
        if row["driver_event"] == "retain_possession"
        and row["outcome_event"] == "positive_territorial_shift"
    )
    audit = evaluate_llm_predictive_mechanism(
        packet,
        {"hypothesis_id": hypothesis["hypothesis_id"], "confidence": 0.8,
         "mechanism_statement": "retention accompanies territorial gain"},
        selected_action="pass", selected_after_action_freeze=True,
    )
    assert predictive_mechanism_audit_is_valid(audit)
    assert audit["association_only"]
    assert not audit["can_update_world_model"]

    rejected = evaluate_llm_predictive_mechanism(
        packet,
        {"hypothesis_id": hypothesis["hypothesis_id"], "confidence": 0.8},
        selected_action="shot", selected_after_action_freeze=True,
    )
    assert rejected["reason"] == "frozen_action_mechanism_mismatch"

    tampered = copy.deepcopy(design)
    tampered["options"][0]["conditional_lift"] = 1.0
    assert not predictive_mechanism_design_is_valid(tampered)


def test_future_joint_outcome_falsifies_against_independence_null():
    packet = _packet()
    packet["predictive_mechanism_design"] = (
        build_predictive_mechanism_design(packet)
    )
    hypothesis = next(
        row for row in packet["predictive_mechanism_design"]["options"]
        if row["driver_event"] == "retain_possession"
        and row["outcome_event"] == "positive_territorial_shift"
    )
    audit = evaluate_llm_predictive_mechanism(
        packet,
        {"hypothesis_id": hypothesis["hypothesis_id"], "confidence": 0.8},
        selected_action="pass", selected_after_action_freeze=True,
    )
    evaluation = score_predictive_mechanism(
        audit,
        {"retained_possession": True, "progress": 0.10,
         "goal_diff_delta": 0.0},
        {"ball_x": 0.50}, attacking_home=True, horizon="60s",
        realized_action="pass", checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    assert predictive_mechanism_evaluation_is_valid(evaluation, audit)
    assert evaluation["observed_joint_cell"] == (
        "driver_true_outcome_true"
    )
    assert evaluation[
        "categorical_log_likelihood_ratio_vs_independence"
    ] > 0.0


def test_predictive_mechanism_tournament_has_a_strict_online_gate():
    packet = _packet()
    packet["predictive_mechanism_design"] = (
        build_predictive_mechanism_design(packet)
    )
    selected = packet["predictive_mechanism_design"]["options"][:2]
    assert len({
        (row["driver_event"], row["outcome_event"]) for row in selected
    }) == 2
    logs = []
    for index in range(4):
        hypothesis = selected[index % 2]
        audit = evaluate_llm_predictive_mechanism(
            packet,
            {"hypothesis_id": hypothesis["hypothesis_id"],
             "confidence": 0.8},
            selected_action="pass", selected_after_action_freeze=True,
        )
        evaluation = score_predictive_mechanism(
            audit,
            {"retained_possession": True, "progress": 0.20,
             "goal_diff_delta": 1.0},
            {"ball_x": 0.50}, attacking_home=True, horizon="60s",
            realized_action="pass", checkpoint_signature="checkpoint-a",
            environment_signature="env-a",
        )
        record = {
            "llm_predictive_mechanism_context": audit,
            "intervention_actual_action": "pass",
            "multi_horizon_regime_outcomes": {"60s": {
                "llm_predictive_mechanism_evaluation": evaluation,
            }},
        }
        logs.append({
            "world_model_decision_adoption": {"records": [record]},
        })
    diagnostics = predictive_mechanism_diagnostics([
        payload["world_model_decision_adoption"]["records"]
        for payload in logs
    ])
    assert diagnostics["scored_joint_outcomes"] == 4
    assert diagnostics["distinct_mechanism_edges"] == 2
    assert diagnostics["mean_log_likelihood_ratio_vs_independence"] > 0.0
    report = aggregate_online_calibration(
        logs, min_transitions=0, require_predictive_mechanisms=True,
    )
    assert report["version"] == 43
    assert report["predictive_mechanisms_ready"]
    assert report["gates"]["predictive_mechanisms"]


def test_cross_match_memory_retains_supported_and_removes_it_from_retesting():
    packet = _packet()
    packet["predictive_mechanism_design"] = (
        build_predictive_mechanism_design(packet)
    )
    hypothesis = next(
        row for row in packet["predictive_mechanism_design"]["options"]
        if row["driver_event"] == "retain_possession"
        and row["outcome_event"] == "positive_territorial_shift"
    )
    logs = []
    for index in range(6):
        audit = evaluate_llm_predictive_mechanism(
            packet, {"hypothesis_id": hypothesis["hypothesis_id"],
                     "confidence": 0.8},
            selected_action="pass", selected_after_action_freeze=True,
        )
        evaluation = score_predictive_mechanism(
            audit, {"retained_possession": True, "progress": 0.10,
                    "goal_diff_delta": 0.0}, {"ball_x": 0.50},
            attacking_home=True, horizon="60s", realized_action="pass",
            checkpoint_signature="checkpoint-a",
            environment_signature="env-a",
        )
        logs.append({"world_model_decision_adoption": {"records": [{
            "decision_id": f"Home:{index}",
            "checkpoint_signature": "checkpoint-a",
            "environment_signature": "env-a",
            "llm_predictive_mechanism_context": audit,
            "multi_horizon_regime_outcomes": {"60s": {
                "llm_predictive_mechanism_evaluation": evaluation,
            }},
        }]}})
    memory = compile_predictive_mechanism_memory(
        logs, checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profiles[(
        "pass", "60s", "retain_possession",
        "positive_territorial_shift", "enabling",
    )]
    assert profile.status == "retained_supported"
    assert profile.matches == 6

    updated = build_predictive_mechanism_design(_packet(), evidence_memory=memory)
    resolved = next(
        row for row in updated["resolved_hypotheses"]
        if row["driver_event"] == "retain_possession"
        and row["outcome_event"] == "positive_territorial_shift"
    )
    assert resolved["cross_match_evidence"]["status"] == (
        "retained_supported"
    )
    assert resolved["hypothesis_id"] not in {
        row["hypothesis_id"] for row in updated["options"]
    }
    assert predictive_mechanism_design_is_valid(updated)
    diagnostics = predictive_mechanism_memory_diagnostics(logs)
    assert diagnostics["retained_profiles"] == 1
    assert diagnostics["all_stopping_rules_machine_owned"]
    report = aggregate_online_calibration(
        logs, min_transitions=0,
        require_predictive_mechanism_memory=True,
    )
    assert report["predictive_mechanism_memory_ready"]
    assert report["gates"]["predictive_mechanism_memory"]


def test_registered_mechanism_is_scored_once_from_natural_future_outcome():
    packet = _packet()
    packet["predictive_mechanism_design"] = (
        build_predictive_mechanism_design(packet)
    )
    hypothesis = next(
        row for row in packet["predictive_mechanism_design"]["options"]
        if row["driver_event"] == "retain_possession"
        and row["outcome_event"] == "positive_territorial_shift"
    )
    audit = evaluate_llm_predictive_mechanism(
        packet, {"hypothesis_id": hypothesis["hypothesis_id"],
                 "confidence": 0.8},
        selected_action="pass", selected_after_action_freeze=True,
    )
    state = SimpleNamespace(
        home=SimpleNamespace(team_id="Home", score=0),
        away=SimpleNamespace(team_id="Away", score=0),
        ball=SimpleNamespace(
            position=[0.50, 0.50], possession_team_id="Home",
        ),
        micro_xg_home=0.0, micro_xg_away=0.0, clock_seconds=100.0,
    )
    baseline = capture_policy_outcome_baseline(state, team_id="Home")
    record = register_coach_action_decision(
        state, team_id="Home", trigger_kind="test",
        llm_selected_action="pass", world_model_recommended_action="pass",
        recommendation_confidence=0.8, horizon_s=10.0,
        intervention_strength=0.2, intervention_enabled=True,
        outcome_horizons_s=(60.0,), checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
        llm_predictive_mechanism_context=audit,
    )
    record_policy_intervention_result(
        state, decision_id=record["decision_id"], actual_action="pass",
        t_sec=100.1, outcome_baseline=baseline,
    )
    state.ball.position[0] = 0.70
    observe_policy_intervention_outcomes(state, t_sec=160.1)
    evaluation = record["multi_horizon_regime_outcomes"]["60s"][
        "llm_predictive_mechanism_evaluation"
    ]
    assert predictive_mechanism_evaluation_is_valid(evaluation, audit)
    assert evaluation["driver_observed"]
    assert evaluation["outcome_observed"]
