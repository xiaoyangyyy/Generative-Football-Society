"""Post-action context stress tests for predictive mechanisms."""

import copy
from types import SimpleNamespace

import numpy as np

from src.match_engine.world_model.mechanism_stress_evaluation import (
    mechanism_stress_test_diagnostics,
    mechanism_stress_evaluation_is_valid,
    score_mechanism_stress_test,
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
from src.match_engine.world_model.mechanism_stress_test import (
    build_mechanism_stress_test_design,
    evaluate_llm_mechanism_stress_test,
    mechanism_stress_test_audit_is_valid,
    mechanism_stress_test_design_is_valid,
)
from src.match_engine.world_model.mechanism_stress_memory import (
    compile_mechanism_stress_memory,
    mechanism_stress_memory_diagnostics,
)
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.predictive_mechanism import (
    build_predictive_mechanism_design,
)
from src.match_engine.world_model.state_scales import predictive_mechanism_edges


def _future(current, *, independent=False):
    future = np.repeat(current[None, :], 8, axis=0)
    if independent:
        future[2:4, 200] = 0.70
        future[4:6, 209] = 1.0
        future[6:8, 209] = 1.0
        future[6:8, 200] = 0.70
    else:
        future[:4, 209] = 1.0
        future[:4, 200] = 0.70
    return future


def _packet(current):
    packet = {
        "team_id": "Home", "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a", "horizon_s": 10.0,
        "candidates": [{
            "action": "pass", "target": [0.65, 0.5],
            "physics_prior": 0.75,
            "multi_horizon_predictions": {"60s": {
                "horizon_s": 60.0, "rollout_steps": 1,
                "state_scales": {"predictive_mechanism_edges": (
                    predictive_mechanism_edges(
                        current, _future(current), ensemble_trained=True,
                    )
                )},
            }},
        }],
    }
    packet["predictive_mechanism_design"] = (
        build_predictive_mechanism_design(packet)
    )
    return packet


class _StressRuntime:
    def __init__(self, current):
        self.current = current
        self.calls = 0

    def encode_state(self, state, *, attacking_home):
        return self.current.copy()

    def planner_confidence(self, observation, *, kind):
        return 0.8

    def imagine(self, observation, action, **kwargs):
        self.calls += 1
        score_neutral = bool(
            np.allclose(observation[[204, 205, 206]], [0.0, 0.0, 0.5])
        )
        return SimpleNamespace(uncertainty_samples={
            "transition_states": _future(
                observation, independent=score_neutral,
            ),
            "transition_ensemble_trained": True,
        })


def _setup():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    current[200] = 0.50
    current[204] = 0.20
    current[206] = 4.0 / 6.0
    current[207] = 0.25
    current[208] = 0.70
    current[298:302] = [0.7, 0.6, 0.55, 0.4]
    current[302:306] = [0.3, 0.4, 0.35, 0.5]
    current[-1] = 1.0
    return current, _StressRuntime(current), _packet(current)


def test_engine_precomputes_bounded_stress_menu_after_action_freeze():
    _, runtime, packet = _setup()
    state = SimpleNamespace(home=SimpleNamespace(team_id="Home"))
    design = build_mechanism_stress_test_design(
        runtime, state, packet, team_id="Home", selected_action="pass",
    )
    packet["mechanism_stress_test_design"] = design

    assert mechanism_stress_test_design_is_valid(design)
    assert 0 < design["model_calls"] <= design["model_call_budget"] == 10
    assert runtime.calls == design["model_calls"]
    score_test = next(
        row for row in design["options"]
        if row["context_factor"] == "score_context"
    )
    assert score_test["classification"] in {"context_sensitive", "fragile"}
    assert score_test["joint_total_variation"] > 0.0

    audit = evaluate_llm_mechanism_stress_test(
        packet,
        {"stress_test_id": score_test["stress_test_id"], "confidence": 0.8,
         "failure_condition": "the score context is neutralized"},
        selected_action="pass", selected_after_action_freeze=True,
    )
    assert mechanism_stress_test_audit_is_valid(audit)
    assert not audit["can_change_current_action"]
    assert not audit["can_update_world_model"]

    tampered = copy.deepcopy(design)
    tampered["options"][0]["fragility_score"] = 0.0
    assert not mechanism_stress_test_design_is_valid(tampered)


def test_natural_outcome_scores_observed_context_against_neutralized_model():
    _, runtime, packet = _setup()
    design = build_mechanism_stress_test_design(
        runtime, SimpleNamespace(home=SimpleNamespace(team_id="Home")),
        packet, team_id="Home", selected_action="pass",
    )
    packet["mechanism_stress_test_design"] = design
    option = next(
        row for row in design["options"]
        if row["context_factor"] == "score_context"
    )
    audit = evaluate_llm_mechanism_stress_test(
        packet, {"stress_test_id": option["stress_test_id"],
                 "confidence": 0.8},
        selected_action="pass", selected_after_action_freeze=True,
    )
    evaluation = score_mechanism_stress_test(
        audit, {"retained_possession": True, "progress": 0.20,
                "goal_diff_delta": 1.0}, {"ball_x": 0.50},
        attacking_home=True, horizon="60s", realized_action="pass",
        checkpoint_signature="checkpoint-a", environment_signature="env-a",
    )
    assert mechanism_stress_evaluation_is_valid(evaluation, audit)
    assert evaluation["log_likelihood_ratio_observed_vs_neutralized"] > 0.0
    assert evaluation["brier_skill_observed_vs_neutralized"] > 0.0
    assert not evaluation["causal_interpretation"]


def test_mechanism_stress_tests_have_an_independent_strict_online_gate():
    _, runtime, packet = _setup()
    design = build_mechanism_stress_test_design(
        runtime, SimpleNamespace(home=SimpleNamespace(team_id="Home")),
        packet, team_id="Home", selected_action="pass",
    )
    packet["mechanism_stress_test_design"] = design
    selected = []
    for factor in ("score_context", "match_phase"):
        option = next((
            row for row in design["options"]
            if row["context_factor"] == factor
        ), None)
        if option is not None:
            selected.append(option)
    assert len(selected) == 2
    logs = []
    for index in range(4):
        option = selected[index % 2]
        audit = evaluate_llm_mechanism_stress_test(
            packet, {"stress_test_id": option["stress_test_id"],
                     "confidence": 0.8},
            selected_action="pass", selected_after_action_freeze=True,
        )
        evaluation = score_mechanism_stress_test(
            audit, {"retained_possession": True, "progress": 0.20,
                    "goal_diff_delta": 1.0}, {"ball_x": 0.50},
            attacking_home=True, horizon="60s", realized_action="pass",
            checkpoint_signature="checkpoint-a",
            environment_signature="env-a",
        )
        record = {
            "llm_mechanism_stress_test_context": audit,
            "intervention_actual_action": "pass",
            "multi_horizon_regime_outcomes": {"60s": {
                "llm_mechanism_stress_test_evaluation": evaluation,
            }},
        }
        logs.append({
            "world_model_decision_adoption": {"records": [record]},
        })
    clusters = [
        payload["world_model_decision_adoption"]["records"]
        for payload in logs
    ]
    diagnostics = mechanism_stress_test_diagnostics(clusters)
    assert diagnostics["scored_natural_outcomes"] == 4
    assert diagnostics["distinct_context_factors"] == 2
    report = aggregate_online_calibration(
        logs, min_transitions=0, require_mechanism_stress_tests=True,
    )
    assert report["version"] == 47
    assert report["mechanism_stress_tests_ready"]
    assert report["gates"]["mechanism_stress_tests"]


def test_registered_stress_test_scores_the_natural_future_only_once():
    _, runtime, packet = _setup()
    design = build_mechanism_stress_test_design(
        runtime, SimpleNamespace(home=SimpleNamespace(team_id="Home")),
        packet, team_id="Home", selected_action="pass",
    )
    packet["mechanism_stress_test_design"] = design
    option = next(
        row for row in design["options"]
        if row["context_factor"] == "score_context"
    )
    audit = evaluate_llm_mechanism_stress_test(
        packet, {"stress_test_id": option["stress_test_id"],
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
        llm_mechanism_stress_test_context=audit,
    )
    record_policy_intervention_result(
        state, decision_id=record["decision_id"], actual_action="pass",
        t_sec=100.1, outcome_baseline=baseline,
    )
    state.ball.position[0] = 0.70
    state.home.score = 1
    observe_policy_intervention_outcomes(state, t_sec=160.1)
    evaluation = record["multi_horizon_regime_outcomes"]["60s"][
        "llm_mechanism_stress_test_evaluation"
    ]
    assert mechanism_stress_evaluation_is_valid(evaluation, audit)
    assert record["llm_mechanism_stress_test_context"]["audit_digest"] == (
        audit["audit_digest"]
    )


def test_cross_match_stress_memory_validates_and_skips_resolved_rollout():
    _, runtime, packet = _setup()
    initial = build_mechanism_stress_test_design(
        runtime, SimpleNamespace(home=SimpleNamespace(team_id="Home")),
        packet, team_id="Home", selected_action="pass",
    )
    packet["mechanism_stress_test_design"] = initial
    option = next(
        row for row in initial["options"]
        if row["context_factor"] == "score_context"
    )
    logs = []
    for index in range(6):
        audit = evaluate_llm_mechanism_stress_test(
            packet, {"stress_test_id": option["stress_test_id"],
                     "confidence": 0.8},
            selected_action="pass", selected_after_action_freeze=True,
        )
        evaluation = score_mechanism_stress_test(
            audit, {"retained_possession": True, "progress": 0.20,
                    "goal_diff_delta": 1.0}, {"ball_x": 0.50},
            attacking_home=True, horizon="60s", realized_action="pass",
            checkpoint_signature="checkpoint-a",
            environment_signature="env-a",
        )
        logs.append({"world_model_decision_adoption": {"records": [{
            "decision_id": f"Home:{index}",
            "checkpoint_signature": "checkpoint-a",
            "environment_signature": "env-a",
            "llm_mechanism_stress_test_context": audit,
            "multi_horizon_regime_outcomes": {"60s": {
                "llm_mechanism_stress_test_evaluation": evaluation,
            }},
        }]}})
    memory = compile_mechanism_stress_memory(
        logs, checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    key = (
        option["action"], option["horizon"], option["driver_event"],
        option["outcome_event"], option["observed_relationship"],
        option["context_factor"],
    )
    profile = memory.profiles[key]
    assert profile.status == "validated_context_dependence"
    assert profile.matches == 6

    _, learned_runtime, learned_packet = _setup()
    learned = build_mechanism_stress_test_design(
        learned_runtime,
        SimpleNamespace(home=SimpleNamespace(team_id="Home")),
        learned_packet, team_id="Home", selected_action="pass",
        evidence_memory=memory,
    )
    assert learned["resolved_tests_skipped_before_rollout"] >= 1
    assert learned["model_calls"] < initial["model_calls"]
    assert learned["stress_memory_summary"]["validated_profiles"] == 1
    assert not any(
        row["context_factor"] == option["context_factor"]
        and row["driver_event"] == option["driver_event"]
        and row["outcome_event"] == option["outcome_event"]
        for row in learned["options"]
    )
    assert mechanism_stress_test_design_is_valid(learned)

    diagnostics = mechanism_stress_memory_diagnostics(logs)
    assert diagnostics["validated_profiles"] == 1
    assert diagnostics["all_match_clustered"]
    assert diagnostics["all_stopping_rules_machine_owned"]
    report = aggregate_online_calibration(
        logs, min_transitions=0, require_mechanism_stress_memory=True,
    )
    assert report["version"] == 47
    assert report["mechanism_stress_memory_ready"]
    assert report["gates"]["mechanism_stress_memory"]


def test_stress_memory_invalidates_negative_evidence_and_caps_inconclusive():
    _, runtime, packet = _setup()
    design = build_mechanism_stress_test_design(
        runtime, SimpleNamespace(home=SimpleNamespace(team_id="Home")),
        packet, team_id="Home", selected_action="pass",
    )
    packet["mechanism_stress_test_design"] = design
    negative = next(
        row for row in design["options"]
        if row["context_factor"] == "score_context"
    )
    neutral = next(
        row for row in design["options"]
        if row["context_factor"] == "match_phase"
    )
    logs = []
    for option, count, progress in (
        (negative, 2, 0.10),
        (neutral, 20, 0.20),
    ):
        for index in range(count):
            audit = evaluate_llm_mechanism_stress_test(
                packet, {"stress_test_id": option["stress_test_id"],
                         "confidence": 0.8},
                selected_action="pass", selected_after_action_freeze=True,
            )
            evaluation = score_mechanism_stress_test(
                audit, {"retained_possession": True, "progress": progress,
                        "goal_diff_delta": 0.0}, {"ball_x": 0.50},
                attacking_home=True, horizon="60s", realized_action="pass",
                checkpoint_signature="checkpoint-a",
                environment_signature="env-a",
            )
            logs.append({"world_model_decision_adoption": {"records": [{
                "decision_id": f"{option['context_factor']}:{index}",
                "checkpoint_signature": "checkpoint-a",
                "environment_signature": "env-a",
                "llm_mechanism_stress_test_context": audit,
                "multi_horizon_regime_outcomes": {"60s": {
                    "llm_mechanism_stress_test_evaluation": evaluation,
                }},
            }]}})
    memory = compile_mechanism_stress_memory(
        logs, checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    negative_profile = next(
        row for row in memory.profiles.values()
        if row.context_factor == "score_context"
        and row.driver_event == negative["driver_event"]
        and row.outcome_event == negative["outcome_event"]
    )
    neutral_profile = next(
        row for row in memory.profiles.values()
        if row.context_factor == "match_phase"
        and row.driver_event == neutral["driver_event"]
        and row.outcome_event == neutral["outcome_event"]
    )
    assert negative_profile.status == "invalidated_context_dependence"
    assert negative_profile.matches == 2
    assert neutral_profile.status == "retired_inconclusive"
    assert neutral_profile.matches == 20
