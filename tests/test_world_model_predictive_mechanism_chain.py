"""Three-event predictive-chain contracts and sequential evidence."""

import copy
from types import SimpleNamespace

import numpy as np

from src.match_engine.world_model.observation import OBS_DIM
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
from src.match_engine.world_model.predictive_mechanism_chain import (
    build_predictive_mechanism_chain_design,
    evaluate_llm_predictive_mechanism_chain,
    predictive_mechanism_chain_audit_is_valid,
    predictive_mechanism_chain_design_is_valid,
)
from src.match_engine.world_model.predictive_mechanism_chain_evaluation import (
    predictive_mechanism_chain_diagnostics,
    predictive_mechanism_chain_evaluation_is_valid,
    score_predictive_mechanism_chain,
)
from src.match_engine.world_model.predictive_mechanism_chain_memory import (
    compile_predictive_mechanism_chain_memory,
    predictive_mechanism_chain_memory_diagnostics,
)
from src.match_engine.world_model.predictive_mechanism_chain_fusion_memory import (
    compile_predictive_mechanism_chain_fusion_memory,
    predictive_mechanism_chain_fusion_diagnostics,
)
from src.match_engine.world_model.state_scales import (
    predictive_mechanism_paths,
)


def _packet():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    current[200] = 0.50
    current[-1] = 1.0
    future = np.repeat(current[None, :], 8, axis=0)
    future[2:4, 209] = 1.0
    future[4:6, 200] = 0.60
    future[6:8, 209] = 1.0
    future[6:8, 200] = 0.70
    return {
        "team_id": "Home", "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a",
        "candidates": [{
            "action": "pass", "multi_horizon_predictions": {
                "60s": {"state_scales": {
                    "predictive_mechanism_paths": predictive_mechanism_paths(
                        current, future, ensemble_trained=True,
                    ),
                }},
            },
        }],
    }


def _audit(
    index=0, *, forecast_memory=None, llm_probability=0.80,
    llm_signature="llm-predictive-chain-test",
):
    packet = _packet()
    design = build_predictive_mechanism_chain_design(packet)
    packet["predictive_mechanism_chain_design"] = design
    chain = design["options"][index]
    audit = evaluate_llm_predictive_mechanism_chain(
        packet,
        {"chain_id": chain["chain_id"], "confidence": 0.82,
         "chain_completion_probability": llm_probability,
         "mediator_statement": "the middle event conditions the endpoint"},
        selected_action="pass", selected_after_action_freeze=True,
        forecast_memory=forecast_memory, llm_signature=llm_signature,
    )
    return packet, design, audit


def _score(audit, outcome=None):
    return score_predictive_mechanism_chain(
        audit,
        outcome or {"retained_possession": True, "progress": 0.20,
                    "goal_diff_delta": 1.0},
        {"ball_x": 0.50}, attacking_home=True, horizon="60s",
        realized_action="pass", checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )


def _log(audit, evaluation):
    return {"world_model_decision_adoption": {"records": [{
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a",
        "intervention_actual_action": "pass",
        "llm_predictive_mechanism_chain_context": audit,
        "multi_horizon_regime_outcomes": {"60s": {
            "llm_predictive_mechanism_chain_evaluation": evaluation,
        }},
    }]}}


def test_llm_can_only_articulate_an_exact_machine_generated_chain():
    packet, design, audit = _audit()
    assert predictive_mechanism_chain_design_is_valid(design)
    assert predictive_mechanism_chain_audit_is_valid(audit)
    assert audit["higher_order_dependence_only"]
    assert not audit["causal_interpretation"]

    rejected = evaluate_llm_predictive_mechanism_chain(
        packet,
        {"chain_id": audit["chain"]["chain_id"], "confidence": 0.8,
         "chain_completion_probability": 0.80},
        selected_action="shot", selected_after_action_freeze=True,
    )
    assert rejected["reason"] == "frozen_action_chain_mismatch"

    tampered = copy.deepcopy(design)
    tampered["options"][0]["conditional_mutual_information_bits"] += 0.1
    assert not predictive_mechanism_chain_design_is_valid(tampered)


def test_natural_three_event_outcome_scores_full_joint_against_markov_null():
    _, _, audit = _audit()
    evaluation = _score(audit)
    assert predictive_mechanism_chain_evaluation_is_valid(evaluation, audit)
    assert evaluation["observed_joint_cell"] == "a1_b1_c1"
    assert evaluation[
        "categorical_log_likelihood_ratio_vs_pairwise_markov_null"
    ] > 0.0
    assert evaluation["brier_skill_vs_pairwise_markov_null"] > 0.0

    tampered = dict(evaluation)
    tampered["observed_joint_cell"] = "a0_b0_c0"
    assert not predictive_mechanism_chain_evaluation_is_valid(tampered, audit)


def test_chain_tournament_and_memory_use_separate_strict_gates():
    rows = []
    for index in (0, 1, 0, 1):
        _, _, audit = _audit(index)
        evaluation = _score(audit)
        rows.append(_log(audit, evaluation))
    diagnostics = predictive_mechanism_chain_diagnostics([
        payload["world_model_decision_adoption"]["records"]
        for payload in rows
    ])
    assert diagnostics["matches"] == 4
    assert diagnostics["distinct_mechanism_paths"] >= 2
    report = aggregate_online_calibration(
        rows, min_transitions=0, require_predictive_mechanism_chains=True,
    )
    assert report["version"] == 47
    assert report["predictive_mechanism_chains_ready"]
    assert report["gates"]["predictive_mechanism_chains"]


def test_match_clustered_chain_memory_stops_at_preregistered_evidence_boundary():
    _, _, audit = _audit()
    evaluation = _score(audit)
    logs = [_log(audit, evaluation) for _ in range(8)]
    memory = compile_predictive_mechanism_chain_memory(
        logs, checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = next(iter(memory.profiles.values()))
    assert profile.status == "retained_higher_order_dependence"
    assert profile.matches == 8
    learned = build_predictive_mechanism_chain_design(
        _packet(), evidence_memory=memory,
    )
    assert predictive_mechanism_chain_design_is_valid(learned)
    assert audit["chain"]["chain_id"] not in {
        row["chain_id"] for row in learned["options"]
    }
    diagnostics = predictive_mechanism_chain_memory_diagnostics(logs)
    assert diagnostics["resolved_profiles"] == 1
    report = aggregate_online_calibration(
        logs, min_transitions=0,
        require_predictive_mechanism_chain_memory=True,
    )
    assert report["predictive_mechanism_chain_memory_ready"]
    assert report["gates"]["predictive_mechanism_chain_memory"]


def test_registered_chain_is_scored_from_the_natural_future_window():
    _, _, audit = _audit()
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
        llm_predictive_mechanism_chain_context=audit,
    )
    record_policy_intervention_result(
        state, decision_id=record["decision_id"], actual_action="pass",
        t_sec=100.1, outcome_baseline=baseline,
    )
    state.ball.position[0] = 0.70
    state.home.score = 1
    observe_policy_intervention_outcomes(state, t_sec=160.1)
    evaluation = record["multi_horizon_regime_outcomes"]["60s"][
        "llm_predictive_mechanism_chain_evaluation"
    ]
    assert predictive_mechanism_chain_evaluation_is_valid(evaluation, audit)
    assert evaluation["first_event_observed"]
    assert evaluation["mediator_event_observed"]
    assert evaluation["outcome_event_observed"]


def test_llm_chain_probability_earns_only_held_out_bounded_fusion_weight():
    logs = []
    for _ in range(6):
        _, _, audit = _audit()
        logs.append(_log(audit, _score(audit)))
    memory = compile_predictive_mechanism_chain_fusion_memory(
        logs, checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
        llm_signature="llm-predictive-chain-test",
    )
    profile = next(iter(memory.profiles.values()))
    assert profile.training_matches == 3
    assert profile.validation_matches == 3
    assert profile.validation_skill >= 0.02
    assert profile.active
    assert 0.0 < profile.fitted_llm_weight <= 0.35

    _, _, fused_audit = _audit(forecast_memory=memory)
    fusion = fused_audit["chain_forecast_fusion"]
    assert fusion["active"]
    assert fusion["reason"] == "match_held_out_chain_forecast_gain"
    assert fused_audit["world_model_prediction_mutated"] is False
    assert (
        fused_audit["world_model_chain_completion_probability"]
        < fused_audit["fused_chain_completion_probability"]
        < fused_audit["llm_chain_completion_probability"]
    )
    evaluation = _score(fused_audit)
    assert evaluation["fused_brier_skill_vs_world_model"] > 0.0

    diagnostics = predictive_mechanism_chain_fusion_diagnostics(logs)
    assert diagnostics["active_profiles"] == 1
    assert diagnostics["all_chronological_match_held_out"]
    assert diagnostics["all_active_improve_world_model"]
    assert diagnostics["all_authority_bounded"]
    report = aggregate_online_calibration(
        logs, min_transitions=0,
        require_predictive_mechanism_chain_fusion=True,
    )
    assert report["version"] == 47
    assert report["predictive_mechanism_chain_fusion_ready"]
    assert report["gates"]["predictive_mechanism_chain_fusion"]

    _, _, incompatible = _audit(
        forecast_memory=memory, llm_signature="different-llm-contract",
    )
    assert not incompatible["chain_forecast_fusion"]["active"]
    assert incompatible["chain_forecast_fusion"]["llm_weight"] == 0.0


def test_harmful_llm_chain_forecasts_keep_zero_fusion_authority():
    logs = []
    for _ in range(6):
        _, _, audit = _audit(llm_probability=0.01)
        logs.append(_log(audit, _score(audit)))
    memory = compile_predictive_mechanism_chain_fusion_memory(
        logs, checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
        llm_signature="llm-predictive-chain-test",
    )
    profile = next(iter(memory.profiles.values()))
    assert not profile.active
    assert profile.fitted_llm_weight == 0.0
    _, _, audit = _audit(
        forecast_memory=memory, llm_probability=0.01,
    )
    assert not audit["chain_forecast_fusion"]["active"]
    assert audit["fused_chain_completion_probability"] == (
        audit["world_model_chain_completion_probability"]
    )
    report = aggregate_online_calibration(
        logs, min_transitions=0,
        require_predictive_mechanism_chain_fusion=True,
    )
    assert not report["predictive_mechanism_chain_fusion_ready"]
    assert not report["gates"]["predictive_mechanism_chain_fusion"]
