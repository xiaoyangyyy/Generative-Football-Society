"""Temporal dependence is scored prospectively against realized paths."""

import copy
from types import SimpleNamespace

import numpy as np

from src.match_engine.world_model.residual_memory import (
    compile_contextual_residual_memory,
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
from src.match_engine.world_model.temporal_calibration import (
    freeze_temporal_path_forecast,
    score_temporal_path_forecast,
    temporal_path_forecast_is_valid,
    temporal_path_score_is_valid,
)
from src.match_engine.world_model.temporal_calibration_evaluation import (
    temporal_path_calibration_diagnostics,
)


def _distribution(members, residuals, coupling):
    members = np.asarray(members, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    scenarios = (members[:, None] + residuals[None, :]).reshape(-1)
    return {
        "available": True,
        "counts_trusted": True,
        "member_identity_preserved": True,
        "distribution_scope": "calibrated_predictive",
        "predictive_distribution_available": True,
        "member_values": members.tolist(),
        "epistemic_member_values": members.tolist(),
        "residual_scenario_offsets": residuals.tolist(),
        "residual_quantile_levels": [0.1, 0.5, 0.9],
        "decision_scenario_values": scenarios.tolist(),
        "mean_utility": float(scenarios.mean()),
        "residual_calibration_samples": 8,
        "residual_quantiles_split": "held_out_calibration",
        "uncertainty_decomposition": {
            "epistemic_member_variance": float(members.var()),
            "residual_outcome_variance": float(residuals.var()),
            "predictive_lattice_variance": float(scenarios.var()),
            "additive_identity_error": abs(float(
                scenarios.var() - members.var() - residuals.var()
            )),
            "axes_statistically_independent_claimed": False,
        },
        "temporal_residual_rank_coupling": copy.deepcopy(coupling),
    }


def _predictions():
    coupling = {
        "version": 2,
        "available": True,
        "horizon_keys": ["60s", "180s"],
        "rank_levels": [0.1, 0.5, 0.9],
        "rank_templates": [[0, 2], [2, 0], [0, 2], [2, 0]],
        "calibration_samples": 4,
        "mean_absolute_rank_correlation": 1.0,
        "shared_across_candidate_actions": True,
        "split": "reference_then_held_out_calibration",
        "dependence_source": "observed_multi_horizon_residual_ranks",
        "temporal_dependence_empirical": True,
        "temporal_joint_calibrated": False,
    }
    return {
        "60s": {"distributional_policy_utility": _distribution(
            [-0.05, 0.10], [-0.20, 0.0, 0.20], coupling,
        )},
        "180s": {"distributional_policy_utility": _distribution(
            [-0.10, 0.15], [-0.30, 0.0, 0.30], coupling,
        )},
    }


def _forecast():
    return freeze_temporal_path_forecast(
        _predictions(), checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )


def _record(forecast, path):
    outcomes = {
        horizon: {
            "policy_utility": float(value),
            "world_model_prediction": {
                "raw_policy_utility": 0.0,
                "policy_utility": 0.0,
            },
        }
        for horizon, value in zip(forecast["primary"]["horizon_keys"], path)
    }
    score = score_temporal_path_forecast(
        forecast, outcomes,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    return {
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "policy_utility_version": 1,
        "team_id": "Home",
        "intervention_actual_action": "pass",
        "outcome_baseline": {
            "opponent_team_id": "Away",
            "zone": "middle",
            "score_state": "level",
            "match_phase": "early",
        },
        "world_model_temporal_path_forecast": copy.deepcopy(forecast),
        "world_model_temporal_path_evaluation": score,
        "multi_horizon_regime_outcomes": outcomes,
    }


def _logs_for_paths(forecast, paths):
    return [{
        "world_model_decision_adoption": {
            "records": [_record(forecast, path)],
        },
    } for path in paths]


def test_forecast_freezes_empirical_paths_and_honest_fallback_benchmark():
    forecast = _forecast()

    assert temporal_path_forecast_is_valid(forecast)
    assert forecast["primary"]["temporal_dependence_learned"]
    assert forecast["comonotonic_benchmark"][
        "temporal_dependence_learned"
    ] is False
    assert forecast["primary"]["horizon_keys"] == ["60s", "180s"]

    outcomes = {
        "60s": {"policy_utility": 0.1},
        "180s": {"policy_utility": -0.1},
    }
    score = score_temporal_path_forecast(
        forecast, outcomes,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    assert temporal_path_score_is_valid(score, forecast)
    assert score["primary_score"]["observed_events"][
        "positive_to_negative_reversal"
    ]
    assert score["comonotonic_benchmark_score"] is not None
    assert not score["joint_temporal_probability_claimed"]

    tampered = copy.deepcopy(forecast)
    tampered["primary"]["scenario_utility_paths"][0][0] += 0.1
    assert not temporal_path_forecast_is_valid(tampered)
    assert score_temporal_path_forecast(
        forecast, {"60s": {"policy_utility": 0.1}},
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    ) is None


def test_epistemic_only_path_forecast_is_scored_without_residual_benchmark():
    predictions = {
        "60s": {"distributional_policy_utility": {
            "distribution_scope": "epistemic_member_only",
            "member_identity_preserved": True,
            "member_values": [-0.1, 0.2],
        }},
        "180s": {"distributional_policy_utility": {
            "distribution_scope": "epistemic_member_only",
            "member_identity_preserved": True,
            "member_values": [-0.2, 0.3],
        }},
    }
    forecast = freeze_temporal_path_forecast(
        predictions, checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    assert temporal_path_forecast_is_valid(forecast)
    assert forecast["comonotonic_benchmark"] is None
    assert not forecast["benchmark_required"]

    score = score_temporal_path_forecast(
        forecast,
        {"60s": {"policy_utility": -0.05},
         "180s": {"policy_utility": 0.1}},
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    assert temporal_path_score_is_valid(score, forecast)
    assert score["comonotonic_benchmark_score"] is None


def test_empirical_path_validation_compares_against_same_marginal_fallback():
    forecast = _forecast()
    empirical_paths = forecast["primary"]["scenario_utility_paths"]
    validated_logs = _logs_for_paths(forecast, empirical_paths)

    validated = temporal_path_calibration_diagnostics(validated_logs)
    assert validated["scored_paths"] == 8
    assert validated["matches"] == 8
    assert validated["empirical_validation_status"] == "validated"
    assert validated["empirical_mean_event_calibration_gap"] == 0.0
    assert validated["match_clustered_empirical_mean_event_brier"] <= (
        validated["match_clustered_comonotonic_mean_event_brier"] + 0.02
    )

    single_match = [{"world_model_decision_adoption": {"records": [
        payload["world_model_decision_adoption"]["records"][0]
        for payload in validated_logs
    ]}}]
    clustered = temporal_path_calibration_diagnostics(single_match)
    assert clustered["empirical_temporal_paths"] == 8
    assert clustered["empirical_temporal_matches"] == 1
    assert clustered["empirical_validation_status"] == "insufficient_evidence"

    tampered_logs = copy.deepcopy(validated_logs)
    tampered_logs[0]["world_model_decision_adoption"]["records"][0][
        "world_model_temporal_path_evaluation"
    ]["primary_score"]["mean_event_brier_score"] += 0.1
    tampered = temporal_path_calibration_diagnostics(tampered_logs)
    assert tampered["malformed_scores"] == 1

    missing_logs = copy.deepcopy(validated_logs)
    missing_logs[0]["world_model_decision_adoption"]["records"][0].pop(
        "world_model_temporal_path_evaluation"
    )
    missing = temporal_path_calibration_diagnostics(missing_logs)
    assert missing["missing_scores"] == 1

    fallback_paths = forecast["comonotonic_benchmark"][
        "scenario_utility_paths"
    ]
    degraded_logs = _logs_for_paths(
        forecast,
        [fallback_paths[index % len(fallback_paths)] for index in range(8)],
    )
    degraded = temporal_path_calibration_diagnostics(degraded_logs)
    assert degraded["empirical_validation_status"] == "degraded"


def test_validated_path_evidence_flows_into_contextual_temporal_memory():
    forecast = _forecast()
    logs = _logs_for_paths(
        forecast, forecast["primary"]["scenario_utility_paths"],
    )
    memory = compile_contextual_residual_memory(
        logs,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
        min_samples=8,
    )
    memory.temporal_rank_memory.drift = {"status": "stable"}
    coupling = memory.temporal_rank_coupling(
        context={
            "team_id": "Home", "opponent_team_id": "Away",
            "zone": "middle", "score_state": "level",
            "match_phase": "early",
        },
        horizon_keys=("60s", "180s"),
    )

    assert coupling["available"]
    assert coupling["validation"]["empirical_validation_status"] == (
        "validated"
    )

    memory.temporal_rank_memory.validation = {
        "empirical_validation_status": "degraded",
    }
    closed = memory.temporal_rank_coupling(
        context={}, horizon_keys=("60s", "180s"),
    )
    assert not closed["available"]
    assert closed["reason"] == "temporal_rank_memory_validation_gate_closed"


def test_strict_online_gate_requires_validated_temporal_path_skill():
    forecast = _forecast()
    validated_logs = _logs_for_paths(
        forecast, forecast["primary"]["scenario_utility_paths"],
    )
    report = aggregate_online_calibration(
        validated_logs, min_transitions=0, min_residual_samples=8,
        require_temporal_path_calibration=True,
    )
    assert report["version"] == 39
    assert report["temporal_path_calibration_ready"]
    assert report["gates"]["temporal_path_calibration"]

    fallback_paths = forecast["comonotonic_benchmark"][
        "scenario_utility_paths"
    ]
    degraded = aggregate_online_calibration(
        _logs_for_paths(forecast, fallback_paths),
        min_transitions=0, min_residual_samples=8,
        require_temporal_path_calibration=True,
    )
    assert not degraded["temporal_path_calibration_ready"]
    assert not degraded["gates"]["temporal_path_calibration"]


def test_executed_action_freezes_then_scores_one_complete_realized_path():
    state = SimpleNamespace(
        clock_seconds=10.0,
        home=SimpleNamespace(team_id="A", score=0),
        away=SimpleNamespace(team_id="B", score=0),
        ball=SimpleNamespace(position=[0.50, 0.50], possession_team_id="A"),
        micro_xg_home=0.2,
        micro_xg_away=0.1,
    )
    record = register_coach_action_decision(
        state, team_id="A", trigger_kind="clock",
        llm_selected_action="pass", world_model_recommended_action="pass",
        recommendation_confidence=0.8, horizon_s=10.0,
        outcome_horizons_s=(60.0, 180.0),
        action_predictions={"pass": _predictions()},
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    baseline = capture_policy_outcome_baseline(state, team_id="A")
    record_policy_intervention_result(
        state, decision_id=record["decision_id"], actual_action="pass",
        t_sec=11.0, outcome_baseline=baseline,
    )
    assert temporal_path_forecast_is_valid(
        record["world_model_temporal_path_forecast"]
    )

    state.ball.position[0] = 0.62
    observe_policy_intervention_outcomes(state, t_sec=71.0)
    assert "world_model_temporal_path_evaluation" not in record
    state.ball.position[0] = 0.45
    observe_policy_intervention_outcomes(state, t_sec=191.0)

    evaluation = record["world_model_temporal_path_evaluation"]
    assert temporal_path_score_is_valid(
        evaluation, record["world_model_temporal_path_forecast"],
    )
    assert evaluation["horizon_keys"] == ["60s", "180s"]
