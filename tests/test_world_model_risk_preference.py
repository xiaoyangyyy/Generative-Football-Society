"""LLM risk preferences are explicit, multi-horizon and model checked."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.match_engine.cognitive.schemas import validate_coach_plan
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
from src.match_engine.world_model.risk_preference import (
    evaluate_llm_risk_preference,
    prospect_value,
    risk_preference_audit_is_valid,
    validate_llm_risk_preference,
)
from src.match_engine.world_model.risk_preference_evaluation import (
    risk_preference_diagnostics,
    score_llm_risk_preference,
)


def _distribution(values):
    members = np.asarray(values, dtype=float)
    residuals = np.zeros(3)
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
            "residual_outcome_variance": 0.0,
            "predictive_lattice_variance": float(scenarios.var()),
            "additive_identity_error": 0.0,
            "axes_statistically_independent_claimed": False,
        },
    }


def _packet():
    values = {
        "hold": [0.02, 0.04, 0.06, 0.08],
        "pass": [0.10, 0.15, 0.20, 0.25],
        "cross": [-0.05, 0.08, 0.12, 0.22],
        "shot": [-0.50, -0.20, 0.15, 0.30],
    }
    return {
        "available": True,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "candidates": [{
            "action": action,
            "multi_horizon_predictions": {
                horizon: {"distributional_policy_utility": _distribution(row)}
                for horizon in ("60s", "180s")
            },
        } for action, row in values.items()],
    }


def _rationalization_packet():
    packet = _packet()
    shot = next(
        candidate for candidate in packet["candidates"]
        if candidate["action"] == "shot"
    )
    shot["multi_horizon_predictions"] = {
        horizon: {
            "distributional_policy_utility": _distribution(
                [-0.50, -0.20, 0.40, 0.90]
            )
        }
        for horizon in ("60s", "180s")
    }
    return packet


def _empirical_temporal_packet():
    packet = _packet()
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
    for candidate in packet["candidates"]:
        for prediction in candidate["multi_horizon_predictions"].values():
            prediction["distributional_policy_utility"][
                "temporal_residual_rank_coupling"
            ] = copy.deepcopy(coupling)
    return packet


def _preference(**updates):
    value = {
        "selected_action": "pass",
        "distribution_scope": "calibrated_predictive",
        "horizon_weights": {"60s": 0.6, "180s": 0.4},
        "loss_aversion": 2.0,
        "diminishing_sensitivity": 0.8,
        "max_acceptable_regret": 0.10,
        "confidence": 0.8,
        "rationale": "Prefer robust gains under one stable risk attitude.",
    }
    value.update(updates)
    return value


def test_preference_schema_is_bounded_and_preserved_by_coach_schema():
    assert validate_llm_risk_preference(_preference())["reference_utility"] == 0.0
    assert validate_llm_risk_preference(
        _preference(horizon_weights={"60s": 0.7, "180s": 0.4})
    ) is None
    assert validate_llm_risk_preference(
        _preference(loss_aversion=5.0)
    ) is None
    assert validate_llm_risk_preference(
        _preference(diminishing_sensitivity=0.4)
    ) is None
    assert validate_llm_risk_preference(
        _preference(horizon_weights={"60s": 1.0})
    ) is None
    plan = validate_coach_plan({
        "world_model_action": "pass",
        "world_model_risk_preference": _preference(),
    })
    assert plan["world_model_risk_preference"]["selected_action"] == "pass"


def test_world_model_recomputes_scenario_values_and_multi_horizon_regret():
    audit = evaluate_llm_risk_preference(
        _packet(), _preference(), selected_action="pass",
        preference_signature="llm-risk-preference:test",
    )

    assert audit["accepted"]
    assert audit["preference_consistent"]
    assert risk_preference_audit_is_valid(audit)
    assert audit["results"]["aggregate_preferred_actions"] == ["pass"]
    assert audit["results"]["selected_aggregate_preference_regret"] == 0.0
    assert set(audit["results"]["horizons"]) == {"60s", "180s"}
    assert not audit["can_change_selected_action"]
    assert audit["preference_robust"]
    assert audit["preference_temporally_robust"]
    assert audit["temporal_preference"][
        "pathwise_comparison_order"
    ] == "compare_actions_within_scenario_then_aggregate_scenarios"
    assert audit["temporal_preference"][
        "pathwise_regret_within_declared_rate"
    ] == 1.0
    assert not audit["temporal_preference"]["temporal_joint_calibrated"]
    assert audit["robustness"]["joint_stress_cases"] == 72
    assert audit["robustness"]["member_axis_jointly_aligned"]
    assert audit["robustness"]["residual_quantile_axis_jointly_aligned"]
    transformed_loss = prospect_value([-0.25], audit["preference"])[0]
    transformed_gain = prospect_value([0.25], audit["preference"])[0]
    assert abs(transformed_loss) == pytest.approx(2.0 * transformed_gain)

    fragile = evaluate_llm_risk_preference(
        _rationalization_packet(),
        _preference(max_acceptable_regret=0.05),
        selected_action="pass",
    )
    assert fragile["preference_consistent"]
    assert not fragile["preference_robust"]
    assert 0.90 < fragile["robustness"]["robustness_rate"] < 1.0
    assert fragile["robustness"]["rationalization_fragility"] > 0.0
    assert len(fragile["robustness"]["failed_cases"]) == 3
    assert not fragile["preference_temporally_robust"]
    assert fragile["temporal_preference"][
        "pathwise_regret_within_declared_rate"
    ] == 0.5

    tampered = copy.deepcopy(audit)
    tampered["results"]["selected_aggregate_preference_regret"] = 0.2
    assert not risk_preference_audit_is_valid(tampered)
    tampered_robustness = copy.deepcopy(audit)
    tampered_robustness["robustness"]["robustness_rate"] = 0.5
    assert not risk_preference_audit_is_valid(tampered_robustness)
    tampered_temporal = copy.deepcopy(audit)
    tampered_temporal["temporal_preference"][
        "pathwise_regret_within_declared_rate"
    ] = 0.5
    assert not risk_preference_audit_is_valid(tampered_temporal)
    wrong_version = copy.deepcopy(audit)
    wrong_version["version"] = 1
    assert not risk_preference_audit_is_valid(wrong_version)

    inconsistent = evaluate_llm_risk_preference(
        _packet(), _preference(selected_action="shot"),
        selected_action="shot",
    )
    assert inconsistent["accepted"]
    assert not inconsistent["preference_consistent"]
    assert inconsistent["results"][
        "selected_aggregate_preference_regret"
    ] > inconsistent["preference"]["max_acceptable_regret"]


def test_preference_paths_share_empirical_temporal_templates_across_actions():
    audit = evaluate_llm_risk_preference(
        _empirical_temporal_packet(), _preference(), selected_action="pass",
    )

    assert audit["accepted"]
    assert risk_preference_audit_is_valid(audit)
    temporal = audit["temporal_preference"]
    assert temporal["temporal_dependence_learned"]
    assert temporal["temporal_dependence_validation"][
        "empirical_validation_status"
    ] == "insufficient_evidence"
    assert temporal["temporal_coupling_source"] == (
        "member_identity_x_empirical_residual_rank_templates"
    )
    assert temporal[
        "empirical_residual_rank_templates_jointly_aligned_across_actions"
    ]
    assert not temporal[
        "residual_quantile_axis_jointly_aligned_across_actions"
    ]


def test_preference_fails_closed_for_scope_action_and_incomplete_evidence():
    mismatch = evaluate_llm_risk_preference(
        _packet(), _preference(selected_action="shot"),
        selected_action="pass",
    )
    assert mismatch["reason"] == "risk_preference_action_must_equal_plan"

    wrong_scope = evaluate_llm_risk_preference(
        _packet(), _preference(distribution_scope="epistemic_member_only"),
        selected_action="pass",
    )
    assert wrong_scope["reason"] == (
        "risk_preference_scope_or_evidence_mismatch"
    )

    incomplete = _packet()
    incomplete["candidates"][0]["multi_horizon_predictions"].pop("180s")
    closed = evaluate_llm_risk_preference(
        incomplete, _preference(), selected_action="pass",
    )
    assert closed["reason"] == "risk_preference_scenarios_unavailable"

    misaligned = _packet()
    misaligned["candidates"][0]["multi_horizon_predictions"]["60s"][
        "distributional_policy_utility"
    ]["residual_quantile_levels"] = [0.2, 0.5, 0.9]
    axis_closed = evaluate_llm_risk_preference(
        misaligned, _preference(), selected_action="pass",
    )
    assert axis_closed["reason"] == "risk_preference_robustness_unavailable"


def test_realized_preference_scores_are_recomputed_and_strictly_gated():
    audit = evaluate_llm_risk_preference(
        _empirical_temporal_packet(), _preference(), selected_action="pass",
        preference_signature="llm-risk-preference:test",
    )
    logs = []
    for observed in (0.10, 0.12, 0.18, 0.50):
        score = score_llm_risk_preference(
            audit, {"policy_utility": observed}, horizon="60s",
            checkpoint_signature="checkpoint:test",
            environment_signature="environment:test",
        )
        logs.append({"world_model_decision_adoption": {"records": [{
            "intervention_actual_action": "pass",
            "llm_risk_preference_context": copy.deepcopy(audit),
            "multi_horizon_regime_outcomes": {
                "60s": {"llm_risk_preference_evaluation": score}
            },
        }]}})

    diagnostics = risk_preference_diagnostics(logs)
    assert diagnostics["realized_preference_values"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics["malformed_preference_evaluations"] == 0
    assert diagnostics["match_clustered_central_80_coverage"] == 0.75
    assert diagnostics["all_predictive_distributions_calibrated"]
    assert diagnostics["match_clustered_prospective_temporal_robustness"] == 1.0
    assert diagnostics[
        "match_clustered_empirical_temporal_dependence_rate"
    ] == 1.0
    assert diagnostics["provenance_compatible"]

    report = aggregate_online_calibration(
        logs, min_transitions=0, min_residual_samples=2,
        require_llm_risk_preferences=True,
    )
    assert report["version"] == 43
    assert report["llm_risk_preferences_ready"]
    assert report["gates"]["calibrated_llm_risk_preferences"]

    tampered_logs = copy.deepcopy(logs)
    tampered_logs[0]["world_model_decision_adoption"]["records"][0][
        "multi_horizon_regime_outcomes"
    ]["60s"]["llm_risk_preference_evaluation"][
        "transformed_scenario_values"
    ][0] += 0.01
    malformed = risk_preference_diagnostics(tampered_logs)
    assert malformed["malformed_preference_evaluations"] == 1

    malformed_context_logs = copy.deepcopy(logs)
    malformed_context_logs[0]["world_model_decision_adoption"]["records"][0][
        "llm_risk_preference_context"
    ]["scenario_evidence"] = {"60s": {"pass": None}}
    malformed_context = risk_preference_diagnostics(malformed_context_logs)
    assert malformed_context["malformed_preference_contexts"] == 1


def test_preference_is_scored_only_after_exact_action_horizon_realizes():
    audit = evaluate_llm_risk_preference(
        _packet(), _preference(),
        selected_action="pass",
        preference_signature="llm-risk-preference:test",
    )
    state = SimpleNamespace(
        clock_seconds=10.0,
        home=SimpleNamespace(team_id="A", score=0),
        away=SimpleNamespace(team_id="B", score=0),
        ball=SimpleNamespace(position=[0.50, 0.50], possession_team_id="A"),
        micro_xg_home=0.2, micro_xg_away=0.1,
    )
    record = register_coach_action_decision(
        state, team_id="A", trigger_kind="clock",
        llm_selected_action="pass", world_model_recommended_action="pass",
        recommendation_confidence=0.8, horizon_s=10.0,
        outcome_horizons_s=(60.0,), checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
        llm_risk_preference_context=audit,
    )
    baseline = capture_policy_outcome_baseline(state, team_id="A")
    record_policy_intervention_result(
        state, decision_id=record["decision_id"], actual_action="pass",
        t_sec=11.0, outcome_baseline=baseline,
    )
    state.ball.position[0] = 0.65
    observe_policy_intervention_outcomes(state, t_sec=71.0)

    assert "llm_risk_preference_evaluation" not in record[
        "multi_horizon_regime_outcomes"
    ]["transition"]
    score = record["multi_horizon_regime_outcomes"]["60s"][
        "llm_risk_preference_evaluation"
    ]
    assert score["paired_same_action_horizon"]
    assert score["audit_digest"] == audit["audit_digest"]
