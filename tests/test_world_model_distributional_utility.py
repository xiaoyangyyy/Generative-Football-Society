import numpy as np
import pytest

from src.match_engine.cognitive.schemas import validate_coach_plan
from src.match_engine.world_model.distributional_utility import (
    align_distribution_location,
    build_distributional_action_frontiers,
    member_policy_utility_distribution,
)
from src.match_engine.world_model.predictive_distribution import (
    build_predictive_scenario_lattice,
)
from src.match_engine.world_model.distributional_claim import (
    distributional_claim_diagnostics,
    evaluate_llm_distributional_claim,
    score_distributional_claim,
    validate_llm_distributional_claim,
)
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)
from src.match_engine.world_model.policy_prediction import (
    build_multi_horizon_policy_predictions,
)


def _distribution(values):
    values = np.asarray(values, dtype=float)
    scenarios = np.repeat(values, 3)
    ordered = np.sort(scenarios)
    quantiles = np.quantile(scenarios, [0.1, 0.25, 0.5, 0.75, 0.9])
    tail = max(1, int(np.ceil(0.25 * len(scenarios))))
    return {
        "version": 2,
        "available": True,
        "counts_trusted": True,
        "member_identity_preserved": True,
        "member_values": values.tolist(),
        "epistemic_member_values": values.tolist(),
        "decision_scenario_values": scenarios.tolist(),
        "predictive_scenario_values": scenarios.tolist(),
        "predictive_distribution_available": True,
        "distribution_scope": "calibrated_predictive",
        "residual_scenario_offsets": [0.0, 0.0, 0.0],
        "residual_calibration_samples": 8,
        "residual_quantiles_split": "held_out_calibration",
        "uncertainty_decomposition": {
            "epistemic_member_variance": float(values.var()),
            "residual_outcome_variance": 0.0,
            "predictive_lattice_variance": float(scenarios.var()),
            "additive_identity_error": 0.0,
            "axes_statistically_independent_claimed": False,
        },
        "mean_utility": float(scenarios.mean()),
        "utility_std": float(scenarios.std()),
        "lower_tail_cvar_25": float(ordered[:tail].mean()),
        "upside_probability": float(
            (np.sum(scenarios > 0) + 0.5) / (len(scenarios) + 1)
        ),
        "quantiles": dict(zip(
            ("q10", "q25", "q50", "q75", "q90"), map(float, quantiles),
        )),
    }


def _packet():
    distributions = {
        "pass": _distribution([-0.1, 0.1, 0.2, 0.3]),
        "hold": _distribution([0.02, 0.04, 0.06, 0.08]),
        "shot": _distribution([-0.4, -0.2, 0.4, 0.9]),
    }
    return {
        "available": True,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "candidates": [
            {
                "action": action,
                "multi_horizon_predictions": {
                    "60s": {"distributional_policy_utility": distribution}
                },
            }
            for action, distribution in distributions.items()
        ],
    }


def _claim(**updates):
    value = {
        "selected_action": "pass",
        "alternative_action": "hold",
        "horizon": "60s",
        "criterion": "mean_utility",
        "distribution_scope": "calibrated_predictive",
        "relation": "selected_better",
        "confidence": 0.8,
        "rationale": "Pass has the stronger member-average utility.",
    }
    value.update(updates)
    return value


def test_member_policy_utility_distribution_is_transparent_and_trained_only():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    current[200] = 0.5
    current[204:206] = 0.2
    current[209] = 1.0
    future = np.repeat(current[None, :], 4, axis=0)
    future[:, 200] = [0.4, 0.5, 0.6, 0.7]
    future[:, 209] = [0.0, 0.4, 0.8, 1.0]
    progress = np.asarray([
        [[-0.2]], [[0.0]], [[0.1]], [[0.2]],
    ], dtype=np.float32)

    distribution = member_policy_utility_distribution(
        current, future[:, None, :], progress,
        attacking_home=True, rollout_steps=2, ensemble_trained=True,
    )

    assert distribution["available"]
    assert distribution["ensemble_members"] == 4
    assert len(distribution["member_values"]) == 4
    assert distribution["minimum_utility"] < distribution["maximum_utility"]
    assert distribution["lower_tail_cvar_25"] == pytest.approx(
        min(distribution["member_values"])
    )
    assert list(distribution["quantiles"].values()) == sorted(
        distribution["quantiles"].values()
    )
    assert 0.0 < distribution["upside_probability"] < 1.0
    assert distribution["member_identity_preserved"]
    assert distribution["distribution_scope"] == "epistemic_member_only"
    assert not distribution["causal_interpretation"]

    neutral = member_policy_utility_distribution(
        current, future[:1], progress[:1],
        attacking_home=True, rollout_steps=1, ensemble_trained=False,
    )
    assert not neutral["available"]
    assert neutral["member_values"] == []


def test_distributional_frontier_keeps_mean_tail_and_upside_tradeoffs():
    packet = _packet()
    frontiers = build_distributional_action_frontiers(packet["candidates"])
    horizon = frontiers["horizons"]["60s"]

    assert horizon["available"]
    assert horizon["criterion_leaders"]["mean_utility"] == ["shot"]
    assert horizon["criterion_leaders"]["lower_tail_cvar_25"] == ["hold"]
    assert set(horizon["pareto_actions"]) >= {"hold", "shot"}
    assert not frontiers["authority_active"]


def test_location_alignment_preserves_spread_and_matches_calibrated_mean():
    original = _distribution([-0.2, 0.0, 0.1, 0.3])
    aligned = align_distribution_location(
        original, 0.25, source="test_residual_memory",
    )

    assert aligned["mean_utility"] == pytest.approx(0.25)
    assert np.std(aligned["member_values"]) == pytest.approx(
        np.std(original["member_values"])
    )
    assert aligned["location_alignment_shift"] == pytest.approx(
        0.25 - original["mean_utility"]
    )
    assert aligned["spread_preserved_by_location_alignment"]
    closed = align_distribution_location(
        original, 2.0, source="invalid_large_shift",
    )
    assert not closed["available"]
    assert closed["reason"] == "distribution_location_shift_gate_closed"


def test_predictive_lattice_separates_epistemic_and_residual_variance():
    epistemic = _distribution([-0.1, 0.1])
    epistemic.update({
        "predictive_distribution_available": False,
        "distribution_scope": "epistemic_member_only",
    })
    predictive = build_predictive_scenario_lattice(epistemic, {
        "calibration_samples": 8,
        "residual_quantile_levels": [0.1, 0.5, 0.9],
        "residual_quantiles": [-0.2, 0.0, 0.2],
        "residual_quantiles_split": "held_out_calibration",
        "residual_quantiles_observed": True,
        "drift": {"status": "stable"},
    })

    assert predictive["predictive_distribution_available"]
    assert predictive["distribution_scope"] == "calibrated_predictive"
    assert predictive["member_values"] == [-0.1, 0.1]
    assert predictive["epistemic_distribution_summary"][
        "mean_utility"
    ] == pytest.approx(0.0)
    assert len(predictive["predictive_scenario_values"]) == 6
    decomposition = predictive["uncertainty_decomposition"]
    assert decomposition["epistemic_member_variance"] == pytest.approx(0.01)
    assert decomposition["residual_outcome_variance"] > 0.0
    assert decomposition["additive_identity_error"] < 1e-12
    assert not decomposition["axes_statistically_independent_claimed"]

    insufficient = build_predictive_scenario_lattice(epistemic, {
        "calibration_samples": 2,
        "residual_quantile_levels": [0.1, 0.5, 0.9],
        "residual_quantiles": [-0.2, 0.0, 0.2],
        "residual_quantiles_split": "held_out_calibration",
        "residual_quantiles_observed": True,
        "drift": {"status": "stable"},
    })
    assert not insufficient["predictive_distribution_available"]
    assert insufficient["distribution_scope"] == "epistemic_member_only"


def test_policy_prediction_composes_residual_scenarios_after_point_alignment():
    class _Runtime:
        def predict_policy_utility(self, *args, **kwargs):
            return {
                "policy_utility": 0.0,
                "uncertainty": 0.1,
                "distributional_policy_utility": {
                    "version": 2,
                    "available": True,
                    "counts_trusted": True,
                    "member_identity_preserved": True,
                    "member_values": [-0.1, 0.1],
                    "mean_utility": 0.0,
                    "utility_std": 0.1,
                    "tail_member_count": 1,
                },
            }

    class _ResidualMemory:
        def calibrate(self, prediction, **kwargs):
            return {
                **prediction,
                "policy_utility": 0.05,
                "residual_memory_trust_factor": 1.0,
                "residual_memory": {
                    "calibration_samples": 8,
                    "residual_quantile_levels": [0.1, 0.5, 0.9],
                    "residual_quantiles": [-0.2, 0.0, 0.2],
                    "residual_quantiles_split": "held_out_calibration",
                    "residual_quantiles_observed": True,
                    "drift": {"status": "stable"},
                },
            }

    predictions = build_multi_horizon_policy_predictions(
        _Runtime(), np.zeros(OBS_DIM), action_name="pass",
        target=np.zeros(2), physics_prior=0.1, confidence=0.8,
        attacking_home=True, base_horizon_s=5.0,
        outcome_horizons_s=(60.0,), residual_memory=_ResidualMemory(),
    )
    distribution = predictions["60s"]["distributional_policy_utility"]
    assert distribution["distribution_scope"] == "calibrated_predictive"
    assert distribution["epistemic_member_values"] == pytest.approx(
        [-0.05, 0.15]
    )
    assert len(distribution["decision_scenario_values"]) == 6


def test_distributional_claim_is_schema_limited_and_model_checked():
    assert validate_llm_distributional_claim(_claim())["criterion"] == "mean_utility"
    assert validate_llm_distributional_claim(
        _claim(alternative_action="pass")
    ) is None
    assert validate_llm_distributional_claim(_claim(criterion="vibes")) is None
    assert validate_llm_distributional_claim(_claim(confidence=0.4)) is None
    assert validate_llm_distributional_claim(
        _claim(distribution_scope="outcome-ish")
    ) is None
    scope_mismatch = evaluate_llm_distributional_claim(
        _packet(), _claim(distribution_scope="epistemic_member_only"),
        selected_action="pass",
    )
    assert scope_mismatch["reason"] == "distribution_scope_mismatch"
    plan = validate_coach_plan({
        "world_model_action": "pass",
        "world_model_distributional_claim": _claim(),
    })
    assert plan["world_model_distributional_claim"]["criterion"] == (
        "mean_utility"
    )

    audit = evaluate_llm_distributional_claim(
        _packet(), _claim(), selected_action="pass",
        claim_signature="llm-distributional-claim:test",
    )
    assert audit["accepted"]
    assert audit["directionally_faithful"]
    assert audit["criterion_delta"] > 0.0
    assert not audit["can_change_selected_action"]
    assert not audit["policy_mutated"]

    false_claim = evaluate_llm_distributional_claim(
        _packet(), _claim(relation="alternative_better"),
        selected_action="pass",
    )
    assert false_claim["accepted"]
    assert not false_claim["directionally_faithful"]


def test_distribution_score_uses_crps_quantiles_and_provenance():
    audit = evaluate_llm_distributional_claim(
        _packet(), _claim(), selected_action="pass",
        claim_signature="llm-distributional-claim:test",
    )
    score = score_distributional_claim(
        audit, {"policy_utility": 0.15},
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )

    assert score["paired_same_action_horizon"]
    assert score["predictive_distribution_calibrated"]
    assert score["empirical_crps"] >= 0.0
    assert score["central_80_covered"]
    assert score["issuance_evaluation_provenance_compatible"]
    assert not score["authority_active"]

    logs = [{"world_model_decision_adoption": {"records": [{
        "multi_horizon_regime_outcomes": {
            "60s": {"llm_distributional_claim_evaluation": dict(score)}
        }
    }]}} for _ in range(4)]
    diagnostics = distributional_claim_diagnostics(logs)
    assert diagnostics["realized_distributions"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics["malformed_evaluations"] == 0
    assert diagnostics["match_clustered_directional_faithfulness"] == 1.0
    assert diagnostics["provenance_compatible"]
    assert diagnostics["all_predictive_distributions_calibrated"]

    tampered = dict(score)
    tampered["decision_scenario_values"] = list(
        tampered["decision_scenario_values"]
    )
    tampered["decision_scenario_values"][0] += 0.01
    tampered_logs = [{"world_model_decision_adoption": {"records": [{
        "multi_horizon_regime_outcomes": {
            "60s": {"llm_distributional_claim_evaluation": tampered}
        }
    }]}}]
    assert distributional_claim_diagnostics(tampered_logs)[
        "malformed_evaluations"
    ] == 1

    logs[-1]["world_model_decision_adoption"]["records"][0][
        "multi_horizon_regime_outcomes"
    ]["60s"]["llm_distributional_claim_evaluation"][
        "claim_signature"
    ] = "llm-distributional-claim:mixed"
    assert not distributional_claim_diagnostics(logs)["provenance_compatible"]


def test_strict_online_gate_requires_calibrated_faithful_distributions():
    audit = evaluate_llm_distributional_claim(
        _packet(), _claim(), selected_action="pass",
        claim_signature="llm-distributional-claim:test",
    )
    logs = []
    for observed in (0.0, 0.12, 0.18, 0.40):
        score = score_distributional_claim(
            audit, {"policy_utility": observed},
            checkpoint_signature="checkpoint:test",
            environment_signature="environment:test",
        )
        logs.append({"world_model_decision_adoption": {"records": [{
            "multi_horizon_regime_outcomes": {
                "60s": {"llm_distributional_claim_evaluation": score}
            }
        }]}})

    report = aggregate_online_calibration(
        logs, min_transitions=0, min_residual_samples=2,
        require_llm_distributional_decisions=True,
    )
    diagnostics = report["decision_adoption"][
        "llm_distributional_decisions"
    ]
    assert report["version"] == 19
    assert diagnostics["match_clustered_central_80_coverage"] == 0.75
    assert diagnostics["match_clustered_below_median_rate"] == 0.5
    assert diagnostics["match_clustered_crps"] <= diagnostics[
        "match_clustered_mean_absolute_error"
    ]
    assert report["llm_distributional_decisions_ready"]
    assert report["gates"]["calibrated_llm_distributional_decisions"]

    logs.append({"world_model_decision_adoption": {"records": [{
        "intervention_actual_action": "pass",
        "llm_distributional_claim_context": audit,
        "multi_horizon_regime_outcomes": {"60s": {}},
    }]}})
    incomplete = aggregate_online_calibration(
        logs, min_transitions=0, min_residual_samples=2,
        require_llm_distributional_decisions=True,
    )
    assert incomplete["decision_adoption"][
        "llm_distributional_decisions"
    ]["unscored_eligible_claims"] == 1
    assert not incomplete["llm_distributional_decisions_ready"]
