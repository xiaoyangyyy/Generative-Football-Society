from types import SimpleNamespace

import pytest

from src.match_engine.cognitive.schemas import validate_coach_plan
from src.match_engine.world_model.risk_certificate import (
    apply_llm_risk_constraint,
    risk_certificate_diagnostics,
    score_llm_risk_certificate,
    validate_llm_risk_constraint,
    wilson_upper_bound,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)


def _constraint(**updates):
    value = {
        "selected_action": "pass",
        "horizon": "60s",
        "downside_event": "lose_possession",
        "max_violation_probability": 0.60,
        "confidence": 0.8,
        "rationale": "Limit the member-implied possession downside.",
    }
    value.update(updates)
    return value


def _packet(*, rollout_steps=1, trained=True, temporal=False):
    modes = {
        "version": 1,
        "available": trained,
        "counts_trusted": trained,
        "causal_interpretation": False,
        "ensemble_members": 8,
        "mode_count": 2,
        "normalized_mode_entropy": 0.54,
        "downside_event_counts": {"lose_possession": 1},
        "downside_event_probabilities": {"lose_possession": 1.5 / 9.0},
        "modes": [
            {
                "mode_id": "mode_0",
                "probability": 0.875,
                "downside_events": {"lose_possession": 0.0},
            },
            {
                "mode_id": "mode_1",
                "probability": 0.125,
                "downside_events": {"lose_possession": 1.0},
            },
        ],
    }
    if temporal:
        modes.update({
            "trajectory_steps": 2,
            "temporal_path_available": True,
            "path_counts_trusted": True,
            "path_downside_event_counts": {"lose_possession": 2},
            "path_downside_event_probabilities": {
                "lose_possession": 2.5 / 9.0,
            },
        })
        modes["modes"][1]["path_downside_events"] = {
            "lose_possession": 1.0,
        }
    return {
        "available": True,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "candidates": [{
            "action": "pass",
            "multi_horizon_predictions": {
                "60s": {
                    "rollout_steps": rollout_steps,
                    "state_scales": {"trajectory_modes": modes},
                },
            },
        }],
    }


class _Runtime:
    def __init__(self, active=True):
        self.active = active

    def two_step_planning_gate(self):
        return {"active": self.active, "authority": 0.4}


def test_risk_constraint_schema_is_strict_and_preserved_by_coach_schema():
    assert validate_llm_risk_constraint(_constraint())["selected_action"] == "pass"
    assert validate_llm_risk_constraint(_constraint(selected_action="none")) is None
    assert validate_llm_risk_constraint(_constraint(horizon="later")) is None
    assert validate_llm_risk_constraint(_constraint(downside_event="injury")) is None
    assert validate_llm_risk_constraint(
        _constraint(max_violation_probability=0.01)
    ) is None
    assert validate_llm_risk_constraint(_constraint(confidence=0.49)) is None
    assert validate_llm_risk_constraint(_constraint(risk_scope="sometime")) is None
    plan = validate_coach_plan({
        "world_model_action": "pass",
        "world_model_risk_constraint": _constraint(),
    })
    assert plan["world_model_risk_constraint"]["horizon"] == "60s"


def test_wilson_upper_bound_is_conservative_and_validates_counts():
    assert wilson_upper_bound(0, 3) == pytest.approx(0.4742, abs=1e-3)
    assert wilson_upper_bound(1, 8) > 1 / 8
    assert wilson_upper_bound(2, 8) > wilson_upper_bound(1, 8)
    with pytest.raises(ValueError):
        wilson_upper_bound(2, 1)


def test_shadow_certificate_distinguishes_nominal_and_conservative_safety():
    packet = _packet()
    audit = apply_llm_risk_constraint(
        _Runtime(), packet, _constraint(), selected_action="pass",
        certificate_signature="llm-risk-certificate:test",
    )
    assert audit["accepted"]
    assert audit["nominal_constraint_satisfied"]
    assert audit["conservatively_certified"]
    assert audit["certificate_can_veto_action"] is False
    assert audit["policy_mutated"] is False
    assert packet["llm_risk_certificate_audit"] is audit

    cautious = apply_llm_risk_constraint(
        _Runtime(), _packet(),
        _constraint(max_violation_probability=0.20),
        selected_action="pass",
    )
    assert cautious["nominal_constraint_satisfied"]
    assert not cautious["conservatively_certified"]


def test_certificate_fails_closed_for_unvalidated_evidence():
    mismatch = apply_llm_risk_constraint(
        _Runtime(), _packet(), _constraint(), selected_action="shot",
    )
    assert mismatch["reason"] == "risk_constraint_action_must_equal_selected_action"
    untrained = apply_llm_risk_constraint(
        _Runtime(), _packet(trained=False), _constraint(), selected_action="pass",
    )
    assert untrained["reason"] == "trained_trajectory_modes_unavailable"
    deep = apply_llm_risk_constraint(
        _Runtime(), _packet(rollout_steps=3), _constraint(), selected_action="pass",
    )
    assert deep["reason"] == "risk_constraint_rollout_depth_not_validated"
    closed = apply_llm_risk_constraint(
        _Runtime(active=False), _packet(rollout_steps=2), _constraint(),
        selected_action="pass",
    )
    assert closed["reason"] == "risk_certificate_planning_gate_closed"
    missing_path = apply_llm_risk_constraint(
        _Runtime(), _packet(rollout_steps=2),
        _constraint(risk_scope="within_horizon"),
        selected_action="pass",
    )
    assert missing_path["reason"] == (
        "member_consistent_temporal_path_unavailable"
    )


def test_ever_risk_uses_member_consistent_path_and_dense_live_monitor():
    audit = apply_llm_risk_constraint(
        _Runtime(), _packet(rollout_steps=2, temporal=True),
        _constraint(
            risk_scope="within_horizon",
            max_violation_probability=0.7,
        ),
        selected_action="pass",
        certificate_signature="llm-risk-certificate:test",
    )
    assert audit["accepted"]
    assert audit["risk_scope"] == "within_horizon"
    assert audit["member_violations"] == 2
    assert audit["projected_violation_probability"] == pytest.approx(2.5 / 9.0)
    assert audit["member_trajectory_identity_preserved"]

    score = score_llm_risk_certificate(
        audit,
        {
            "retained_possession": True,
            "progress": 0.1,
            "goal_diff_delta": 0.0,
        },
        {"ball_x": 0.4},
        attacking_home=True,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
        interval_monitor={
            "complete": True,
            "downside_observed": True,
            "observations": 12,
            "max_gap_s": 5.0,
        },
    )
    assert score["downside_observed"]
    assert score["risk_scope"] == "within_horizon"
    assert score["interval_monitor_complete"]
    assert score_llm_risk_certificate(
        audit,
        {
            "retained_possession": True,
            "progress": 0.1,
            "goal_diff_delta": 0.0,
        },
        {"ball_x": 0.4},
        attacking_home=True,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
        interval_monitor={"complete": False},
    ) is None


def test_realized_certificate_scoring_and_match_clustered_diagnostics():
    audit = apply_llm_risk_constraint(
        _Runtime(), _packet(), _constraint(), selected_action="pass",
        certificate_signature="llm-risk-certificate:test",
    )
    baseline = {"ball_x": 0.4}
    safe_outcome = {
        "retained_possession": True,
        "progress": 0.1,
        "goal_diff_delta": 0.0,
    }
    score = score_llm_risk_certificate(
        audit, safe_outcome, baseline, attacking_home=True,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    assert score["downside_observed"] is False
    assert score["violation_brier"] == pytest.approx((1.5 / 9.0) ** 2)
    assert score["false_safe_certificate"] is False

    violated = score_llm_risk_certificate(
        audit, {**safe_outcome, "retained_possession": False}, baseline,
        attacking_home=True, checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    assert violated["false_safe_certificate"]
    logs = []
    for index in range(4):
        evaluation = score if index else violated
        logs.append({"world_model_decision_adoption": {"records": [{
            "multi_horizon_regime_outcomes": {
                "60s": {"llm_risk_certificate_evaluation": dict(evaluation)}
            }
        }]}})
    diagnostics = risk_certificate_diagnostics(logs)
    assert diagnostics["realized_certificates"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics["certified_outcomes"] == 4
    assert diagnostics["false_safe_certificates"] == 1
    assert diagnostics["provenance_compatible"]

    logs[-1]["world_model_decision_adoption"]["records"][0][
        "multi_horizon_regime_outcomes"
    ]["60s"]["llm_risk_certificate_evaluation"][
        "certificate_signature"
    ] = "llm-risk-certificate:mixed"
    assert not risk_certificate_diagnostics(logs)["provenance_compatible"]


def test_online_gate_requires_realized_calibrated_risk_certificates():
    audit = apply_llm_risk_constraint(
        _Runtime(), _packet(), _constraint(), selected_action="pass",
        certificate_signature="llm-risk-certificate:test",
    )
    baseline = {"ball_x": 0.4}
    logs = []
    for _ in range(4):
        evaluation = score_llm_risk_certificate(
            audit,
            {
                "retained_possession": True,
                "progress": 0.1,
                "goal_diff_delta": 0.0,
            },
            baseline,
            attacking_home=True,
            checkpoint_signature="checkpoint:test",
            environment_signature="environment:test",
        )
        logs.append({"world_model_decision_adoption": {"records": [{
            "multi_horizon_regime_outcomes": {
                "60s": {"llm_risk_certificate_evaluation": evaluation}
            }
        }]}})
    report = aggregate_online_calibration(
        logs, min_transitions=0, min_residual_samples=2,
        require_llm_risk_certificates=True,
    )
    assert report["version"] == 28
    assert report["llm_risk_certificates_ready"]
    assert report["gates"]["realized_llm_risk_certificates"]
    assert report["decision_adoption"]["llm_risk_certificates"][
        "certified_matches"
    ] == 4

    logs.append({"world_model_decision_adoption": {"records": [{
        "intervention_actual_action": "pass",
        "llm_risk_certificate_context": audit,
        "multi_horizon_regime_outcomes": {"60s": {}},
    }]}})
    incomplete = aggregate_online_calibration(
        logs, min_transitions=0, min_residual_samples=2,
        require_llm_risk_certificates=True,
    )
    diagnostics = incomplete["decision_adoption"]["llm_risk_certificates"]
    assert diagnostics["unscored_eligible_certificates"] == 1
    assert not incomplete["llm_risk_certificates_ready"]
