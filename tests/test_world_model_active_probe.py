"""Safe exploration becomes a pre-registered, falsifiable model probe."""

import copy
from types import SimpleNamespace

from src.match_engine.world_model.active_probe import (
    active_probe_audit_is_valid,
    active_probe_design_is_valid,
    active_probe_evaluation_is_valid,
    build_active_probe_design,
    evaluate_llm_active_probe,
    score_active_probe,
)
from src.match_engine.world_model.active_probe_evaluation import (
    active_probe_diagnostics,
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


def _packet():
    return {
        "team_id": "Home",
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a",
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
                    "transition": {"retention_probability": 0.35},
                    "60s": {"retention_probability": 0.42},
                },
            },
            {
                "action": "pass",
                "multi_horizon_predictions": {
                    "transition": {"retention_probability": 0.82},
                    "60s": {"retention_probability": 0.70},
                },
            },
        ],
    }


def _accepted_probe():
    packet = _packet()
    design = build_active_probe_design(packet)
    packet["active_probe_design"] = design
    option = design["options"][0]
    audit = evaluate_llm_active_probe(
        packet,
        {
            "probe_id": option["probe_id"],
            "confidence": 0.86,
            "rationale": "This endpoint most clearly separates the forecasts.",
        },
        selected_action="pass",
        decision_mode="explore",
        selected_after_action_freeze=True,
    )
    return packet, audit


def test_active_probe_is_exact_safe_and_post_action_only():
    packet = _packet()
    design = build_active_probe_design(packet)
    packet["active_probe_design"] = design

    assert active_probe_design_is_valid(design)
    assert design["available"]
    assert all(option["action"] == "pass" for option in design["options"])
    assert all(option["null_action"] == "shot" for option in design["options"])
    option = design["options"][0]

    pre_action = evaluate_llm_active_probe(
        packet,
        {"probe_id": option["probe_id"], "confidence": 0.8},
        selected_action="pass",
        decision_mode="explore",
        selected_after_action_freeze=False,
    )
    mismatch = evaluate_llm_active_probe(
        packet,
        {"probe_id": option["probe_id"], "confidence": 0.8},
        selected_action="shot",
        decision_mode="exploit",
        selected_after_action_freeze=True,
    )
    assert not pre_action["accepted"]
    assert not mismatch["accepted"]

    _, audit = _accepted_probe()
    assert active_probe_audit_is_valid(audit)
    assert not audit["can_change_current_action"]
    assert not audit["can_change_tactical_controls"]
    assert not audit["can_schedule_future_action"]


def test_realized_probe_scores_against_pre_registered_exploit_null():
    _, audit = _accepted_probe()
    score = score_active_probe(
        audit,
        {"retained_possession": True},
        horizon=audit["probe"]["horizon"],
        realized_action="pass",
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )

    assert score is not None
    assert active_probe_evaluation_is_valid(score, audit)
    assert score["brier_skill_vs_null"] > 0.0
    assert score["log_likelihood_ratio_vs_null"] > 0.0
    assert not score["falsification_signal"]
    tampered = copy.deepcopy(score)
    tampered["brier_skill_vs_null"] = 1.0
    assert not active_probe_evaluation_is_valid(tampered, audit)
    assert score_active_probe(
        audit,
        {"retained_possession": True},
        horizon=audit["probe"]["horizon"],
        realized_action="shot",
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    ) is None


def test_probe_survives_registration_and_is_scored_at_declared_horizon():
    _, audit = _accepted_probe()
    home = SimpleNamespace(team_id="Home", score=0)
    away = SimpleNamespace(team_id="Away", score=0)
    state = SimpleNamespace(
        home=home,
        away=away,
        ball=SimpleNamespace(
            position=[0.5, 0.5], possession_team_id="Home",
        ),
        micro_xg_home=0.0,
        micro_xg_away=0.0,
        clock_seconds=100.0,
    )
    baseline = capture_policy_outcome_baseline(state, team_id="Home")
    record = register_coach_action_decision(
        state,
        team_id="Home",
        trigger_kind="test",
        llm_selected_action="pass",
        world_model_recommended_action="shot",
        recommendation_confidence=0.8,
        horizon_s=10.0,
        intervention_strength=0.2,
        intervention_enabled=True,
        experiment_control_rate=0.0,
        outcome_horizons_s=(0.0, 60.0),
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
        llm_active_probe_context=audit,
    )
    assert record["llm_active_probe_context"] == audit
    record_policy_intervention_result(
        state,
        decision_id=record["decision_id"],
        actual_action="pass",
        t_sec=100.1,
        outcome_baseline=baseline,
    )
    observe_policy_intervention_outcomes(state, t_sec=100.1)
    evaluation = record["multi_horizon_regime_outcomes"][
        "transition"
    ]["llm_active_probe_evaluation"]
    assert evaluation["probe_id"] == audit["probe"]["probe_id"]
    assert evaluation["brier_skill_vs_null"] > 0.0


def test_cross_match_probe_gate_requires_realized_scored_evidence():
    _, audit = _accepted_probe()
    probe = audit["probe"]
    score = score_active_probe(
        audit,
        {"retained_possession": True},
        horizon=audit["probe"]["horizon"],
        realized_action="pass",
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    clusters = []
    logs = []
    for index in range(4):
        record = {
            "decision_id": f"Home:{index}",
            "team_id": "Home",
            "llm_active_probe_context": copy.deepcopy(audit),
            "intervention_actual_action": "pass",
            "multi_horizon_regime_outcomes": {
                probe["horizon"]: {
                    "llm_active_probe_evaluation": copy.deepcopy(score),
                },
            },
        }
        clusters.append([record])
        logs.append({
            "world_model_decision_adoption": {"records": [record]},
        })

    diagnostics = active_probe_diagnostics(clusters)
    assert diagnostics["scored_probe_outcomes"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics["mean_brier_skill_vs_exploit_null"] > 0.0
    report = aggregate_online_calibration(
        logs,
        min_transitions=0,
        require_active_probe_design=True,
    )
    assert report["version"] == 46
    assert report["active_probe_design_ready"]
    assert report["gates"]["active_probe_design"]
