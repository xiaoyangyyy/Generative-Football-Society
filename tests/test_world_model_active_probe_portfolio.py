"""One safe action can carry a bounded nonredundant probe portfolio."""

import copy
from types import SimpleNamespace

from src.match_engine.world_model.active_probe import (
    build_active_probe_design,
    score_active_probe,
)
from src.match_engine.world_model.active_probe_memory import (
    compile_active_probe_discovery_memory,
)
from src.match_engine.world_model.active_probe_portfolio import (
    active_probe_portfolio_audit_is_valid,
    active_probe_portfolio_design_is_valid,
    build_active_probe_portfolio_design,
    evaluate_llm_active_probe_portfolio,
)
from src.match_engine.world_model.active_probe_portfolio_evaluation import (
    active_probe_portfolio_diagnostics,
)
from src.match_engine.world_model.decision_adoption import (
    record_policy_intervention_result,
    register_coach_action_decision,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)
from src.match_engine.world_model.policy_outcomes import (
    capture_policy_outcome_baseline,
    observe_policy_intervention_outcomes,
)
from src.match_engine.cognitive.schemas import validate_coach_plan


def _packet():
    packet = {
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
                    "transition": {"retention_probability": 0.10},
                    "60s": {"retention_probability": 0.20},
                    "180s": {"retention_probability": 0.30},
                },
            },
            {
                "action": "pass",
                "multi_horizon_predictions": {
                    "transition": {"retention_probability": 0.90},
                    "60s": {"retention_probability": 0.80},
                    "180s": {"retention_probability": 0.70},
                },
            },
        ],
    }
    packet["active_probe_design"] = build_active_probe_design(packet)
    packet["active_probe_portfolio_design"] = (
        build_active_probe_portfolio_design(packet)
    )
    return packet


def _accepted_portfolio():
    packet = _packet()
    design = packet["active_probe_portfolio_design"]
    audit = evaluate_llm_active_probe_portfolio(
        packet,
        {
            "portfolio_id": design["recommended_portfolio_id"],
            "confidence": 0.84,
            "rationale": "Use the strongest nonredundant observation bundle.",
        },
        selected_action="pass",
        decision_mode="explore",
        selected_after_action_freeze=True,
    )
    return packet, audit


def _portfolio_record(index: int):
    _, audit = _accepted_portfolio()
    outcomes = {}
    for probe_audit in audit["probe_audits"]:
        probe = probe_audit["probe"]
        evaluation = score_active_probe(
            probe_audit,
            {"retained_possession": True},
            horizon=probe["horizon"],
            realized_action="pass",
            checkpoint_signature="checkpoint-a",
            environment_signature="env-a",
        )
        outcomes[probe["horizon"]] = {
            "llm_active_probe_portfolio_evaluation": evaluation,
        }
    return {
        "decision_id": f"Home:{index}",
        "team_id": "Home",
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a",
        "llm_active_probe_portfolio_context": audit,
        "intervention_actual_action": "pass",
        "multi_horizon_regime_outcomes": outcomes,
    }


def test_portfolio_is_exact_bounded_and_post_action_only():
    packet = _packet()
    design = packet["active_probe_portfolio_design"]
    assert active_probe_portfolio_design_is_valid(
        design, packet["active_probe_design"],
    )
    assert design["available"]
    assert all(
        1 <= option["observation_count"] <= 2
        for option in design["options"]
    )
    recommended = next(
        option for option in design["options"]
        if option["portfolio_id"] == design["recommended_portfolio_id"]
    )
    assert recommended["observation_count"] == 2
    assert recommended["redundancy_penalty"] > 0.0
    assert recommended["incremental_action_regret"] == 0.0
    validated_plan = validate_coach_plan({
        "world_model_active_probe_portfolio": {
            "portfolio_id": recommended["portfolio_id"],
            "confidence": 0.8,
            "rationale": "Bounded joint evidence.",
        },
    })
    assert validated_plan["world_model_active_probe_portfolio"][
        "portfolio_id"
    ] == recommended["portfolio_id"]

    rejected = evaluate_llm_active_probe_portfolio(
        packet,
        {"portfolio_id": recommended["portfolio_id"], "confidence": 0.8},
        selected_action="pass",
        decision_mode="explore",
        selected_after_action_freeze=False,
    )
    assert not rejected["accepted"]
    _, audit = _accepted_portfolio()
    assert active_probe_portfolio_audit_is_valid(audit)
    assert len(audit["probe_audits"]) == 2
    assert not audit["can_schedule_future_action"]


def test_registered_portfolio_scores_each_declared_horizon_once():
    _, audit = _accepted_portfolio()
    state = SimpleNamespace(
        home=SimpleNamespace(team_id="Home", score=0),
        away=SimpleNamespace(team_id="Away", score=0),
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
        outcome_horizons_s=(0.0, 60.0, 180.0),
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
        llm_active_probe_portfolio_context=audit,
    )
    record_policy_intervention_result(
        state,
        decision_id=record["decision_id"],
        actual_action="pass",
        t_sec=100.1,
        outcome_baseline=baseline,
    )
    for now in (100.1, 160.1, 280.1):
        observe_policy_intervention_outcomes(state, t_sec=now)
    for probe_audit in audit["probe_audits"]:
        horizon = probe_audit["probe"]["horizon"]
        evaluation = record["multi_horizon_regime_outcomes"][horizon][
            "llm_active_probe_portfolio_evaluation"
        ]
        assert evaluation["probe_id"] == probe_audit["probe"]["probe_id"]
    assert not record["llm_active_probe_context"]


def test_portfolio_gate_and_discovery_memory_use_all_valid_subprobes():
    records = [_portfolio_record(index) for index in range(8)]
    clusters = [[record] for record in records]
    diagnostics = active_probe_portfolio_diagnostics(clusters)
    assert diagnostics["completed_portfolios"] == 8
    assert diagnostics["completed_multi_horizon_portfolios"] == 8
    assert diagnostics["scored_portfolio_probes"] == 16
    logs = [
        {"world_model_decision_adoption": {"records": [record]}}
        for record in records
    ]
    memory = compile_active_probe_discovery_memory(
        logs,
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    assert len(memory.profiles) == 2
    assert all(profile.active for profile in memory.profiles.values())
    report = aggregate_online_calibration(
        logs,
        min_transitions=0,
        require_active_probe_design=True,
        require_active_probe_portfolios=True,
    )
    assert report["version"] == 47
    assert report["active_probe_portfolios_ready"]
    assert report["active_probe_design_ready"]
    assert report["gates"]["active_probe_design"]
    assert report["gates"]["active_probe_portfolios"]

    tampered = copy.deepcopy(records)
    tampered[0]["llm_active_probe_portfolio_context"]["portfolio"][
        "observation_budget"
    ] = 3
    rejected = active_probe_portfolio_diagnostics([[tampered[0]]])
    assert rejected["malformed_portfolio_audits"] == 1
