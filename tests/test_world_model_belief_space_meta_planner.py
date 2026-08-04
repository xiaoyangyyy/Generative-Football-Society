"""Resolved answers route only bounded future world-model computation."""

import copy

from src.match_engine.world_model.belief_space_meta_planner import (
    belief_space_compute_preference_is_valid,
    belief_space_meta_plan_audit_is_valid,
    belief_space_meta_plan_is_valid,
    build_belief_space_meta_plan,
    evaluate_llm_belief_space_meta_plan,
)
from src.match_engine.world_model.belief_space_meta_planner_evaluation import (
    belief_space_meta_planner_diagnostics,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)
from src.match_engine.world_model.opponent_information_feedback import (
    build_opponent_information_feedback,
)
from src.match_engine.world_model.opponent_information_policy import (
    evaluate_llm_opponent_information_policy,
)
from src.match_engine.world_model.opponent_information_query_outcomes import (
    score_opponent_information_query,
)
from tests.test_world_model_opponent_information_query import (
    _packet,
    _packet_with_cognitive_policy,
)


def _resolved_feedback():
    packet = _packet_with_cognitive_policy()
    recommendation = packet["opponent_information_cognitive_policy"][
        "recommendations_by_action"
    ]["pass"]
    policy_audit = evaluate_llm_opponent_information_policy(
        packet,
        {
            "decision": "ask",
            "proposal_id": recommendation["recommended_proposal_id"],
            "confidence": 0.9,
            "rationale": "Acquire an answer before routing future compute.",
        },
        selected_action="pass",
        query_signature="llm-opponent-information-query:test",
        selected_after_action_freeze=True,
    )
    query_audit = policy_audit["query_audit"]
    score = score_opponent_information_query(
        query_audit,
        {
            "pressing_intensity": 0.8,
            "risk_budget": 0.4,
            "line_height": 0.7,
            "rotation_aggressiveness": 0.3,
        },
        horizon=query_audit["query"]["horizon"],
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    record = {
        "decision_id": "A:10:0",
        "team_id": "A",
        "created_t_sec": 10.0,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "intervention_actual_action": "pass",
        "llm_opponent_information_query_context": query_audit,
        "llm_opponent_information_policy_context": policy_audit,
        "multi_horizon_regime_outcomes": {
            query_audit["query"]["horizon"]: {
                "observed_t_sec": 70.0,
                "llm_opponent_information_query_evaluation": score,
            },
        },
    }
    return build_opponent_information_feedback(
        [record],
        team_id="A",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
        as_of_t_sec=80.0,
    )


def _meta_packet():
    packet = _packet()
    packet.update({
        "team_id": "A",
        "decision_context": {"clock_seconds": 80.0},
        "opponent_information_feedback": _resolved_feedback(),
    })
    return packet


def test_resolved_answer_builds_exact_compute_route_and_llm_preference():
    packet = _meta_packet()
    plan = build_belief_space_meta_plan(
        packet, default_branch_budget=48,
    )
    packet["belief_space_meta_plan"] = plan

    assert belief_space_meta_plan_is_valid(plan)
    assert plan["answer_conditioned_route_available"]
    answer_option = next(
        row for row in plan["options"]
        if row["mode"] == "answer_conditioned_route"
    )
    route = answer_option["route"]
    assert len(route["target_first_actions"]) == 2
    assert len(route["hypothesis_priority"]) == 7
    assert not route["can_change_current_action"]

    audit = evaluate_llm_belief_space_meta_plan(
        packet,
        {
            "option_id": plan["recommended_option_id"],
            "confidence": 0.85,
            "rationale": "Use the answer to focus future shadow rollouts.",
        },
        selected_after_action_freeze=True,
    )
    assert audit["accepted"]
    assert audit["matched_model_recommendation"]
    assert belief_space_meta_plan_audit_is_valid(audit)
    assert belief_space_compute_preference_is_valid(
        audit["compute_preference"]
    )

    replay = build_belief_space_meta_plan(
        packet,
        default_branch_budget=48,
        stored_preference=audit["compute_preference"],
    )
    assert replay["preference_status"] == (
        "prior_llm_compute_preference_applied"
    )
    assert replay["applied_option"] == audit["selected_option"]


def test_meta_plan_rejects_pre_action_and_wrong_scope_preferences():
    packet = _meta_packet()
    plan = build_belief_space_meta_plan(packet, default_branch_budget=48)
    packet["belief_space_meta_plan"] = plan
    raw = {
        "option_id": plan["recommended_option_id"],
        "confidence": 0.8,
        "rationale": "This must happen only after action freeze.",
    }
    rejected = evaluate_llm_belief_space_meta_plan(
        packet, raw, selected_after_action_freeze=False,
    )
    assert not rejected["accepted"]

    accepted = evaluate_llm_belief_space_meta_plan(
        packet, raw, selected_after_action_freeze=True,
    )
    wrong_scope = copy.deepcopy(accepted["compute_preference"])
    wrong_scope["team_id"] = "other"
    next_plan = build_belief_space_meta_plan(
        packet, default_branch_budget=48, stored_preference=wrong_scope,
    )
    assert next_plan["preference_status"] == (
        "model_recommended_current_route"
    )


def test_cross_decision_route_fidelity_and_strict_gate():
    packet = _meta_packet()
    plan = build_belief_space_meta_plan(packet, default_branch_budget=48)
    packet["belief_space_meta_plan"] = plan
    answer_option = next(
        row for row in plan["options"]
        if row["mode"] == "answer_conditioned_route"
    )
    audit = evaluate_llm_belief_space_meta_plan(
        packet,
        {
            "option_id": answer_option["option_id"],
            "confidence": 0.9,
            "rationale": "Concentrate future rollouts on the resolved belief.",
        },
        selected_after_action_freeze=True,
    )
    trajectory = {
        "active": True,
        "branch_evaluation_budget": answer_option[
            "branch_evaluation_budget"
        ],
        "branch_evaluations": answer_option["branch_evaluation_budget"],
        "belief_space_route_applied": True,
        "belief_space_route": answer_option["route"],
        "broad_coverage_pairs": 16,
        "causal_interpretation": False,
    }
    logs = []
    for match_index in range(4):
        records = [
            {
                "decision_id": f"A:{match_index}:0",
                "team_id": "A",
                "created_t_sec": 80.0,
                "llm_belief_space_meta_plan_context": copy.deepcopy(audit),
            },
            {
                "decision_id": f"A:{match_index}:1",
                "team_id": "A",
                "created_t_sec": 90.0,
                "opponent_response_context": {
                    "trajectory_rollout": copy.deepcopy(trajectory),
                },
            },
        ]
        logs.append({"world_model_decision_adoption": {"records": records}})
    diagnostics = belief_space_meta_planner_diagnostics([
        payload["world_model_decision_adoption"]["records"]
        for payload in logs
    ])
    assert diagnostics["accepted_meta_plan_audits"] == 4
    assert diagnostics["applied_next_decision_preferences"] == 4
    assert diagnostics["answer_conditioned_routes_applied"] == 4
    assert diagnostics["match_clustered_route_application_rate"] == 1.0
    assert diagnostics["compute_budget_violations"] == 0
    assert diagnostics["broad_coverage_failures"] == 0

    report = aggregate_online_calibration(
        logs,
        min_transitions=0,
        require_belief_space_meta_planning=True,
    )
    assert report["version"] == 44
    assert report["belief_space_meta_planning_ready"]
    assert report["gates"]["belief_space_meta_planning"]
