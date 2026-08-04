"""LLM tactical questions are converted into model-owned value of information."""

import copy
from types import SimpleNamespace

import pytest

from src.match_engine.cognitive.schemas import validate_coach_plan
from src.match_engine.world_model.opponent_contract import OPPONENT_HYPOTHESES
from src.match_engine.world_model.opponent_information_query import (
    build_opponent_information_question_menu,
    evaluate_llm_opponent_information_query,
    opponent_information_question_menu_is_valid,
    opponent_information_query_audit_is_valid,
    validate_llm_opponent_information_query,
)
from src.match_engine.world_model.opponent_information_query_outcomes import (
    opponent_information_query_score_is_valid,
    score_opponent_information_query,
)
from src.match_engine.world_model.opponent_information_policy import (
    build_opponent_information_cognitive_policy,
    evaluate_llm_opponent_information_policy,
    opponent_information_cognitive_policy_is_valid,
    opponent_information_policy_audit_is_valid,
    opponent_information_policy_diagnostics,
)
from src.match_engine.world_model.opponent_information_query_evaluation import (
    opponent_information_query_diagnostics,
)
from src.match_engine.world_model.opponent_information_calibration import (
    compile_opponent_information_calibration_memory,
)
from src.match_engine.world_model.opponent_information_feedback import (
    build_opponent_information_feedback,
    opponent_information_feedback_diagnostics,
    opponent_information_feedback_is_valid,
)
from src.match_engine.world_model.opponent_information_adaptation import (
    evaluate_llm_opponent_information_adaptation,
    opponent_information_adaptation_audit_is_valid,
    validate_llm_opponent_information_adaptation,
)
from src.match_engine.world_model.opponent_information_adaptation_outcomes import (
    opponent_information_adaptation_score_is_valid,
    score_opponent_information_adaptation,
)
from src.match_engine.world_model.opponent_information_adaptation_evaluation import (
    opponent_information_adaptation_diagnostics,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)
from src.match_engine.world_model.decision_adoption import (
    decision_adoption_diagnostics,
    record_policy_intervention_result,
    register_coach_action_decision,
)
from src.match_engine.world_model.policy_outcomes import (
    capture_policy_outcome_baseline,
    observe_policy_intervention_outcomes,
)


def _query(**updates):
    value = {
        "selected_action": "pass",
        "feature": "line_height",
        "horizon": "60s",
        "purpose": "resolve_action_choice",
        "action_if_high": "pass",
        "action_if_low": "shot",
        "confidence": 0.8,
        "rationale": "Check whether the opponent will keep a high line.",
    }
    value.update(updates)
    return value


def _packet():
    posterior = {
        name: value for name, value in zip(
            OPPONENT_HYPOTHESES,
            [0.14, 0.18, 0.16, 0.14, 0.14, 0.12, 0.12],
        )
    }
    candidates = []
    for action_index, action in enumerate(("hold", "pass", "cross", "shot")):
        values = {
            name: (
                0.02 * action_index
                + 0.03 * hypothesis_index
                + (0.16 if action == "pass" and hypothesis_index < 3 else 0.0)
                + (0.22 if action == "shot" and hypothesis_index >= 3 else 0.0)
            )
            for hypothesis_index, name in enumerate(OPPONENT_HYPOTHESES)
        }
        candidates.append({
            "action": action,
            "opponent_hypothesis_values": values,
            "multi_horizon_predictions": {"60s": {"policy_utility": 0.1}},
            "opponent_response_prediction": (
                {"response_posterior": posterior} if action == "pass" else {}
            ),
        })
    return {
        "available": True,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "opponent_belief": {"posterior": posterior},
        "candidates": candidates,
    }


def _packet_with_question_menu():
    packet = _packet()
    packet["opponent_information_question_menu"] = (
        build_opponent_information_question_menu(packet)
    )
    return packet


def _packet_with_cognitive_policy(feedback=None):
    packet = _packet()
    if feedback is not None:
        packet["opponent_information_feedback"] = feedback
    packet["opponent_information_question_menu"] = (
        build_opponent_information_question_menu(packet)
    )
    packet["opponent_information_cognitive_policy"] = (
        build_opponent_information_cognitive_policy(packet)
    )
    return packet


def _proposed_query(packet, *, purpose="reduce_opponent_uncertainty"):
    proposal = next(
        row for row in packet["opponent_information_question_menu"][
            "proposals"
        ]
        if row["selected_action"] == "pass" and row["purpose"] == purpose
    )
    return {
        key: proposal[key] for key in (
            "proposal_id", "selected_action", "feature", "horizon",
            "purpose", "action_if_high", "action_if_low",
        )
    } | {
        "confidence": 0.85,
        "rationale": "Select the model-proposed question with highest VOI.",
    }


def test_query_schema_is_bounded_and_preserved_by_coach_plan():
    validated = validate_llm_opponent_information_query(_query())
    assert validated["feature"] == "line_height"
    assert "threshold" not in validated
    assert validate_llm_opponent_information_query(
        _query(feature="formation_name")
    ) is None
    assert validate_llm_opponent_information_query(
        _query(horizon="transition")
    ) is None
    assert validate_llm_opponent_information_query(
        _query(action_if_high="none")
    ) is None
    plan = validate_coach_plan({
        "world_model_action": "pass",
        "opponent_information_query": _query(),
        "opponent_information_policy": {
            "decision": "stop",
            "proposal_id": "",
            "confidence": 0.8,
            "rationale": "The bounded information budget is exhausted.",
        },
        "opponent_information_adaptation": {
            "prior_decision_id": "A:10:0",
            "adaptation_kind": "change_query_feature",
            "confidence": 0.8,
            "rationale": "Respond to the prior scored query.",
        },
    })
    assert plan["opponent_information_query"]["purpose"] == (
        "resolve_action_choice"
    )
    assert plan["opponent_information_adaptation"]["adaptation_kind"] == (
        "change_query_feature"
    )
    assert plan["opponent_information_policy"]["decision"] == "stop"


def test_world_model_computes_all_queries_and_independent_voi():
    audit = evaluate_llm_opponent_information_query(
        _packet(), _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )

    assert audit["accepted"]
    assert opponent_information_query_audit_is_valid(audit)
    assert len(audit["query_leaderboard"]) == 4
    assert {row["feature"] for row in audit["query_leaderboard"]} == {
        "pressing_intensity", "risk_budget", "line_height",
        "rotation_aggressiveness",
    }
    selected = audit["selected_query_report"]
    assert 0.0 <= selected["forecast_high_rate"] <= 1.0
    assert selected["expected_normalized_information_gain"] >= 0.0
    assert selected["expected_decision_value_of_information"] >= 0.0
    assert sum(selected["posterior_if_high"].values()) == pytest.approx(1.0)
    assert sum(selected["posterior_if_low"].values()) == pytest.approx(1.0)
    assert audit["model_ranking_objective"] == (
        "decision_value_then_information_gain"
    )
    assert 0.0 <= audit["selected_query_objective_efficiency"] <= 1.0
    assert audit["model_evidence"]["posterior_source"] == (
        "selected_action_response_posterior"
    )
    assert not audit["can_change_current_action"]
    assert not audit["can_schedule_future_action"]
    contingent = audit["contingent_policy"]
    assert contingent["expected_policy_regret"] >= 0.0
    assert contingent["worst_branch_regret"] >= 0.0
    assert contingent["regret_thresholds_model_owned"]
    assert contingent["expected_regret_identity_error"] <= 1e-12

    uncertainty = evaluate_llm_opponent_information_query(
        _packet(), _query(purpose="reduce_opponent_uncertainty"),
        selected_action="pass",
    )
    assert uncertainty["model_ranking_objective"] == (
        "information_gain_then_decision_value"
    )

    tampered = copy.deepcopy(audit)
    tampered["selected_query_report"]["forecast_high_rate"] = 0.99
    assert not opponent_information_query_audit_is_valid(tampered)
    tampered_contract = copy.deepcopy(audit)
    tampered_contract["model_evidence"]["observation_sigma"] = 0.01
    assert not opponent_information_query_audit_is_valid(tampered_contract)
    tampered_policy = copy.deepcopy(audit)
    tampered_policy["contingent_policy"]["expected_policy_regret"] = 0.0
    assert not opponent_information_query_audit_is_valid(tampered_policy)


def test_world_model_proposes_exact_questions_for_post_action_llm_selection():
    packet = _packet_with_question_menu()
    menu = packet["opponent_information_question_menu"]
    assert opponent_information_question_menu_is_valid(menu, packet)
    assert menu["proposal_count"] == 16
    assert all(row["shadow_only"] for row in menu["proposals"])
    assert all(not row["can_change_current_action"] for row in menu[
        "proposals"
    ])

    query = _proposed_query(packet)
    audit = evaluate_llm_opponent_information_query(
        packet, query, selected_action="pass",
        query_signature="llm-opponent-information-query:test",
        selected_after_action_freeze=True,
    )
    assert audit["accepted"]
    assert opponent_information_query_audit_is_valid(audit)
    selection = audit["question_selection"]
    assert selection["selection_source"] == "world_model_question_menu"
    assert selection["proposal_fields_exact"]
    assert selection["llm_selected_after_action_freeze"]

    legacy_phase = evaluate_llm_opponent_information_query(
        packet, query, selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    assert legacy_phase["accepted"]
    assert opponent_information_query_audit_is_valid(legacy_phase)
    assert not legacy_phase["question_selection"][
        "llm_selected_after_action_freeze"
    ]

    tampered = dict(query)
    tampered["feature"] = next(
        feature for feature in (
            "pressing_intensity", "risk_budget", "line_height",
            "rotation_aggressiveness",
        ) if feature != query["feature"]
    )
    rejected = evaluate_llm_opponent_information_query(
        packet, tampered, selected_action="pass",
    )
    assert not rejected["accepted"]
    assert rejected["reason"] == "question_proposal_fields_must_be_exact"


def test_finite_horizon_policy_makes_ask_and_stop_explicit_after_freeze():
    packet = _packet_with_cognitive_policy()
    policy = packet["opponent_information_cognitive_policy"]
    assert opponent_information_cognitive_policy_is_valid(policy, packet)
    recommendation = policy["recommendations_by_action"]["pass"]
    assert recommendation["recommended_decision"] == "ask"
    assert recommendation["round_index"] == 1

    audit = evaluate_llm_opponent_information_policy(
        packet,
        {
            "decision": "ask",
            "proposal_id": recommendation["recommended_proposal_id"],
            "confidence": 0.85,
            "rationale": "Acquire the highest marginal information value.",
        },
        selected_action="pass",
        query_signature="llm-opponent-information-query:test",
        selected_after_action_freeze=True,
    )
    assert audit["accepted"]
    assert audit["model_checked_consistent"]
    assert opponent_information_policy_audit_is_valid(audit)
    assert opponent_information_query_audit_is_valid(audit["query_audit"])

    unsafe_phase = evaluate_llm_opponent_information_policy(
        packet,
        {
            "decision": "ask",
            "proposal_id": recommendation["recommended_proposal_id"],
            "confidence": 0.85,
            "rationale": "This was selected before action freeze.",
        },
        selected_action="pass",
        query_signature="llm-opponent-information-query:test",
        selected_after_action_freeze=False,
    )
    assert not unsafe_phase["accepted"]
    assert unsafe_phase["reason"] == (
        "post_action_valid_policy_evidence_required"
    )


def test_resolved_answers_drive_second_question_then_budgeted_stop():
    observed = {
        "pressing_intensity": 0.8,
        "risk_budget": 0.4,
        "line_height": 0.7,
        "rotation_aggressiveness": 0.3,
    }

    def issue(packet, *, issued, observed_t, decision_id):
        recommendation = packet[
            "opponent_information_cognitive_policy"
        ]["recommendations_by_action"]["pass"]
        audit = evaluate_llm_opponent_information_policy(
            packet,
            {
                "decision": "ask",
                "proposal_id": recommendation["recommended_proposal_id"],
                "confidence": 0.85,
                "rationale": "Follow the finite-horizon acquisition policy.",
            },
            selected_action="pass",
            query_signature="llm-opponent-information-query:test",
            selected_after_action_freeze=True,
        )
        query_audit = audit["query_audit"]
        score = score_opponent_information_query(
            query_audit,
            observed,
            horizon=query_audit["query"]["horizon"],
            checkpoint_signature="checkpoint:test",
            environment_signature="environment:test",
        )
        return {
            "decision_id": decision_id,
            "team_id": "A",
            "created_t_sec": issued,
            "checkpoint_signature": "checkpoint:test",
            "environment_signature": "environment:test",
            "intervention_actual_action": "pass",
            "llm_opponent_information_query_context": query_audit,
            "llm_opponent_information_policy_context": audit,
            "multi_horizon_regime_outcomes": {
                query_audit["query"]["horizon"]: {
                    "observed_t_sec": observed_t,
                    "llm_opponent_information_query_evaluation": score,
                },
            },
        }

    first_packet = _packet_with_cognitive_policy()
    first_record = issue(
        first_packet, issued=10.0, observed_t=70.0, decision_id="A:10:0",
    )
    first_feedback = build_opponent_information_feedback(
        [first_record], team_id="A",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test", as_of_t_sec=80.0,
    )
    first_resolution = first_feedback["recent_resolutions"][0]
    assert first_resolution["question_policy"]["round_index"] == 1

    second_packet = _packet_with_cognitive_policy(first_feedback)
    second_recommendation = second_packet[
        "opponent_information_cognitive_policy"
    ]["recommendations_by_action"]["pass"]
    assert second_recommendation["round_index"] == 2
    assert second_recommendation["recommended_decision"] == "ask"
    second_proposal = next(
        row for row in second_packet["opponent_information_question_menu"][
            "proposals"
        ] if row["proposal_id"]
        == second_recommendation["recommended_proposal_id"]
    )
    assert second_proposal["feature"] != first_resolution["feature"]

    second_record = issue(
        second_packet, issued=80.0, observed_t=140.0,
        decision_id="A:80:1",
    )
    second_feedback = build_opponent_information_feedback(
        [first_record, second_record], team_id="A",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test", as_of_t_sec=150.0,
    )
    third_packet = _packet_with_cognitive_policy(second_feedback)
    stop = third_packet["opponent_information_cognitive_policy"][
        "recommendations_by_action"
    ]["pass"]
    assert stop["round_index"] == 3
    assert stop["recommended_decision"] == "stop"
    assert stop["stop_reason"] == "episode_question_budget_exhausted"

    stop_audit = evaluate_llm_opponent_information_policy(
        third_packet,
        {
            "decision": "stop", "proposal_id": "", "confidence": 0.9,
            "rationale": "The two-question evidence budget is exhausted.",
        },
        selected_action="pass",
        query_signature="llm-opponent-information-query:test",
        selected_after_action_freeze=True,
    )
    assert stop_audit["accepted"]
    assert stop_audit["model_checked_consistent"]
    assert not stop_audit["query_audit"]
    assert opponent_information_policy_audit_is_valid(stop_audit)

    stop_record = {
        "decision_id": "A:150:2",
        "team_id": "A",
        "created_t_sec": 150.0,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "intervention_actual_action": "pass",
        "llm_opponent_information_policy_context": stop_audit,
        "multi_horizon_regime_outcomes": {},
    }
    logs = [
        {"world_model_decision_adoption": {"records": copy.deepcopy([
            first_record, second_record, stop_record,
        ])}}
        for _ in range(4)
    ]
    diagnostics = opponent_information_policy_diagnostics([
        payload["world_model_decision_adoption"]["records"]
        for payload in logs
    ])
    assert diagnostics["accepted_policy_audits"] == 12
    assert diagnostics["multi_round_episodes"] == 4
    assert diagnostics["completed_stop_episodes"] == 4
    assert diagnostics["realized_policy_answers"] == 8
    assert diagnostics["question_budget_violations"] == 0
    assert diagnostics["redundant_second_questions"] == 0
    assert diagnostics["all_policy_decisions_post_action_shadow_only"]

    report = aggregate_online_calibration(
        logs,
        min_transitions=0,
        require_multi_round_question_policy=True,
    )
    assert report["version"] == 45
    assert report["multi_round_question_policy_ready"]
    assert report["gates"]["multi_round_question_policy"]


def test_query_fails_closed_for_action_horizon_and_incomplete_values():
    action = evaluate_llm_opponent_information_query(
        _packet(), _query(selected_action="shot"), selected_action="pass",
    )
    assert action["reason"] == "information_query_action_must_equal_plan"
    horizon = evaluate_llm_opponent_information_query(
        _packet(), _query(horizon="180s"), selected_action="pass",
    )
    assert horizon["reason"] == "query_horizon_not_evaluated"
    incomplete = _packet()
    incomplete["candidates"][0]["opponent_hypothesis_values"].pop(
        OPPONENT_HYPOTHESES[0]
    )
    closed = evaluate_llm_opponent_information_query(
        incomplete, _query(), selected_action="pass",
    )
    assert closed["reason"] == "complete_hypothesis_values_required"


def test_realized_tactical_observation_scores_frozen_query_forecasts():
    audit = evaluate_llm_opponent_information_query(
        _packet(), _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    observed = {
        "pressing_intensity": 0.8,
        "risk_budget": 0.4,
        "line_height": 0.7,
        "rotation_aggressiveness": 0.3,
    }
    score = score_opponent_information_query(
        audit, observed, horizon="60s",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )

    assert opponent_information_query_score_is_valid(score, audit)
    assert score["observed_high_events"]["line_height"]
    assert 0.0 <= score["selected_feature_brier_score"] <= 1.0
    assert len(score["feature_brier_scores"]) == 4
    assert score["resolved_observation_branch"] in {"high", "low"}
    assert score["proposed_continuation_action"] in {
        "hold", "pass", "cross", "shot",
    }
    assert score["observed_branch_policy_regret"] >= 0.0
    assert not score["hidden_opponent_intent_observed"]

    tampered = copy.deepcopy(score)
    tampered["selected_feature_brier_score"] = 0.0
    assert not opponent_information_query_score_is_valid(tampered, audit)


def test_cross_match_query_diagnostics_and_strict_readiness_gate():
    packet = _packet_with_question_menu()
    audit = evaluate_llm_opponent_information_query(
        packet, _proposed_query(packet),
        selected_action="pass",
        query_signature="llm-opponent-information-query:test",
        selected_after_action_freeze=True,
    )
    observed = {
        row["feature"]: (0.8 if row["forecast_high_rate"] >= 0.5 else 0.2)
        for row in audit["query_leaderboard"]
    }
    logs = []
    for _ in range(4):
        score = score_opponent_information_query(
            audit, observed, horizon="60s",
            checkpoint_signature="checkpoint:test",
            environment_signature="environment:test",
        )
        logs.append({"world_model_decision_adoption": {"records": [{
            "intervention_actual_action": "pass",
            "llm_opponent_information_query_context": copy.deepcopy(audit),
            "multi_horizon_regime_outcomes": {"60s": {
                "llm_opponent_information_query_evaluation": score,
            }},
        }]}})

    diagnostics = opponent_information_query_diagnostics(logs)
    assert diagnostics["realized_query_scores"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics["match_clustered_model_top_query_rate"] == 1.0
    assert diagnostics["match_clustered_query_objective_efficiency"] == 1.0
    assert diagnostics["match_clustered_supported_query_purpose_rate"] == 1.0
    assert diagnostics[
        "match_clustered_contingent_policy_consistency_rate"
    ] == 1.0
    assert diagnostics["match_clustered_observed_branch_policy_regret"] == 0.0
    assert diagnostics[
        "match_clustered_world_model_question_selection_rate"
    ] == 1.0
    assert diagnostics[
        "match_clustered_post_action_question_selection_rate"
    ] == 1.0
    assert diagnostics["match_clustered_bayesian_answer_feedback_rate"] == 1.0
    assert diagnostics["all_answers_prevent_live_belief_double_counting"]
    assert diagnostics["provenance_compatible"]

    report = aggregate_online_calibration(
        logs, min_transitions=0,
        require_opponent_information_queries=True,
        require_world_model_question_loop=True,
    )
    assert report["version"] == 45
    assert report["opponent_information_queries_ready"]
    assert report["gates"]["opponent_information_queries"]
    assert report["world_model_question_loop_ready"]
    assert report["gates"]["world_model_question_loop"]
    assert report["decision_adoption"][
        "opponent_information_feedback"
    ]["feedback_contexts"] == 0

    tampered = copy.deepcopy(logs)
    tampered[0]["world_model_decision_adoption"]["records"][0][
        "multi_horizon_regime_outcomes"
    ]["60s"]["llm_opponent_information_query_evaluation"][
        "selected_feature_brier_score"
    ] += 0.1
    malformed = opponent_information_query_diagnostics(tampered)
    assert malformed["malformed_query_scores"] == 1
    rejected = aggregate_online_calibration(
        tampered, min_transitions=0,
        require_opponent_information_queries=True,
    )
    assert not rejected["opponent_information_queries_ready"]
    assert not rejected["gates"]["opponent_information_queries"]


def test_due_horizon_observes_opponent_controls_and_scores_query():
    audit = evaluate_llm_opponent_information_query(
        _packet(), _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    home = SimpleNamespace(
        team_id="A", score=0,
        coach=SimpleNamespace(
            tactical_current={}, tactical_base={"line_height": 0.5},
        ),
    )
    away_controls = {
        "pressing_intensity": 0.8,
        "risk_budget": 0.4,
        "line_height": 0.7,
        "rotation_aggressiveness": 0.3,
    }
    away = SimpleNamespace(
        team_id="B", score=0,
        coach=SimpleNamespace(
            tactical_current=away_controls, tactical_base=away_controls,
        ),
    )
    state = SimpleNamespace(
        clock_seconds=10.0, home=home, away=away,
        ball=SimpleNamespace(position=[0.5, 0.5], possession_team_id="A"),
        micro_xg_home=0.2, micro_xg_away=0.1,
    )
    record = register_coach_action_decision(
        state, team_id="A", trigger_kind="clock",
        llm_selected_action="pass", world_model_recommended_action="pass",
        recommendation_confidence=0.8, horizon_s=10.0,
        outcome_horizons_s=(60.0,), checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
        llm_opponent_information_query_context=audit,
    )
    baseline = capture_policy_outcome_baseline(state, team_id="A")
    record_policy_intervention_result(
        state, decision_id=record["decision_id"], actual_action="pass",
        t_sec=11.0, outcome_baseline=baseline,
    )
    observe_policy_intervention_outcomes(state, t_sec=71.0)

    score = record["multi_horizon_regime_outcomes"]["60s"][
        "llm_opponent_information_query_evaluation"
    ]
    assert opponent_information_query_score_is_valid(score, audit)
    assert score["observed_features"] == away_controls
    diagnostics = decision_adoption_diagnostics(state)
    assert diagnostics["opponent_information_queries"][
        "realized_query_scores"
    ] == 1


def test_held_out_query_calibration_changes_likelihoods_coherently():
    packet = _packet()
    audit = evaluate_llm_opponent_information_query(
        packet, _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    observed = {feature: 0.8 for feature in (
        "pressing_intensity", "risk_budget", "line_height",
        "rotation_aggressiveness",
    )}
    logs = []
    for _ in range(16):
        score = score_opponent_information_query(
            audit, observed, horizon="60s",
            checkpoint_signature="checkpoint:test",
            environment_signature="environment:test",
        )
        logs.append({"world_model_decision_adoption": {"records": [{
            "intervention_actual_action": "pass",
            "llm_opponent_information_query_context": copy.deepcopy(audit),
            "multi_horizon_regime_outcomes": {"60s": {
                "llm_opponent_information_query_evaluation": score,
            }},
        }]}})

    memory = compile_opponent_information_calibration_memory(
        logs, checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    correction = memory.lookup(horizon="60s", feature="line_height")
    assert correction["available"]
    assert correction["reference_samples"] == 8
    assert correction["validation_samples"] == 8
    assert correction["validation_skill_vs_raw"] > 0.0

    repeated_inside_one_match = [{"world_model_decision_adoption": {
        "records": [
            copy.deepcopy(payload["world_model_decision_adoption"]["records"][0])
            for payload in logs
        ],
    }}]
    clustered = compile_opponent_information_calibration_memory(
        repeated_inside_one_match,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    assert clustered.rows == 4
    assert clustered.summary()["active_groups"] == 0

    calibrated_packet = copy.deepcopy(packet)
    calibrated_packet["opponent_information_query_calibration"] = {
        "version": 1,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "horizons": {"60s": memory.contract(horizon="60s")},
    }
    calibrated = evaluate_llm_opponent_information_query(
        calibrated_packet, _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    assert opponent_information_query_audit_is_valid(calibrated)
    report = calibrated["selected_query_report"]
    assert report["calibration_applied"]
    assert report["forecast_high_rate"] > report["raw_forecast_high_rate"]
    assert sum(report["posterior_if_high"].values()) == pytest.approx(1.0)
    assert sum(report["posterior_if_low"].values()) == pytest.approx(1.0)
    assert report["hypothesis_high_likelihoods"] != (
        report["raw_hypothesis_high_likelihoods"]
    )

    score = score_opponent_information_query(
        calibrated, observed, horizon="60s",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    assert score["any_query_calibration_applied"]
    assert score["all_feature_mean_brier_score"] < (
        score["raw_all_feature_mean_brier_score"]
    )


def test_query_calibration_rejects_tampered_or_wrong_scope_history():
    audit = evaluate_llm_opponent_information_query(
        _packet(), _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    observed = {feature: 0.8 for feature in (
        "pressing_intensity", "risk_budget", "line_height",
        "rotation_aggressiveness",
    )}
    score = score_opponent_information_query(
        audit, observed, horizon="60s",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    tampered = copy.deepcopy(score)
    tampered["raw_selected_feature_brier_score"] = 0.0
    logs = [{"world_model_decision_adoption": {"records": [{
        "llm_opponent_information_query_context": copy.deepcopy(audit),
        "multi_horizon_regime_outcomes": {"60s": {
            "llm_opponent_information_query_evaluation": (
                tampered if index == 0 else copy.deepcopy(score)
            ),
        }},
    }]}} for index in range(16)]
    memory = compile_opponent_information_calibration_memory(
        logs, checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    assert not memory.lookup(horizon="60s", feature="line_height")[
        "available"
    ]
    wrong_scope = compile_opponent_information_calibration_memory(
        logs, checkpoint_signature="checkpoint:other",
        environment_signature="environment:test",
    )
    assert wrong_scope.rows == 0


def test_resolved_query_becomes_safe_feedback_for_the_next_llm_turn():
    audit = evaluate_llm_opponent_information_query(
        _packet(), _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    observed = {
        "pressing_intensity": 0.8,
        "risk_budget": 0.4,
        "line_height": 0.7,
        "rotation_aggressiveness": 0.3,
    }
    score = score_opponent_information_query(
        audit, observed, horizon="60s",
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
        "llm_opponent_information_query_context": audit,
        "multi_horizon_regime_outcomes": {"60s": {
            "observed_t_sec": 70.0,
            "llm_opponent_information_query_evaluation": score,
        }},
    }
    feedback = build_opponent_information_feedback(
        [record], team_id="A", checkpoint_signature="checkpoint:test",
        environment_signature="environment:test", as_of_t_sec=80.0,
    )

    assert feedback["available"]
    assert opponent_information_feedback_is_valid(feedback)
    resolution = feedback["recent_resolutions"][0]
    assert resolution["observed_feature_value"] == 0.7
    assert resolution["observed_high"]
    assert resolution["signed_probability_surprise"] == pytest.approx(
        1.0 - resolution["forecast_high_rate"]
    )
    assert not resolution["branch_action_executed"]
    assert not resolution["counterfactual_outcome_observed"]
    assert not resolution["hidden_opponent_intent_observed"]
    answer = resolution["resolved_answer"]
    assert answer["model_owned_bayesian_update"]
    assert answer["feeds_next_question"]
    assert not answer["can_directly_mutate_live_belief"]
    assert answer["live_observation_already_assimilated"]
    assert sum(answer["posterior_after_answer"].values()) == pytest.approx(1.0)
    assert feedback["feature_profiles"]["line_height"][
        "resolved_queries"
    ] == 1
    premature = build_opponent_information_feedback(
        [record], team_id="A", checkpoint_signature="checkpoint:test",
        environment_signature="environment:test", as_of_t_sec=60.0,
    )
    assert not premature["available"]
    assert premature["future_dated_queries_excluded"] == 1
    next_decision = {
        "team_id": "A",
        "created_t_sec": 80.0,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "opponent_information_feedback_context": feedback,
    }
    diagnostics = opponent_information_feedback_diagnostics([
        [next_decision]
    ])
    assert diagnostics["available_feedback_contexts"] == 1
    assert diagnostics["resolution_exposures"] == 1
    assert diagnostics["bayesian_answer_exposures"] == 1
    assert diagnostics["all_feedback_temporally_prospective"]
    leaked = copy.deepcopy(next_decision)
    leaked["created_t_sec"] = 60.0
    leak_diagnostics = opponent_information_feedback_diagnostics([[leaked]])
    assert leak_diagnostics["future_information_leaks"] == 1
    assert not leak_diagnostics["all_feedback_temporally_prospective"]

    tampered_feedback = copy.deepcopy(feedback)
    tampered_feedback["recent_resolutions"][0][
        "observed_feature_value"
    ] = 0.1
    assert not opponent_information_feedback_is_valid(tampered_feedback)

    tampered_record = copy.deepcopy(record)
    tampered_record["multi_horizon_regime_outcomes"]["60s"][
        "llm_opponent_information_query_evaluation"
    ]["selected_feature_brier_score"] = 0.0
    rejected = build_opponent_information_feedback(
        [tampered_record], team_id="A",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test", as_of_t_sec=80.0,
    )
    assert not rejected["available"]
    assert rejected["malformed_queries"] == 1


def test_query_feedback_separates_pending_ineligible_and_wrong_scope():
    audit = evaluate_llm_opponent_information_query(
        _packet(), _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    base = {
        "team_id": "A",
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "llm_opponent_information_query_context": audit,
        "multi_horizon_regime_outcomes": {},
    }
    pending = {**base, "intervention_actual_action": None}
    ineligible = {**base, "intervention_actual_action": "shot"}
    feedback = build_opponent_information_feedback(
        [pending, ineligible], team_id="A",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test", as_of_t_sec=80.0,
    )
    assert feedback["pending_queries"] == 1
    assert feedback["ineligible_action_queries"] == 1
    assert feedback["malformed_queries"] == 0


def test_llm_feedback_adaptation_is_model_checked_then_realized():
    seed = evaluate_llm_opponent_information_query(
        _packet(), _query(), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    prior_feature = seed["model_recommended_feature"]
    prior_audit = evaluate_llm_opponent_information_query(
        _packet(), _query(feature=prior_feature), selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    prior_rate = prior_audit["selected_query_report"]["forecast_high_rate"]
    prior_observed = {
        "pressing_intensity": 0.5,
        "risk_budget": 0.5,
        "line_height": 0.5,
        "rotation_aggressiveness": 0.5,
    }
    prior_observed[prior_feature] = 0.2 if prior_rate >= 0.5 else 0.8
    prior_score = score_opponent_information_query(
        prior_audit, prior_observed, horizon="60s",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    prior_record = {
        "decision_id": "A:10:0",
        "team_id": "A",
        "created_t_sec": 10.0,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "intervention_actual_action": "pass",
        "llm_opponent_information_query_context": prior_audit,
        "multi_horizon_regime_outcomes": {"60s": {
            "observed_t_sec": 70.0,
            "llm_opponent_information_query_evaluation": prior_score,
        }},
    }
    feedback = build_opponent_information_feedback(
        [prior_record], team_id="A",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test", as_of_t_sec=80.0,
    )
    assert "forecast_surprise" in feedback["recent_resolutions"][0][
        "attention_flags"
    ]

    current_feature = next(
        feature for feature in prior_observed if feature != prior_feature
    )
    current_audit = evaluate_llm_opponent_information_query(
        _packet(), _query(feature=current_feature),
        selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    raw_adaptation = {
        "prior_decision_id": "A:10:0",
        "adaptation_kind": "change_query_feature",
        "confidence": 0.8,
        "rationale": "The previous feature forecast was surprising.",
    }
    assert validate_llm_opponent_information_adaptation(raw_adaptation)
    packet = _packet()
    packet["opponent_information_feedback"] = feedback
    adaptation = evaluate_llm_opponent_information_adaptation(
        packet, raw_adaptation, current_audit,
        adaptation_signature="llm-opponent-information-adaptation:test",
    )
    assert adaptation["accepted"]
    assert adaptation["evidence_supported"]
    assert adaptation["structural_change_verified"]
    assert adaptation["model_checked_consistent"]
    assert opponent_information_adaptation_audit_is_valid(
        adaptation, current_audit, feedback,
    )
    adoption_state = SimpleNamespace(clock_seconds=80.0)
    registered = register_coach_action_decision(
        adoption_state, team_id="A", trigger_kind="xg_swing",
        llm_selected_action="pass", world_model_recommended_action="pass",
        recommendation_confidence=0.8, horizon_s=10.0,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
        llm_opponent_information_query_context=current_audit,
        opponent_information_feedback_context=feedback,
        llm_opponent_information_adaptation_context=adaptation,
    )
    assert registered[
        "llm_opponent_information_adaptation_context"
    ] == adaptation

    current_rate = current_audit["selected_query_report"][
        "forecast_high_rate"
    ]
    current_observed = {
        "pressing_intensity": 0.5,
        "risk_budget": 0.5,
        "line_height": 0.5,
        "rotation_aggressiveness": 0.5,
    }
    current_observed[current_feature] = 0.8 if current_rate >= 0.5 else 0.2
    current_score = score_opponent_information_query(
        current_audit, current_observed, horizon="60s",
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
    )
    score = score_opponent_information_adaptation(
        adaptation, current_audit, feedback, current_score,
    )
    assert opponent_information_adaptation_score_is_valid(
        score, adaptation, current_audit, feedback, current_score,
    )
    assert score["adaptation_target_metric"] == (
        "selected_feature_brier_reduction"
    )
    assert score["adaptation_target_improvement"] > 0.0
    assert score["adaptation_improved"]
    assert not score["counterfactual_outcome_observed"]

    current_record = {
        "team_id": "A",
        "created_t_sec": 80.0,
        "checkpoint_signature": "checkpoint:test",
        "environment_signature": "environment:test",
        "intervention_actual_action": "pass",
        "llm_opponent_information_query_context": current_audit,
        "opponent_information_feedback_context": feedback,
        "llm_opponent_information_adaptation_context": adaptation,
        "multi_horizon_regime_outcomes": {"60s": {
            "llm_opponent_information_query_evaluation": current_score,
            "llm_opponent_information_adaptation_evaluation": score,
        }},
    }
    diagnostics = opponent_information_adaptation_diagnostics([{
        "world_model_decision_adoption": {"records": [current_record]},
    }])
    assert diagnostics["realized_adaptation_scores"] == 1
    assert diagnostics[
        "match_clustered_adaptation_improvement_rate"
    ] == 1.0
    assert diagnostics["match_clustered_feedback_adaptation_rate"] == 1.0
    assert diagnostics["provenance_compatible"]

    adaptation_logs = [{
        "world_model_decision_adoption": {
            "records": [copy.deepcopy(current_record)],
        },
    } for _ in range(4)]
    report = aggregate_online_calibration(
        adaptation_logs, min_transitions=0,
        require_opponent_information_adaptation=True,
    )
    assert report["opponent_information_adaptation_ready"]
    assert report["gates"]["opponent_information_adaptation"]

    malformed_logs = copy.deepcopy(adaptation_logs)
    malformed_logs[0]["world_model_decision_adoption"]["records"][0][
        "multi_horizon_regime_outcomes"
    ]["60s"]["llm_opponent_information_adaptation_evaluation"][
        "adaptation_target_improvement"
    ] -= 0.1
    rejected = aggregate_online_calibration(
        malformed_logs, min_transitions=0,
        require_opponent_information_adaptation=True,
    )
    assert not rejected["opponent_information_adaptation_ready"]
    assert not rejected["gates"]["opponent_information_adaptation"]

    tampered = copy.deepcopy(adaptation)
    tampered["model_checked_consistent"] = False
    assert not opponent_information_adaptation_audit_is_valid(
        tampered, current_audit, feedback,
    )
    tampered_state = SimpleNamespace(clock_seconds=80.0)
    sanitized = register_coach_action_decision(
        tampered_state, team_id="A", trigger_kind="xg_swing",
        llm_selected_action="pass", world_model_recommended_action="pass",
        recommendation_confidence=0.8, horizon_s=10.0,
        checkpoint_signature="checkpoint:test",
        environment_signature="environment:test",
        llm_opponent_information_query_context=current_audit,
        opponent_information_feedback_context=feedback,
        llm_opponent_information_adaptation_context=tampered,
    )
    assert sanitized["llm_opponent_information_adaptation_context"] == {}


def test_feedback_adaptation_rejects_unsupported_or_unavailable_claims():
    current = evaluate_llm_opponent_information_query(
        _packet(), _query(feature="pressing_intensity"),
        selected_action="pass",
    )
    unavailable = evaluate_llm_opponent_information_adaptation(
        _packet(), {
            "prior_decision_id": "missing",
            "adaptation_kind": "change_query_feature",
            "confidence": 0.8,
        }, current,
    )
    assert not unavailable["accepted"]
    assert unavailable["reason"] == "compatible_feedback_and_query_required"
    assert validate_llm_opponent_information_adaptation({
        "prior_decision_id": "x",
        "adaptation_kind": "rewrite_world_model_probability",
        "confidence": 0.8,
    }) is None
