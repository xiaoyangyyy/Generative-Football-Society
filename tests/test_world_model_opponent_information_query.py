"""LLM tactical questions are converted into model-owned value of information."""

import copy
from types import SimpleNamespace

import pytest

from src.match_engine.cognitive.schemas import validate_coach_plan
from src.match_engine.world_model.opponent_contract import OPPONENT_HYPOTHESES
from src.match_engine.world_model.opponent_information_query import (
    evaluate_llm_opponent_information_query,
    opponent_information_query_audit_is_valid,
    validate_llm_opponent_information_query,
)
from src.match_engine.world_model.opponent_information_query_outcomes import (
    opponent_information_query_score_is_valid,
    score_opponent_information_query,
)
from src.match_engine.world_model.opponent_information_query_evaluation import (
    opponent_information_query_diagnostics,
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
    })
    assert plan["opponent_information_query"]["purpose"] == (
        "resolve_action_choice"
    )


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
    seed = evaluate_llm_opponent_information_query(
        _packet(), _query(purpose="reduce_opponent_uncertainty"),
        selected_action="pass",
        query_signature="llm-opponent-information-query:test",
    )
    audit = evaluate_llm_opponent_information_query(
        _packet(),
        _query(
            feature=seed["model_recommended_feature"],
            purpose="reduce_opponent_uncertainty",
            action_if_high=seed["query_leaderboard"][0][
                "best_action_if_high"
            ],
            action_if_low=seed["query_leaderboard"][0][
                "best_action_if_low"
            ],
        ),
        selected_action="pass",
        query_signature="llm-opponent-information-query:test",
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
    assert diagnostics["provenance_compatible"]

    report = aggregate_online_calibration(
        logs, min_transitions=0,
        require_opponent_information_queries=True,
    )
    assert report["version"] == 26
    assert report["opponent_information_queries_ready"]
    assert report["gates"]["opponent_information_queries"]

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
