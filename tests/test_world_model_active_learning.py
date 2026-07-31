"""Active learning is informative, budgeted, and subordinate to safety."""

import pytest

from src.match_engine.world_model.active_learning import (
    active_learning_diagnostics,
    build_active_learning_advice,
)


def _candidate(action, value, uncertainty, turnover):
    return {
        "action": action,
        "risk_adjusted_value": value,
        "effective_confidence": 0.7,
        "uncertainty": uncertainty,
        "event_probabilities": {"turnover": turnover},
        "multi_horizon_predictions": {
            "transition": {
                "policy_utility": value,
                "uncertainty": uncertainty,
            },
            "60s": {
                "policy_utility": value + 0.05,
                "uncertainty": uncertainty,
            },
        },
    }


def _context():
    return {
        "team_id": "Home",
        "opponent_team_id": "Away",
        "zone": "middle",
        "score_state": "level",
        "match_phase": "early",
    }


def test_selects_high_information_action_only_inside_low_regret_safe_set():
    candidates = [
        _candidate("hold", 0.50, 0.10, 0.10),
        _candidate("pass", 0.46, 0.80, 0.18),
        _candidate("shot", 0.20, 1.00, 0.60),
    ]
    advice = build_active_learning_advice(
        candidates, [], context=_context(),
    )

    assert advice["eligible"]
    assert advice["exploit_action"] == "hold"
    assert advice["exploration_action"] == "pass"
    assert advice["estimated_regret"] == pytest.approx(0.04)
    shot = next(
        item for item in advice["candidate_evidence"]
        if item["action"] == "shot"
    )
    assert not shot["safe_to_explore"]
    assert candidates[1]["active_learning"]["information_value"] > (
        candidates[0]["active_learning"]["information_value"]
    )


def test_exploration_budget_blocks_an_otherwise_safe_opportunity():
    records = [
        {"active_learning": {"decision_mode": mode}}
        for mode in ("explore", "explore", "exploit", "exploit")
    ]
    advice = build_active_learning_advice(
        [
            _candidate("hold", 0.50, 0.10, 0.10),
            _candidate("pass", 0.46, 0.80, 0.18),
        ],
        records,
        context=_context(),
        config={"budget_fraction": 0.25},
    )

    assert not advice["eligible"]
    assert advice["reason"] == "exploration_budget_exhausted"
    assert advice["budget"]["exploration_decisions"] == 2


def test_compatible_cross_match_memory_reduces_false_novelty():
    uncovered = _candidate("pass", 0.46, 0.5, 0.18)
    covered = _candidate("pass", 0.46, 0.5, 0.18)
    for prediction in covered["multi_horizon_predictions"].values():
        prediction["residual_memory"] = {
            "scope": "action_context", "samples": 24,
        }
    uncovered_advice = build_active_learning_advice(
        [_candidate("hold", 0.50, 0.1, 0.1), uncovered],
        [], context=_context(),
    )
    covered_advice = build_active_learning_advice(
        [_candidate("hold", 0.50, 0.1, 0.1), covered],
        [], context=_context(),
    )
    uncovered_pass = next(
        item for item in uncovered_advice["candidate_evidence"]
        if item["action"] == "pass"
    )
    covered_pass = next(
        item for item in covered_advice["candidate_evidence"]
        if item["action"] == "pass"
    )

    assert covered_pass["action_samples"] == 24
    assert covered_pass["context_action_samples"] == 24
    assert covered_pass["information_value"] < uncovered_pass[
        "information_value"
    ]


def test_diagnostics_measure_later_uncertainty_reduction_in_same_context():
    baseline = {
        "opponent_team_id": "Away",
        "zone": "middle",
        "score_state": "level",
        "match_phase": "early",
    }
    exploration = {
        "team_id": "Home",
        "outcome_baseline": baseline,
        "intervention_actual_action": "pass",
        "active_learning": {
            "decision_mode": "explore", "exploration_action": "pass",
        },
        "action_outcome_predictions": {
            "pass": {"60s": {"uncertainty": 0.8}},
        },
    }
    later = {
        "team_id": "Home",
        "outcome_baseline": baseline,
        "intervention_actual_action": "hold",
        "active_learning": {"decision_mode": "exploit"},
        "action_outcome_predictions": {
            "pass": {"60s": {"uncertainty": 0.4}},
        },
    }

    report = active_learning_diagnostics([[exploration, later]])
    assert report["exploration_decisions"] == 1
    assert report["executed_explorations"] == 1
    assert report["explored_actions"] == ["pass"]
    assert report["uncertainty_reduction_pairs"] == 1
    assert report["mean_later_uncertainty_reduction"] == pytest.approx(0.4)
    assert report["positive_uncertainty_reduction_rate"] == 1.0
