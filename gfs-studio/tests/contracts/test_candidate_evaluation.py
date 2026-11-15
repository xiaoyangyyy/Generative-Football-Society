from types import SimpleNamespace

import pytest

from gfs_studio.contracts.product import CandidateEvaluation
from gfs_studio.plugins.product.studio.candidates import CandidateEvaluator, candidate_record


def record(candidate, seed, utility):
    summary = SimpleNamespace(
        micro_xg_home=utility,
        micro_xg_away=0.0,
        shots_home=0,
        shots_away=0,
        goals_micro_home=0,
        goals_micro_away=0,
        passes_home=0,
        passes_away=0,
    )
    return candidate_record(candidate, summary, {}, seed)


def test_candidate_evaluation_uses_paired_seed_sets_without_causal_claim():
    evaluator = CandidateEvaluator()
    assert isinstance(evaluator, CandidateEvaluation)
    observations = [
        record("preserve_current_vector", 42, 0.2),
        record("preserve_current_vector", 43, 0.4),
        record("attacking_pressure", 42, 0.5),
        record("attacking_pressure", 43, 0.6),
    ]
    result = evaluator.evaluate(observations)
    assert result["status"] == "complete"
    assert result["best_candidate"] == "attacking_pressure"
    assert result["candidates"]["attacking_pressure"]["samples"] == 2
    assert result["paired_comparisons"]["attacking_pressure"]["mean_utility_delta"] == pytest.approx(0.25)
    assert result["causal_claim"] is False
    assert result["outcome_improvement_claim"] is False


def test_candidate_evaluation_rejects_unpaired_or_insufficient_samples():
    evaluator = CandidateEvaluator()
    assert evaluator.evaluate([
        record("preserve_current_vector", 42, 0.2),
    ])["best_candidate"] is None
    result = evaluator.evaluate([
        record("preserve_current_vector", 42, 0.2),
        record("preserve_current_vector", 43, 0.2),
        record("attacking_pressure", 42, 0.5),
        record("attacking_pressure", 44, 0.5),
    ])
    assert result["status"] == "insufficient_paired_samples"
    assert result["best_candidate"] is None


def test_candidate_duplicate_replay_is_idempotent_and_conflict_fails():
    observations = [
        record("preserve_current_vector", 42, 0.2),
        record("preserve_current_vector", 43, 0.2),
        record("attacking_pressure", 42, 0.5),
        record("attacking_pressure", 43, 0.5),
    ]
    result = CandidateEvaluator().evaluate(observations + [observations[-1]])
    assert result["candidates"]["attacking_pressure"]["samples"] == 2
    conflicting = dict(observations[-1])
    conflicting["utility"] = 99.0
    with pytest.raises(ValueError, match="conflicts"):
        CandidateEvaluator().evaluate(observations + [conflicting])
