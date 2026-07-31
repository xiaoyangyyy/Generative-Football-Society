"""Outcome-linked evaluation for world-model/LLM tactical fusion."""

from __future__ import annotations

import json

import pytest

from src.simulation.fusion_audit import (
    append_fusion_audits,
    build_outcome_linked_fusion_record,
    evaluate_fusion_audits,
    evaluate_matched_seed_tactical_policy,
    fusion_audit_path,
    load_fusion_audits,
)


def _audit(*, selected="balanced", recommended="balanced", available=True):
    return {
        "evidence_available": available,
        "evidence_reason": "ok" if available else "quality_gate_closed",
        "recommended_tactical_preset": recommended if available else "none",
        "selected_tactical_preset": selected,
        "disagreed_with_recommendation": available and selected != recommended,
        "selection_constrained": False,
        "observation_coverage": 0.9,
        "recommendation_confidence": 0.7,
        "recommendation_margin": 0.05,
        "selected_evidence": {"risk_adjusted_value": 0.3},
        "balanced_evidence": {"risk_adjusted_value": 0.2},
        "rationale": "Opponent shape justifies the bounded selection.",
    }


def _record(index, *, selected="balanced", recommended="balanced", result="win"):
    return build_outcome_linked_fusion_record(
        decision_id=f"group:A:B:{index}",
        team="A",
        opponent="B",
        stage="Group",
        decision_audit=_audit(
            selected=selected, recommended=recommended,
        ),
        outcomes={
            "result": result,
            "goal_diff": 1 if result == "win" else -1,
            "xg_for": 1.4,
            "xg_against": 0.8,
            "fatigue_delta": 0.2,
        },
    )


def test_fusion_audit_round_trip_is_json_safe_and_deduplicated(tmp_path):
    first = _record(1)
    duplicate = _record(1, selected="low_block", result="loss")
    duplicate["decision"]["nonfinite"] = float("nan")
    append_fusion_audits(tmp_path, [first])
    # Records built by the public builder sanitize non-finite values.
    duplicate = build_outcome_linked_fusion_record(
        decision_id=duplicate["decision_id"],
        team="A", opponent="B", stage="Group",
        decision_audit=duplicate["decision"],
        outcomes={"result": "loss", "goal_diff": -1},
    )
    append_fusion_audits(tmp_path, [duplicate])
    loaded = load_fusion_audits(fusion_audit_path(tmp_path))
    assert len(loaded) == 1
    assert loaded[0]["outcome"]["result"] == "loss"
    assert loaded[0]["decision"]["nonfinite"] == 0.0
    json.dumps(loaded, allow_nan=False)


def test_fusion_policy_evaluation_separates_observation_from_causality():
    records = [
        _record(1, result="win"),
        _record(2, result="draw"),
        _record(3, selected="low_block", result="loss"),
        _record(4, selected="low_block", result="win"),
    ]
    report = evaluate_fusion_audits(records, min_records=4)
    assert report["ready_for_observational_comparison"]
    assert report["gate_open_rate"] == pytest.approx(1.0)
    assert report["agreement_rate"] == pytest.approx(0.5)
    assert report["mean_selected_advantage_vs_balanced"] == pytest.approx(0.1)
    assert report["observed_outcomes"]["agreement"]["samples"] == 2
    assert report["observed_outcomes"]["disagreement"]["samples"] == 2
    assert report["causal_interpretation"] is False


def test_matched_seed_policy_evaluation_recovers_simulator_effect():
    def run(seed, preset):
        shared_noise = (seed % 17) * 0.01
        return shared_noise + (0.25 if preset == "gegenpress" else 0.0)

    report = evaluate_matched_seed_tactical_policy(
        run,
        baseline_preset="balanced",
        treatment_preset="gegenpress",
        samples=8,
    )
    assert report["average_treatment_effect"] == pytest.approx(0.25)
    assert report["directionally_supported"]
    assert report["causal_interpretation"] == "within_simulator_only"
