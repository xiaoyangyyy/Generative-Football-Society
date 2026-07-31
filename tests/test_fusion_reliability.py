"""Historical trust remains conservative and evidence-tier aware."""

from __future__ import annotations

from src.simulation.counterfactual_evidence import (
    append_counterfactual_evidence,
    counterfactual_evidence_path,
    load_counterfactual_evidence,
)
from src.simulation.fusion_reliability import (
    build_fusion_reliability_profile,
    enrich_decision_packet_with_history,
    summarize_team_strategy_memory,
)
from src.simulation.fusion_audit import append_fusion_audits
from src.simulation.tournament_setup import TournamentSetupMixin


def _audit(index, *, team="A", opponent="B", preset="balanced"):
    return {
        "schema_version": 1,
        "decision_id": f"d-{index}",
        "team": team,
        "opponent": opponent,
        "stage": "Group",
        "decision": {
            "evidence_available": True,
            "selected_tactical_preset": preset,
            "disagreed_with_recommendation": False,
        },
        "outcome": {
            "result": "win",
            "points": 3.0,
            "goal_diff": 1.0,
            "xg_diff": 0.4,
            "fatigue_delta": 0.2,
        },
    }


def _counterfactual(*, low=0.05, high=0.3, effect=0.15, samples=8):
    return {
        "team": "A",
        "opponent": "B",
        "baseline_preset": "balanced",
        "treatment_preset": "gegenpress",
        "samples": samples,
        "average_treatment_effect": effect,
        "confidence_interval": [low, high],
        "directionally_supported": low > 0 or high < 0,
        "match_seconds": 900,
        "fast_mode": False,
        "root_seed": 42,
        "causal_interpretation": "within_simulator_only",
    }


def test_counterfactual_evidence_round_trip_deduplicates_configuration(tmp_path):
    report = _counterfactual()
    append_counterfactual_evidence(tmp_path, report)
    report["average_treatment_effect"] = 0.2
    append_counterfactual_evidence(tmp_path, report)
    loaded = load_counterfactual_evidence(
        counterfactual_evidence_path(tmp_path)
    )
    assert len(loaded) == 1
    assert loaded[0]["average_treatment_effect"] == 0.2


def test_strategy_memory_is_team_scoped_and_marked_noncausal():
    records = [
        _audit(1, preset="balanced"),
        _audit(2, opponent="C", preset="gegenpress"),
        _audit(3, team="Other"),
    ]
    memory = summarize_team_strategy_memory(
        records, team="A", opponent="B",
    )
    assert memory["matches"] == 2
    assert memory["opponent_specific_matches"] == 1
    assert memory["by_selected_tactical_preset"]["balanced"]["samples"] == 1
    assert memory["recent_relevant_decisions"][0]["opponent"] == "B"
    assert memory["causal_interpretation"] is False


def test_matched_seed_support_changes_trust_but_cannot_open_gate():
    positive = _counterfactual()
    profile = build_fusion_reliability_profile(
        team="A",
        recommended_preset="gegenpress",
        raw_recommendation_confidence=0.6,
        audit_records=[],
        counterfactual_records=[positive],
        quality_gate_open=True,
    )
    assert profile["evidence_tier"] == "matched_seed_simulator"
    assert profile["guidance"] == "moderate_support"
    assert profile["adjusted_recommendation_trust"] > 0.6
    closed = build_fusion_reliability_profile(
        team="A",
        recommended_preset="gegenpress",
        raw_recommendation_confidence=0.9,
        audit_records=[],
        counterfactual_records=[positive],
        quality_gate_open=False,
    )
    assert closed["adjusted_recommendation_trust"] == 0.0
    assert closed["can_open_quality_gate"] is False


def test_observational_history_never_becomes_causal_support():
    profile = build_fusion_reliability_profile(
        team="A",
        recommended_preset="balanced",
        raw_recommendation_confidence=0.8,
        audit_records=[_audit(index) for index in range(5)],
        counterfactual_records=[],
        quality_gate_open=True,
    )
    assert profile["evidence_tier"] == "observational_only"
    assert profile["guidance"] == "advisory_only"
    assert profile["causal_scope"] == "none"
    assert profile["adjusted_recommendation_trust"] < 0.8


def test_fast_or_short_counterfactual_is_not_promoted_to_trust_evidence():
    exploratory = _counterfactual()
    exploratory["fast_mode"] = True
    profile = build_fusion_reliability_profile(
        team="A",
        recommended_preset="gegenpress",
        raw_recommendation_confidence=0.8,
        audit_records=[],
        counterfactual_records=[exploratory],
        quality_gate_open=True,
    )
    assert profile["evidence_tier"] == "insufficient_history"
    assert profile["matched_seed_samples"] == 0


def test_packet_enrichment_keeps_raw_recommendation_and_adds_history():
    packet = {
        "available": True,
        "recommended_tactical_preset": "gegenpress",
        "recommendation_confidence": 0.6,
        "candidates": [],
    }
    enriched = enrich_decision_packet_with_history(
        packet,
        team="A",
        opponent="B",
        audit_records=[_audit(1)],
        counterfactual_records=[_counterfactual()],
    )
    assert packet.get("strategy_memory") is None
    assert enriched["recommended_tactical_preset"] == "gegenpress"
    assert enriched["strategy_memory"]["matches"] == 1
    assert enriched["fusion_reliability"]["evidence_tier"] == (
        "matched_seed_simulator"
    )


def test_tournament_enrichment_reads_persistent_cross_process_evidence(tmp_path):
    append_fusion_audits(tmp_path, [_audit(1)])
    append_counterfactual_evidence(tmp_path, _counterfactual())
    manager = TournamentSetupMixin()
    manager.base_dir = str(tmp_path)
    packets = {
        "A": {
            "available": True,
            "recommended_tactical_preset": "gegenpress",
            "recommendation_confidence": 0.6,
            "candidates": [],
        },
        "B": {
            "available": False,
            "recommended_tactical_preset": "none",
            "candidates": [],
        },
    }
    enriched = manager._enrich_prematch_packets_with_history(
        packets, t1_name="A", t2_name="B",
    )
    assert enriched["A"]["strategy_memory"]["matches"] == 1
    assert enriched["A"]["fusion_reliability"]["evidence_tier"] == (
        "matched_seed_simulator"
    )
    assert enriched["B"]["fusion_reliability"][
        "adjusted_recommendation_trust"
    ] == 0.0
