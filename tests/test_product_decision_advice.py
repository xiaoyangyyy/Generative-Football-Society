import copy
import hashlib
import json

import pytest

from src.product.decision_advice import (
    build_manager_advice_adoption,
    build_manager_advice_comparison,
    build_manager_decision_advice,
    validate_manager_advice_adoption,
    validate_manager_advice_comparison,
    validate_manager_decision_advice,
)
from src.product.match_plan import PLAYABLE_TACTICS


def _packet():
    candidates = []
    for index, tactic in enumerate(PLAYABLE_TACTICS):
        value = 0.7 - index * 0.05
        candidates.append({
            "tactical_preset": tactic,
            "risk_adjusted_value": value,
            "effective_confidence": 0.72 - index * 0.01,
            "uncertainty": 0.18 + index * 0.01,
            "fatigue_cost_proxy": 0.2 + index * 0.03,
            "structural_risk_proxy": 0.3 + index * 0.02,
            "event_probabilities": {
                "retain": 0.45, "turnover": 0.22, "shot": 0.18,
                "foul": 0.10, "out": 0.05,
            },
        })
    return {
        "available": True,
        "evaluation_scope": "short_horizon_tactical_policy_proxy",
        "horizon_s": 20.0,
        "observation_coverage": 0.93,
        "checkpoint_signature": "sha256:" + "d" * 64,
        "candidates": candidates,
        "fusion_reliability": {
            "evidence_tier": "insufficient_history",
            "guidance": "low_authority",
            "adjusted_recommendation_trust": 0.54,
            "matched_seed_samples": 0,
            "historical_match_records": 2,
            "causal_scope": "none",
        },
    }


def _snapshots():
    return {
        "Brazil": {
            "source": "persisted_carryover", "source_identity": "a" * 64,
        },
        "Argentina": {
            "source": "deterministic_roster_baseline",
            "source_identity": "b" * 64,
        },
    }


def _advice():
    return build_manager_decision_advice(
        packet=_packet(), season_id="season-0001", fixture_id="md01-fx01",
        home="Brazil", away="Argentina", manager_team="Brazil",
        mode="research", match_seed=142, base_revision=3,
        checkpoint_artifact="data/world_model/candidate.pt",
        checkpoint_sha256="d" * 64, state_snapshots=_snapshots(),
    )


def _rehash(payload, field):
    frozen = copy.deepcopy(payload)
    frozen.pop(field, None)
    payload[field] = hashlib.sha256(json.dumps(
        frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def test_manager_advice_compares_every_product_tactic_and_binds_state():
    advice = _advice()
    validate_manager_decision_advice(advice)

    assert advice["issued_revision"] == 4
    assert [row["rank"] for row in advice["candidates"]] == list(
        range(1, len(PLAYABLE_TACTICS) + 1)
    )
    assert {row["tactic"] for row in advice["candidates"]} == set(
        PLAYABLE_TACTICS
    )
    assert advice["recommended_tactic"] == "team_identity"
    assert advice["representative_state"]["carryover_sources"]["Brazil"][
        "source_identity"
    ] == "a" * 64


def test_manager_advice_rejects_rehashed_ranking_tampering():
    advice = _advice()
    advice["candidates"][0]["risk_adjusted_value"] = -9.0
    _rehash(advice, "advice_identity")

    with pytest.raises(ValueError, match="recommendation replay mismatch"):
        validate_manager_decision_advice(advice)


def test_manager_advice_adoption_binds_explicit_intent_and_selected_tactic():
    advice = _advice()
    adoption = build_manager_advice_adoption(
        advice, selected_tactic="gegenpress", intent="reviewed_then_selected",
    )
    validate_manager_advice_adoption(
        adoption, advice=advice, selected_tactic="gegenpress",
    )
    assert not adoption["aligned_with_recommendation"]

    with pytest.raises(ValueError, match="does not match tactic"):
        build_manager_advice_adoption(
            advice, selected_tactic="gegenpress", intent="adopt_recommendation",
        )


def test_manager_advice_rejects_stable_mode_and_partial_candidate_set():
    packet = _packet()
    packet["candidates"].pop()
    with pytest.raises(ValueError, match="every playable tactic"):
        build_manager_decision_advice(
            packet=packet, season_id="season-0001", fixture_id="md01-fx01",
            home="Brazil", away="Argentina", manager_team="Brazil",
            mode="research", match_seed=142, base_revision=3,
            checkpoint_artifact="data/world_model/candidate.pt",
            checkpoint_sha256="d" * 64, state_snapshots=_snapshots(),
        )


def test_manager_advice_comparison_explains_selected_tactic_tradeoffs():
    advice = _advice()
    comparison = build_manager_advice_comparison(
        advice, selected_tactic="gegenpress",
    )
    validate_manager_advice_comparison(
        comparison, advice=advice, selected_tactic="gegenpress",
    )

    selected = next(
        row for row in advice["candidates"] if row["tactic"] == "gegenpress"
    )
    recommended = advice["candidates"][0]
    assert comparison["aligned_with_recommendation"] is False
    assert comparison["authority"]["level"] == "exploratory_only"
    assert "insufficient_history" in comparison["authority"]["reasons"]
    assert comparison["authority"]["automatic_adoption_authorized"] is False
    assert comparison["recommended_minus_selected"][
        "risk_adjusted_value"
    ] == round(
        recommended["risk_adjusted_value"] - selected["risk_adjusted_value"], 9,
    )
    assert set(comparison["recommended_minus_selected"][
        "event_probabilities"
    ]) == {"retain", "turnover", "shot", "foul", "out"}
    assert len(comparison["comparison_identity"]) == 64


def test_manager_advice_comparison_marks_low_confidence_and_rejects_tampering():
    advice = _advice()
    advice["candidates"][0]["effective_confidence"] = 0.05
    advice["recommendation_confidence"] = 0.05
    _rehash(advice, "advice_identity")
    validate_manager_decision_advice(advice)
    comparison = build_manager_advice_comparison(
        advice, selected_tactic=advice["recommended_tactic"],
    )

    assert comparison["authority"]["level"] == "exploratory_only"
    assert "low_model_confidence" in comparison["authority"]["reasons"]
    assert all(
        value == 0.0
        for key, value in comparison["recommended_minus_selected"].items()
        if key != "event_probabilities"
    )

    comparison["selected_rank"] = 99
    _rehash(comparison, "comparison_identity")
    with pytest.raises(ValueError, match="comparison replay mismatch"):
        validate_manager_advice_comparison(
            comparison, advice=advice,
            selected_tactic=advice["recommended_tactic"],
        )
    with pytest.raises(ValueError, match="requires research mode"):
        build_manager_decision_advice(
            packet=_packet(), season_id="season-0001", fixture_id="md01-fx01",
            home="Brazil", away="Argentina", manager_team="Brazil",
            mode="stable", match_seed=142, base_revision=3,
            checkpoint_artifact="data/world_model/candidate.pt",
            checkpoint_sha256="d" * 64, state_snapshots=_snapshots(),
        )
