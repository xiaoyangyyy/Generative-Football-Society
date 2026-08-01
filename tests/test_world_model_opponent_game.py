from __future__ import annotations

import pytest

from src.match_engine.world_model.opponent_contract import OPPONENT_HYPOTHESES
from src.match_engine.world_model.opponent_game import (
    apply_llm_response_hypothesis,
    attach_second_order_game,
    validate_llm_response_hypothesis,
)


def _posterior(primary: str, probability: float = 0.94):
    residual = (1.0 - probability) / (len(OPPONENT_HYPOTHESES) - 1)
    return {
        name: probability if name == primary else residual
        for name in OPPONENT_HYPOTHESES
    }


def _candidate(action: str, immediate: float, **values: float):
    hypotheses = {name: 0.0 for name in OPPONENT_HYPOTHESES}
    hypotheses.update(values)
    return {
        "action": action,
        "effective_confidence": 0.8,
        "risk_adjusted_value": immediate,
        "opponent_hypothesis_values": hypotheses,
    }


class _ActionResponseMemory:
    def predict(self, current_posterior, action):
        target = "low_block" if action == "hold" else "gegenpress"
        return {
            "version": 1,
            "action": action,
            "source": "validated_action_conditioned_response_memory",
            "learned_active": True,
            "trust": 0.4,
            "response_posterior": _posterior(target),
            "structural_posterior": dict(current_posterior),
            "normalized_entropy": 0.18,
        }

    def summary(self):
        return {"version": 1, "active_actions": 2}


def test_two_ply_game_can_rerank_with_action_conditioned_response():
    candidates = [
        _candidate("hold", 0.80, low_block=0.0, gegenpress=0.2),
        _candidate("pass", 0.70, low_block=-0.1, gegenpress=1.0),
    ]
    game = attach_second_order_game(
        candidates,
        {"posterior": _posterior("balanced")},
        _ActionResponseMemory(),
    )

    assert game["available"]
    assert game["recommended_action"] == "pass"
    assert candidates[0]["opponent_response_prediction"][
        "response_posterior"
    ]["low_block"] > 0.9
    assert candidates[1]["opponent_response_prediction"][
        "response_posterior"
    ]["gegenpress"] > 0.9
    assert candidates[1]["two_ply_risk_adjusted_value"] > candidates[0][
        "two_ply_risk_adjusted_value"
    ]


def test_continuation_is_one_action_chosen_against_belief_not_hidden_truth():
    candidates = [
        _candidate("hold", 0.3, low_block=1.0, gegenpress=0.0),
        _candidate("pass", 0.3, low_block=0.0, gegenpress=0.9),
    ]
    game = attach_second_order_game(
        candidates,
        {"posterior": _posterior("balanced")},
        _ActionResponseMemory(),
    )

    assert all("best_continuation_action" in branch for branch in game["branches"])
    assert len(candidates[0]["continuation_options"]) == 2
    assert isinstance(candidates[0]["best_continuation_action"], str)
    assert "hidden truth" in game["limitations"][2]


def test_llm_response_branch_is_bounded_audited_and_non_persistent():
    candidates = [
        _candidate("hold", 0.4, low_block=0.8, gegenpress=0.1),
        _candidate("pass", 0.3, low_block=0.2, gegenpress=0.7),
    ]
    packet = {
        "candidates": candidates,
        "opponent_belief": {"posterior": _posterior("balanced")},
    }
    packet["second_order_game"] = attach_second_order_game(
        candidates, packet["opponent_belief"], _ActionResponseMemory(),
    )
    audit = apply_llm_response_hypothesis(
        packet,
        {
            "if_action": "hold",
            "response_preset": "low_block",
            "confidence": 1.0,
            "rationale": "conditional stress",
        },
        _ActionResponseMemory(),
    )

    assert audit["accepted"]
    assert 0.0 < audit["bounded_influence"] <= 0.15
    assert not audit["can_update_response_memory"]
    assert not audit["causal_interpretation"]
    assert "bounded_llm_branch_hypothesis" in candidates[0][
        "opponent_response_prediction"
    ]["source"]


def test_response_hypothesis_validation_and_closed_gate_are_safe():
    assert validate_llm_response_hypothesis({"if_action": "teleport"}) is None
    assert not attach_second_order_game([], {"posterior": _posterior("balanced")})[
        "available"
    ]
    packet = {"candidates": [], "opponent_belief": {"posterior": _posterior("balanced")}}
    audit = apply_llm_response_hypothesis(
        packet,
        {"if_action": "pass", "response_preset": "low_block"},
    )
    assert not audit["accepted"]
    assert audit["reason"] == "response_branch_not_evaluated"


def test_structural_only_llm_branch_has_half_evidence_authority():
    candidates = [_candidate("pass", 0.4, low_block=0.7)]
    packet = {
        "candidates": candidates,
        "opponent_belief": {"posterior": _posterior("low_block")},
    }
    packet["second_order_game"] = attach_second_order_game(
        candidates, packet["opponent_belief"], None,
    )
    audit = apply_llm_response_hypothesis(
        packet,
        {
            "if_action": "pass",
            "response_preset": "low_block",
            "confidence": 1.0,
        },
        None,
    )

    assert audit["evidence_tier"] == "structural_response_prior"
    assert audit["bounded_influence"] == pytest.approx(0.075)
