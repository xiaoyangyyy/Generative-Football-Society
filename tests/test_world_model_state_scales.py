"""World-model state scales and LLM event claims stay numeric and falsifiable."""

import numpy as np
import pytest

from src.match_engine.world_model.llm_event_hypothesis import (
    apply_llm_event_hypothesis,
    score_llm_event_hypothesis,
    validate_llm_event_hypothesis,
)
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.state_scales import multiscale_state_forecast


def _forecast():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    current[200:202] = [0.5, 0.5]
    current[204:206] = [0.2, 0.2]
    current[207] = 0.5
    current[209] = 1.0
    current[102] = 0.8
    samples = np.repeat(current[None, :], 3, axis=0)
    samples[:, 200] = [0.70, 0.80, 0.50]
    samples[:, 209] = [0.80, 0.20, 0.90]
    samples[0, 204] = 0.40
    return multiscale_state_forecast(
        current, samples, attacking_home=True, horizon_s=60.0,
    )


def _packet():
    forecast = _forecast()
    return {
        "recommended_action": "pass",
        "candidates": [{
            "action": "pass",
            "risk_adjusted_value": 0.2,
            "multi_horizon_predictions": {
                "60s": {
                    "state_scales": forecast,
                    "semantic_event_probabilities": forecast[
                        "semantic_event_probabilities"
                    ],
                },
            },
        }],
    }


def test_multiscale_projection_uses_member_frequency_without_false_certainty():
    forecast = _forecast()

    assert forecast["primary_scale"] == "tactical"
    assert set(forecast) >= {"short", "tactical", "strategic"}
    events = forecast["semantic_event_probabilities"]
    assert events["retain_possession"] == pytest.approx(0.625)
    assert events["enter_final_third"] == pytest.approx(0.625)
    assert events["positive_territorial_shift"] == pytest.approx(0.625)
    assert events["improve_scoreline"] == pytest.approx(0.375)
    assert all(0.0 < value < 1.0 for value in events.values())
    assert not forecast["claims"]["observed"]
    assert not forecast["claims"]["causal"]


def test_multiscale_projection_respects_attacking_orientation():
    current = np.zeros(OBS_DIM, dtype=np.float32)
    current[200] = 0.7
    samples = np.repeat(current[None, :], 2, axis=0)
    samples[:, 200] = 0.5

    forecast = multiscale_state_forecast(
        current, samples, attacking_home=False, horizon_s=180.0,
    )

    assert forecast["primary_scale"] == "strategic"
    assert forecast["short"]["mean_progress_delta"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="members, 307"):
        multiscale_state_forecast(
            current, np.zeros((2, 4)), attacking_home=False, horizon_s=10.0,
        )

    neutral = multiscale_state_forecast(
        current,
        samples,
        attacking_home=False,
        horizon_s=180.0,
        ensemble_trained=False,
    )
    assert set(neutral["semantic_event_probabilities"].values()) == {0.5}
    assert neutral["claims"]["probability_source"] == (
        "untrained_ensemble_neutral"
    )


def test_llm_event_hypothesis_is_grounded_and_shadow_only():
    packet = _packet()
    raw = {
        "action": "pass",
        "horizon": "60s",
        "event": "enter_final_third",
        "expectation": "occur",
        "confidence": 0.8,
        "evidence_scales": ["tactical", "strategic"],
        "rationale": "Territorial trajectory supports entry.",
    }

    audit = apply_llm_event_hypothesis(
        packet, raw, selected_action="pass",
    )

    assert audit["accepted"]
    assert not audit["authority_active"]
    assert not audit["policy_mutated"]
    assert audit["llm_event_probability"] == pytest.approx(0.8)
    assert audit["world_model_event_probability"] == pytest.approx(0.625)
    assert packet["recommended_action"] == "pass"
    assert packet["candidates"][0]["risk_adjusted_value"] == 0.2


def test_llm_event_contract_rejects_weak_or_ungrounded_claims():
    raw = {
        "action": "pass",
        "horizon": "60s",
        "event": "enter_final_third",
        "expectation": "occur",
        "confidence": 0.2,
        "evidence_scales": ["tactical"],
    }
    assert validate_llm_event_hypothesis(raw) is None

    raw["confidence"] = 0.8
    audit = apply_llm_event_hypothesis(
        _packet(), raw, selected_action="shot",
    )
    assert not audit["accepted"]
    assert audit["reason"] == "event_action_must_equal_selected_action"

    packet = _packet()
    packet["candidates"][0]["multi_horizon_predictions"]["60s"][
        "semantic_event_probabilities"
    ]["enter_final_third"] = float("nan")
    invalid_probability = apply_llm_event_hypothesis(
        packet, raw, selected_action="pass",
    )
    assert not invalid_probability["accepted"]
    assert invalid_probability["reason"] == "event_probability_not_finite"


def test_llm_and_world_model_event_probabilities_share_one_outcome_score():
    audit = apply_llm_event_hypothesis(
        _packet(),
        {
            "action": "pass",
            "horizon": "60s",
            "event": "enter_final_third",
            "expectation": "occur",
            "confidence": 0.8,
            "evidence_scales": ["tactical"],
        },
        selected_action="pass",
    )
    score = score_llm_event_hypothesis(
        audit,
        {
            "progress": 0.20,
            "retained_possession": True,
            "goal_diff_delta": 0.0,
        },
        {"ball_x": 0.50},
        attacking_home=True,
    )

    assert score["observed"]
    assert score["llm_brier"] == pytest.approx(0.04)
    assert score["world_model_brier"] == pytest.approx(0.140625)
    assert score["llm_brier_gain_vs_world_model"] > 0.0
