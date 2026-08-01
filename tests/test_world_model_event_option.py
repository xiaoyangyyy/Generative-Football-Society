"""LLM conditional options remain bounded, auditable shadow proposals."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.match_engine.cognitive.schemas import validate_coach_plan
from src.match_engine.world_model.event_option import (
    evaluate_llm_event_option,
    validate_llm_event_option,
)
from src.match_engine.world_model.observation import OBS_DIM


def _option(**updates):
    value = {
        "first_action": "pass",
        "horizon": "60s",
        "event": "enter_final_third",
        "on_occurrence": "shot",
        "on_absence": "hold",
        "confidence": 0.8,
        "rationale": "Attack if territory is won; recycle otherwise.",
    }
    value.update(updates)
    return value


def _packet(*, rollout_steps=2, event_probability=0.5):
    return {
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "environment-a",
        "horizon_s": 10.0,
        "candidates": [{
            "action": "pass",
            "target": [0.68, 0.5],
            "physics_prior": 0.72,
            "risk_adjusted_value": 0.3,
            "multi_horizon_predictions": {
                "60s": {
                    "horizon_s": 60.0,
                    "rollout_steps": rollout_steps,
                    "semantic_event_probabilities": {
                        "enter_final_third": event_probability,
                    },
                },
            },
        }],
    }


class _OptionRuntime:
    def __init__(self, *, gate_active=True):
        self.gate_active = gate_active
        self.calls = []

    def two_step_planning_gate(self):
        return {
            "active": self.gate_active,
            "authority": 0.35 if self.gate_active else 0.0,
            "reason": "test_contract",
        }

    def semantic_event_head_gate(self, event, *, rollout_steps):
        return {"active": False, "authority": 0.0, "reason": "projection_only"}

    def encode_state(self, state, *, attacking_home):
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        observation[200] = 0.55
        observation[209] = 1.0
        observation[-1] = float(attacking_home)
        return observation

    def imagine(self, observation, action, **kwargs):
        members = np.repeat(observation[None, None, :], 4, axis=0)
        members[:, 0, 200] = [0.80, 0.75, 0.60, 0.55]
        return SimpleNamespace(
            uncertainty_samples={"transition_states": members},
        )

    def evaluate_action_from_observation(
        self, observation, action, *, action_kind, attacking_home, horizon_s,
    ):
        self.calls.append(action_kind)
        x = float(observation[200])
        value = x if action_kind == "shot" else 1.2 - x
        return {
            "expected_value": value,
            "model_calls": 1,
            "future": SimpleNamespace(
                state_uncertainty=0.1,
                event_probabilities={"turnover": 0.1},
            ),
        }


def test_event_option_contract_is_strict_and_survives_plan_sanitization():
    assert validate_llm_event_option(_option())["on_occurrence"] == "shot"
    assert validate_llm_event_option(_option(on_absence="shot")) is None
    assert validate_llm_event_option(_option(confidence=0.49)) is None

    plan = validate_coach_plan({
        "world_model_action": "pass",
        "world_model_event_option": _option(),
    })
    assert plan["world_model_event_option"]["event"] == "enter_final_third"


def test_member_conditioned_option_is_evaluated_with_hard_shadow_budget():
    runtime = _OptionRuntime()
    packet = _packet()
    audit = evaluate_llm_event_option(
        runtime,
        SimpleNamespace(home=SimpleNamespace(team_id="A")),
        packet,
        _option(),
        team_id="A",
        selected_action="pass",
        option_signature="llm-event-option:test",
        max_member_evaluations=8,
    )

    assert audit["accepted"]
    assert audit["member_evaluations"] == 8
    assert audit["member_evaluation_budget"] == 8
    assert runtime.calls == ["shot"] * 4 + ["hold"] * 4
    assert audit["conditional_gain_vs_best_fixed"] > 0.0
    assert audit["shadow_only"]
    assert not audit["authority_active"]
    assert not audit["policy_mutated"]
    assert not audit["can_execute_future_action"]
    assert packet["llm_event_option_audit"] is audit


@pytest.mark.parametrize(
    ("runtime", "packet", "budget", "reason"),
    [
        (_OptionRuntime(gate_active=False), _packet(), 8,
         "option_predicted_state_gate_closed"),
        (_OptionRuntime(), _packet(rollout_steps=3), 8,
         "option_rollout_depth_not_validated"),
        (_OptionRuntime(), _packet(), 7,
         "option_member_budget_insufficient"),
        (_OptionRuntime(), _packet(event_probability=float("nan")), 8,
         "option_event_probability_not_finite"),
    ],
)
def test_event_option_fails_closed(runtime, packet, budget, reason):
    audit = evaluate_llm_event_option(
        runtime,
        SimpleNamespace(home=SimpleNamespace(team_id="A")),
        packet,
        _option(),
        team_id="A",
        selected_action="pass",
        max_member_evaluations=budget,
    )
    assert not audit["accepted"]
    assert audit["reason"] == reason
    assert audit["shadow_only"]
