"""LLM explanations are checked against bounded world-model interventions."""

from types import SimpleNamespace

import numpy as np

from src.match_engine.cognitive.schemas import validate_coach_plan
from src.match_engine.world_model.contrastive_explanation import (
    contrastive_explanation_diagnostics,
    evaluate_llm_contrastive_claim,
    validate_llm_contrastive_claim,
)
from src.match_engine.world_model.observation import OBS_DIM


def _claim(**updates):
    value = {
        "selected_action": "pass",
        "alternative_action": "hold",
        "horizon": "transition",
        "factor": "score_context",
        "effect": "supports_selected",
        "confidence": 0.8,
        "rationale": "The lead makes circulation preferable to waiting.",
    }
    value.update(updates)
    return value


def _candidate(action, rollout_steps=1):
    return {
        "action": action,
        "target": [0.65, 0.5],
        "physics_prior": 0.75,
        "multi_horizon_predictions": {
            "transition": {
                "horizon_s": 10.0,
                "rollout_steps": rollout_steps,
            },
        },
    }


def _packet(rollout_steps=1):
    return {
        "available": True,
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "environment-a",
        "horizon_s": 10.0,
        "candidates": [
            _candidate("pass", rollout_steps),
            _candidate("hold", rollout_steps),
        ],
    }


class _ContrastiveRuntime:
    def __init__(self, *, two_step_active=True, neutral_phase=False):
        self.model = SimpleNamespace(transition_member_count=3)
        self.two_step_active = two_step_active
        self.neutral_phase = neutral_phase
        self.calls = []

    def encode_state(self, state, *, attacking_home):
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        observation[204] = 0.2
        observation[205] = 0.0
        observation[206] = 4.0 / 6.0
        observation[207] = 0.5 if self.neutral_phase else 0.25
        observation[208] = 0.7
        observation[298:302] = [0.7, 0.6, 0.55, 0.4]
        observation[302:306] = [0.3, 0.4, 0.35, 0.5]
        observation[-1] = float(attacking_home)
        return observation

    def two_step_planning_gate(self):
        return {
            "active": self.two_step_active,
            "authority": 0.35 if self.two_step_active else 0.0,
            "reason": "test_contract",
        }

    def planner_confidence(self, observation, *, kind):
        return 0.8

    def predict_policy_utility(
        self, observation, action, *, action_kind, attacking_home, horizon_s,
    ):
        self.calls.append((action_kind, float(observation[206])))
        score_edge = float(observation[206]) - 0.5
        utility = (
            0.20 + 0.48 * score_edge
            if action_kind == "pass" else 0.25
        )
        return {
            "policy_utility": utility,
            "uncertainty": 0.1,
            "retention_probability": 0.8,
        }


def _state():
    return SimpleNamespace(home=SimpleNamespace(team_id="A"))


def test_contrastive_claim_schema_rejects_non_contrastive_or_weak_claims():
    assert validate_llm_contrastive_claim(_claim())["factor"] == "score_context"
    assert validate_llm_contrastive_claim(
        _claim(alternative_action="pass")
    ) is None
    assert validate_llm_contrastive_claim(_claim(confidence=0.49)) is None
    assert validate_llm_contrastive_claim(_claim(factor="hidden_momentum")) is None
    plan = validate_coach_plan({
        "world_model_action": "pass",
        "world_model_contrastive_claim": _claim(),
    })
    assert plan["world_model_contrastive_claim"]["alternative_action"] == "hold"


def test_world_model_checks_declared_factor_against_action_margin():
    runtime = _ContrastiveRuntime()
    packet = _packet()
    audit = evaluate_llm_contrastive_claim(
        runtime,
        _state(),
        packet,
        _claim(),
        team_id="A",
        selected_action="pass",
        contrastive_signature="llm-contrastive:test",
    )

    assert audit["accepted"]
    assert audit["directionally_faithful"]
    assert audit["declared_factor_effect_on_margin"] > 0.0
    assert audit["masked_feature_indices"] == [204, 205, 206]
    assert audit["model_calls"] == 4
    assert audit["probe_value_source"] == (
        "runtime_policy_utility_without_cross_match_residual_memory"
    )
    assert audit["trajectory_member_paths"] == 12
    assert len(runtime.calls) == 4
    assert audit["shadow_only"]
    assert not audit["can_change_selected_action"]
    assert not audit["world_model_prediction_mutated"]
    assert packet["llm_contrastive_explanation_audit"] is audit


def test_unfaithful_direction_is_preserved_without_policy_authority():
    audit = evaluate_llm_contrastive_claim(
        _ContrastiveRuntime(),
        _state(),
        _packet(),
        _claim(effect="opposes_selected"),
        team_id="A",
        selected_action="pass",
    )
    assert audit["accepted"]
    assert not audit["directionally_faithful"]
    assert audit["claimed_directional_effect"] < 0.0
    assert not audit["authority_active"]


def test_tactical_factor_indices_follow_attacking_team_orientation():
    away_state = SimpleNamespace(home=SimpleNamespace(team_id="A"))
    audit = evaluate_llm_contrastive_claim(
        _ContrastiveRuntime(),
        away_state,
        _packet(),
        _claim(factor="own_tactics"),
        team_id="B",
        selected_action="pass",
    )
    assert audit["accepted"]
    assert audit["masked_feature_indices"] == [302, 303, 304, 305]


def test_contrastive_probe_fails_closed_on_neutral_factor_gate_or_budget():
    unavailable_packet = _packet()
    unavailable_packet["available"] = False
    unavailable = evaluate_llm_contrastive_claim(
        _ContrastiveRuntime(),
        _state(),
        unavailable_packet,
        _claim(),
        team_id="A",
        selected_action="pass",
    )
    assert not unavailable["accepted"]
    assert unavailable["reason"] == (
        "contrastive_world_model_quality_gate_closed"
    )

    neutral = evaluate_llm_contrastive_claim(
        _ContrastiveRuntime(neutral_phase=True),
        _state(),
        _packet(),
        _claim(factor="match_phase"),
        team_id="A",
        selected_action="pass",
    )
    assert not neutral["accepted"]
    assert neutral["reason"] == "contrastive_factor_already_neutral"

    closed = evaluate_llm_contrastive_claim(
        _ContrastiveRuntime(two_step_active=False),
        _state(),
        _packet(rollout_steps=2),
        _claim(),
        team_id="A",
        selected_action="pass",
    )
    assert not closed["accepted"]
    assert closed["reason"] == "contrastive_planning_gate_closed"

    budgeted = evaluate_llm_contrastive_claim(
        _ContrastiveRuntime(),
        _state(),
        _packet(rollout_steps=2),
        _claim(),
        team_id="A",
        selected_action="pass",
        max_member_trajectory_paths=23,
    )
    assert not budgeted["accepted"]
    assert budgeted["reason"] == "contrastive_trajectory_budget_insufficient"
    assert budgeted["trajectory_member_paths_required"] == 24


def test_contrastive_diagnostics_are_match_clustered_and_provenance_scoped():
    logs = []
    for index in range(4):
        audit = evaluate_llm_contrastive_claim(
            _ContrastiveRuntime(),
            _state(),
            _packet(),
            _claim(effect=(
                "opposes_selected" if index == 3 else "supports_selected"
            )),
            team_id="A",
            selected_action="pass",
            contrastive_signature="llm-contrastive:test",
        )
        logs.append({
            "world_model_decision_adoption": {
                "records": [{
                    "llm_contrastive_explanation_context": audit,
                }],
            },
        })

    diagnostics = contrastive_explanation_diagnostics(logs)
    assert diagnostics["claims"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics[
        "match_clustered_directional_faithfulness"
    ] == 0.75
    assert diagnostics["budgets_respected"]
    assert diagnostics["provenance_compatible"]

    first = logs[0]["world_model_decision_adoption"]["records"][0][
        "llm_contrastive_explanation_context"
    ]
    first["directionally_faithful"] = False
    malformed = contrastive_explanation_diagnostics(logs)
    assert malformed["malformed_claim_audits"] == 1
    first["directionally_faithful"] = True

    logs[-1]["world_model_decision_adoption"]["records"][0][
        "llm_contrastive_explanation_context"
    ]["contrastive_signature"] = "llm-contrastive:mixed"
    mixed = contrastive_explanation_diagnostics(logs)
    assert not mixed["provenance_compatible"]
