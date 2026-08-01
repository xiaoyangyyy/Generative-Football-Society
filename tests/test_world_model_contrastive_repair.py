"""World-model counterevidence may repair explanation, never the action."""

from types import SimpleNamespace

import numpy as np

from src.match_engine.world_model.contrastive_explanation import (
    evaluate_llm_contrastive_claim,
)
from src.match_engine.world_model.contrastive_repair import (
    attempt_contrastive_explanation_repair,
    contrastive_repair_diagnostics,
)
from src.match_engine.world_model.observation import OBS_DIM


def _claim(effect="opposes_selected", selected="pass"):
    return {
        "selected_action": selected,
        "alternative_action": "hold",
        "horizon": "transition",
        "factor": "score_context",
        "effect": effect,
        "confidence": 0.8,
        "rationale": "A falsifiable explanation.",
    }


def _packet():
    candidates = []
    for action in ("pass", "hold"):
        candidates.append({
            "action": action,
            "target": [0.65, 0.5],
            "physics_prior": 0.75,
            "multi_horizon_predictions": {
                "transition": {"horizon_s": 10.0, "rollout_steps": 1},
            },
        })
    return {
        "available": True,
        "horizon_s": 10.0,
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "environment-a",
        "candidates": candidates,
    }


class _Runtime:
    def __init__(self):
        self.model = SimpleNamespace(transition_member_count=3)

    def encode_state(self, state, *, attacking_home):
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        observation[204:207] = [0.2, 0.0, 4.0 / 6.0]
        observation[207] = 0.25
        observation[208] = 0.7
        observation[298:306] = [0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6]
        observation[-1] = float(attacking_home)
        return observation

    def planner_confidence(self, observation, *, kind):
        return 0.8

    def predict_policy_utility(
        self, observation, action, *, action_kind, attacking_home, horizon_s,
    ):
        edge = float(observation[206]) - 0.5
        return {
            "policy_utility": 0.2 + 0.48 * edge if action_kind == "pass" else 0.25,
            "uncertainty": 0.1,
            "retention_probability": 0.8,
        }


class _RepairLLM:
    def __init__(self, revised_effect="supports_selected", selected="pass"):
        self.revised_effect = revised_effect
        self.selected = selected
        self.calls = []

    def revise_world_model_contrastive_claim(self, **kwargs):
        self.calls.append(kwargs)
        return _claim(effect=self.revised_effect, selected=self.selected)


def _state():
    return SimpleNamespace(home=SimpleNamespace(team_id="A"))


def _initial_audit():
    audit = evaluate_llm_contrastive_claim(
        _Runtime(),
        _state(),
        _packet(),
        _claim(effect="opposes_selected"),
        team_id="A",
        selected_action="pass",
        contrastive_signature="llm-contrastive:test",
    )
    assert audit["accepted"] and not audit["directionally_faithful"]
    return audit


def test_one_counterevidence_turn_repairs_only_the_explanation():
    llm = _RepairLLM()
    repair = attempt_contrastive_explanation_repair(
        llm,
        _Runtime(),
        _state(),
        _packet(),
        _initial_audit(),
        team_id="A",
        selected_action="pass",
        enabled=True,
        repair_signature="llm-contrastive-repair:test",
        total_member_trajectory_path_budget=128,
    )

    assert repair["repair_attempted"]
    assert repair["llm_calls"] == 1
    assert repair["revision_accepted"]
    assert repair["repair_successful"]
    assert repair["effective_audit_source"] == "revision"
    assert repair["effective_claim"]["effect"] == "supports_selected"
    assert repair["directional_effect_gain"] > 0.0
    assert repair["total_member_trajectory_paths"] == 24
    assert repair["total_member_trajectory_paths"] <= 128
    assert repair["selected_action_immutable"]
    assert not repair["action_mutated"]
    assert not repair["control_fields_mutated"]
    assert not repair["can_change_selected_action"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["immutable_selected_action"] == "pass"


def test_repair_rejects_action_change_and_never_reopens_control():
    repair = attempt_contrastive_explanation_repair(
        _RepairLLM(selected="shot"),
        _Runtime(),
        _state(),
        _packet(),
        _initial_audit(),
        team_id="A",
        selected_action="pass",
        enabled=True,
    )
    assert repair["repair_attempted"]
    assert not repair["revision_accepted"]
    assert not repair["repair_successful"]
    assert repair["reason"] == "contrastive_repair_attempted_action_change"
    assert not repair["selected_action_immutable"]
    assert not repair["action_mutated"]
    assert not repair["authority_active"]


def test_repair_is_opt_in_and_checks_cumulative_budget_before_llm_call():
    disabled_llm = _RepairLLM()
    disabled = attempt_contrastive_explanation_repair(
        disabled_llm,
        _Runtime(),
        _state(),
        _packet(),
        _initial_audit(),
        team_id="A",
        selected_action="pass",
        enabled=False,
    )
    assert not disabled["repair_attempted"]
    assert disabled["reason"] == "contrastive_repair_disabled"
    assert not disabled_llm.calls

    budget_llm = _RepairLLM()
    exhausted = attempt_contrastive_explanation_repair(
        budget_llm,
        _Runtime(),
        _state(),
        _packet(),
        _initial_audit(),
        team_id="A",
        selected_action="pass",
        enabled=True,
        total_member_trajectory_path_budget=12,
    )
    assert not exhausted["repair_attempted"]
    assert exhausted["reason"] == "contrastive_repair_budget_exhausted"
    assert not budget_llm.calls


def test_failed_revision_probe_is_conservatively_charged_to_compute_budget():
    class FailingRuntime(_Runtime):
        def __init__(self):
            super().__init__()
            self.prediction_calls = 0

        def predict_policy_utility(self, *args, **kwargs):
            self.prediction_calls += 1
            if self.prediction_calls == 2:
                raise RuntimeError("synthetic probe failure")
            return super().predict_policy_utility(*args, **kwargs)

    repair = attempt_contrastive_explanation_repair(
        _RepairLLM(),
        FailingRuntime(),
        _state(),
        _packet(),
        _initial_audit(),
        team_id="A",
        selected_action="pass",
        enabled=True,
        total_member_trajectory_path_budget=128,
    )
    assert repair["repair_attempted"]
    assert not repair["revision_accepted"]
    assert not repair["repair_successful"]
    assert repair["revised_member_trajectory_paths"] == 12
    assert repair["total_member_trajectory_paths"] == 24
    assert repair["total_member_trajectory_paths"] <= 128


def test_repair_diagnostics_are_match_clustered_and_provenance_scoped():
    logs = []
    for index in range(4):
        repair = attempt_contrastive_explanation_repair(
            _RepairLLM(
                revised_effect=(
                    "opposes_selected" if index == 3 else "supports_selected"
                )
            ),
            _Runtime(),
            _state(),
            _packet(),
            _initial_audit(),
            team_id="A",
            selected_action="pass",
            enabled=True,
            repair_signature="llm-contrastive-repair:test",
        )
        logs.append({
            "world_model_decision_adoption": {
                "records": [{"llm_contrastive_repair_context": repair}],
            },
        })

    diagnostics = contrastive_repair_diagnostics(logs)
    assert diagnostics["attempts"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics["repairs_successful"] == 3
    assert diagnostics["match_clustered_repair_success_rate"] == 0.75
    assert diagnostics["all_single_call"]
    assert diagnostics["budgets_respected"]
    assert diagnostics["all_actions_immutable"]
    assert diagnostics["provenance_compatible"]

    logs[-1]["world_model_decision_adoption"]["records"][0][
        "llm_contrastive_repair_context"
    ]["repair_signature"] = "llm-contrastive-repair:mixed"
    mixed = contrastive_repair_diagnostics(logs)
    assert not mixed["provenance_compatible"]
