"""Outcome-aligned M2 policy contracts fail closed and preserve actor identity."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_world_model import _policy_utility_validation
import scripts.run_m2_mirrored_policy_study as m2_study
from src.match_engine.world_model.action_adoption import (
    direct_action_adoption_diagnostics,
    record_action_policy_sample,
    register_action_policy_opportunity,
    resolve_action_policy_outcome,
)
from src.match_engine.world_model.action_codec import decode_action_kind
from src.match_engine.world_model.config import world_model_controls_side
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.planner import action_imagination_adjustments
from src.match_engine.world_model.mirrored_policy_evaluation import (
    fixture_stratified_cluster_interval,
    paired_effect_rows,
)
from src.match_engine.world_model.policy_utility import (
    POLICY_UTILITY_VERSION,
    policy_utility_validation_gate,
    transition_policy_utility_numpy,
    transition_policy_utility_tensor,
)


def _observation(*, attacking_home: bool, x: float = 0.5) -> np.ndarray:
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[200] = x
    observation[209] = 1.0 if attacking_home else 0.0
    observation[-1] = float(attacking_home)
    return observation


def test_transition_utility_has_zero_persistence_and_actor_symmetry():
    home_current = _observation(attacking_home=True)
    home_future = home_current.copy()
    home_future[200] += 0.2
    away_current = _observation(attacking_home=False)
    away_future = away_current.copy()
    away_future[200] -= 0.2

    persistence = transition_policy_utility_numpy(
        home_current, home_current, attacking_home=True,
    )
    home = transition_policy_utility_numpy(
        home_current, home_future, attacking_home=True,
    )
    away = transition_policy_utility_numpy(
        away_current, away_future, attacking_home=False,
    )

    assert float(persistence["policy_utility"]) == 0.0
    assert float(home["policy_utility"]) == pytest.approx(
        float(away["policy_utility"])
    )
    assert float(home["policy_utility"]) > 0.0


def test_tensor_utility_broadcast_matches_numpy():
    torch = pytest.importorskip("torch")
    current = _observation(attacking_home=True)
    futures = np.stack([current.copy(), current.copy()])
    futures[1, 200] += 0.1

    actual = transition_policy_utility_tensor(
        torch.from_numpy(current),
        torch.from_numpy(futures),
        attacking_home=True,
    )["policy_utility"].numpy()
    expected = np.asarray([
        transition_policy_utility_numpy(
            current, future, attacking_home=True,
        )["policy_utility"]
        for future in futures
    ])

    assert np.allclose(actual, expected)


def _valid_evidence() -> dict:
    return {
        "version": 1,
        "target_version": POLICY_UTILITY_VERSION,
        "trained_with_policy_utility_objective": True,
        "configured_loss_weight": 0.25,
        "optimization_steps": 10,
        "actions": {
            "pass": {
                "samples": 120,
                "groups": 8,
                "model_mse": 0.08,
                "persistence_mse": 0.10,
                "skill_vs_persistence": 0.20,
                "prediction_target_correlation": 0.30,
            },
        },
    }


def test_policy_utility_evidence_gate_is_action_specific_and_fail_closed():
    assert policy_utility_validation_gate(
        None, action_kind="pass",
    )["authorized"] is False
    assert policy_utility_validation_gate(
        _valid_evidence(), action_kind="cross",
    )["authorized"] is False
    opened = policy_utility_validation_gate(
        _valid_evidence(), action_kind="pass",
    )
    assert opened["authorized"] is True
    assert 0.0 < opened["authority"] <= 1.0


def test_training_validation_reports_separate_action_branches():
    target = np.linspace(-0.3, 0.3, 12)
    predicted = np.stack([target + 0.01, target - 0.01])
    actions = np.array(["pass"] * 6 + ["hold"] * 6)
    groups = np.array([f"g{i}" for i in range(12)])

    report = _policy_utility_validation(
        predicted, target, actions, groups,
        configured_loss_weight=0.25, optimization_steps=4,
    )

    assert report["target_version"] == POLICY_UTILITY_VERSION
    assert report["actions"]["pass"]["samples"] == 6
    assert report["actions"]["hold"]["samples"] == 6
    assert report["actions"]["cross"]["samples"] == 0


class _M2Runtime:
    def __init__(self, *, gate_open: bool):
        self.cfg = SimpleNamespace(
            planner_blend=0.30,
            shot_planner_blend=0.25,
            cross_planner_blend=0.20,
        )
        self.gate_open = gate_open
        self.last_uncertainty = 0.1

    def encode_state(self, state, *, attacking_home):
        return _observation(attacking_home=attacking_home)

    def planner_confidence(self, observation, kind="general"):
        return 0.8 if kind == "pass" else 0.0

    def policy_utility_authority(self, action_kind):
        return {
            "authorized": self.gate_open and action_kind == "pass",
            "authority": 0.75 if self.gate_open else 0.0,
            "reason": (
                "grouped_heldout_policy_utility_gain"
                if self.gate_open else "policy_utility_evidence_gate_closed"
            ),
        }

    def predict_policy_utility(
        self, observation, action, *, action_kind, attacking_home, horizon_s,
    ):
        return {
            "policy_utility": (
                0.30 if decode_action_kind(action) == "pass" else 0.0
            ),
            "policy_utility_version": POLICY_UTILITY_VERSION,
            "uncertainty": 0.1,
        }


def _planner_state():
    return SimpleNamespace(
        ball=SimpleNamespace(position=np.array([0.5, 0.5])),
        clock_seconds=10.0,
        _wm_horizon_s=1.0,
    )


def test_m2_changes_pass_utility_only_when_exact_gate_is_open(monkeypatch):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    monkeypatch.setenv("MATCH_WM_OUTCOME_ALIGNED_POLICY", "1")
    labels = ["pass", "shot", "cross", "hold"]
    base = np.zeros(4)
    carrier = SimpleNamespace(team_id="home", position=np.array([0.5, 0.5]))

    opened = action_imagination_adjustments(
        _M2Runtime(gate_open=True), _planner_state(), carrier, True,
        base, labels, dist_goal=0.5, feasible_actions={"pass", "hold"},
    )
    closed = action_imagination_adjustments(
        _M2Runtime(gate_open=False), _planner_state(), carrier, True,
        base, labels, dist_goal=0.5, feasible_actions={"pass", "hold"},
    )

    assert opened[0] > 0.0
    assert np.array_equal(closed, base)


def test_control_scope_supports_mirrored_one_sided_interventions(monkeypatch):
    monkeypatch.setenv("MATCH_WM_CONTROL_SCOPE", "home")
    assert world_model_controls_side(True)
    assert not world_model_controls_side(False)
    monkeypatch.setenv("MATCH_WM_CONTROL_SCOPE", "away")
    assert not world_model_controls_side(True)
    assert world_model_controls_side(False)
    monkeypatch.setenv("MATCH_WM_CONTROL_SCOPE", "none")
    assert not world_model_controls_side(True)
    assert not world_model_controls_side(False)


def test_sampled_action_is_resolved_against_realized_next_state():
    state = SimpleNamespace()
    labels = ["pass", "hold"]
    register_action_policy_opportunity(
        state,
        team_id="home",
        t_sec=10.0,
        feasible_actions=set(labels),
        base_utilities=[0.0, 0.0],
        adjusted_utilities=[0.1, 0.0],
        labels=labels,
        model_adjustments={"pass": 0.1, "hold": 0.0},
        quality_gates={
            "pass": {
                "open": True,
                "decision_confidence": 0.5,
                "decision_certainty": 0.9,
                "policy_blend": 0.3,
                "model_advantage": 0.2,
                "value_source": "outcome_aligned_policy_utility",
                "pass_value": 0.1,
            },
            "hold": {"open": False},
        },
    )
    record_action_policy_sample(
        state,
        actual_action="pass",
        labels=labels,
        base_probabilities=[0.5, 0.5],
        adjusted_probabilities=[0.6, 0.4],
        sampling_uniform=0.2,
        counterfactual_baseline_action="pass",
    )
    current = _observation(attacking_home=True)
    future = current.copy()
    future[200] += 0.2
    resolve_action_policy_outcome(
        state, current, future, attacking_home=True,
    )

    audit = direct_action_adoption_diagnostics(state)
    outcome = audit["records"][0]["policy_utility_outcome"]
    assert outcome["actual_action"] == "pass"
    assert outcome["realized_policy_utility"] > 0.0
    assert audit["realized_policy_utility"]["count"] == 1
    assert audit["realized_policy_utility"]["attributable_count"] == 1
    assert audit["realized_policy_utility"]["by_action"]["pass"][
        "predicted_count"
    ] == 1


def _study_row(*, sample: int, home_xg: float, away_xg: float) -> dict:
    return {
        "fixture": "A_vs_B",
        "sample_index": sample,
        "micro_xg_home": home_xg,
        "micro_xg_away": away_xg,
        "goals_home": 1.0,
        "goals_away": 0.0,
        "pass_completion_home": 0.8,
        "pass_completion_away": 0.7,
        "shots_home": 10.0,
        "shots_away": 8.0,
        "shots_on_target_home": 4.0,
        "shots_on_target_away": 3.0,
        "wm_action_counterfactual_change_rate": 0.03,
        "wm_action_expected_counterfactual_change_rate": 0.04,
        "wm_realized_policy_utility_count": 5.0,
        "wm_realized_policy_utility_mean": 0.1,
        "wm_attributable_realized_policy_utility_count": 5.0,
        "wm_attributable_realized_policy_utility_mean": 0.1,
    }


def test_mirrored_effects_use_the_controlled_team_perspective():
    baseline = [_study_row(sample=0, home_xg=1.0, away_xg=1.0)]
    home = [_study_row(sample=0, home_xg=1.2, away_xg=1.0)]
    away = [_study_row(sample=0, home_xg=1.0, away_xg=1.3)]

    effects = paired_effect_rows(baseline, home, away)

    assert len(effects) == 2
    assert effects[0]["effects"]["controlled_micro_xg_margin"] == pytest.approx(
        0.2
    )
    assert effects[1]["effects"]["controlled_micro_xg_margin"] == pytest.approx(
        0.3
    )
    interval = fixture_stratified_cluster_interval(
        effects,
        lambda row: row["effects"]["controlled_micro_xg_margin"],
        draws=100,
        seed=7,
    )
    assert interval["point_estimate"] == pytest.approx(0.25)


def test_mirrored_study_rejects_missing_or_misaligned_arms():
    baseline = [_study_row(sample=0, home_xg=1.0, away_xg=1.0)]
    duplicate = baseline + baseline
    with pytest.raises(ValueError, match="duplicate experimental unit"):
        paired_effect_rows(duplicate, baseline, baseline)
    with pytest.raises(ValueError, match="must match exactly"):
        paired_effect_rows(baseline, [], baseline)


def test_candidate_eligibility_reads_the_two_step_active_contract(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"sealed")

    class Runtime:
        pass_quality = 0.2
        cfg = SimpleNamespace(min_planner_quality=0.15)

        def policy_utility_authority(self, action):
            return {"authorized": action == "pass", "authority": 0.4}

        def two_step_planning_gate(self):
            return {"active": True, "authority": 0.3}

    monkeypatch.setattr(
        m2_study.WorldModelRuntime, "load", lambda _path: Runtime(),
    )
    protocol = {
        "candidate": {
            "required_policy_utility_branches": ["pass"],
            "optional_policy_utility_branches": ["cross"],
        },
    }

    eligibility = m2_study.candidate_eligibility(checkpoint, protocol)

    assert eligibility["eligible"] is True
    assert eligibility["two_step_gate"]["active"] is True
