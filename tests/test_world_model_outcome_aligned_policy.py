"""Outcome-aligned M2 policy contracts fail closed and preserve actor identity."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.train_world_model import (
    _bootstrap_transition_loss,
    _policy_utility_validation,
)
import scripts.run_m2_mirrored_policy_study as m2_study
from scripts.estimate_m2_policy_power import estimate as estimate_power
from scripts.preflight_m2_training import readiness_checks
from scripts.validate_world_model import _sealed_evaluation
import src.match_engine.calibration.benchmark_core as benchmark_core
from src.match_engine.world_model.action_adoption import (
    direct_action_adoption_diagnostics,
    record_action_policy_sample,
    register_action_policy_opportunity,
    resolve_action_policy_outcome,
)
from src.match_engine.world_model.action_codec import (
    decode_action_kind,
    decode_action_kinds,
    encode_high_level_action,
)
from src.match_engine.world_model.config import world_model_controls_side
from src.match_engine.world_model.inference import WorldModelRuntime
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.planner import (
    action_imagination_adjustments,
    pass_candidate_policy_probabilities,
)
from src.match_engine.world_model.mirrored_policy_evaluation import (
    fixture_stratified_cluster_interval,
    paired_effect_rows,
    study_execution_identity,
    study_preflight_identity,
)
from src.match_engine.world_model.policy_utility import (
    POLICY_UTILITY_VERSION,
    build_continuation_support_evidence,
    continuation_support_validation_gate,
    policy_utility_validation_gate,
    policy_utility_sequence_validation_gate,
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
    profile = {
        "samples": 128,
        "groups": 8,
        "model_mse": 0.08,
        "persistence_mse": 0.10,
        "skill_vs_persistence": 0.20,
        "prediction_target_correlation": 0.30,
    }
    return {
        "version": 1,
        "target_version": POLICY_UTILITY_VERSION,
        "trained_with_policy_utility_objective": True,
        "configured_loss_weight": 0.25,
        "optimization_steps": 10,
        "actions": {
            "pass": {
                **profile,
                "samples": 256,
                "perspectives": {
                    "home": dict(profile),
                    "away": dict(profile),
                },
            },
        },
    }


def _continuation_support(
    *,
    first_action: str = "pass",
    pass_rows: int = 96,
    shot_rows: int = 32,
    actor_perspective: str = "pooled",
) -> dict:
    total = pass_rows + shot_rows
    first = np.asarray([first_action] * total)
    second = np.asarray(["pass"] * pass_rows + ["shot"] * shot_rows)
    groups = np.asarray([f"g{i % 8}" for i in range(total)])
    actions = np.zeros((total, 18), dtype=np.float32)
    actions[:pass_rows, 0] = 1.0
    actions[pass_rows:, 1] = 1.0
    return build_continuation_support_evidence(
        first,
        second,
        groups,
        actions,
        first_action_kind=first_action,
        actor_perspective=actor_perspective,
    )


def _policy_gate(
    action: str,
    *,
    authorized: bool = True,
    attacking_home: bool | None = None,
) -> dict:
    def profile(*, samples: int, actor_perspective: str) -> dict:
        return {
            "active": authorized,
            "authorized": authorized,
            "authority": 0.70 if authorized else 0.0,
            "action_kind": action,
            "actor_perspective": actor_perspective,
            "target_version": POLICY_UTILITY_VERSION,
            "samples": samples,
            "groups": 8,
            "model_mse": 0.08,
            "persistence_mse": 0.10,
            "skill_vs_persistence": 0.20,
            "prediction_target_correlation": 0.30,
            "metric_identity_verified": True,
            "minimum_samples": 96,
            "minimum_groups": 6,
            "minimum_skill_vs_persistence": 0.02,
            "perspective_conditioned": True,
            "perspective_profile_identity_verified": True,
        }

    if attacking_home is None:
        perspectives = {
            label: _policy_gate(
                action,
                authorized=authorized,
                attacking_home=side,
            )
            for label, side in (("home", True), ("away", False))
        }
        return {
            **profile(samples=256, actor_perspective="pooled"),
            "perspective_gates": perspectives,
        }
    label = "home" if attacking_home else "away"
    return profile(samples=128, actor_perspective=label)


def _sequence_gate(
    action: str,
    *,
    authorized: bool = True,
    attacking_home: bool | None = None,
) -> dict:
    if attacking_home is None:
        perspectives = {
            label: _sequence_gate(
                action,
                authorized=authorized,
                attacking_home=side,
            )
            for label, side in (("home", True), ("away", False))
        }
        return {
            **_policy_gate(action, authorized=authorized),
            "rollout_steps": 2,
            "perspective_gates": perspectives,
            "continuation_support_authorized": authorized,
            "continuation_policy": [],
            "reason": (
                "perspective_conditioned_changing_action_policy_utility_gain"
                if authorized else "perspective_conditioned_sequence_gate_closed"
            ),
        }
    label = "home" if attacking_home else "away"
    support = _continuation_support(
        first_action=action,
        actor_perspective=label,
    )
    policy = []
    if authorized:
        mass = support["supported_probability_mass"]
        policy = [
            {
                "action_kind": kind,
                "weight": row["empirical_probability"] / mass,
                "samples": row["samples"],
                "groups": row["groups"],
                "empirical_probability": row["empirical_probability"],
                "action_prototype": row["action_prototype"],
            }
            for kind, row in support["actions"].items()
            if row["supported"]
        ]
    return {
        **_policy_gate(
            action,
            authorized=authorized,
            attacking_home=attacking_home,
        ),
        "rollout_steps": 2,
        "attacking_home": attacking_home,
        "continuation_support_authorized": authorized,
        "continuation_policy": policy,
        "reason": (
            "grouped_heldout_changing_action_policy_utility_gain"
            if authorized else "policy_utility_sequence_gate_closed"
        ),
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


def test_perspective_policy_gate_rejects_legacy_pooled_checkpoint():
    legacy = _valid_evidence()
    legacy["actions"]["pass"].pop("perspectives")

    pooled = policy_utility_validation_gate(
        legacy, action_kind="pass",
    )
    home = policy_utility_validation_gate(
        legacy, action_kind="pass", attacking_home=True,
    )

    assert pooled["authorized"] is True
    assert pooled["perspective_conditioned"] is False
    assert home["authorized"] is False
    assert home["perspective_conditioned"] is False


def test_perspective_policy_gate_replays_metric_and_partition_identity():
    evidence = _valid_evidence()
    evidence["actions"]["pass"]["samples"] += 1
    assert policy_utility_validation_gate(
        evidence, action_kind="pass", attacking_home=True,
    )["authorized"] is False

    evidence = _valid_evidence()
    evidence["actions"]["pass"]["perspectives"]["home"][
        "skill_vs_persistence"
    ] = 0.30
    gate = policy_utility_validation_gate(
        evidence, action_kind="pass", attacking_home=True,
    )
    assert gate["authorized"] is False
    assert gate["metric_identity_verified"] is False


def test_sequence_utility_gate_requires_joint_changing_action_evidence():
    one_step = _valid_evidence()
    assert policy_utility_sequence_validation_gate(
        one_step, action_kind="pass",
    )["authorized"] is False
    two_step = copy.deepcopy({
        **one_step,
        "objective_scope": "two_step_policy_utility",
        "rollout_steps": 2,
        "action_sequence": "observed_changing_actions",
        "trained_with_action_sequence_objective": True,
        "grouped_holdout": True,
    })
    two_step["actions"]["pass"]["continuation_support"] = (
        _continuation_support()
    )
    two_step["actions"]["pass"]["perspectives"]["home"][
        "continuation_support"
    ] = _continuation_support(actor_perspective="home")
    two_step["actions"]["pass"]["perspectives"]["away"][
        "continuation_support"
    ] = _continuation_support(actor_perspective="away")
    opened = policy_utility_sequence_validation_gate(
        two_step, action_kind="pass",
    )
    assert opened["authorized"] is True
    changed_depth = policy_utility_sequence_validation_gate(
        two_step, action_kind="pass", rollout_steps=3,
    )
    assert changed_depth["authorized"] is False


def test_continuation_support_gate_rejects_unsupported_mass_and_tampering():
    first = np.asarray(["pass"] * 128)
    second = np.asarray(["pass"] * 96 + ["other"] * 32)
    groups = np.asarray([f"g{i % 8}" for i in range(128)])
    actions = np.zeros((128, 18), dtype=np.float32)
    actions[:96, 0] = 1.0
    evidence = build_continuation_support_evidence(
        first, second, groups, actions, first_action_kind="pass",
    )
    closed = continuation_support_validation_gate(
        evidence, first_action_kind="pass", expected_samples=128,
    )
    assert closed["authorized"] is False
    assert closed["reason"] == "continuation_support_mass_insufficient"

    tampered = _continuation_support()
    tampered["actions"]["pass"]["action_prototype"][0] = float("nan")
    rejected = continuation_support_validation_gate(
        tampered, first_action_kind="pass", expected_samples=128,
    )
    assert rejected["authorized"] is False
    assert rejected["reason"] == "continuation_support_contract_invalid"

    mislabeled = _continuation_support(actor_perspective="home")
    rejected = continuation_support_validation_gate(
        mislabeled,
        first_action_kind="pass",
        expected_samples=128,
        actor_perspective="away",
    )
    assert rejected["authorized"] is False
    assert rejected["reason"] == "continuation_support_contract_invalid"


def test_training_validation_reports_separate_action_branches():
    target = np.linspace(-0.3, 0.3, 256)
    predicted = np.stack([target + 0.01, target - 0.01])
    actions = np.array(["pass"] * 128 + ["hold"] * 128)
    groups = np.array([f"g{i % 8}" for i in range(256)])
    continuation_kinds = np.array(
        (["pass"] * 96 + ["shot"] * 32) * 2
    )
    continuation_actions = np.zeros((256, 18), dtype=np.float32)
    continuation_actions[continuation_kinds == "pass", 0] = 1.0
    continuation_actions[continuation_kinds == "shot", 1] = 1.0

    report = _policy_utility_validation(
        predicted, target, actions, groups,
        configured_loss_weight=0.25, optimization_steps=4,
    )

    assert report["target_version"] == POLICY_UTILITY_VERSION
    assert report["actions"]["pass"]["samples"] == 128
    assert report["actions"]["hold"]["samples"] == 128
    assert report["actions"]["cross"]["samples"] == 0

    sequence_report = _policy_utility_validation(
        predicted,
        target,
        actions,
        groups,
        configured_loss_weight=0.25,
        optimization_steps=4,
        objective_scope="two_step_policy_utility",
        rollout_steps=2,
        action_sequence="observed_changing_actions",
        trained_with_action_sequence_objective=True,
        continuation_action_kinds=continuation_kinds,
        continuation_actions=continuation_actions,
    )
    assert sequence_report["grouped_holdout"] is True
    gate = policy_utility_sequence_validation_gate(
        sequence_report, action_kind="pass",
    )
    assert gate["authorized"] is True
    assert [row["action_kind"] for row in gate["continuation_policy"]] == [
        "pass", "shot",
    ]


def test_perspective_validation_keeps_continuations_and_targets_separate():
    target = np.linspace(-0.3, 0.3, 256)
    predicted = np.stack([target + 0.01, target - 0.01])
    actions = np.asarray(["pass"] * 256)
    groups = np.asarray([f"g{i % 8}" for i in range(256)])
    attacking_home = np.asarray([True] * 128 + [False] * 128)
    continuation_kinds = np.asarray(
        ["pass"] * 96 + ["shot"] * 32
        + ["pass"] * 126 + ["shot"] * 2
    )
    continuation_actions = np.zeros((256, 18), dtype=np.float32)
    continuation_actions[continuation_kinds == "pass", 0] = 1.0
    continuation_actions[continuation_kinds == "shot", 1] = 1.0
    continuation_actions[:128, 6] = 0.20
    continuation_actions[128:, 6] = 0.80

    report = _policy_utility_validation(
        predicted,
        target,
        actions,
        groups,
        configured_loss_weight=0.25,
        optimization_steps=4,
        objective_scope="two_step_policy_utility",
        rollout_steps=2,
        action_sequence="observed_changing_actions",
        trained_with_action_sequence_objective=True,
        continuation_action_kinds=continuation_kinds,
        continuation_actions=continuation_actions,
        attacking_home=attacking_home,
    )

    gate = policy_utility_sequence_validation_gate(
        report, action_kind="pass",
    )
    assert gate["authorized"] is True
    assert gate["perspective_conditioned"] is True
    home = gate["perspective_gates"]["home"]
    away = gate["perspective_gates"]["away"]
    assert [row["action_kind"] for row in home["continuation_policy"]] == [
        "pass", "shot",
    ]
    assert [row["action_kind"] for row in away["continuation_policy"]] == [
        "pass",
    ]
    assert home["continuation_policy"][0]["action_prototype"][6] == (
        pytest.approx(0.20)
    )
    assert away["continuation_policy"][0]["action_prototype"][6] == (
        pytest.approx(0.80)
    )


def test_two_step_policy_utility_preserves_member_bootstrap_routing():
    current = torch.zeros((3, OBS_DIM), dtype=torch.float32)
    current[:, 209] = 1.0
    current[:, -1] = 1.0
    predicted = current.unsqueeze(0).repeat(2, 1, 1).clone().requires_grad_()
    target = current.clone()
    target[:, 200] = torch.tensor([0.2, 0.3, 0.4])
    predicted_utility = transition_policy_utility_tensor(
        current, predicted,
    )["policy_utility"]
    target_utility = transition_policy_utility_tensor(
        current, target,
    )["policy_utility"]
    bootstrap = torch.tensor([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    loss, _ = _bootstrap_transition_loss(
        predicted_utility.unsqueeze(-1),
        target_utility.unsqueeze(-1),
        torch.ones(1),
        bootstrap,
    )
    loss.backward()

    assert predicted.grad is not None
    assert predicted.grad[0, :2, 200].abs().sum() > 0
    assert predicted.grad[0, 2, 200] == 0
    assert predicted.grad[1, :2, 200].abs().sum() == 0
    assert predicted.grad[1, 2, 200].abs() > 0


def test_sealed_evaluation_builds_independent_two_step_policy_gate(
    monkeypatch, tmp_path,
):
    import src.data_engine.dataset_registry as dataset_registry
    import src.match_engine.world_model.recorder as recorder

    groups_total = 12
    rows_per_group = 17
    row_count = groups_total * rows_per_group
    observations = np.zeros((row_count, OBS_DIM), dtype=np.float32)
    observations[:, 209] = 1.0
    observations[: 6 * rows_per_group, -1] = 1.0
    groups = []
    for group_index in range(groups_total):
        start = group_index * rows_per_group
        stop = start + rows_per_group
        positions = np.square(np.arange(rows_per_group + 1)) * 0.001
        observations[start:stop, 200] = positions[:-1]
        groups.extend([f"sealed-{group_index}"] * rows_per_group)
    next_observations = observations.copy()
    actions = np.zeros((row_count, 18), dtype=np.float32)
    actions[:, 0] = 1.0
    for group_index in range(groups_total):
        start = group_index * rows_per_group
        stop = start + rows_per_group
        positions = np.square(np.arange(rows_per_group + 1)) * 0.001
        next_observations[start:stop, 200] = positions[1:]
        actions[start:stop, 10] = positions[1:]

    monkeypatch.setattr(dataset_registry, "load_manifest", lambda _path: {})
    monkeypatch.setattr(
        dataset_registry,
        "verify_trace_manifest",
        lambda _manifest, *, base_dir: tmp_path,
    )
    monkeypatch.setattr(
        dataset_registry,
        "files_for_split",
        lambda _manifest, _splits: [],
    )
    monkeypatch.setattr(
        recorder,
        "load_trace_batches",
        lambda *_args, **_kwargs: (
            observations,
            actions,
            next_observations,
            np.asarray(groups),
        ),
    )

    class Model:
        transition_member_count = 3

        @staticmethod
        def transition_predictions(current, clean_actions):
            future = current.clone()
            future[:, 200] = clean_actions[:, 10]
            members = future.unsqueeze(0).repeat(3, 1, 1)
            return torch.zeros((len(current), 1)), members

        @staticmethod
        def transition_rollout_predictions(current, sequences):
            future = current.clone()
            future[:, 200] = sequences[:, 1, 10]
            return future.unsqueeze(0).repeat(3, 1, 1)

    runtime = SimpleNamespace(
        model=Model(),
        meta={"policy_utility_training": {
            "configured_loss_weight": 0.25,
            "optimization_steps": 8,
            "two_step_optimization_steps": 8,
            "trained_with_action_sequence_objective": True,
        }},
    )

    report = _sealed_evaluation(
        runtime,
        tmp_path / "traces",
        tmp_path / "manifest.json",
    )

    assert report["two_step"]["samples"] == 192
    assert report["policy_utility"]["gates"]["pass"]["authorized"] is True
    sequence = report["policy_utility_two_step"]
    assert sequence["actions"]["pass"]["samples"] == 192
    assert sequence["actions"]["pass"]["groups"] == 12
    assert sequence["gates"]["pass"]["reason"] == (
        "perspective_conditioned_changing_action_policy_utility_gain"
    ), sequence["gates"]["pass"]
    assert sequence["gates"]["pass"]["authorized"] is True
    assert all(
        gate["authorized"]
        for gate in sequence["gates"]["pass"]["perspective_gates"].values()
    )
    assert sequence["sealed_test"] is True


def test_vectorized_policy_labels_do_not_alias_intercepts_or_unknowns_to_pass():
    actions = np.zeros((6, 18), dtype=np.float32)
    for index in range(5):
        actions[index, index] = 1.0
    actions[5, 0] = np.nan

    assert decode_action_kinds(actions).tolist() == [
        "pass", "shot", "hold", "cross", "intercept", "other",
    ]


class _M2Runtime:
    def __init__(self, *, gate_open: bool):
        self.cfg = SimpleNamespace(
            planner_blend=0.30,
            shot_planner_blend=0.25,
            cross_planner_blend=0.20,
        )
        self.gate_open = gate_open
        self.last_uncertainty = 0.1
        self.sequence_calls = []

    def encode_state(self, state, *, attacking_home):
        return _observation(attacking_home=attacking_home)

    def planner_confidence(self, observation, kind="general"):
        return 0.8 if kind == "pass" else 0.0

    def policy_utility_authority(
        self, action_kind, *, attacking_home=None,
    ):
        return _policy_gate(
            action_kind,
            authorized=self.gate_open and action_kind == "pass",
            attacking_home=attacking_home,
        )

    def policy_utility_sequence_authority(
        self, action_kind, *, rollout_steps=2, attacking_home=None,
    ):
        return _sequence_gate(
            action_kind,
            authorized=(
                self.gate_open and action_kind == "pass" and rollout_steps == 2
            ),
            attacking_home=attacking_home,
        )

    def two_step_planning_gate(self):
        return {
            "active": self.gate_open,
            "authority": 0.5 if self.gate_open else 0.0,
            "reason": (
                "trained_calibrated_grouped_two_step_gain"
                if self.gate_open else "two_step_evidence_gate_closed"
            ),
        }

    def predict_policy_utility_sequence(
        self, observation, actions, *, action_kind, attacking_home,
    ):
        sequence = [decode_action_kind(action) for action in actions]
        self.sequence_calls.append(sequence)
        return {
            "prediction_source": "explicit_changing_action_sequence_rollout",
            "planning_mode": "open_loop_evidence_supported_continuation_policy",
            "rollout_steps": 2,
            "action_sequence": sequence,
            "policy_utility": (
                0.30 if sequence[0] == "pass" else 0.0
            ) + (0.04 if sequence[1] == "pass" else 0.0),
            "policy_utility_version": POLICY_UTILITY_VERSION,
            "uncertainty": 0.1,
            "sequence_gate": self.policy_utility_sequence_authority(
                action_kind,
                rollout_steps=2,
                attacking_home=attacking_home,
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

    runtime = _M2Runtime(gate_open=True)
    state = _planner_state()
    opened = action_imagination_adjustments(
        runtime, state, carrier, True,
        base, labels, dist_goal=0.5, feasible_actions={"pass", "hold"},
    )
    closed = action_imagination_adjustments(
        _M2Runtime(gate_open=False), _planner_state(), carrier, True,
        base, labels, dist_goal=0.5, feasible_actions={"pass", "hold"},
    )

    assert opened[0] > 0.0
    assert np.array_equal(closed, base)
    assert runtime.sequence_calls == [
        ["pass", "pass"],
        ["pass", "shot"],
    ]
    gate = state._wm_pending_direct_action_adoption["quality_gates"]["pass"]
    assert gate["value_source"] == "outcome_aligned_two_step_policy_utility"
    assert gate["comparison_design"] == "action_vs_zero_persistence_reference"
    assert gate["rollout_steps"] == 2
    assert [row["weight"] for row in gate["continuation_policy"]] == [
        pytest.approx(0.75), pytest.approx(0.25),
    ]
    assert gate["hold_value"] == 0.0
    assert gate["hold_value_gate"]["reason"] == (
        "exact_zero_persistence_reference"
    )


def _pass_target_state():
    carrier = SimpleNamespace(
        team_id="home", player_id="carrier", on_pitch=True,
        position=np.array([0.5, 0.5]),
    )
    receivers = [
        SimpleNamespace(
            team_id="home", player_id=f"receiver-{index}", on_pitch=True,
            position=np.array([target_x, 0.5]),
        )
        for index, target_x in enumerate((0.35, 0.80))
    ]
    team = SimpleNamespace(players=[carrier, *receivers])
    state = _planner_state()
    state.team = lambda team_id: team
    return state, carrier, receivers


def test_m2_pass_target_ranking_uses_the_same_supported_sequence_gate(
    monkeypatch,
):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    monkeypatch.setenv("MATCH_WM_OUTCOME_ALIGNED_POLICY", "1")
    runtime = _M2Runtime(gate_open=True)
    state, carrier, receivers = _pass_target_state()
    metadata = [
        (
            receiver, "through", receiver.position.copy(),
            0.8, 0.1, 0.0, 0.75,
        )
        for receiver in receivers
    ]

    def ranked_sequence(
        observation, actions, *, action_kind, attacking_home,
    ):
        sequence = [decode_action_kind(action) for action in actions]
        runtime.sequence_calls.append(sequence)
        return {
            "prediction_source": "explicit_changing_action_sequence_rollout",
            "planning_mode": "open_loop_evidence_supported_continuation_policy",
            "rollout_steps": 2,
            "action_sequence": sequence,
            "policy_utility": float(actions[0, 6]),
            "policy_utility_version": POLICY_UTILITY_VERSION,
            "uncertainty": 0.1,
                "sequence_gate": runtime.policy_utility_sequence_authority(
                    action_kind,
                    rollout_steps=2,
                    attacking_home=attacking_home,
                ),
        }

    runtime.predict_policy_utility_sequence = ranked_sequence
    base = np.array([0.5, 0.5], dtype=float)

    adjusted, evidence = pass_candidate_policy_probabilities(
        runtime, state, carrier, metadata, True, base,
    )

    assert adjusted[1] > base[1]
    assert runtime.sequence_calls == [
        ["pass", "pass"],
        ["pass", "shot"],
        ["pass", "pass"],
        ["pass", "shot"],
    ]
    assert evidence["value_source"] == (
        "outcome_aligned_two_step_policy_utility"
    )
    assert evidence["planning_mode"] == (
        "open_loop_evidence_supported_continuation_policy"
    )
    assert evidence["policy_utility_gate"]["rollout_steps"] == 2


def test_m2_pass_target_ranking_fails_closed_without_sequence_runtime(
    monkeypatch,
):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    monkeypatch.setenv("MATCH_WM_OUTCOME_ALIGNED_POLICY", "1")
    runtime = _M2Runtime(gate_open=True)
    runtime.predict_policy_utility_sequence = None
    state, carrier, receivers = _pass_target_state()
    metadata = [
        (
            receiver, "through", receiver.position.copy(),
            0.8, 0.1, 0.0, 0.75,
        )
        for receiver in receivers
    ]
    base = np.array([0.4, 0.6], dtype=float)

    adjusted, evidence = pass_candidate_policy_probabilities(
        runtime, state, carrier, metadata, True, base,
    )

    assert np.array_equal(adjusted, base)
    assert evidence["applied"] is False
    assert evidence["reason"] == (
        "policy_utility_sequence_runtime_contract_missing"
    )


def test_m2_action_control_fails_closed_without_sequence_runtime(monkeypatch):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    monkeypatch.setenv("MATCH_WM_OUTCOME_ALIGNED_POLICY", "1")
    runtime = _M2Runtime(gate_open=True)
    runtime.policy_utility_sequence_authority = None
    state = _planner_state()
    base = np.zeros(4)

    adjusted = action_imagination_adjustments(
        runtime,
        state,
        SimpleNamespace(team_id="home", position=np.array([0.5, 0.5])),
        True,
        base,
        ["pass", "shot", "cross", "hold"],
        dist_goal=0.5,
        feasible_actions={"pass", "hold"},
    )

    assert np.array_equal(adjusted, base)
    gate = state._wm_pending_direct_action_adoption["quality_gates"]["pass"]
    assert gate["open"] is False
    assert gate["reason"] == "policy_utility_sequence_runtime_contract_missing"


def test_m2_action_control_rejects_invalid_evidence_continuation_prototype(
    monkeypatch,
):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    monkeypatch.setenv("MATCH_WM_OUTCOME_ALIGNED_POLICY", "1")
    runtime = _M2Runtime(gate_open=True)
    original_authority = runtime.policy_utility_sequence_authority

    def invalid_authority(
        action_kind, *, rollout_steps=2, attacking_home=None,
    ):
        gate = original_authority(
            action_kind,
            rollout_steps=rollout_steps,
            attacking_home=attacking_home,
        )
        gate["continuation_policy"][0]["action_prototype"] = []
        return gate

    runtime.policy_utility_sequence_authority = invalid_authority
    state = _planner_state()
    base = np.zeros(4)

    adjusted = action_imagination_adjustments(
        runtime,
        state,
        SimpleNamespace(team_id="home", position=np.array([0.5, 0.5])),
        True,
        base,
        ["pass", "shot", "cross", "hold"],
        dist_goal=0.5,
        feasible_actions={"pass", "hold"},
    )

    assert np.array_equal(adjusted, base)
    assert runtime.sequence_calls == []
    gate = state._wm_pending_direct_action_adoption["quality_gates"]["pass"]
    assert gate["open"] is False
    assert gate["reason"] == "evidence_supported_continuation_policy_invalid"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("policy_utility_version", "stale", "sequence_prediction_identity_or_gate_invalid"),
        ("action_sequence", ["pass", "shot"], "sequence_prediction_identity_or_gate_invalid"),
        ("uncertainty", float("nan"), "sequence_prediction_non_finite"),
    ],
)
def test_m2_action_control_rejects_stale_or_nonfinite_sequence_predictions(
    monkeypatch, field, value, reason,
):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    monkeypatch.setenv("MATCH_WM_OUTCOME_ALIGNED_POLICY", "1")
    runtime = _M2Runtime(gate_open=True)
    original = runtime.predict_policy_utility_sequence

    def corrupted(*args, **kwargs):
        prediction = original(*args, **kwargs)
        prediction[field] = value
        return prediction

    runtime.predict_policy_utility_sequence = corrupted
    state = _planner_state()
    base = np.zeros(4)
    adjusted = action_imagination_adjustments(
        runtime,
        state,
        SimpleNamespace(team_id="home", position=np.array([0.5, 0.5])),
        True,
        base,
        ["pass", "shot", "cross", "hold"],
        dist_goal=0.5,
        feasible_actions={"pass", "hold"},
    )

    assert np.array_equal(adjusted, base)
    gate = state._wm_pending_direct_action_adoption["quality_gates"]["pass"]
    assert gate["open"] is False
    assert gate["reason"] == reason


def test_runtime_sequence_prediction_uses_explicit_clean_member_rollout():
    class SequenceModel:
        transition_ensemble_trained = True

        def __init__(self):
            self.actions = None

        def transition_rollout_predictions(self, observations, actions):
            self.actions = actions.detach().clone()
            assert tuple(observations.shape) == (1, OBS_DIM)
            assert tuple(actions.shape) == (1, 2, 18)
            members = observations.unsqueeze(0).repeat(3, 1, 1)
            members[0, :, 200] += 0.10
            members[1, :, 200] += 0.20
            members[2, :, 200] += 0.30
            return members

    model = SequenceModel()
    runtime = WorldModelRuntime.__new__(WorldModelRuntime)
    runtime.model = model
    runtime.progress_aleatoric_scale = 0.05
    runtime.last_decision_uncertainty = 1.0
    runtime.last_uncertainty = 1.0
    home_support = _continuation_support(actor_perspective="home")
    away_support = _continuation_support(actor_perspective="away")
    runtime.meta = {
        "validation": {
            "policy_utility_two_step": {
                "version": 1,
                "target_version": POLICY_UTILITY_VERSION,
                "trained_with_policy_utility_objective": True,
                "configured_loss_weight": 0.2,
                "optimization_steps": 8,
                "objective_scope": "two_step_policy_utility",
                "rollout_steps": 2,
                "action_sequence": "observed_changing_actions",
                "trained_with_action_sequence_objective": True,
                "grouped_holdout": True,
                "actions": {
                    "pass": {
                        "samples": 256,
                        "groups": 8,
                        "model_mse": 0.02,
                        "persistence_mse": 0.04,
                        "skill_vs_persistence": 0.5,
                        "prediction_target_correlation": 0.4,
                        "perspectives": {
                            "home": {
                                "samples": 128,
                                "groups": 8,
                                "model_mse": 0.02,
                                "persistence_mse": 0.04,
                                "skill_vs_persistence": 0.5,
                                "prediction_target_correlation": 0.4,
                                "continuation_support": home_support,
                            },
                            "away": {
                                "samples": 128,
                                "groups": 8,
                                "model_mse": 0.02,
                                "persistence_mse": 0.04,
                                "skill_vs_persistence": 0.5,
                                "prediction_target_correlation": 0.4,
                                "continuation_support": away_support,
                            },
                        },
                    },
                },
            },
        },
    }
    first = encode_high_level_action(
        "pass", np.array([0.65, 0.50]), horizon_s=1.0,
    )
    second = encode_high_level_action(
        "pass", np.array([0.65, 0.50]), horizon_s=1.0,
    )
    first[14:17] = 1.0
    second[14:17] = 1.0

    prediction = runtime.predict_policy_utility_sequence(
        _observation(attacking_home=True),
        np.stack((first, second)),
        action_kind="pass",
        attacking_home=True,
    )

    assert prediction["prediction_source"] == (
        "explicit_changing_action_sequence_rollout"
    )
    assert prediction["planning_mode"] == (
        "open_loop_evidence_supported_continuation_policy"
    )
    assert prediction["action_sequence"] == ["pass", "pass"]
    assert prediction["rollout_steps"] == 2
    assert prediction["horizon_s"] == pytest.approx(2.0)
    assert prediction["policy_utility"] == pytest.approx(0.07)
    assert prediction["sequence_gate"]["authorized"] is True
    assert 0.0 <= prediction["uncertainty"] <= 1.0
    assert torch.count_nonzero(model.actions[0, :, 14:17]).item() == 0


def test_runtime_sequence_prediction_rejects_malformed_actions():
    runtime = WorldModelRuntime.__new__(WorldModelRuntime)
    runtime.model = SimpleNamespace(transition_ensemble_trained=True)
    runtime.progress_aleatoric_scale = 0.05
    runtime.meta = {"validation": {}}
    with pytest.raises(ValueError, match=r"shape \[2, 18\]"):
        runtime.predict_policy_utility_sequence(
            _observation(attacking_home=True),
            np.zeros((1, 18), dtype=np.float32),
            action_kind="pass",
            attacking_home=True,
        )
    with pytest.raises(ValueError, match=r"observation must have shape \[307\]"):
        runtime.predict_policy_utility_sequence(
            np.zeros(306, dtype=np.float32),
            np.stack((
                encode_high_level_action("pass"),
                encode_high_level_action("hold"),
            )),
            action_kind="pass",
            attacking_home=True,
        )
    with pytest.raises(ValueError, match="first action does not match authority"):
        runtime.predict_policy_utility_sequence(
            _observation(attacking_home=True),
            np.stack((
                encode_high_level_action("hold"),
                encode_high_level_action("pass"),
            )),
            action_kind="pass",
            attacking_home=True,
        )


def test_m2_action_control_fails_closed_when_sequence_prediction_raises(
    monkeypatch,
):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    monkeypatch.setenv("MATCH_WM_OUTCOME_ALIGNED_POLICY", "1")
    runtime = _M2Runtime(gate_open=True)

    def failed_prediction(*_args, **_kwargs):
        raise RuntimeError("model failure")

    runtime.predict_policy_utility_sequence = failed_prediction
    state = _planner_state()
    base = np.zeros(4)
    adjusted = action_imagination_adjustments(
        runtime,
        state,
        SimpleNamespace(team_id="home", position=np.array([0.5, 0.5])),
        True,
        base,
        ["pass", "shot", "cross", "hold"],
        dist_goal=0.5,
        feasible_actions={"pass", "hold"},
    )
    assert np.array_equal(adjusted, base)
    gate = state._wm_pending_direct_action_adoption["quality_gates"]["pass"]
    assert gate["reason"] == "sequence_prediction_failed_closed"


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
    manifest = tmp_path / "data/world_model/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "data/world_model/traces").mkdir()

    class Runtime:
        pass_quality = 0.2
        cfg = SimpleNamespace(min_planner_quality=0.15)
        model = SimpleNamespace(transition_member_count=3)
        meta = {
            "dataset_manifest": "data/world_model/manifest.json",
            "sealed_test_used": False,
            "training_configuration": {
                **m2_study.FROZEN_TRAINING_CONFIGURATION,
                "dataset_manifest": "data/world_model/manifest.json",
            },
        }

        def policy_utility_authority(self, action):
            return _policy_gate(action, authorized=action == "pass")

        def policy_utility_sequence_authority(self, action, *, rollout_steps=2):
            return _sequence_gate(
                action, authorized=action == "pass" and rollout_steps == 2,
            )

        def predict_policy_utility_sequence(self, *_args, **_kwargs):
            return {}

        def two_step_planning_gate(self):
            return {"active": True, "authority": 0.3}

    monkeypatch.setattr(
        m2_study.WorldModelRuntime, "load", lambda _path: Runtime(),
    )
    monkeypatch.setattr(
        m2_study,
        "_sealed_evaluation",
        lambda *_args: {
            "two_step": {"active": True},
            "policy_utility": {
                "gates": {"pass": _policy_gate("pass")},
            },
            "policy_utility_two_step": {
                "gates": {"pass": _sequence_gate("pass")},
            },
        },
    )
    protocol = {
        "candidate": {
            "dataset_manifest": "data/world_model/manifest.json",
            "trace_dir": "data/world_model/traces",
            "required_policy_utility_branches": ["pass"],
            "optional_policy_utility_branches": ["cross"],
        },
    }

    eligibility = m2_study.candidate_eligibility(
        checkpoint, protocol, root=tmp_path,
    )

    assert eligibility["eligible"] is True
    assert eligibility["two_step_gate"]["active"] is True
    assert eligibility["dataset_manifest_identity_verified"] is True
    assert eligibility["sealed_test_unused_by_training"] is True
    assert eligibility["sealed_two_step_active"] is True
    assert eligibility["sequence_predictor_available"] is True
    assert eligibility["sealed_pass_policy_utility_gate"]["authorized"] is True
    assert eligibility["required_policy_utility_sequence_gates"]["pass"][
        "authorized"
    ] is True
    assert eligibility["sealed_pass_policy_utility_two_step_gate"][
        "authorized"
    ] is True


def test_preflight_identity_binds_protocol_manifest_and_training_code(tmp_path):
    protocol_path = tmp_path / "data/evaluation/m2.json"
    manifest_path = tmp_path / "data/world_model/manifest.json"
    trainer_path = tmp_path / "scripts/train.py"
    shared_path = (
        tmp_path
        / "src/match_engine/world_model/mirrored_policy_evaluation.py"
    )
    for path, content in (
        (manifest_path, "{}"),
        (trainer_path, "trainer = 1"),
        (shared_path, "identity = 1"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    protocol = {
        "candidate": {"dataset_manifest": "data/world_model/manifest.json"},
        "integrity": {"code_identity_files": ["scripts/train.py"]},
    }
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text("{}", encoding="utf-8")

    identity = study_preflight_identity(
        tmp_path, protocol_path, protocol, manifest_path,
    )

    assert identity["manifest_path"] == "data/world_model/manifest.json"
    assert set(identity["code_sha256"]) == {
        "scripts/train.py",
        "src/match_engine/world_model/mirrored_policy_evaluation.py",
    }
    trainer_path.write_text("trainer = 2", encoding="utf-8")
    assert study_preflight_identity(
        tmp_path, protocol_path, protocol, manifest_path,
    ) != identity


def test_current_m2_transitive_identity_reaches_match_runtime_dependencies():
    protocol_path = m2_study.DEFAULT_PROTOCOL
    protocol = m2_study.load_protocol(protocol_path)
    manifest_path = m2_study.ROOT / protocol["candidate"]["dataset_manifest"]

    identity = study_preflight_identity(
        m2_study.ROOT, protocol_path, protocol, manifest_path,
    )

    assert {
        "src/infrastructure/code_identity.py",
        "src/match_engine/match_micro_runner.py",
        "src/match_engine/internal_signals.py",
        "src/simulation/match_pipeline.py",
    }.issubset(identity["code_sha256"])
    assert list(identity["code_sha256"]) == sorted(identity["code_sha256"])


def test_candidate_qualification_receipt_is_noncausal_and_identity_bound():
    identity = {
        "checkpoint_path": "data/world_model/m2.pt",
        "checkpoint_sha256": "a" * 64,
    }
    protocol = {
        "protocol_id": "m2-mirrored-policy-v1",
        "candidate": {"required_policy_utility_branches": ["pass"]},
    }
    eligibility = {
        "eligible": True,
        "required_policy_utility_gates": {
            "pass": _policy_gate("pass"),
        },
        "required_policy_utility_sequence_gates": {
            "pass": _sequence_gate("pass"),
        },
        "sequence_predictor_available": True,
        "two_step_gate": {"active": True},
        "dataset_manifest_identity_verified": True,
        "sealed_test_unused_by_training": True,
        "training_configuration_verified": True,
        "sealed_two_step_active": True,
        "sealed_pass_policy_utility_gate": _policy_gate("pass"),
        "sealed_pass_policy_utility_two_step_gate": _sequence_gate("pass"),
        "sealed_validation": {"executed": True},
    }

    receipt = m2_study.candidate_eligibility_report(
        protocol,
        identity,
        eligibility,
    )

    assert receipt["status"] == "eligible_for_m2_execution"
    assert receipt["execution_identity"] is identity
    assert receipt["candidate_eligibility"] is eligibility
    assert receipt["formal_execution_started"] is False
    assert receipt["formal_result_available"] is False
    assert "no_policy_effect_or_outcome_claim" in receipt["claim_scope"]

    contradictory = copy.deepcopy(receipt)
    contradictory["candidate_eligibility"][
        "sealed_pass_policy_utility_two_step_gate"
    ]["authorized"] = False
    with pytest.raises(ValueError, match="closed qualification gate"):
        m2_study.validate_m2_candidate_receipt(contradictory, protocol)

    mislabeled = copy.deepcopy(receipt)
    mislabeled["candidate_eligibility"][
        "required_policy_utility_sequence_gates"
    ]["pass"]["perspective_gates"]["home"]["continuation_policy"][0][
        "action_prototype"
    ] = (
        [0.0, 1.0] + [0.0] * 16
    )
    with pytest.raises(ValueError, match="closed qualification gate"):
        m2_study.validate_m2_candidate_receipt(mislabeled, protocol)

    mislabeled = copy.deepcopy(receipt)
    mislabeled["candidate_eligibility"][
        "required_policy_utility_sequence_gates"
    ]["pass"]["perspective_gates"]["home"]["actor_perspective"] = "away"
    with pytest.raises(ValueError, match="closed qualification gate"):
        m2_study.validate_m2_candidate_receipt(mislabeled, protocol)


def test_candidate_eligibility_rejects_sealed_policy_failure(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"sealed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    traces = tmp_path / "traces"
    traces.mkdir()

    class Runtime:
        pass_quality = 0.2
        cfg = SimpleNamespace(min_planner_quality=0.15)
        model = SimpleNamespace(transition_member_count=3)
        meta = {
            "dataset_manifest": "manifest.json",
            "sealed_test_used": False,
            "training_configuration": {
                **m2_study.FROZEN_TRAINING_CONFIGURATION,
                "dataset_manifest": "manifest.json",
            },
        }

        def policy_utility_authority(self, action):
            return _policy_gate(action, authorized=action == "pass")

        def policy_utility_sequence_authority(self, action, *, rollout_steps=2):
            return _sequence_gate(
                action, authorized=action == "pass" and rollout_steps == 2,
            )

        def predict_policy_utility_sequence(self, *_args, **_kwargs):
            return {}

        def two_step_planning_gate(self):
            return {"active": True}

    monkeypatch.setattr(m2_study.WorldModelRuntime, "load", lambda _path: Runtime())
    monkeypatch.setattr(
        m2_study,
        "_sealed_evaluation",
        lambda *_args: {
            "two_step": {"active": True},
            "policy_utility": {
                "gates": {"pass": _policy_gate("pass", authorized=False)},
            },
            "policy_utility_two_step": {
                "gates": {"pass": _sequence_gate("pass")},
            },
        },
    )
    protocol = {"candidate": {
        "dataset_manifest": "manifest.json",
        "trace_dir": "traces",
        "required_policy_utility_branches": ["pass"],
        "optional_policy_utility_branches": ["cross"],
    }}

    eligibility = m2_study.candidate_eligibility(
        checkpoint, protocol, root=tmp_path,
    )

    assert eligibility["eligible"] is False
    assert eligibility["sealed_two_step_active"] is True
    assert eligibility["sealed_pass_policy_utility_gate"]["authorized"] is False


def test_candidate_eligibility_rejects_missing_sequence_authority(
    monkeypatch, tmp_path,
):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"sealed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    traces = tmp_path / "traces"
    traces.mkdir()

    class LegacyRuntime:
        pass_quality = 0.2
        cfg = SimpleNamespace(min_planner_quality=0.15)
        model = SimpleNamespace(transition_member_count=3)
        meta = {
            "dataset_manifest": "manifest.json",
            "sealed_test_used": False,
            "training_configuration": {
                **m2_study.FROZEN_TRAINING_CONFIGURATION,
                "dataset_manifest": "manifest.json",
            },
        }

        def policy_utility_authority(self, action):
            return _policy_gate(action, authorized=action == "pass")

        def two_step_planning_gate(self):
            return {"active": True}

    monkeypatch.setattr(
        m2_study.WorldModelRuntime,
        "load",
        lambda _path: LegacyRuntime(),
    )
    protocol = {"candidate": {
        "dataset_manifest": "manifest.json",
        "trace_dir": "traces",
        "required_policy_utility_branches": ["pass"],
        "optional_policy_utility_branches": ["cross"],
    }}

    eligibility = m2_study.candidate_eligibility(
        checkpoint, protocol, root=tmp_path,
    )

    assert eligibility["eligible"] is False
    sequence_gate = eligibility["required_policy_utility_sequence_gates"]["pass"]
    assert sequence_gate["authorized"] is False
    assert sequence_gate["reason"] == (
        "policy_utility_sequence_authority_unavailable"
    )
    assert eligibility["sequence_predictor_available"] is False


def test_formal_rows_prove_checkpoint_and_one_sided_runtime_identity():
    signature = "sha256:" + "a" * 64
    baseline = [{
        "wm_runtime_loaded": False,
        "wm_checkpoint_signature": None,
        "wm_control_scope": "none",
        "wm_outcome_aligned_policy": False,
    }]
    home = [{
        "wm_runtime_loaded": True,
        "wm_checkpoint_signature": signature,
        "wm_control_scope": "home",
        "wm_outcome_aligned_policy": True,
    }]
    away = [{
        "wm_runtime_loaded": True,
        "wm_checkpoint_signature": signature,
        "wm_control_scope": "away",
        "wm_outcome_aligned_policy": True,
    }]

    audit = m2_study.verify_runtime_row_identity(
        baseline, home, away, checkpoint_sha256="a" * 64,
    )

    assert audit["verified"] is True
    corrupted = [dict(home[0], wm_control_scope="both")]
    with pytest.raises(ValueError, match="runtime identity"):
        m2_study.verify_runtime_row_identity(
            baseline, corrupted, away, checkpoint_sha256="a" * 64,
        )


def test_complete_m2_analysis_positive_path_is_reachable_and_fully_audited(
    monkeypatch,
):
    protocol = m2_study.load_protocol(m2_study.DEFAULT_PROTOCOL)
    baseline, home_rows, away_rows = [], [], []
    signature = "sha256:" + "a" * 64
    for home_team, away_team in protocol["design"]["fixtures"]:
        fixture = f"{home_team}_vs_{away_team}"
        for sample_index in range(protocol["design"]["samples_per_fixture"]):
            base = _study_row(
                sample=sample_index, home_xg=1.0, away_xg=1.0,
            )
            base.update({
                "fixture": fixture,
                "wm_runtime_loaded": False,
                "wm_checkpoint_signature": None,
                "wm_control_scope": "none",
                "wm_outcome_aligned_policy": False,
            })
            home = _study_row(
                sample=sample_index, home_xg=1.2, away_xg=1.0,
            )
            home.update({
                "fixture": fixture,
                "wm_runtime_loaded": True,
                "wm_checkpoint_signature": signature,
                "wm_control_scope": "home",
                "wm_outcome_aligned_policy": True,
            })
            away = _study_row(
                sample=sample_index, home_xg=1.0, away_xg=1.2,
            )
            away.update({
                "fixture": fixture,
                "wm_runtime_loaded": True,
                "wm_checkpoint_signature": signature,
                "wm_control_scope": "away",
                "wm_outcome_aligned_policy": True,
            })
            baseline.append(base)
            home_rows.append(home)
            away_rows.append(away)
    identity = {"checkpoint_sha256": "a" * 64}
    state = {
        "execution_identity": identity,
        "candidate_eligibility": {"eligible": True},
        "arms": {
            "M0": {"rows": baseline},
            "M2_home": {"rows": home_rows},
            "M2_away": {"rows": away_rows},
        },
    }
    monkeypatch.setattr(
        m2_study,
        "_continuous_loss_interval",
        lambda *_args, **_kwargs: {
            "method": "synthetic_verified_external_loss",
            "point_delta": 0.0,
            "ci95_low": -0.5,
            "ci95_high": 0.5,
        },
    )

    result = m2_study.analyze(
        protocol, state, expected_identity=identity,
    )

    assert result["decision"] == "promotion_supported"
    assert result["promotion_supported"] is True
    assert all(result["promotion_gates"].values())
    assert result["primary_controlled_micro_xg_effect"][
        "point_estimate"
    ] == pytest.approx(0.2)
    assert result["mechanism_attributable_coverage"]["outcomes"] == 1200
    assert result["experimental_unit_identity"] == {
        "verified": True,
        "required_complete": True,
        "expected_units_per_arm": 120,
        "observed_units_by_arm": {
            "M0": 120, "M2_home": 120, "M2_away": 120,
        },
    }

    interval_reports = iter([
        {
            "point_estimate": 0.05,
            "ci95_low": 0.01,
            "ci95_high": 0.09,
        },
        {"point_estimate": 0.0, "ci95_low": -0.1, "ci95_high": 0.1},
        {"point_estimate": 0.1, "ci95_low": 0.01, "ci95_high": 0.2},
        {"point_estimate": 0.04, "ci95_low": 0.03, "ci95_high": 0.05},
    ])
    monkeypatch.setattr(
        m2_study,
        "fixture_stratified_cluster_interval",
        lambda *_args, **_kwargs: next(interval_reports),
    )
    weak = m2_study.analyze(
        protocol, state, expected_identity=identity,
    )
    assert weak["primary_controlled_micro_xg_effect"]["ci95_low"] > 0.0
    assert weak["promotion_gates"][
        "controlled_micro_xg_effect_is_meaningful"
    ] is False
    assert weak["promotion_supported"] is False
    assert weak["decision"] == "research_only_default_off"


def test_m2_analysis_rejects_foreign_or_incomplete_frozen_units():
    protocol = m2_study.load_protocol(m2_study.DEFAULT_PROTOCOL)
    arms = {arm: {"rows": []} for arm in protocol["arms"]}
    arms["M0"]["rows"] = [{"fixture": "foreign", "sample_index": 0}]
    with pytest.raises(ValueError, match="foreign M2 M0"):
        m2_study.verify_frozen_unit_identity(
            protocol, arms, require_complete=False,
        )

    arms["M0"]["rows"] = []
    with pytest.raises(ValueError, match="frozen unit budget incomplete"):
        m2_study.verify_frozen_unit_identity(
            protocol, arms, require_complete=True,
        )


def test_benchmark_world_ignores_ambient_seed_and_persistence(monkeypatch):
    captured = {}

    def build(root, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(agents={}), None, {}

    monkeypatch.setattr(
        benchmark_core, "build_world_and_tournament", build,
    )
    monkeypatch.setenv("GFS_SEED", "999999")

    rows = benchmark_core.run_micro_benchmark_rows(
        root=".",
        fixtures=[("MissingHome", "MissingAway")],
        samples=3,
        seed_start=260903,
    )

    assert rows == []
    assert captured["initialization_seed"] == 260903
    assert captured["load_persistence_state"] is False


def test_power_budget_uses_historical_variance_not_candidate_mean():
    rows0, rows1 = [], []
    for index, difference in enumerate((-0.4, 0.0, 0.4)):
        base = {
            "fixture": "A_vs_B",
            "sample_index": index,
            "micro_xg_home": 1.0,
            "micro_xg_away": 1.0,
        }
        candidate = dict(base)
        candidate["micro_xg_home"] += 10.0 + difference
        rows0.append(base)
        rows1.append(candidate)
    payload = {
        "arms": {
            "M0": {"rows": rows0},
            "M1": {"rows": rows1},
        },
    }

    report = estimate_power(payload, meaningful_delta=0.1)

    assert report["historical_home_xg_margin_paired_sd"] == pytest.approx(0.4)
    assert report["normal_approximation_required_units"] == 126
    assert report["historical_candidate_mean_effect_used"] is False


def test_m2_training_preflight_requires_real_grouped_action_support():
    manifest = {
        "schema_version": 1,
        "sealed_test_policy": "evaluation-only; never train or tune",
        "files": [
            {"group": "train-a", "split": "train"},
            {"group": "dev-a", "split": "dev"},
            {"group": "sealed-a", "split": "sealed_test"},
        ],
        "summary": {"sealed_test": {"passes": 120, "groups": 6}},
    }
    train = {
        "rows": 256,
        "action_counts": {"pass": 180},
        "action_groups": {"pass": 8},
        "two_step_pairs": 64,
        "two_step_groups": 5,
        "pass_continuation_support": {"sufficient": True},
        "pass_continuation_support_by_perspective": {
            "home": {"sufficient": True},
            "away": {"sufficient": True},
        },
        "pass_utility_persistence_mse": 0.01,
        "pass_utility_standard_deviation": 0.1,
    }
    dev = {
        "rows": 128,
        "action_counts": {"pass": 100},
        "action_groups": {"pass": 6},
        "two_step_pairs": 40,
        "two_step_groups": 4,
        "pass_continuation_support": {"sufficient": True},
        "pass_continuation_support_by_perspective": {
            "home": {"sufficient": True},
            "away": {"sufficient": True},
        },
        "pass_utility_persistence_mse": 0.01,
        "pass_utility_standard_deviation": 0.1,
    }

    checks = readiness_checks(
        manifest=manifest,
        train=train,
        dev=dev,
        epochs=45,
        warmup_fraction=0.2,
        policy_utility_loss_weight=0.25,
        multi_step_loss_weight=0.25,
        transition_ensemble_size=3,
    )
    assert all(checks.values())

    insufficient = copy.deepcopy(dev)
    insufficient["action_counts"]["pass"] = 95
    failed = readiness_checks(
        manifest=manifest,
        train=train,
        dev=insufficient,
        epochs=45,
        warmup_fraction=0.2,
        policy_utility_loss_weight=0.25,
        multi_step_loss_weight=0.25,
        transition_ensemble_size=3,
    )
    assert failed["development_pass_support_sufficient"] is False

    unsupported = copy.deepcopy(dev)
    unsupported["pass_continuation_support"]["sufficient"] = False
    failed = readiness_checks(
        manifest=manifest,
        train=train,
        dev=unsupported,
        epochs=45,
        warmup_fraction=0.2,
        policy_utility_loss_weight=0.25,
        multi_step_loss_weight=0.25,
        transition_ensemble_size=3,
    )
    assert failed["development_pass_continuation_support_sufficient"] is False
    assert failed[
        "two_step_policy_utility_sequence_objective_will_activate"
    ] is False

    unsupported = copy.deepcopy(dev)
    unsupported["pass_continuation_support_by_perspective"]["away"][
        "sufficient"
    ] = False
    failed = readiness_checks(
        manifest=manifest,
        train=train,
        dev=unsupported,
        epochs=45,
        warmup_fraction=0.2,
        policy_utility_loss_weight=0.25,
        multi_step_loss_weight=0.25,
        transition_ensemble_size=3,
    )
    assert failed["development_pass_perspective_support_sufficient"] is False
    assert failed[
        "two_step_policy_utility_sequence_objective_will_activate"
    ] is False


def test_m2_execution_identity_is_portable_complete_and_root_bounded(tmp_path):
    root = tmp_path / "project"
    protocol_path = root / "data/evaluation/protocol.json"
    checkpoint = root / "data/world_model/candidate.pt"
    code = root / "src/controller.py"
    input_path = root / "data/calibration/target.json"
    for path, content in (
        (protocol_path, b"{}"),
        (checkpoint, b"checkpoint"),
        (code, b"controller"),
        (input_path, b"{}"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    protocol = {
        "integrity": {
            "code_identity_files": ["src/controller.py"],
            "input_identity_globs": ["data/calibration/*.json"],
        },
    }

    identity = study_execution_identity(
        root, protocol_path, protocol, checkpoint,
    )

    assert identity["protocol_path"] == "data/evaluation/protocol.json"
    assert identity["checkpoint_path"] == "data/world_model/candidate.pt"
    assert set(identity["code_sha256"]) == {"src/controller.py"}
    assert set(identity["input_sha256"]) == {"data/calibration/target.json"}
    code.write_bytes(b"changed")
    assert study_execution_identity(
        root, protocol_path, protocol, checkpoint,
    ) != identity

    external = tmp_path / "outside.pt"
    external.write_bytes(b"outside")
    with pytest.raises(ValueError, match="contained by the project root"):
        study_execution_identity(root, protocol_path, protocol, external)


def test_m2_protocol_validator_rejects_preregistered_threshold_drift():
    protocol = m2_study.load_protocol(m2_study.DEFAULT_PROTOCOL)
    assert m2_study.validate_protocol(protocol) == {
        "fixtures": 6, "units": 120, "runs": 360,
    }
    assert protocol["integrity"]["code_identity_mode"] == (
        "transitive_local_imports_v1"
    )
    assert protocol["integrity"]["identity_amendment"]["version"] == 5
    assert protocol["candidate"]["required_sealed_validation"] == [
        "two_step",
        "policy_utility.pass",
        "policy_utility.pass.perspectives.home",
        "policy_utility.pass.perspectives.away",
        "policy_utility_two_step.pass",
        "policy_utility_two_step.pass.perspectives.home.continuation_support",
        "policy_utility_two_step.pass.perspectives.away.continuation_support",
    ]
    assert protocol["candidate"]["continuation_support_contract"] == {
        "source": "grouped_holdout_observed_second_actions",
        "minimum_samples": 32,
        "minimum_groups": 4,
        "minimum_supported_probability_mass": 0.95,
        "action_source": "perspective_development_mean_leakage_cleaned_vector",
        "coordinate_system": "absolute_pitch_with_attacking_home_flag",
        "runtime_partition": "attacking_home",
        "qualification_requires": ["home", "away"],
        "reference_action": "same_state_zero_transition_utility",
        "runtime_consumers": [
            "high_level_pass_utility",
            "pass_target_ranking",
        ],
    }
    changed = copy.deepcopy(protocol)
    changed["analysis"]["minimum_meaningful_delta"] = 0.09
    with pytest.raises(ValueError, match="threshold is frozen"):
        m2_study.validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["integrity"]["code_identity_mode"] = "explicit_files_v1"
    with pytest.raises(ValueError, match="transitive code identity"):
        m2_study.validate_protocol(changed)
