from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.match_engine.world_model.cross_validation import (
    CROSS_MIN_GROUPS,
    CROSS_MIN_SAMPLES,
    build_cross_action_validation,
    replay_cross_action_validation,
)
from src.match_engine.world_model.config import WorldModelConfig
from src.match_engine.world_model.inference import WorldModelRuntime
from src.match_engine.world_model.observation import OBS_DIM


def _rows(samples: int = 192):
    current = np.zeros((samples, 4), dtype=float)
    actual = np.ones((samples, 4), dtype=float)
    predicted = np.full((samples, 4), 0.7, dtype=float)
    groups = [f"match-{index % 8}" for index in range(samples)]
    return current, predicted, actual, np.ones(4), groups


def test_grouped_cross_skill_builds_replayable_bounded_quality():
    evidence = build_cross_action_validation(*_rows())
    replay = replay_cross_action_validation(evidence)

    assert evidence["eligible"] is True
    assert evidence["external_football_validity"] is False
    assert evidence["skill_vs_persistence"] > 0.0
    assert 0.0 < evidence["planner_quality"] <= 1.0
    assert replay["valid"] is True
    assert replay["quality"] == evidence["planner_quality"]


@pytest.mark.parametrize(
    ("samples", "groups", "reason"),
    [
        (CROSS_MIN_SAMPLES - 1, CROSS_MIN_GROUPS, "insufficient_cross_samples"),
        (CROSS_MIN_SAMPLES, CROSS_MIN_GROUPS - 1, "insufficient_cross_groups"),
    ],
)
def test_cross_gate_requires_sample_and_group_support(samples, groups, reason):
    current, predicted, actual, weights, _ = _rows(samples)
    labels = [f"match-{index % groups}" for index in range(samples)]
    evidence = build_cross_action_validation(
        current, predicted, actual, weights, labels,
    )

    assert evidence["eligible"] is False
    assert evidence["planner_quality"] == 0.0
    assert evidence["reason"] == reason


def test_cross_gate_rejects_model_that_does_not_beat_persistence():
    current, _, actual, weights, groups = _rows()
    evidence = build_cross_action_validation(
        current, np.full_like(current, 3.0), actual, weights, groups,
    )

    assert evidence["skill_vs_persistence"] < 0.0
    assert evidence["eligible"] is False
    assert evidence["planner_quality"] == 0.0
    assert evidence["reason"] == "cross_skill_below_minimum"


def test_cross_gate_rejects_rehashed_or_relabelled_quality_claims():
    evidence = build_cross_action_validation(*_rows())
    tampered = copy.deepcopy(evidence)
    tampered["planner_quality"] = 1.0
    relabelled = copy.deepcopy(evidence)
    relabelled["external_football_validity"] = True

    assert replay_cross_action_validation(tampered)["valid"] is False
    assert replay_cross_action_validation(tampered)["quality"] == 0.0
    assert replay_cross_action_validation(relabelled) == {
        "valid": False,
        "quality": 0.0,
        "reason": "cross_validation_contract_mismatch",
    }


def _runtime(validation):
    model = SimpleNamespace(
        checkpoint_version=9,
        transition_kind="gru",
        transition_ensemble_trained=True,
    )
    return WorldModelRuntime(
        model, WorldModelConfig(), {"validation": validation},
    )


def test_checkpoint_without_cross_evidence_has_zero_cross_authority():
    runtime = _runtime({
        "planner_quality": 0.8,
        "pass_planner_quality": 0.8,
        "shot_planner_quality": 0.8,
    })
    authority = runtime.planner_authority(
        np.ones(OBS_DIM, dtype=np.float32), kind="cross",
    )

    assert runtime.cross_quality == 0.0
    assert runtime.cross_validation["reason"] == "cross_validation_unavailable"
    assert authority["authorized"] is False
    assert authority["decision_confidence"] == 0.0


def test_only_replayable_cross_evidence_can_authorize_runtime():
    evidence = build_cross_action_validation(*_rows())
    runtime = _runtime({"cross_action_validation": evidence})
    tampered = copy.deepcopy(evidence)
    tampered["planner_quality"] = 0.99
    rejected = _runtime({"cross_action_validation": tampered})

    assert runtime.cross_quality == evidence["planner_quality"]
    assert runtime.planner_authority(
        np.ones(OBS_DIM, dtype=np.float32), kind="cross",
    )["authorized"] is True
    assert rejected.cross_quality == 0.0
