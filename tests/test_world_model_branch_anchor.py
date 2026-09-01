from types import SimpleNamespace

import pytest

from src.match_engine.match_micro_runner import run_match_micro_simulation
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.world_model.config import (
    world_model_plan_clock, world_model_plan_enabled,
)
from src.simulation.runtime import environment_override
from tests.test_affective_phase1b import _FakeAgent


def test_world_model_policy_activates_only_at_configured_match_time():
    with environment_override({
        "MATCH_WORLD_MODEL": "1",
        "MATCH_WM_PLAN": "1",
        "MATCH_WM_PLAN_START_SEC": "60",
    }):
        assert not world_model_plan_enabled(SimpleNamespace(clock_seconds=59.999))
        assert world_model_plan_enabled(SimpleNamespace(clock_seconds=60.0))
        assert world_model_plan_enabled()  # compatibility/configuration query


def test_world_model_plan_clock_is_nested_and_does_not_leak():
    with environment_override({
        "MATCH_WORLD_MODEL": "1",
        "MATCH_WM_PLAN": "1",
        "MATCH_WM_PLAN_START_SEC": "60",
    }):
        assert world_model_plan_enabled()
        with world_model_plan_clock(59.0):
            assert not world_model_plan_enabled()
            with world_model_plan_clock(60.0):
                assert world_model_plan_enabled()
            assert not world_model_plan_enabled()
        assert world_model_plan_enabled()


@pytest.mark.parametrize("value", ["nan", "-1", "5401", "invalid"])
def test_world_model_policy_rejects_invalid_start_time(value):
    with environment_override({
        "MATCH_WORLD_MODEL": "1", "MATCH_WM_PLAN": "1",
        "MATCH_WM_PLAN_START_SEC": value,
    }):
        with pytest.raises(ValueError):
            world_model_plan_enabled(SimpleNamespace(clock_seconds=60.0))


def _anchored_summary(seed=7):
    cfg = MicroMatchConfig.fast_demo()
    with environment_override({
        "MATCH_WORLD_MODEL": "0",
        "MATCH_WM_BRANCH_AT_SEC": "60",
        "MATCH_AUTHORITATIVE_TICK_CLOCK": "1",
    }):
        return run_match_micro_simulation(
            _FakeAgent("Brazil"), _FakeAgent("Argentina"),
            goals_home=0, goals_away=0, xg_home=1.0, xg_away=0.8,
            config=cfg, seed=seed, writeback_agents=False,
            match_seconds=120.0,
        )


def test_deterministic_replay_produces_identical_pre_intervention_anchor():
    left = _anchored_summary()
    right = _anchored_summary()
    assert left.world_model_branch_anchor["available"]
    assert left.world_model_branch_anchor["requested_sec"] == 60.0
    assert left.world_model_branch_anchor["actual_sec"] == 60.0
    assert left.world_model_branch_anchor["engine_state_clock_sec"] == 60.0
    assert left.world_model_branch_anchor["clock_contract"] == (
        "authoritative_tick_v2"
    )
    assert left.simulation_clock["authoritative_tick_clock"]
    assert left.simulation_clock["final_state_clock_sec"] == 120.0
    assert left.world_model_branch_anchor["resume_capability"] == (
        "deterministic_replay_only"
    )
    assert left.world_model_branch_anchor["state_identity"] == (
        right.world_model_branch_anchor["state_identity"]
    )


def test_different_seed_changes_branch_identity():
    assert _anchored_summary(7).world_model_branch_anchor["state_identity"] != (
        _anchored_summary(8).world_model_branch_anchor["state_identity"]
    )


def test_runner_exposes_exact_tick_clock_and_delayed_policy_to_actions(
    monkeypatch, tmp_path,
):
    observed = []

    def record_action_boundary(**kwargs):
        observed.append((
            float(kwargs["state"].clock_seconds),
            world_model_plan_enabled(),
        ))

    monkeypatch.setattr(
        "src.match_engine.match_micro_runner._execute_tick_actions",
        record_action_boundary,
    )
    with environment_override({
        "MATCH_WORLD_MODEL": "1",
        "MATCH_WORLD_MODEL_REQUIRED": "0",
        "MATCH_WM_CHECKPOINT": str(tmp_path / "missing.pt"),
        "MATCH_WM_PLAN": "1",
        "MATCH_WM_PLAN_START_SEC": "60",
        "MATCH_WM_BRANCH_AT_SEC": "60",
        "MATCH_AUTHORITATIVE_TICK_CLOCK": "1",
    }):
        summary = run_match_micro_simulation(
            _FakeAgent("Brazil"), _FakeAgent("Argentina"),
            goals_home=0, goals_away=0, xg_home=1.0, xg_away=0.8,
            config=MicroMatchConfig.fast_demo(), seed=9,
            writeback_agents=False, match_seconds=120.0,
            base_dir=tmp_path,
        )
    assert observed == [
        (30.0, False), (60.0, True), (90.0, True), (120.0, True),
    ]
    assert summary.world_model_branch_anchor["actual_sec"] == 60.0
    assert summary.simulation_clock["final_logical_sec"] == 120.0
    assert summary.simulation_clock["final_state_clock_sec"] == 120.0
