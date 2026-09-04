from __future__ import annotations

import copy

import numpy as np
import pytest

from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.manager_plan import (
    InMatchInstruction, InMatchPlan, InMatchPlanRuntime,
)
from src.match_engine.player_match_stats import PlayerMatchStatsTracker
from src.match_engine.match_micro_runner import run_match_micro_simulation
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.tactical_profile import apply_locked_tactical_preset
from tests.test_affective_phase1b import _FakeAgent


def _state():
    return build_match_affective_state(
        _FakeAgent("Home"), _FakeAgent("Away"),
        rng=np.random.default_rng(42),
    )


def test_plan_contract_is_bounded_and_lineup_scoped():
    plan = InMatchPlan("Home", (
        InMatchInstruction("trail-60", 60, "trailing", tactic="gegenpress"),
        InMatchInstruction(
            "sub-70", 70, "drawing", substitute_off="s1", substitute_on="b1",
        ),
    ))
    plan.validate_lineup(starters=["s1", "s2"], bench=["b1"])
    assert plan.controls_substitutions is True
    assert InMatchPlan.from_payload(plan.as_dict()) == plan

    with pytest.raises(ValueError, match="minutes must be unique"):
        InMatchPlan("Home", (
            InMatchInstruction("a", 60, tactic="balanced"),
            InMatchInstruction("b", 60, tactic="gegenpress"),
        ))
    with pytest.raises(ValueError, match="frozen starting XI"):
        plan.validate_lineup(starters=["other"], bench=["b1"])


def test_runtime_evaluates_real_score_once_and_switches_tactic():
    state = _state()
    state.home.score, state.away.score = 0, 1
    plan = InMatchPlan("Home", (
        InMatchInstruction("trail-60", 60, "trailing", tactic="gegenpress"),
    ))
    runtime = InMatchPlanRuntime(plan)
    tracker = PlayerMatchStatsTracker()

    notes = runtime.step(
        state, clock_sec=60 * 60, tracker=tracker, subs_done={},
    )
    second = runtime.step(
        state, clock_sec=61 * 60, tracker=tracker, subs_done={},
    )

    assert len(notes) == 1
    assert second == []
    assert runtime.diagnostics()["applied"] == 1
    assert state.home.coach.tactical_current["pressing_intensity"] > 0.7


def test_failed_condition_is_resolved_without_late_trigger():
    state = _state()
    state.home.score, state.away.score = 1, 0
    runtime = InMatchPlanRuntime(InMatchPlan("Home", (
        InMatchInstruction("trail-60", 60, "trailing", tactic="gegenpress"),
    )))
    tracker = PlayerMatchStatsTracker()

    runtime.step(state, clock_sec=60 * 60, tracker=tracker, subs_done={})
    state.home.score, state.away.score = 1, 2
    runtime.step(state, clock_sec=70 * 60, tracker=tracker, subs_done={})

    diagnostics = runtime.diagnostics()
    assert diagnostics["skipped"] == 1
    assert diagnostics["applied"] == 0


def test_runtime_applies_exact_substitution_and_updates_limit_counter():
    state = _state()
    off_player = next(
        player for player in state.home.players
        if player.on_pitch and player.role != "GK"
    )
    on_player = copy.deepcopy(off_player)
    on_player.player_id = "Home_explicit_bench"
    on_player.name = "Explicit Bench"
    on_player.on_pitch = False
    state.home.players.append(on_player)
    runtime = InMatchPlanRuntime(InMatchPlan("Home", (
        InMatchInstruction(
            "sub-10", 10, "always",
            substitute_off=off_player.player_id,
            substitute_on=on_player.player_id,
        ),
    )))
    tracker = PlayerMatchStatsTracker()
    subs_done = {}

    runtime.step(
        state, clock_sec=10 * 60, tracker=tracker, subs_done=subs_done,
    )

    assert off_player.on_pitch is False
    assert on_player.on_pitch is True
    assert subs_done["Home"] == 1
    assert runtime.diagnostics()["outcomes"][0]["applied_substitution"] == {
        "off": off_player.player_id, "on": on_player.player_id,
    }


def test_micro_runner_executes_and_exports_frozen_plan():
    cfg = MicroMatchConfig.fast_demo()
    summary = run_match_micro_simulation(
        _FakeAgent("Home"), _FakeAgent("Away"),
        goals_home=0, goals_away=0, xg_home=0.3, xg_away=0.3,
        config=cfg, seed=91, writeback_agents=False,
        match_seconds=70.0,
        in_match_plan_home=InMatchPlan("Home", (
            InMatchInstruction("minute-one", 1, "always", tactic="gegenpress"),
        )).as_dict(),
    )

    audit = summary.in_match_management["home"]
    assert audit["applied"] == 1
    assert audit["outcomes"][0]["score_for"] >= 0
    assert any("MANAGER Home" in item for item in summary.timeline_snippet)


def test_micro_runner_exports_locked_tactical_vector_execution():
    home, away = _FakeAgent("Home"), _FakeAgent("Away")
    for agent in (home, away):
        if not isinstance(getattr(agent, "semantic_memory", None), dict):
            agent.semantic_memory = {}
    apply_locked_tactical_preset(
        home, "gegenpress", source="studio_user_intervention",
    )
    apply_locked_tactical_preset(
        away, "low_block_counter", source="studio_user_intervention",
    )

    summary = run_match_micro_simulation(
        home, away, goals_home=0, goals_away=0,
        xg_home=0.2, xg_away=0.2,
        config=MicroMatchConfig.fast_demo(), seed=117,
        writeback_agents=False, match_seconds=5.0,
    )

    home_execution = summary.tactical_execution["home"]
    away_execution = summary.tactical_execution["away"]
    assert home_execution["applied_tactic"] == "gegenpress"
    assert away_execution["applied_tactic"] == "low_block_counter"
    assert home_execution["preset_locked"] is True
    assert home_execution["source"] == "studio_user_intervention"
    assert len(home_execution["initial_vector"]) == 22
    assert set(home_execution["changed_controls"]) <= set(
        home_execution["initial_vector"]
    )
