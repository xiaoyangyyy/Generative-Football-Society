"""Structural guards for the micro-match orchestration boundary."""

import inspect

from src.match_engine import match_micro_runner
from src.match_engine import event_schedule
from src.match_engine import micro_events
from src.match_engine.passing_engine import PassingEngine
from src.match_engine.shot_engine import ShotEngine


def test_micro_match_orchestrator_delegates_setup_and_summary():
    source_lines, _ = inspect.getsourcelines(
        match_micro_runner.run_match_micro_simulation
    )


def test_passing_step_stays_stage_driven():
    source_lines, _ = inspect.getsourcelines(PassingEngine.step)
    assert len(source_lines) <= 100
    assert {
        "_select_pass_candidate",
        "_resolve_pass_delivery",
        "_record_pass_statistics",
        "_apply_pass_outcome",
        "_record_pass_result",
    }.issubset(PassingEngine.__dict__)


def test_event_schedule_stays_stage_driven():
    source_lines, _ = inspect.getsourcelines(event_schedule.build_event_schedule)
    assert len(source_lines) <= 65
    assert all(
        hasattr(event_schedule, helper)
        for helper in (
            "_build_schedule_times",
            "_append_atmosphere_events",
            "_append_goal_events",
            "_append_shot_events",
            "_append_discipline_events",
            "_append_background_events",
        )
    )


def test_shot_resolution_stays_stage_driven():
    source_lines, _ = inspect.getsourcelines(ShotEngine.resolve_shot)
    assert len(source_lines) <= 75
    assert {
        "_select_shot_kind",
        "_simulate_shot_trajectory",
        "_resolve_shot_probabilities",
        "_record_shot_statistics",
        "_build_shot_events",
        "_accumulate_micro_xg",
        "_log_shot_path",
    }.issubset(ShotEngine.__dict__)


def test_micro_event_application_stays_stage_driven():
    source_lines, _ = inspect.getsourcelines(micro_events.apply_micro_event)
    assert len(source_lines) <= 45
    assert all(
        hasattr(micro_events, helper)
        for helper in (
            "_apply_goal_event",
            "_apply_player_event",
            "_apply_discipline_event",
            "_apply_atmosphere_event",
        )
    )
    assert len(source_lines) <= 200
    assert all(
        hasattr(match_micro_runner, helper)
        for helper in (
            "_build_micro_engines",
            "_build_temporal_adapters",
            "_initialize_micro_match_state",
            "_step_affective_discipline",
            "_step_possession_spatial",
            "_execute_tick_actions",
            "_record_tick_meso",
            "_resolve_final_micro_score",
            "_build_micro_match_summary",
        )
    )
