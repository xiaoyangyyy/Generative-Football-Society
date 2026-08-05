import pytest

from scripts.benchmark_sim_vs_open_data import _bootstrap_mean_intervals
from scripts.merge_formal_benchmark_shards import merge_reports
from scripts.plan_m0_calibration import build_plan
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES


def test_bootstrap_intervals_are_deterministic_and_cover_observed_mean():
    rows = [
        {"fixture": "a", "shots_per_team_match": 8.0, "pass_completion": 0.7},
        {"fixture": "b", "shots_per_team_match": 12.0, "pass_completion": 0.8},
        {"fixture": "c", "shots_per_team_match": 16.0, "pass_completion": 0.9},
    ]
    first = _bootstrap_mean_intervals(rows, seed=42, draws=500)
    second = _bootstrap_mean_intervals(rows, seed=42, draws=500)
    assert first == second
    shots = first["shots_per_team_match"]
    assert shots["mean"] == pytest.approx(12.0)
    assert shots["ci95_low"] <= shots["mean"] <= shots["ci95_high"]
    assert "fixture" not in first


def test_formal_merge_rejects_incomplete_fixture_or_seed_coverage():
    rows = [
        {"fixture": f"{home}_vs_{away}", "pass_completion": 0.8}
        for home, away in DEFAULT_FIXTURES
    ]
    with pytest.raises(ValueError, match="three seeds"):
        merge_reports([{"raw_rows": rows}])

    with pytest.raises(ValueError, match="fixture coverage"):
        merge_reports([{"raw_rows": rows[:1] * 3}])


def test_calibration_planner_does_not_tune_a_passing_formal_baseline():
    plan = build_plan({
        "protocol": "six_fixtures_three_seeds_full_90min",
        "samples_total": 18,
        "all_pass": True,
        "failed_metrics": [],
        "failed_soft": [],
    })
    assert plan["action"] == "no_op_freeze_parameters"
    assert plan["parameter_families"] == []


def test_calibration_planner_routes_failures_to_bounded_families():
    plan = build_plan({
        "all_pass": False,
        "failed_metrics": ["shots_on_target_rate", "fouls_committed_per_team_match"],
        "failed_soft": [],
    })
    assert plan["action"] == "staged_family_search"
    assert set(plan["parameter_families"]) == {
        "discipline", "goalkeeper", "pressure", "shot_geometry",
    }
