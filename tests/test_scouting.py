import copy

import pytest

from src.simulation.scouting import (
    append_scouting_report, scouting_observation, scouting_view,
    validate_scouting_registry,
)


def _entry(player_id):
    return {"player_id": player_id, "player": {"player_id": player_id}}


def test_baseline_and_scouted_intervals_cover_truth_and_scouting_is_narrower():
    baseline = scouting_observation(
        market_id="free-market-0003-team-pool", team="A", player_id="p1",
        true_quality=0.67, level="baseline",
    )
    scouted = scouting_observation(
        market_id="free-market-0003-team-pool", team="A", player_id="p1",
        true_quality=0.67, level="scouted",
    )
    assert baseline["quality_low"] <= 0.67 <= baseline["quality_high"]
    assert scouted["quality_low"] <= 0.67 <= scouted["quality_high"]
    assert scouted["interval_width"] < baseline["interval_width"]
    assert scouted == scouting_observation(
        market_id="free-market-0003-team-pool", team="A", player_id="p1",
        true_quality=0.67, level="scouted",
    )


def test_report_budget_is_two_idempotent_and_tamper_evident():
    registry = None
    for player_id in ("p1", "p2"):
        registry, report = append_scouting_report(
            registry, market_id="free-market-0003-team-pool", team="A",
            entry=_entry(player_id), true_quality=0.6,
        )
        assert report["observation"]["level"] == "scouted"
    same, _ = append_scouting_report(
        registry, market_id="free-market-0003-team-pool", team="A",
        entry=_entry("p1"), true_quality=0.6,
    )
    assert same == registry
    with pytest.raises(ValueError, match="budget exhausted"):
        append_scouting_report(
            registry, market_id="free-market-0003-team-pool", team="A",
            entry=_entry("p3"), true_quality=0.6,
        )
    assert scouting_view(registry)["report_count"] == 2
    tampered = copy.deepcopy(registry)
    tampered["reports"][0]["observation"]["quality_low"] = 0.8
    with pytest.raises(ValueError, match="observation interval"):
        validate_scouting_registry(tampered)
