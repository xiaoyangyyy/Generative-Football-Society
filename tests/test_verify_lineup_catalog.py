from pathlib import Path

from scripts.verify_lineup_catalog import verify_lineup_catalog


def test_frozen_roster_catalog_has_only_ready_or_explicit_safe_degradation():
    report = verify_lineup_catalog(Path(__file__).resolve().parents[1])
    assert report["passed"]
    assert report["summary"] == {
        "roster_teams": 48,
        "player_level_ready": 46,
        "safe_degraded": 2,
        "invalid": 0,
    }
    degraded = {
        row["team"]: row["reason"]
        for row in report["teams"] if row["state"] == "safe_degraded"
    }
    assert degraded == {
        "Algeria": "roster_missing_goalkeeper",
        "Iran": "roster_missing_goalkeeper",
    }
    assert not report["external_calls_made"]
    assert report["matches_executed"] == 0
    assert not report["training_executed"]
