from pathlib import Path

from gfs_core.api import run_monte_carlo

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_monte_carlo_matches_fixed_seed_reference():
    rows = run_monte_carlo(4, project_root=PROJECT_ROOT, seed=42)
    assert len(rows) == 48
    assert sum(row["wins"] for row in rows) == 4
    assert sum(row["probability"] for row in rows) == 1.0
    assert {
        row["team"]: row["wins"] for row in rows if row["wins"]
    } == {
        "Brazil": 1,
        "France": 1,
        "New Zealand": 1,
        "United States": 1,
    }
    assert rows[0]["probability"] == 0.25
    assert all("status_score" in row for row in rows)
