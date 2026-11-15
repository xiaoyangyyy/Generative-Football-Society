from pathlib import Path

import pytest

from gfs_core.api import load_status_snapshot, load_status_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def snapshot():
    return load_status_snapshot(PROJECT_ROOT)


def test_real_status_pipeline_preserves_reference_results(snapshot):
    table = snapshot.table
    assert table.shape == (333, 24)
    assert len(snapshot.team_matches) % 2 == 0
    assert set(snapshot.source.raw) == {
        "results", "goalscorers", "shootouts", "former_names",
    }
    ranked = snapshot.ranked(3)
    assert ranked.index.tolist() == ["Argentina", "Brazil", "France"]
    assert ranked.loc["Argentina", "final_status_score"] == 100.0
    assert ranked.loc["Brazil", "final_status_score"] == pytest.approx(96.4385157745)
    assert ranked.loc["France", "tier"] == "Core Power"
    assert ranked.loc["Argentina", "sample_confidence"] == "high"


def test_public_compatibility_api_returns_table_matches_and_raw_data(snapshot):
    table, team_matches, raw = load_status_table(PROJECT_ROOT)
    assert table.equals(snapshot.table)
    assert team_matches.equals(snapshot.team_matches)
    assert raw["results"].shape[0] > 40_000
    assert raw["goalscorers"].shape[0] > 40_000


def test_ranked_view_validates_limit_and_has_stable_columns(snapshot):
    assert snapshot.ranked(1).shape == (1, 7)
    with pytest.raises(ValueError, match="must be positive"):
        snapshot.ranked(0)
