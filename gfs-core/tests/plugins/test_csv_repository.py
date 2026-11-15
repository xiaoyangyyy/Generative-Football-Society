import pandas as pd
import pytest

from gfs_core.plugins.data.sources.csv_repository import CSVResultsRepository


def test_repository_validates_required_files_and_columns(tmp_path):
    repository = CSVResultsRepository(tmp_path)
    with pytest.raises(FileNotFoundError, match="results dataset"):
        repository.load()

    for kind in ("results", "goalscorers", "shootouts", "former_names"):
        (tmp_path / f"{kind}.csv").write_text("unexpected\nvalue\n", encoding="utf-8")
    repository = CSVResultsRepository(tmp_path)
    with pytest.raises(ValueError, match="invalid results dataset"):
        repository.load()


def test_repository_loads_and_normalizes_minimal_dataset(tmp_path):
    frames = {
        "results": pd.DataFrame({
            "date": ["2024-01-01"],
            "home_team": ["Korea Republic"],
            "away_team": ["Brazil"],
            "home_score": [1],
            "away_score": [2],
            "tournament": ["Friendly"],
        }),
        "goalscorers": pd.DataFrame({
            "date": ["2024-01-01"], "home_team": ["Korea Republic"], "away_team": ["Brazil"],
        }),
        "shootouts": pd.DataFrame({
            "date": ["2024-01-01"], "home_team": ["Korea Republic"],
            "away_team": ["Brazil"], "winner": ["Brazil"],
        }),
        "former_names": pd.DataFrame({"former": ["Old Brazil"], "current": ["Brazil"]}),
    }
    for kind, frame in frames.items():
        frame.to_csv(tmp_path / f"{kind}.csv", index=False)
    first = CSVResultsRepository(tmp_path).load()
    second = CSVResultsRepository(tmp_path).load()
    assert first.results.loc[0, "home_team"] == "South Korea"
    assert first.results.loc[0, "home_result"] == "loss"
    assert first.results.loc[0, "winner_90min"] == "Brazil"
    pd.testing.assert_frame_equal(first.results, second.results)
