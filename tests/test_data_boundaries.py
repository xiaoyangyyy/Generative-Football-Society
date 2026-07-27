import pandas as pd
import pytest

from src.data_engine.cleaner import clean_results
from src.data_engine.loader import load_data


def test_clean_results_rejects_invalid_rows_and_duplicates():
    rows = pd.DataFrame(
        [
            {"date": "2026-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0, "tournament": "Cup"},
            {"date": "2026-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0, "tournament": "Cup"},
            {"date": "bad", "home_team": "A", "away_team": "A", "home_score": -1, "away_score": 0, "tournament": "Cup"},
        ]
    )
    cleaned = clean_results(rows)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["winner_90min"] == "A"


def test_loader_reports_schema_errors_at_boundary(tmp_path):
    for name in ("results", "goalscorers", "shootouts", "former_names"):
        pd.DataFrame({"wrong": []}).to_csv(tmp_path / f"{name}.csv", index=False)
    with pytest.raises(ValueError, match="Invalid results dataset"):
        load_data(str(tmp_path))
