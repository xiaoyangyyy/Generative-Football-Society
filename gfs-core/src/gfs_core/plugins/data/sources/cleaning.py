from __future__ import annotations

import pandas as pd


def clean_results(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["home_score"] = pd.to_numeric(cleaned["home_score"], errors="coerce")
    cleaned["away_score"] = pd.to_numeric(cleaned["away_score"], errors="coerce")
    cleaned = cleaned.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()
    cleaned = cleaned[
        cleaned["home_team"].astype(str).str.strip()
        != cleaned["away_team"].astype(str).str.strip()
    ].copy()
    cleaned = cleaned[
        (cleaned["home_score"] >= 0) & (cleaned["away_score"] >= 0)
    ].copy()
    cleaned = cleaned.drop_duplicates(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
        ],
        keep="last",
    )
    cleaned["goal_diff_home"] = cleaned["home_score"] - cleaned["away_score"]
    cleaned["home_result"] = cleaned["goal_diff_home"].apply(_home_result)
    cleaned["away_result"] = cleaned["home_result"].map(
        {"win": "loss", "loss": "win", "draw": "draw"}
    )
    cleaned["winner_90min"] = cleaned.apply(_winner_90min, axis=1)
    cleaned["is_draw"] = cleaned["home_result"] == "draw"
    return cleaned


def _home_result(goal_difference: float) -> str:
    if goal_difference > 0:
        return "win"
    if goal_difference < 0:
        return "loss"
    return "draw"


def _winner_90min(row: pd.Series) -> str | None:
    if row["home_result"] == "win":
        return row["home_team"]
    if row["home_result"] == "loss":
        return row["away_team"]
    return None
