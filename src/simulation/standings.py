"""Pure tournament standings rules, separated from orchestration and I/O."""

from __future__ import annotations

from typing import MutableMapping


def apply_group_result(
    standings: MutableMapping[str, MutableMapping[str, int]],
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
) -> None:
    if home not in standings or away not in standings:
        raise KeyError(f"Unknown team in standings: {home} vs {away}")
    if home == away or min(home_goals, away_goals) < 0:
        raise ValueError("A result requires distinct teams and non-negative goals")
    home_row, away_row = standings[home], standings[away]
    home_row["pts"] += 3 if home_goals > away_goals else (1 if home_goals == away_goals else 0)
    away_row["pts"] += 3 if away_goals > home_goals else (1 if home_goals == away_goals else 0)
    home_row["gf"] += home_goals
    home_row["ga"] += away_goals
    away_row["gf"] += away_goals
    away_row["ga"] += home_goals
    home_row["gd"] = home_row["gf"] - home_row["ga"]
    away_row["gd"] = away_row["gf"] - away_row["ga"]
