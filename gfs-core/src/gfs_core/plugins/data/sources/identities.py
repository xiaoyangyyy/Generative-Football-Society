from __future__ import annotations

import pandas as pd

from .aliases import TEAM_ALIASES


def normalize_identities(
    frame: pd.DataFrame, former_names: pd.DataFrame,
) -> pd.DataFrame:
    normalized = frame.copy()
    name_map = dict(zip(former_names["former"], former_names["current"]))
    normalized["home_team_original"] = normalized["home_team"]
    normalized["away_team_original"] = normalized["away_team"]
    normalized["home_team"] = normalized["home_team"].replace(name_map).replace(TEAM_ALIASES)
    normalized["away_team"] = normalized["away_team"].replace(name_map).replace(TEAM_ALIASES)
    normalized["home_score"] = normalized["home_score"].astype(float)
    normalized["away_score"] = normalized["away_score"].astype(float)
    normalized["goal_diff_home"] = normalized["home_score"] - normalized["away_score"]
    normalized["goal_diff_abs"] = normalized["goal_diff_home"].abs()
    normalized["total_goals"] = normalized["home_score"] + normalized["away_score"]
    normalized["is_draw"] = normalized["home_score"] == normalized["away_score"]
    normalized["home_result"] = normalized["goal_diff_home"].apply(_home_result)
    normalized["away_result"] = normalized["goal_diff_home"].apply(_away_result)
    normalized["winner_90min"] = normalized.apply(_winner_90min, axis=1)
    return normalized


def _home_result(goal_difference: float) -> str:
    if goal_difference > 0:
        return "win"
    if goal_difference < 0:
        return "loss"
    return "draw"


def _away_result(goal_difference: float) -> str:
    if goal_difference < 0:
        return "win"
    if goal_difference > 0:
        return "loss"
    return "draw"


def _winner_90min(row: pd.Series) -> str:
    if row["goal_diff_home"] > 0:
        return row["home_team"]
    if row["goal_diff_home"] < 0:
        return row["away_team"]
    return "draw"
