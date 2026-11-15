from __future__ import annotations

import numpy as np
import pandas as pd

from .weights import MAJOR_TOURNAMENTS, time_decay, tournament_weight

MATCH_COLUMNS = [
    "date", "team", "opponent", "result", "goal_diff", "tournament",
    "is_draw", "winner_90min",
]


def compute_team_status(
    frame: pd.DataFrame,
    shootouts: pd.DataFrame,
    *,
    current_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_matches = _team_match_rows(frame)
    team_matches = _add_match_weights(team_matches, shootouts, current_year)
    return _aggregate_status(team_matches), team_matches


def _team_match_rows(frame: pd.DataFrame) -> pd.DataFrame:
    home = frame[[
        "date", "home_team", "away_team", "home_result", "goal_diff_home",
        "tournament", "is_draw", "winner_90min",
    ]].copy()
    home.columns = MATCH_COLUMNS
    home["is_home"] = True

    away = frame[[
        "date", "away_team", "home_team", "away_result", "goal_diff_home",
        "tournament", "is_draw", "winner_90min",
    ]].copy()
    away["goal_diff_home"] = -away["goal_diff_home"]
    away.columns = MATCH_COLUMNS
    away["is_home"] = False
    return pd.concat([home, away], ignore_index=True)


def _add_match_weights(
    team_matches: pd.DataFrame,
    shootouts: pd.DataFrame,
    current_year: int,
) -> pd.DataFrame:
    matches = team_matches.copy()
    matches["years_ago"] = current_year - matches["date"].dt.year
    matches["t_weight"] = matches["tournament"].apply(tournament_weight)
    matches["time_decay"] = time_decay(matches["years_ago"], matches["t_weight"])
    matches["points"] = matches["result"].map({"win": 3, "draw": 1, "loss": 0})
    matches = _attach_shootouts(matches, shootouts)
    matches["w_points"] = matches["points"] * matches["t_weight"] * matches["time_decay"]
    matches["w_max_points"] = 3 * matches["t_weight"] * matches["time_decay"]
    matches["gd_score"] = np.tanh(matches["goal_diff"] / 3.0)
    matches["w_gd_score"] = matches["gd_score"] * matches["t_weight"] * matches["time_decay"]
    matches["w_gd_base"] = matches["t_weight"] * matches["time_decay"]
    matches["is_major"] = matches["tournament"].isin(MAJOR_TOURNAMENTS)
    matches["w_major_exp"] = (
        matches["is_major"].astype(int) * matches["t_weight"] * matches["time_decay"]
    )
    pressure_event = (matches["t_weight"] >= 0.7) & (
        matches["shootout_win"]
        | ((matches["result"] == "win") & (matches["goal_diff"] >= 2))
    )
    matches["w_pressure"] = (
        pressure_event.astype(int) * matches["t_weight"] * matches["time_decay"]
    )
    return matches


def _attach_shootouts(
    matches: pd.DataFrame, shootouts: pd.DataFrame,
) -> pd.DataFrame:
    if shootouts is None or shootouts.empty:
        matches["shootout_winner"] = np.nan
        matches["shootout_win"] = False
        return matches
    shootouts = shootouts.copy()
    shootouts["date"] = pd.to_datetime(shootouts["date"])
    matches = matches.merge(
        shootouts[["date", "home_team", "winner"]],
        how="left",
        left_on=["date", "team"],
        right_on=["date", "home_team"],
    )
    matches.rename(columns={"winner": "winner_h"}, inplace=True)
    matches = matches.merge(
        shootouts[["date", "away_team", "winner"]],
        how="left",
        left_on=["date", "team"],
        right_on=["date", "away_team"],
    )
    matches.rename(columns={"winner": "winner_a"}, inplace=True)
    matches["shootout_winner"] = matches["winner_h"].combine_first(matches["winner_a"])
    matches.drop(
        columns=["home_team_x", "home_team_y", "away_team", "winner_h", "winner_a"],
        inplace=True,
        errors="ignore",
    )
    matches["shootout_win"] = matches["shootout_winner"] == matches["team"]
    return matches


def _aggregate_status(matches: pd.DataFrame) -> pd.DataFrame:
    grouped = matches.groupby("team")
    stats = pd.DataFrame()
    stats["total_matches"] = grouped.size()
    stats["c1_win_rate"] = _ratio(grouped, "w_points", "w_max_points") * 100.0
    stats["c2_gd"] = (_ratio(grouped, "w_gd_score", "w_gd_base") + 1.0) / 2.0 * 100.0

    major_exposure = grouped["w_major_exp"].sum()
    stats["c3_major_exp"] = major_exposure / max(1.0, major_exposure.max()) * 100.0
    pressure = grouped["w_pressure"].sum()
    stats["c5_pressure"] = pressure / max(1.0, pressure.max()) * 100.0

    global_matches = matches[matches["tournament"].isin(MAJOR_TOURNAMENTS)]
    global_grouped = global_matches.groupby("team")
    stats["global_win_rate"] = _ratio(
        global_grouped, "w_points", "w_max_points",
    ).fillna(0) * 100.0
    stats["global_exp"] = global_grouped["w_major_exp"].sum()
    stats["global_exp"] = stats["global_exp"] / max(1.0, stats["global_exp"].max()) * 100.0
    stats["true_global_score"] = (
        0.5 * stats["global_win_rate"] + 0.5 * stats["global_exp"].fillna(0)
    )

    top_teams = stats["true_global_score"].nlargest(20).index.tolist()
    matches["is_top_opponent"] = matches["opponent"].isin(top_teams)
    strong_opponents = matches[matches["is_top_opponent"]].groupby("team")
    stats["c4_strong_opp"] = _ratio(
        strong_opponents, "w_points", "w_max_points",
    ).fillna(0) * 100.0
    stats["raw_score"] = (
        0.30 * stats["c1_win_rate"]
        + 0.20 * stats["c2_gd"]
        + 0.25 * stats["c3_major_exp"]
        + 0.15 * stats["c4_strong_opp"]
        + 0.10 * stats["c5_pressure"]
    )
    _add_modern_power(stats, matches)
    _normalize_scores(stats)
    return stats


def _ratio(grouped, numerator: str, denominator: str):
    return grouped[numerator].sum() / grouped[denominator].sum().clip(lower=1e-9)


def _add_modern_power(stats: pd.DataFrame, matches: pd.DataFrame) -> None:
    modern = matches[matches["years_ago"] <= 15]
    grouped = modern.groupby("team")
    stats["modern_win_rate"] = _ratio(grouped, "w_points", "w_max_points") * 100.0
    stats["modern_gd"] = (_ratio(grouped, "w_gd_score", "w_gd_base") + 1.0) / 2.0 * 100.0
    exposure = grouped["w_major_exp"].sum()
    stats["modern_exp"] = exposure / max(1.0, exposure.max()) * 100.0
    pressure = grouped["w_pressure"].sum()
    stats["modern_pressure"] = pressure / max(1.0, pressure.max()) * 100.0
    strong = modern[modern["is_top_opponent"]].groupby("team")
    stats["modern_strong_opp"] = _ratio(strong, "w_points", "w_max_points") * 100.0
    stats["modern_power"] = (
        0.30 * stats["modern_win_rate"].fillna(0)
        + 0.20 * stats["modern_gd"].fillna(0)
        + 0.25 * stats["modern_exp"].fillna(0)
        + 0.15 * stats["modern_strong_opp"].fillna(0)
        + 0.10 * stats["modern_pressure"].fillna(0)
    )
    stats["historical_base"] = stats["raw_score"]
    stats["final_status_score"] = (
        0.30 * stats["historical_base"] + 0.70 * stats["modern_power"].fillna(0)
    )


def _normalize_scores(stats: pd.DataFrame) -> None:
    stats["final_status_score"] = (
        stats["final_status_score"] / stats["final_status_score"].max() * 100.0
    )
    stats["historical_base"] = stats["historical_base"] / stats["historical_base"].max() * 100.0
    stats["modern_power"] = stats["modern_power"] / stats["modern_power"].max() * 100.0
