from __future__ import annotations

import numpy as np
import pandas as pd


def add_exposure_gate(
    stats: pd.DataFrame, team_matches: pd.DataFrame,
) -> pd.DataFrame:
    gated = stats.copy()
    world_cup = team_matches[team_matches["tournament"] == "FIFA World Cup"]
    gated["wc_matches"] = world_cup.groupby("team").size().reindex(
        gated.index, fill_value=0,
    )
    top_opponents = team_matches[team_matches["is_top_opponent"]]
    gated["top20_matches"] = top_opponents.groupby("team").size().reindex(
        gated.index, fill_value=0,
    )
    gated["global_exposure_gate"] = gated.apply(_exposure_label, axis=1)
    gated["tier"] = gated.apply(_tier, axis=1)
    return gated


def add_sample_confidence(
    stats: pd.DataFrame, *, threshold: int,
) -> pd.DataFrame:
    if threshold < 1:
        raise ValueError("sample confidence threshold must be positive")
    confident = stats.copy()
    score = np.minimum(1.0, np.sqrt(confident["total_matches"] / threshold))
    confident["sample_confidence_score"] = score
    confident["sample_confidence"] = score.apply(_confidence_label)
    return confident


def _exposure_label(row: pd.Series) -> str:
    if row["wc_matches"] >= 10 and row["top20_matches"] >= 30:
        return "pass"
    if row["wc_matches"] >= 3 or row["top20_matches"] >= 10:
        return "partial_pass"
    return "fail"


def _tier(row: pd.Series) -> str:
    score = row["final_status_score"]
    gate = row["global_exposure_gate"]
    if score >= 75 and gate == "pass":
        return "Core Power"
    if score >= 60 and gate in {"pass", "partial_pass"}:
        return "Semi-Core"
    if score >= 40:
        return "Regional Power"
    if score >= 25 and row["modern_power"] > row["historical_base"] + 10:
        return "Emerging Actor"
    return "Peripheral Actor"


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"
