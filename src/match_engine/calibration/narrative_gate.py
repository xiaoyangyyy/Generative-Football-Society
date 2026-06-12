"""Narrative anomaly gate — same heuristics as simulation/narrative_debug."""

from __future__ import annotations

from typing import Any


def anomalies_from_match(
    *,
    score_home: int,
    score_away: int,
    shots_home: int,
    shots_away: int,
    micro_xg_home: float,
    micro_xg_away: float,
) -> list[str]:
    flags: list[str] = []
    total = score_home + score_away
    if total >= 6:
        flags.append(f"high_total_goals:{total}")
    if score_home >= 5 or score_away >= 5:
        flags.append(f"team_goals_spike:{score_home}-{score_away}")
    mxg = float(micro_xg_home + micro_xg_away)
    if mxg > 0.05 and total / mxg > 2.4:
        flags.append(f"goals_vs_micro_xg:{total:.0f}/{mxg:.2f}")
    sh, sa = int(shots_home), int(shots_away)
    if (sh == 0 and sa >= 12) or (sa == 0 and sh >= 12):
        flags.append(f"shot_imbalance:{sh}-{sa}")
    if sh + sa >= 40:
        flags.append(f"shot_volume_spike:{sh + sa}")
    return flags


def anomalies_from_row(row: dict[str, Any]) -> list[str]:
    return anomalies_from_match(
        score_home=int(row.get("goals_home", 0)),
        score_away=int(row.get("goals_away", 0)),
        shots_home=int(row.get("shots_home", 0)),
        shots_away=int(row.get("shots_away", 0)),
        micro_xg_home=float(row.get("micro_xg_home", 0.0)),
        micro_xg_away=float(row.get("micro_xg_away", 0.0)),
    )


def evaluate_narrative_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = []
    for row in rows:
        anom = anomalies_from_row(row)
        if anom:
            flagged.append({"fixture": row.get("fixture", ""), "anomalies": anom, **row})
    n = len(rows)
    rate = len(flagged) / max(1, n)
    goals = [float(r.get("goals_home", 0) + r.get("goals_away", 0)) for r in rows]
    shots = [float(r.get("shots_home", 0) + r.get("shots_away", 0)) for r in rows]
    gxg = [float(r.get("goals_to_micro_xg_ratio", 0.0)) for r in rows if r.get("goals_to_micro_xg_ratio")]
    max_goals = max(goals) if goals else 0.0
    max_shots = max(shots) if shots else 0.0
    # Gate thresholds (derived from clean full-run distribution)
    checks = {
        "anomaly_rate_max_0.20": rate <= 0.20,
        "max_match_goals_le_6": max_goals <= 6.0,
        "max_match_shots_le_36": max_shots <= 36.0,
        "avg_goals_per_match_le_3.5": (sum(goals) / max(1, n)) <= 3.5,
        "avg_gxg_ratio_le_2.2": (sum(gxg) / max(1, len(gxg))) <= 2.2 if gxg else True,
    }
    all_pass = all(checks.values())
    return {
        "matches": n,
        "anomaly_matches": len(flagged),
        "anomaly_rate": round(rate, 4),
        "avg_goals_per_match": round(sum(goals) / max(1, n), 3),
        "avg_shots_per_match": round(sum(shots) / max(1, n), 2),
        "avg_gxg_ratio": round(sum(gxg) / max(1, len(gxg)), 3) if gxg else 0.0,
        "max_match_goals": max_goals,
        "max_match_shots": max_shots,
        "checks": checks,
        "all_pass": all_pass,
        "flagged_sample": flagged[:12],
    }
