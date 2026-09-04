"""Cross-check roster abilities vs Transfermarkt market value and caps."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_engine.roster_loader import load_roster_json, roster_path_for_team
from src.data_engine.roster_builder import citizenship_labels


def _load_tm_players(base_dir: str) -> pd.DataFrame:
    path = Path(base_dir) / "data" / "external" / "transfermarkt" / "players.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(f, low_memory=False)


def _expected_ability_from_mv(mv_eur: float, caps: int) -> float:
    """Monotonic prior: higher MV + caps → higher blended tech (0–1)."""
    mv = max(0.0, float(mv_eur))
    log_mv = math.log1p(mv / 1e6)
    cap_term = 0.08 * min(120, int(caps))
    raw = 0.38 + 0.12 * log_mv + cap_term
    return float(np.clip(raw / 1.35, 0.25, 0.92))


def verify_player_record(
    player: Dict[str, Any],
    tm_row: Optional[pd.Series],
) -> List[str]:
    issues: List[str] = []
    ab = player.get("abilities") or {}
    tech = float(ab.get("tech", 0.5))
    mv = float(player.get("market_value_eur") or 0)
    caps = int(player.get("international_caps") or 0)
    if tm_row is not None and not pd.isna(tm_row.get("market_value_in_eur")):
        tm_mv = float(tm_row["market_value_in_eur"])
        if mv > 0 and tm_mv > 0:
            ratio = mv / tm_mv
            if ratio < 0.5 or ratio > 2.0:
                issues.append(f"market_value mismatch roster={mv:.0f} tm={tm_mv:.0f}")
    expected = _expected_ability_from_mv(mv, caps)
    if abs(tech - expected) > 0.22:
        issues.append(f"tech={tech:.2f} vs mv_prior={expected:.2f}")
    if tech > 0.95 or tech < 0.15:
        issues.append(f"tech out of sane range: {tech:.2f}")
    return issues


def verify_team_roster(
    base_dir: str,
    team_name: str,
    tm_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    path = roster_path_for_team(base_dir, team_name)
    roster = load_roster_json(path)
    if not roster:
        return {"team": team_name, "ok": False, "error": "missing roster"}
    if tm_df is None:
        tm_df = _load_tm_players(base_dir)
    labels = citizenship_labels(team_name)
    team_tm = pd.DataFrame()
    if not tm_df.empty and "country_of_citizenship" in tm_df.columns:
        team_tm = tm_df[tm_df["country_of_citizenship"].isin(labels)]

    player_issues: Dict[str, List[str]] = {}
    missing_tm = 0
    for p in roster.get("players", []):
        pid = str(p.get("player_id", ""))
        name = str(p.get("name", ""))
        tm_row = None
        if not team_tm.empty and "player_id" in team_tm.columns:
            tid = pid.replace("tm_", "")
            hits = team_tm[team_tm["player_id"].astype(str) == tid]
            if hits.empty and name:
                hits = team_tm[team_tm["name"].astype(str).str.lower() == name.lower()]
            if not hits.empty:
                tm_row = hits.iloc[0]
            else:
                missing_tm += 1
        issues = verify_player_record(p, tm_row)
        if issues:
            player_issues[pid or name] = issues

    starters = [p for p in roster.get("players", []) if p.get("squad_role") == "starter"]
    return {
        "team": team_name,
        "ok": len(player_issues) == 0,
        "squad_size": len(roster.get("players", [])),
        "starters": len(starters),
        "missing_tm_links": missing_tm,
        "player_issues": player_issues,
    }


def verify_all_rosters(base_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    roster_dir = Path(base_dir) / "data" / "rosters"
    tm_df = _load_tm_players(base_dir)
    reports: List[Dict[str, Any]] = []
    for path in sorted(roster_dir.glob("*.json")):
        team = json.loads(path.read_text(encoding="utf-8")).get("team_id", path.stem)
        reports.append(verify_team_roster(base_dir, str(team), tm_df))
    summary = {
        "teams": len(reports),
        "teams_ok": sum(1 for r in reports if r.get("ok")),
        "teams_with_issues": sum(1 for r in reports if r.get("player_issues")),
        "total_player_flags": sum(len(r.get("player_issues", {})) for r in reports),
    }
    return reports, summary


def write_verification_report(base_dir: str, out_name: str = "roster_verification.json") -> str:
    reports, summary = verify_all_rosters(base_dir)
    out_path = Path(base_dir) / "data" / "persistence" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "teams": reports}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)
