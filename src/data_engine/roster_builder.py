"""Build World Cup squads from Transfermarkt open player data (not FM)."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_engine.team_name_resolver import canonical_team
from src.match_engine.formation import roles_for_formation
from src.match_engine.tactical_catalog import normalize_formation_key

# GFS team name -> Transfermarkt country_of_citizenship values
CITIZENSHIP_ALIASES: Dict[str, List[str]] = {
    "Ivory Coast": ["Ivory Coast", "Cote d'Ivoire", "Côte d'Ivoire"],
    "South Korea": ["Korea, South", "South Korea", "Korea Republic"],
    "DR Congo": ["DR Congo", "Congo DR", "Democratic Republic of the Congo", "Congo-Kinshasa"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina", "Bosnia and Herzegovina"],
    "Czech Republic": ["Czech Republic", "Czechia"],
    "United States": ["United States", "USA"],
    "Turkey": ["Turkey", "Türkiye"],
    "Curaçao": ["Curaçao", "Curacao"],
    "Cape Verde": ["Cape Verde"],
    "Haiti": ["Haiti"],
    "Qatar": ["Qatar"],
    "Jordan": ["Jordan"],
    "Iraq": ["Iraq"],
    "Iran": ["Iran"],
    "Uzbekistan": ["Uzbekistan"],
    "Panama": ["Panama"],
    "New Zealand": ["New Zealand"],
}

TM_SUBPOS_TO_ROLE: Dict[str, str] = {
    "goalkeeper": "GK",
    "centre-back": "CB",
    "left-back": "LB",
    "right-back": "RB",
    "defensive midfield": "DM",
    "central midfield": "CM",
    "attacking midfield": "AM",
    "left midfield": "LM",
    "right midfield": "RM",
    "left winger": "LW",
    "right winger": "RW",
    "centre-forward": "ST",
    "second striker": "ST",
}

ROLE_PRIORITY = ["GK", "CB", "LB", "RB", "DM", "CM", "AM", "LM", "RM", "LW", "RW", "ST"]


def citizenship_labels(team_name: str) -> List[str]:
    canon = canonical_team(team_name)
    return CITIZENSHIP_ALIASES.get(canon, CITIZENSHIP_ALIASES.get(team_name, [canon, team_name]))


def _norm_subpos(s: Any) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return str(s).strip().lower()


def tm_role(sub_position: str, position: str = "") -> str:
    sp = _norm_subpos(sub_position)
    if sp in TM_SUBPOS_TO_ROLE:
        return TM_SUBPOS_TO_ROLE[sp]
    if "goalkeeper" in sp or str(position).lower() == "goalkeeper":
        return "GK"
    if "back" in sp:
        return "CB"
    if "wing" in sp:
        return "LW" if "left" in sp else "RW"
    if "midfield" in sp:
        return "CM"
    if "forward" in sp or "striker" in sp:
        return "ST"
    return "CM"


def _age_from_dob(dob: Any) -> Optional[int]:
    if dob is None or (isinstance(dob, float) and math.isnan(dob)):
        return None
    try:
        dt = pd.to_datetime(dob)
        return int(datetime.now().year - dt.year)
    except Exception:
        return None


def abilities_from_row(row: pd.Series, squad_max_mv: float, role: str) -> Dict[str, float]:
    """Legacy helper — returns blended abilities from full player dynamics."""
    return build_player_record_dynamics(row, squad_max_mv, role)["abilities"]


def build_player_record_dynamics(row: pd.Series, squad_max_mv: float, role: str) -> Dict[str, Any]:
    from src.data_engine.player_loader import build_player_profile_from_observables

    mv = float(row.get("market_value_in_eur") or 0)
    caps = int(row.get("international_caps") or 0)
    height = float(row.get("height_in_cm") or 180)
    name = str(row.get("name") or f"{row.get('first_name', '')} {row.get('last_name', '')}".strip())
    pid = f"tm_{int(row['player_id'])}"
    return build_player_profile_from_observables(
        player_id=pid,
        name=name,
        team_id="",
        role=role,
        market_value_eur=mv,
        international_caps=caps,
        height_cm=height,
        squad_max_mv=squad_max_mv,
        age=_age_from_dob(row.get("date_of_birth")),
        transfermarkt_url=str(row.get("url") or ""),
    )


def filter_team_pool(players: pd.DataFrame, team_name: str, min_season: int = 2024) -> pd.DataFrame:
    labels = citizenship_labels(team_name)
    pool = players[players["country_of_citizenship"].isin(labels)].copy()
    if "last_season" in pool.columns:
        pool = pool[pool["last_season"] >= min_season]
    pool["tm_role"] = pool.apply(
        lambda r: tm_role(r.get("sub_position", ""), r.get("position", "")), axis=1
    )
    pool["market_value_in_eur"] = pd.to_numeric(pool["market_value_in_eur"], errors="coerce").fillna(0)
    pool["international_caps"] = pd.to_numeric(pool["international_caps"], errors="coerce").fillna(0)
    pool["pick_score"] = pool["market_value_in_eur"] + pool["international_caps"] * 2_500_000
    return pool.sort_values("pick_score", ascending=False)


def select_starting_eleven(pool: pd.DataFrame, formation_key: str) -> List[pd.Series]:
    roles_needed = roles_for_formation(normalize_formation_key(formation_key))
    used_idx = set()
    picked: List[pd.Series] = []

    for need in roles_needed:
        candidates = pool[~pool.index.isin(used_idx)]
        # exact role first
        exact = candidates[candidates["tm_role"] == need]
        if exact.empty and need in ("LM", "RM"):
            exact = candidates[candidates["tm_role"].isin(["LW", "RW", "CM", "AM"])]
        if exact.empty and need in ("LW", "RW"):
            exact = candidates[candidates["tm_role"].isin(["LM", "RM", "ST", "AM"])]
        if exact.empty and need == "DM":
            exact = candidates[candidates["tm_role"].isin(["CM", "CB"])]
        if exact.empty:
            exact = candidates
        if exact.empty:
            continue
        row = exact.iloc[0]
        used_idx.add(row.name)
        picked.append(row)

    return picked


def player_record_from_row(row: pd.Series, team_id: str, squad_max_mv: float, shirt: int) -> Dict[str, Any]:
    role = str(row["tm_role"])
    rec = build_player_record_dynamics(row, squad_max_mv, role)
    rec["team_id"] = team_id
    rec["shirt_number"] = shirt
    rec["club"] = str(row.get("current_club_name") or "")
    return rec


def build_squad_payload(
    team_name: str,
    players: pd.DataFrame,
    *,
    formation: str = "4-3-3",
    squad_size: int = 23,
    coach_mental: Optional[Dict[str, float]] = None,
    fifa_ranking: Optional[float] = None,
) -> Dict[str, Any]:
    pool = filter_team_pool(players, team_name)
    if pool.empty:
        return {
            "team_id": team_name,
            "source": "transfermarkt",
            "formation": formation,
            "players": [],
            "error": "no_players_matched",
        }

    squad_max = float(pool["market_value_in_eur"].max()) if len(pool) else 1.0
    squad_rows = pool.head(squad_size)
    xi_rows = select_starting_eleven(squad_rows, formation)

    # remaining squad members not in XI
    xi_ids = {r.name for r in xi_rows}
    bench = squad_rows[~squad_rows.index.isin(xi_ids)]

    out_players: List[Dict[str, Any]] = []
    for i, row in enumerate(xi_rows, start=1):
        rec = player_record_from_row(row, team_name, squad_max, i)
        rec["squad_role"] = "starter"
        out_players.append(rec)

    for j, (_, row) in enumerate(bench.iterrows(), start=12):
        rec = player_record_from_row(row, team_name, squad_max, j)
        rec["squad_role"] = "bench"
        out_players.append(rec)

    from src.data_engine.entity_dynamics import team_state_vector

    team_state = team_state_vector(
        [p["abilities"] for p in out_players],
        coach_mental or {},
        fifa_ranking=float(fifa_ranking) if fifa_ranking is not None else 50.0,
        squad_conditions=[p.get("condition", {}) for p in out_players],
    )

    return {
        "team_id": team_name,
        "source": "transfermarkt",
        "formation": formation,
        "squad_size": len(out_players),
        "players": out_players,
        "team_dynamics": team_state,
        "dynamics_model": "entity_dynamics_v1",
    }


def write_all_wc_rosters(
    base_dir: Path,
    teams: List[str],
    tactics: Dict[str, Any],
    players_path: Optional[Path] = None,
    coaches: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    players_path = players_path or base_dir / "data" / "external" / "transfermarkt" / "players.csv.gz"
    players = pd.read_csv(players_path)
    out_dir = base_dir / "data" / "rosters"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}
    for team in teams:
        tac = tactics.get(team, {})
        raw_form = str(tac.get("formation", "4-3-3"))
        formation = raw_form.split(" - ")[0].strip() if " - " in raw_form else raw_form.strip()
        coach_entry = (coaches or {}).get(team) or {}
        coach_mental = coach_entry.get("mental") if isinstance(coach_entry, dict) else None
        tm = coach_entry.get("team_meta") or {}
        fifa = tm.get("fifa_ranking")
        payload = build_squad_payload(
            team,
            players,
            formation=formation,
            coach_mental=coach_mental,
            fifa_ranking=fifa,
        )
        safe = re.sub(r"[^\w\-]+", "_", team)
        path = out_dir / f"{safe}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        stats[team] = len(payload.get("players", []))
    return stats
