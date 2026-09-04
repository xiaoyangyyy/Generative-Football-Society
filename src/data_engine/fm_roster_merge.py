"""Merge FM export abilities into Transfermarkt roster JSON."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

import numpy as np

from src.data_engine.player_loader import build_player_profile_from_observables
from src.match_engine.tactical_catalog import normalize_formation_key
from src.match_engine.formation import roles_for_formation


def normalize_player_name(name: str) -> str:
    raw = unicodedata.normalize("NFKD", str(name).lower())
    ascii_name = raw.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", "", ascii_name)
    return re.sub(r"\s+", " ", s).strip()


def fm_position_to_role(pos: str) -> str:
    p = str(pos or "").upper()
    mapping = {
        "GK": "GK",
        "D": "CB",
        "DC": "CB",
        "DL": "LB",
        "DR": "RB",
        "DM": "DM",
        "M": "CM",
        "MC": "CM",
        "ML": "LM",
        "MR": "RM",
        "AM": "AM",
        "AML": "LW",
        "AMR": "RW",
        "ST": "ST",
        "W": "LW",
    }
    for key, role in mapping.items():
        if key in p.split():
            return role
    if "GOAL" in p:
        return "GK"
    if "STRIK" in p or "FORW" in p:
        return "ST"
    if "WING" in p:
        return "LW"
    if "MID" in p:
        return "CM"
    if "DEF" in p or "BACK" in p:
        return "CB"
    return "CM"


def fm_row_to_abilities(fm: Dict[str, Any]) -> Dict[str, float]:
    """Map FM attributes + CA to sim abilities block."""
    ab = fm.get("abilities_fm") or {}
    ca = int(fm.get("ca") or 0)
    ca_n = float(np.clip(ca / 200.0, 0.35, 0.95)) if ca else 0.55

    def g(*keys: str, default: float = 0.5) -> float:
        vals = [float(ab[k]) for k in keys if k in ab]
        if vals:
            return float(np.clip(np.mean(vals), 0.0, 1.0))
        return default

    pace = g("pace", "acceleration", default=ca_n)
    tech = g("technique", "passing", "finishing", default=ca_n)
    vision = g("vision", "decisions", default=tech)
    mental = g("composure", "decisions", default=ca_n)
    press = g("tackling", "positioning", default=ca_n * 0.95)
    aerial = g("heading", "aerial_reach", default=ca_n * 0.85)
    gk = g("reflexes", "handling", default=0.0)

    return {
        "tech": tech,
        "phys": g("strength", "stamina", default=ca_n),
        "mental": mental,
        "pace": pace,
        "vision": vision,
        "spatial": vision,
        "press": press,
        "aerial": aerial,
        "heading": aerial,
        "pass_skill": g("passing", "technique", default=tech),
        "shot": g("finishing", "technique", default=tech * 0.92),
        "curve": g("technique", default=tech * 0.85),
        "power": g("strength", default=ca_n),
        "gk": gk,
        "gk_reflex": gk if gk > 0.1 else 0.0,
        "gk_aerial": g("aerial_reach", "handling", default=gk * 0.95 if gk > 0.1 else 0.0),
    }


def _index_roster_by_name(players: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in players:
        key = normalize_player_name(p.get("name", ""))
        if key:
            out[key] = p
    return out


def merge_fm_into_roster(
    roster: Dict[str, Any],
    fm_players: List[Dict[str, Any]],
    *,
    overwrite_dynamics: bool = False,
) -> Dict[str, Any]:
    """
    Match FM rows to roster players by normalized name; override abilities.
    Unmatched FM players are appended as bench if squad has room.
    """
    out = dict(roster)
    players = [dict(p) for p in out.get("players") or []]
    by_name = _index_roster_by_name(players)
    matched = 0

    for fm in fm_players:
        key = normalize_player_name(fm.get("name", ""))
        if not key:
            continue
        abilities = fm_row_to_abilities(fm)
        if key in by_name:
            rec = by_name[key]
            rec["abilities"] = {**rec.get("abilities", {}), **abilities}
            rec["fm_ca"] = int(fm.get("ca") or 0)
            rec["fm_pa"] = int(fm.get("pa") or 0)
            rec["source"] = "transfermarkt+fm"
            matched += 1
            continue

        role = fm_position_to_role(fm.get("position", ""))
        ca = int(fm.get("ca") or 140)
        mv_est = float(ca) * 500_000.0
        prof = build_player_profile_from_observables(
            player_id=f"fm_{key.replace(' ', '_')[:40]}",
            name=str(fm.get("name", "")),
            team_id=str(out.get("team_id", "")),
            role=role,
            market_value_eur=mv_est,
            international_caps=0,
            height_cm=180.0,
            squad_max_mv=max(mv_est, 1.0),
            age=int(fm.get("age") or 26),
            squad_role="bench",
            club=str(fm.get("club", "")),
        )
        prof["abilities"] = {**prof.get("abilities", {}), **abilities}
        prof["fm_ca"] = ca
        prof["source"] = "fm_export"
        players.append(prof)
        by_name[key] = prof

    if overwrite_dynamics:
        for p in players:
            if p.get("fm_ca"):
                role = str(p.get("role", "CM"))
                ca = int(p["fm_ca"])
                mv = float(p.get("market_value_eur") or ca * 500_000)
                from src.data_engine.entity_dynamics import build_player_dynamics_payload

                dyn = build_player_dynamics_payload(
                    role=role,
                    log_market_value=float(np.log1p(mv)),
                    international_caps=float(p.get("international_caps") or 0),
                    height_cm=180.0,
                    squad_max_mv=mv,
                    age=float(p.get("age") or 26),
                )
                p.update({k: v for k, v in dyn.items() if k not in ("player_id", "name", "team_id")})

    out["players"] = players
    out["fm_merge_matched"] = matched
    out["fm_merge_total"] = len(fm_players)
    if matched:
        out["source"] = str(out.get("source", "transfermarkt")) + "+fm"
    return out


def build_fm_only_roster(
    team_id: str,
    fm_players: List[Dict[str, Any]],
    formation: str = "4-3-3",
) -> Dict[str, Any]:
    """Build roster JSON when no TM file exists (FM-only squad)."""
    fkey = normalize_formation_key(formation)
    roles_needed = roles_for_formation(fkey)
    sorted_fm = sorted(fm_players, key=lambda x: int(x.get("ca") or 0), reverse=True)

    out_players: List[Dict[str, Any]] = []
    for i, fm in enumerate(sorted_fm[:23]):
        role = fm_position_to_role(fm.get("position", ""))
        if i < len(roles_needed):
            role = roles_needed[i]
        ca = int(fm.get("ca") or 140)
        mv = float(ca) * 500_000.0
        prof = build_player_profile_from_observables(
            player_id=f"fm_{normalize_player_name(fm.get('name',''))[:36]}",
            name=str(fm.get("name", "")),
            team_id=team_id,
            role=role,
            market_value_eur=mv,
            international_caps=0,
            height_cm=180.0,
            squad_max_mv=float(max(p.get("ca", 140) for p in sorted_fm) * 500_000),
            age=int(fm.get("age") or 26),
            squad_role="starter" if i < 11 else "bench",
            club=str(fm.get("club", "")),
        )
        prof["abilities"] = fm_row_to_abilities(fm)
        prof["fm_ca"] = ca
        prof["source"] = "fm_export"
        out_players.append(prof)

    from src.data_engine.entity_dynamics import team_state_vector

    return {
        "team_id": team_id,
        "source": "fm_export",
        "formation": formation,
        "squad_size": len(out_players),
        "players": out_players,
        "team_dynamics": team_state_vector(
            [p["abilities"] for p in out_players],
            {},
            fifa_ranking=50.0,
            squad_conditions=[p.get("condition", {}) for p in out_players],
        ),
    }
