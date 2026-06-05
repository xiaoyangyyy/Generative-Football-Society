"""Map GFS / World Cup team names to Transfermarkt country labels."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from src.data_engine.team_aliases import TEAM_ALIASES

# GFS name -> Transfermarkt country_name / name substrings
WC_TO_TM: Dict[str, List[str]] = {
    "DR Congo": ["Congo DR", "DR Congo", "Democratic Republic"],
    "United States": ["United States", "USA"],
    "Ivory Coast": ["Ivory Coast", "Cote d'Ivoire", "Côte d'Ivoire"],
    "Curaçao": ["Curacao", "Curaçao"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina", "Bosnia and Herzegovina"],
    "South Korea": ["Korea Republic", "South Korea"],
    "Czech Republic": ["Czechia", "Czech Republic"],
    "Turkey": ["Turkey", "Türkiye"],
    "Cape Verde": ["Cape Verde"],
}


def canonical_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def resolve_national_team_row(nt: pd.DataFrame, team_name: str) -> Optional[pd.Series]:
    """Return first matching row in Transfermarkt national_teams table."""
    canon = canonical_team(team_name)
    for col in ("country_name", "name"):
        hit = nt[nt[col].astype(str).str.lower() == canon.lower()]
        if not hit.empty:
            return hit.iloc[0]

    for alt in WC_TO_TM.get(team_name, WC_TO_TM.get(canon, [])):
        m = nt["country_name"].astype(str).str.contains(alt, case=False, na=False) | nt[
            "name"
        ].astype(str).str.contains(alt, case=False, na=False)
        if m.any():
            return nt[m].iloc[0]

    # loose: strip accents/punctuation
    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    target = norm(canon)
    for _, row in nt.iterrows():
        for col in ("country_name", "name"):
            if norm(str(row[col])) == target:
                return row
    return None
