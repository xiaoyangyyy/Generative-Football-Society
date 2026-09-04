#!/usr/bin/env python
"""
Download open football baseline data and build World Cup 2026 coach seeds.

- Transfermarkt: national_teams, countries, players (CC-friendly weekly dump)
- Coaches: public WC2026 head-coach list (FIFPlay, May 2026) + TM FIFA rank / market value

FM proprietary databases are NOT downloaded here. Use export from your FM install:
  scripts/import_fm_export.py  (see docs/COACH_AND_FM_DATA.md)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_engine.coach_loader import build_coach_profile_payload
from src.data_engine.team_name_resolver import resolve_national_team_row
from src.simulation.tournament_2026 import WORLD_CUP_2026_GROUPS

TM_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
FILES = ("national_teams.csv.gz", "countries.csv.gz", "players.csv.gz")

# Source: https://www.fifplay.com/world-cup-2026-head-coaches/ (accessed 2026-05)
FIFPLAY_WC2026_COACHES: List[Dict[str, Any]] = [
    {"team": "Algeria", "name": "Vladimir Petković", "nationality": "Bosnia & Herzegovina", "since": 2024},
    {"team": "Argentina", "name": "Lionel Scaloni", "nationality": "Argentina", "since": 2018},
    {"team": "Australia", "name": "Tony Popovic", "nationality": "Australia", "since": 2024},
    {"team": "Austria", "name": "Ralf Rangnick", "nationality": "Germany", "since": 2022},
    {"team": "Belgium", "name": "Rudi Garcia", "nationality": "France", "since": 2025},
    {"team": "Bosnia and Herzegovina", "name": "Sergej Barbarez", "nationality": "Bosnia & Herzegovina", "since": 2024},
    {"team": "Brazil", "name": "Carlo Ancelotti", "nationality": "Italy", "since": 2025},
    {"team": "Canada", "name": "Jesse Marsch", "nationality": "United States", "since": 2024},
    {"team": "Cape Verde", "name": "Bubista", "nationality": "Cape Verde", "since": 2020},
    {"team": "Colombia", "name": "Néstor Lorenzo", "nationality": "Argentina", "since": 2022},
    {"team": "Croatia", "name": "Zlatko Dalić", "nationality": "Croatia", "since": 2017},
    {"team": "Curaçao", "name": "Dick Advocaat", "nationality": "Netherlands", "since": 2024},
    {"team": "Czech Republic", "name": "Miroslav Koubek", "nationality": "Czech Republic", "since": 2024},
    {"team": "DR Congo", "name": "Sébastien Desabre", "nationality": "France", "since": 2022},
    {"team": "Ecuador", "name": "Sebastián Beccacece", "nationality": "Argentina", "since": 2024},
    {"team": "Egypt", "name": "Hossam Hassan", "nationality": "Egypt", "since": 2024},
    {"team": "England", "name": "Thomas Tuchel", "nationality": "Germany", "since": 2025},
    {"team": "France", "name": "Didier Deschamps", "nationality": "France", "since": 2012},
    {"team": "Germany", "name": "Julian Nagelsmann", "nationality": "Germany", "since": 2023},
    {"team": "Ghana", "name": "Otto Addo", "nationality": "Ghana", "since": 2024},
    {"team": "Haiti", "name": "Sébastien Migné", "nationality": "France", "since": 2024},
    {"team": "Iran", "name": "Amir Ghalenoei", "nationality": "Iran", "since": 2023},
    {"team": "Iraq", "name": "Graham Arnold", "nationality": "Australia", "since": 2025},
    {"team": "Ivory Coast", "name": "Emerse Faé", "nationality": "Ivory Coast", "since": 2024},
    {"team": "Japan", "name": "Hajime Moriyasu", "nationality": "Japan", "since": 2018},
    {"team": "Jordan", "name": "Jamal Sellami", "nationality": "Morocco", "since": 2024},
    {"team": "South Korea", "name": "Hong Myung-bo", "nationality": "South Korea", "since": 2024},
    {"team": "Mexico", "name": "Javier Aguirre", "nationality": "Mexico", "since": 2024},
    {"team": "Morocco", "name": "Walid Regragui", "nationality": "Morocco", "since": 2022},
    {"team": "Netherlands", "name": "Ronald Koeman", "nationality": "Netherlands", "since": 2023},
    {"team": "New Zealand", "name": "Darren Bazeley", "nationality": "England", "since": 2023},
    {"team": "Norway", "name": "Ståle Solbakken", "nationality": "Norway", "since": 2020},
    {"team": "Panama", "name": "Thomas Christiansen", "nationality": "Spain", "since": 2020},
    {"team": "Paraguay", "name": "Gustavo Alfaro", "nationality": "Argentina", "since": 2024},
    {"team": "Portugal", "name": "Roberto Martínez", "nationality": "Spain", "since": 2023},
    {"team": "Qatar", "name": "Julen Lopetegui", "nationality": "Spain", "since": 2025},
    {"team": "Saudi Arabia", "name": "Giorgos Donis", "nationality": "Greece", "since": 2025},
    {"team": "Scotland", "name": "Steve Clarke", "nationality": "Scotland", "since": 2019},
    {"team": "Senegal", "name": "Pape Thiaw", "nationality": "Senegal", "since": 2024},
    {"team": "South Africa", "name": "Hugo Broos", "nationality": "Belgium", "since": 2021},
    {"team": "Spain", "name": "Luis de la Fuente", "nationality": "Spain", "since": 2022},
    {"team": "Sweden", "name": "Graham Potter", "nationality": "England", "since": 2025},
    {"team": "Switzerland", "name": "Murat Yakin", "nationality": "Switzerland", "since": 2021},
    {"team": "Tunisia", "name": "Sami Trabelsi", "nationality": "Tunisia", "since": 2025},
    {"team": "Turkey", "name": "Vincenzo Montella", "nationality": "Italy", "since": 2023},
    {"team": "Uruguay", "name": "Marcelo Bielsa", "nationality": "Argentina", "since": 2023},
    {"team": "United States", "name": "Mauricio Pochettino", "nationality": "Argentina", "since": 2024},
    {"team": "Uzbekistan", "name": "Fabio Cannavaro", "nationality": "Italy", "since": 2025},
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100:
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  download: {dest.name}")
    subprocess.run(
        ["curl.exe", "-L", "-o", str(dest), url],
        check=True,
        cwd=str(ROOT),
    )


def _load_tactics_style(base_dir: Path, team: str) -> str:
    path = base_dir / "data" / "tactics_final_en.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data.get(team, {})
    return str(entry.get("style_desc", ""))


def build_coaches(base_dir: Path) -> Dict[str, Any]:
    ext = base_dir / "data" / "external" / "transfermarkt"
    nt = pd.read_csv(ext / "national_teams.csv.gz")
    teams = sorted({t for g in WORLD_CUP_2026_GROUPS.values() for t in g})
    by_team = {c["team"]: c for c in FIFPLAY_WC2026_COACHES}

    out: Dict[str, Any] = {}
    missing_tm: List[str] = []
    for team in teams:
        coach_row = by_team.get(team)
        if not coach_row:
            print(f"WARN: no FIFPlay coach for {team}")
            continue
        tm = resolve_national_team_row(nt, team)
        tm_meta = {}
        fifa = None
        if tm is not None:
            fifa = tm.get("fifa_ranking")
            tm_meta = {
                "transfermarkt_id": int(tm["national_team_id"]) if pd.notna(tm.get("national_team_id")) else None,
                "squad_size": int(tm["squad_size"]) if pd.notna(tm.get("squad_size")) else None,
                "total_market_value": str(tm.get("total_market_value", "")),
                "average_age": float(tm["average_age"]) if pd.notna(tm.get("average_age")) else None,
                "fifa_ranking": int(fifa) if pd.notna(fifa) else None,
            }
        else:
            missing_tm.append(team)

        style = _load_tactics_style(base_dir, team)
        out[team] = build_coach_profile_payload(
            team,
            coach_row["name"],
            nationality=coach_row.get("nationality", ""),
            in_charge_since=coach_row.get("since"),
            previous_role=coach_row.get("previous_role", ""),
            source="fifplay_wc2026+transfermarkt+dynamics",
            fifa_ranking=float(fifa) if fifa is not None and pd.notna(fifa) else None,
            style_desc=style,
            team_meta=tm_meta,
        )

    if missing_tm:
        print("TM rows missing for:", ", ".join(missing_tm).encode("ascii", "replace").decode("ascii"))
    return out


def main() -> None:
    base = ROOT
    ext = base / "data" / "external" / "transfermarkt"
    for fname in FILES:
        _download(f"{TM_BASE}/{fname}", ext / fname)

    coaches = build_coaches(base)
    tactics_path = base / "data" / "tactics_final_en.json"
    tactics = {}
    if tactics_path.exists():
        tactics = json.loads(tactics_path.read_text(encoding="utf-8"))
    from src.data_engine.roster_builder import write_all_wc_rosters
    from src.simulation.tournament_2026 import WORLD_CUP_2026_GROUPS

    teams = sorted({t for g in WORLD_CUP_2026_GROUPS.values() for t in g})
    roster_stats = write_all_wc_rosters(base, teams, tactics, coaches=coaches)
    ok_rosters = sum(1 for n in roster_stats.values() if n >= 11)
    print(f"Rosters: {ok_rosters}/{len(teams)} teams with 11+ players (Transfermarkt)")

    out_path = base / "data" / "coaches" / "wc2026_coaches.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coaches, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(coaches)} coaches -> {out_path}")

    meta_path = base / "data" / "external" / "transfermarkt" / "manifest.json"
    meta_path.write_text(
        json.dumps(
            {
                "source": "dcaribou/transfermarkt-datasets",
                "base_url": TM_BASE,
                "files": list(FILES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
