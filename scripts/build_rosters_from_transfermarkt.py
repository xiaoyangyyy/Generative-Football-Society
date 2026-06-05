#!/usr/bin/env python
"""Generate data/rosters/*.json for all World Cup 2026 teams from Transfermarkt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.roster_builder import write_all_wc_rosters
from src.simulation.tournament_2026 import WORLD_CUP_2026_GROUPS


def main() -> None:
    teams = sorted({t for g in WORLD_CUP_2026_GROUPS.values() for t in g})
    tactics_path = ROOT / "data" / "tactics_final_en.json"
    tactics = {}
    if tactics_path.exists():
        tactics = json.loads(tactics_path.read_text(encoding="utf-8"))

    coaches_path = ROOT / "data" / "coaches" / "wc2026_coaches.json"
    coaches = {}
    if coaches_path.exists():
        coaches = json.loads(coaches_path.read_text(encoding="utf-8"))

    stats = write_all_wc_rosters(ROOT, teams, tactics, coaches=coaches)
    ok = sum(1 for n in stats.values() if n >= 11)
    empty = [t for t, n in stats.items() if n < 11]
    print(f"Rosters written: {ok}/{len(teams)} teams with 11+ players")
    if empty:
        print("Sparse rosters:", ", ".join(f"{t}({stats[t]})" for t in empty))


if __name__ == "__main__":
    main()
