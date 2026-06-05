#!/usr/bin/env python
"""Import user-exported FM CSV/HTML and merge into data/rosters/ (overrides TM abilities)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.fm_import import filter_players_by_nation, parse_fm_csv, parse_fm_html
from src.data_engine.fm_roster_merge import build_fm_only_roster, merge_fm_into_roster
from src.data_engine.roster_loader import load_roster_json, roster_path_for_team


def main() -> None:
    p = argparse.ArgumentParser(description="Import FM player export into sim rosters JSON")
    p.add_argument("export_path", help="Path to FM .csv or .html export")
    p.add_argument("--nation", required=True, help="Nation filter, e.g. Brazil")
    p.add_argument("--team-id", default=None, help="Output team_id (defaults to nation)")
    p.add_argument(
        "--fm-only",
        action="store_true",
        help="Build roster from FM only when no TM file exists",
    )
    p.add_argument(
        "--rebuild-dynamics",
        action="store_true",
        help="Rebuild condition/channels from FM-matched players",
    )
    args = p.parse_args()

    path = Path(args.export_path)
    if path.suffix.lower() == ".html":
        players = parse_fm_html(path)
    else:
        players = parse_fm_csv(path)

    nation_players = filter_players_by_nation(players, args.nation)
    if not nation_players:
        print(f"No players matched nation={args.nation!r}")
        sys.exit(1)

    team_id = args.team_id or args.nation
    base_dir = str(ROOT)
    roster_path = roster_path_for_team(base_dir, team_id)
    existing = load_roster_json(roster_path)

    if existing:
        merged = merge_fm_into_roster(
            existing,
            nation_players,
            overwrite_dynamics=args.rebuild_dynamics,
        )
        print(
            f"Merged FM into {roster_path}: "
            f"{merged.get('fm_merge_matched', 0)}/{merged.get('fm_merge_total', 0)} matched"
        )
    elif args.fm_only:
        tac_path = ROOT / "data" / "tactics_final_en.json"
        formation = "4-3-3"
        if tac_path.exists():
            tac = json.loads(tac_path.read_text(encoding="utf-8"))
            entry = tac.get(team_id, {})
            formation = str(entry.get("formation", "4-3-3")).split(" - ")[0].strip()
        merged = build_fm_only_roster(team_id, nation_players, formation=formation)
        print(f"Built FM-only roster with {len(merged.get('players', []))} players")
    else:
        print(
            f"No roster at {roster_path}. Run build_rosters_from_transfermarkt.py first, "
            "or pass --fm-only."
        )
        sys.exit(2)

    out_dir = ROOT / "data" / "rosters"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{team_id.replace(' ', '_')}.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(merged.get('players', []))} players -> {out_path}")


if __name__ == "__main__":
    main()
