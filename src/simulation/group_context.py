"""Group-stage standings snapshots and qualification context for coach LLM."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple


def snapshot_standings_table(standings: Dict[str, Dict[str, Dict[str, int]]], group_name: str) -> Dict[str, Dict[str, int]]:
    return copy.deepcopy(standings.get(group_name, {}))


def _sorted_table(table: Dict[str, Dict[str, int]]) -> List[Tuple[str, Dict[str, int]]]:
    return sorted(
        table.items(),
        key=lambda x: (x[1].get("pts", 0), x[1].get("gd", 0), x[1].get("gf", 0)),
        reverse=True,
    )


def format_group_table(table: Dict[str, Dict[str, int]]) -> str:
    rows = []
    for i, (team, st) in enumerate(_sorted_table(table), start=1):
        rows.append(
            f"{i}. {team}: {st.get('pts', 0)}pts "
            f"({st.get('gf', 0)}-{st.get('ga', 0)}, GD {st.get('gd', 0):+d})"
        )
    return " | ".join(rows) if rows else "empty"


def qualification_scenario(
    table: Dict[str, Dict[str, int]],
    team: str,
    opponent: str,
    *,
    matchday: int,
    total_matchdays: int = 3,
) -> str:
    """Heuristic must-win / can-draw text from pre-matchday table."""
    if team not in table or opponent not in table:
        return "qualification context unavailable"
    ranked = _sorted_table(table)
    order = {t: i for i, (t, _) in enumerate(ranked)}
    pos = order.get(team, 3)
    opp_pos = order.get(opponent, 3)
    pts = table[team].get("pts", 0)
    opp_pts = table[opponent].get("pts", 0)
    remaining = max(0, total_matchdays - matchday + 1)

    if pos <= 1 and pts >= opp_pts + 3:
        need = "can afford a draw; pressure to control tempo"
    elif pos >= 2 and remaining <= 1:
        need = "likely need maximum points today"
    elif pos == 2 and pts == opp_pts:
        need = "six-pointer tone — head-to-head swing matters"
    else:
        need = "every point matters for top-two / best-third race"
    return (
        f"pre-matchday {matchday}/{total_matchdays}; ranked {pos + 1}/4; "
        f"opponent {opponent} ranked {opp_pos + 1}/4 ({opp_pts}pts vs our {pts}pts); {need}"
    )


def build_coach_match_context(
    agent_context: str,
    *,
    group_name: str,
    matchday: int,
    team: str,
    opponent: str,
    standings_snapshot: Optional[Dict[str, Dict[str, int]]] = None,
    venue_label: str = "",
) -> str:
    parts = [agent_context]
    if venue_label:
        parts.append(f"Venue: {venue_label}")
    if standings_snapshot is not None and group_name.startswith("Group"):
        parts.append(f"GROUP TABLE (before this matchday): {format_group_table(standings_snapshot)}")
        parts.append(qualification_scenario(standings_snapshot, team, opponent, matchday=matchday))
    return " || ".join(parts)
