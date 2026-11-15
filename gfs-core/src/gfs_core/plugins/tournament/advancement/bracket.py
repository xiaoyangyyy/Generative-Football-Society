"""Build Round-of-32 fixtures with basic group-separation heuristics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class QualifiedEntry:
    team: str
    group: str
    rank: int  # 1, 2, or 3


def _group_of(qualified: List[QualifiedEntry], team: str) -> Optional[str]:
    for q in qualified:
        if q.team == team:
            return q.group
    return None


def build_qualified_entries(
    groups: Dict[str, List[str]],
    standings: Dict[str, Dict[str, Dict[str, int]]],
) -> List[QualifiedEntry]:
    entries: List[QualifiedEntry] = []
    thirds_meta = []
    for g_name, stats in standings.items():
        ranked = sorted(
            stats.keys(),
            key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"]),
            reverse=True,
        )
        entries.append(QualifiedEntry(ranked[0], g_name, 1))
        entries.append(QualifiedEntry(ranked[1], g_name, 2))
        thirds_meta.append((g_name, ranked[2], stats[ranked[2]]))
    thirds_meta.sort(key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gf"]), reverse=True)
    for g_name, team, _ in thirds_meta[:8]:
        entries.append(QualifiedEntry(team, g_name, 3))
    return entries


def _pick_opponent(
    pool: List[QualifiedEntry],
    avoid_group: str,
    used: set[str],
    rng: random.Random,
) -> Optional[QualifiedEntry]:
    cands = [x for x in pool if x.group != avoid_group and x.team not in used]
    if not cands:
        cands = [x for x in pool if x.team not in used]
    return rng.choice(cands) if cands else None


def build_r32_pairings(
    entries: List[QualifiedEntry],
    *,
    rng: random.Random | None = None,
) -> List[Tuple[str, str]]:
    """
    Exactly 16 R32 fixtures (32 teams):
    - 8 group winners vs 8 best third-place teams (different groups)
    - 4 remaining winners vs 4 runners-up
    - 4 runner-up vs runner-up ties from the other 8 runners
    """
    rng = rng or random.Random(42)
    winners = [e for e in entries if e.rank == 1]
    runners = [e for e in entries if e.rank == 2]
    thirds = [e for e in entries if e.rank == 3]
    rng.shuffle(winners)
    rng.shuffle(thirds)
    rng.shuffle(runners)

    fixtures: List[Tuple[str, str]] = []
    used_thirds: set[str] = set()
    used_runners: set[str] = set()

    for w in winners[:8]:
        t = _pick_opponent(thirds, w.group, used_thirds, rng)
        if t is None:
            continue
        used_thirds.add(t.team)
        fixtures.append((w.team, t.team))

    avail_runners = [r for r in runners]
    for w in winners[8:12]:
        cands = [r for r in avail_runners if r.group != w.group and r.team not in used_runners]
        if not cands:
            cands = [r for r in avail_runners if r.team not in used_runners]
        if not cands:
            break
        r = rng.choice(cands)
        used_runners.add(r.team)
        fixtures.append((w.team, r.team))

    remaining = [r for r in runners if r.team not in used_runners]
    rng.shuffle(remaining)
    i = 0
    while i + 1 < len(remaining):
        a, b = remaining[i], remaining[i + 1]
        if a.group == b.group:
            swapped = False
            for j in range(i + 2, len(remaining)):
                if remaining[j].group != a.group:
                    remaining[i + 1], remaining[j] = remaining[j], remaining[i + 1]
                    b = remaining[i + 1]
                    swapped = True
                    break
            if not swapped and i + 2 < len(remaining):
                remaining[i], remaining[i + 2] = remaining[i + 2], remaining[i]
                a, b = remaining[i], remaining[i + 1]
        fixtures.append((a.team, b.team))
        i += 2

    if len(fixtures) != 16:
        raise ValueError(f"R32 bracket must have 16 fixtures, got {len(fixtures)}")
    rng.shuffle(fixtures)
    return fixtures


def order_knockout_round(
    current_round_teams: List[str],
    fixtures: Optional[List[Tuple[str, str]]] = None,
) -> List[str]:
    """Return team list ordered as [a1,b1,a2,b2,...] for simulate_knockout_round."""
    if fixtures:
        ordered = []
        for a, b in fixtures:
            ordered.extend([a, b])
        return ordered
    return list(current_round_teams)
