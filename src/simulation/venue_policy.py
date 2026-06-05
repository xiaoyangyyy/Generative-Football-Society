"""2026 host-nation venue rules: only USA / Canada / Mexico get home advantage."""

from __future__ import annotations

from typing import Tuple

HOST_NATIONS = frozenset({"Mexico", "Canada", "United States"})


def resolve_match_venue(
    team_a: str,
    team_b: str,
    *,
    matchday: int = 0,
    fixture_seed: int = 0,
) -> Tuple[str, str, bool, str]:
    """
    Returns (home_team, away_team, neutral_venue, venue_label).

    - Single host vs visitor: host is micro home.
    - Both hosts: rotate home by matchday + seed (still a host stadium).
    - Neither host: neutral (50/50 micro prior, no crowd home boost).
    """
    a_host = team_a in HOST_NATIONS
    b_host = team_b in HOST_NATIONS

    if a_host and not b_host:
        return team_a, team_b, False, f"{team_a} (host)"
    if b_host and not a_host:
        return team_b, team_a, False, f"{team_b} (host)"
    if a_host and b_host:
        flip = ((matchday + fixture_seed) % 2) == 1
        if flip:
            return team_b, team_a, False, f"{team_b} (host, dual-host fixture)"
        return team_a, team_b, False, f"{team_a} (host, dual-host fixture)"
    # Neutral — stable ordering for micro labels only
    if team_a <= team_b:
        return team_a, team_b, True, "neutral venue (non-host fixture)"
    return team_b, team_a, True, "neutral venue (non-host fixture)"
