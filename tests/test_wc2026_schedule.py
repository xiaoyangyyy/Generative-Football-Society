"""Official WC2026 group-stage fixture schedule."""

from __future__ import annotations

from src.simulation.tournament_2026 import WORLD_CUP_2026_GROUPS
from src.simulation.wc2026_schedule import official_group_matchdays, validate_groups


def test_group_a_opener_mexico_south_africa():
    md1 = official_group_matchdays("Group A")[0]
    assert ("Mexico", "South Africa") in md1
    assert md1[0] == ("Mexico", "South Africa")


def test_all_groups_validate_against_roster():
    validate_groups(WORLD_CUP_2026_GROUPS)


def test_each_team_plays_three_times():
    for g_name, teams in WORLD_CUP_2026_GROUPS.items():
        counts = {t: 0 for t in teams}
        for md in official_group_matchdays(g_name):
            assert len(md) == 2
            for home, away in md:
                counts[home] += 1
                counts[away] += 1
        assert all(v == 3 for v in counts.values()), f"{g_name} {counts}"
