"""Official FIFA WC2026 group-stage matchdays (Final Draw, Dec 2025).

Each group has 3 matchdays; within a matchday both fixtures kick off in parallel.
Fixture tuples are (designated home, away) per FIFA schedule listing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

FixturePair = Tuple[str, str]
MatchdayFixtures = List[FixturePair]

# Keys must match WORLD_CUP_2026_GROUPS in tournament_2026.py
OFFICIAL_GROUP_MATCHDAYS: Dict[str, List[MatchdayFixtures]] = {
    "Group A": [
        [("Mexico", "South Africa"), ("South Korea", "Czech Republic")],
        [("Czech Republic", "South Africa"), ("Mexico", "South Korea")],
        [("Czech Republic", "Mexico"), ("South Africa", "South Korea")],
    ],
    "Group B": [
        [("Canada", "Bosnia and Herzegovina"), ("Qatar", "Switzerland")],
        [("Switzerland", "Bosnia and Herzegovina"), ("Canada", "Qatar")],
        [("Switzerland", "Canada"), ("Bosnia and Herzegovina", "Qatar")],
    ],
    "Group C": [
        [("Haiti", "Scotland"), ("Brazil", "Morocco")],
        [("Brazil", "Haiti"), ("Scotland", "Morocco")],
        [("Scotland", "Brazil"), ("Morocco", "Haiti")],
    ],
    "Group D": [
        [("United States", "Paraguay"), ("Australia", "Turkey")],
        [("Turkey", "Paraguay"), ("United States", "Australia")],
        [("Turkey", "United States"), ("Paraguay", "Australia")],
    ],
    "Group E": [
        [("Ivory Coast", "Ecuador"), ("Germany", "Curaçao")],
        [("Germany", "Ivory Coast"), ("Ecuador", "Curaçao")],
        [("Curaçao", "Ivory Coast"), ("Ecuador", "Germany")],
    ],
    "Group F": [
        [("Netherlands", "Japan"), ("Sweden", "Tunisia")],
        [("Netherlands", "Sweden"), ("Tunisia", "Japan")],
        [("Japan", "Sweden"), ("Tunisia", "Netherlands")],
    ],
    "Group G": [
        [("Iran", "New Zealand"), ("Belgium", "Egypt")],
        [("Belgium", "Iran"), ("New Zealand", "Egypt")],
        [("Egypt", "Iran"), ("New Zealand", "Belgium")],
    ],
    "Group H": [
        [("Saudi Arabia", "Uruguay"), ("Spain", "Cape Verde")],
        [("Uruguay", "Cape Verde"), ("Spain", "Saudi Arabia")],
        [("Cape Verde", "Saudi Arabia"), ("Uruguay", "Spain")],
    ],
    "Group I": [
        [("France", "Senegal"), ("Iraq", "Norway")],
        [("Norway", "Senegal"), ("France", "Iraq")],
        [("Norway", "France"), ("Senegal", "Iraq")],
    ],
    "Group J": [
        [("Argentina", "Algeria"), ("Austria", "Jordan")],
        [("Argentina", "Austria"), ("Jordan", "Algeria")],
        [("Algeria", "Austria"), ("Jordan", "Argentina")],
    ],
    "Group K": [
        [("Portugal", "DR Congo"), ("Uzbekistan", "Colombia")],
        [("Portugal", "Uzbekistan"), ("Colombia", "DR Congo")],
        [("Colombia", "Portugal"), ("DR Congo", "Uzbekistan")],
    ],
    "Group L": [
        [("Ghana", "Panama"), ("England", "Croatia")],
        [("England", "Ghana"), ("Panama", "Croatia")],
        [("Panama", "England"), ("Croatia", "Ghana")],
    ],
}


@dataclass(frozen=True)
class FixtureMeta:
    match_number: int
    date: str  # ISO yyyy-mm-dd
    city: str
    venue: str


# Selected metadata for host-nation fixtures (optional log enrichment).
FIXTURE_META: Dict[Tuple[str, str, str], FixtureMeta] = {
    ("Group A", "Mexico", "South Africa"): FixtureMeta(1, "2026-06-11", "Mexico City", "Estadio Azteca"),
    ("Group A", "South Korea", "Czech Republic"): FixtureMeta(2, "2026-06-11", "Guadalajara", "Estadio Akron"),
    ("Group A", "Czech Republic", "South Africa"): FixtureMeta(25, "2026-06-18", "Atlanta", "Mercedes-Benz Stadium"),
    ("Group A", "Mexico", "South Korea"): FixtureMeta(28, "2026-06-18", "Guadalajara", "Estadio Akron"),
    ("Group A", "Czech Republic", "Mexico"): FixtureMeta(53, "2026-06-24", "Mexico City", "Estadio Azteca"),
    ("Group A", "South Africa", "South Korea"): FixtureMeta(54, "2026-06-24", "Monterrey", "Estadio BBVA"),
}


def official_group_matchdays(group_name: str) -> List[MatchdayFixtures]:
    if group_name not in OFFICIAL_GROUP_MATCHDAYS:
        raise KeyError(f"No official schedule for {group_name}")
    return OFFICIAL_GROUP_MATCHDAYS[group_name]


def validate_groups(groups: Dict[str, List[str]]) -> None:
    """Ensure roster keys match official fixture participants."""
    for g_name, teams in groups.items():
        roster = set(teams)
        for md in OFFICIAL_GROUP_MATCHDAYS.get(g_name, []):
            for home, away in md:
                if home not in roster or away not in roster:
                    raise ValueError(
                        f"{g_name} roster {sorted(roster)} missing fixture participant {home} vs {away}"
                    )


def fixture_meta(group_name: str, home: str, away: str) -> FixtureMeta | None:
    return FIXTURE_META.get((group_name, home, away))
