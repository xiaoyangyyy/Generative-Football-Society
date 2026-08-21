import copy

import pytest

from src.product.club_strategy import (
    build_season_club_strategies, resolve_fixture_club_strategy,
    validate_club_strategy_snapshot,
)


def _roster(team, strengths=None):
    strengths = strengths or {}
    roles = ("GK", "GK", "CB", "CB", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "ST")
    players = []
    for index, role in enumerate(roles):
        abilities = {
            key: strengths.get(key, 0.55)
            for key in (
                "tech", "pass_skill", "vision", "spatial", "pace", "press",
                "shot", "power", "aerial", "mental", "gk_reflex", "gk_aerial",
            )
        }
        players.append({
            "player_id": f"{team.lower()}-{index}", "name": f"{team} {index}",
            "team_id": team, "role": role, "abilities": abilities,
        })
    return {"team_id": team, "formation": "4-3-3", "players": players}


def _archive():
    return {
        "schema_version": 1, "season_id": "season-0001",
        "final_standings": [
            {"position": 1, "team": "A", "points": 12},
            {"position": 2, "team": "B", "points": 7},
            {"position": 3, "team": "C", "points": 4},
            {"position": 4, "team": "D", "points": 1},
        ],
    }


def test_strategy_snapshot_is_roster_derived_rank_aware_and_deterministic():
    rosters = {
        "A": _roster("A", {"press": 0.88, "pace": 0.84, "mental": 0.82, "power": 0.78}),
        "B": _roster("B", {"tech": 0.88, "pass_skill": 0.9, "vision": 0.87, "spatial": 0.86}),
        "C": _roster("C"),
        "D": _roster("D", {"press": 0.78, "power": 0.86, "aerial": 0.88, "mental": 0.84}),
    }
    snapshot = build_season_club_strategies(
        season_id="season-0002", teams=("A", "B", "C", "D"),
        manager_team="A", rosters=rosters, source_archive=_archive(),
        recruitment_moves={"B": 1, "D": 2},
    )

    assert snapshot == build_season_club_strategies(
        season_id="season-0002", teams=("A", "B", "C", "D"),
        manager_team="A", rosters=rosters, source_archive=_archive(),
        recruitment_moves={"B": 1, "D": 2},
    )
    profiles = {profile["team"]: profile for profile in snapshot["profiles"]}
    assert profiles["A"]["primary_tactic"] == "gegenpress"
    assert profiles["A"]["risk_posture"] == "proactive"
    assert profiles["B"]["primary_tactic"] == "tiki_taka"
    assert profiles["D"]["risk_posture"] == "conservative"
    assert profiles["D"]["away_tactic"] == "low_block_counter"
    assert "post_window_roster:2_moves" in profiles["D"]["reasons"]


def test_previous_identity_has_bounded_inertia_and_snapshot_tampering_fails():
    initial = build_season_club_strategies(
        season_id="season-0001", teams=("A", "B"), manager_team="A",
        rosters={"A": _roster("A"), "B": _roster("B")},
    )
    evolved = build_season_club_strategies(
        season_id="season-0002", teams=("A", "B"), manager_team="A",
        rosters={"A": _roster("A"), "B": _roster("B")},
        source_archive={
            "season_id": "season-0001",
            "final_standings": [
                {"position": 1, "team": "A"}, {"position": 2, "team": "B"},
            ],
        },
        previous_snapshot=initial,
    )
    b = next(profile for profile in evolved["profiles"] if profile["team"] == "B")
    assert b["previous_primary_tactic"] == "balanced"
    assert "identity_inertia:balanced" in b["reasons"]

    tampered = copy.deepcopy(evolved)
    tampered["profiles"][0]["style_scores"]["balanced"] = float("nan")
    with pytest.raises(ValueError, match="profile evidence"):
        validate_club_strategy_snapshot(tampered)


def test_fixture_resolution_applies_ai_identities_manager_override_and_adaptation():
    snapshot = build_season_club_strategies(
        season_id="season-0001", teams=("A", "B"), manager_team="A",
        rosters={
            "A": _roster("A"),
            "B": _roster("B", {"tech": 0.9, "pass_skill": 0.9, "vision": 0.9, "spatial": 0.9}),
        },
    )
    ai_fixture = resolve_fixture_club_strategy(snapshot, home="A", away="B")
    assert ai_fixture["home_tactic"] == "balanced"
    assert ai_fixture["away_tactic"] == "tiki_taka"

    identity_response = resolve_fixture_club_strategy(
        snapshot, home="A", away="B",
        manager_decision={"team": "A", "tactic": "gegenpress"},
        opponent_preparation={"selected_tactic": "team_identity"},
    )
    assert identity_response["home_tactic"] == "gegenpress"
    assert identity_response["away_tactic"] == "tiki_taka"
    assert identity_response["opponent_adaptation_applied"] is False

    adapted = resolve_fixture_club_strategy(
        snapshot, home="A", away="B",
        manager_decision={"team": "A", "tactic": "gegenpress"},
        opponent_preparation={"selected_tactic": "direct_vertical"},
    )
    assert adapted["away_tactic"] == "direct_vertical"
    assert adapted["opponent_adaptation_applied"] is True
