import copy

import pytest

from src.product.league_ecosystem import (
    ai_club_recruitment_decision, append_league_transition,
    league_ecosystem_view, validate_league_ecosystem,
)


def _archive(position=4):
    teams = ["A", "B", "C", "D"]
    order = [team for team in teams if team != "B"]
    order.insert(position - 1, "B")
    return {
        "schema_version": 1,
        "season_id": "season-0001",
        "plan": {"teams": teams, "manager_team": "A"},
        "final_standings": [
            {"position": index, "team": team, "points": 12 - index}
            for index, team in enumerate(order, start=1)
        ],
    }


def _roster(team="B"):
    roles = (
        "GK", "GK", "CB", "CB", "CB", "LB", "RB", "DM",
        "CM", "CM", "AM", "LW", "RW", "ST", "ST", "CB",
    )
    players = []
    for index, role in enumerate(roles):
        quality = 0.50 + 0.012 * index
        players.append({
            "player_id": f"{team.lower()}-{index:02d}",
            "name": f"{team} Player {index}", "team_id": team,
            "role": role,
            "abilities": {
                key: quality for key in (
                    "tech", "pass_skill", "vision", "spatial", "pace",
                    "press", "shot", "power", "aerial", "mental",
                    "gk_reflex", "gk_aerial",
                )
            },
        })
    return {"team_id": team, "formation": "4-3-3", "players": players}


def test_ai_rebuild_is_deterministic_cash_bounded_and_explainable():
    decision = ai_club_recruitment_decision(
        _archive(), _roster(), team="B", next_season_index=2, allowance=8,
    )

    assert decision == ai_club_recruitment_decision(
        _archive(), _roster(), team="B", next_season_index=2, allowance=8,
    )
    assert decision["strategy"] == "rebuild"
    assert decision["maximum_moves"] == 2
    assert 1 <= len(decision["selected_moves"]) <= 2
    assert sum(move["cost"] for move in decision["selected_moves"]) <= 8
    assert all(
        move["quality_improvement"] >= decision["minimum_quality_improvement"]
        for move in decision["selected_moves"]
    )
    assert decision["plan"]["market_id"] == decision["market_id"]


def test_ai_policy_handles_selective_zero_cash_and_missing_roster():
    selective = ai_club_recruitment_decision(
        _archive(position=1), _roster(), team="B",
        next_season_index=2, allowance=8,
    )
    assert selective["strategy"] == "selective"
    assert len(selective["selected_moves"]) <= 1
    assert selective["spending_cap"] == 5

    no_cash = ai_club_recruitment_decision(
        _archive(), _roster(), team="B", next_season_index=2, allowance=0,
    )
    assert no_cash["plan"] is None
    assert no_cash["reason"] == "no_eligible_upgrade"
    unavailable = ai_club_recruitment_decision(
        _archive(), None, team="B", next_season_index=2, allowance=8,
    )
    assert unavailable["available"] is False
    assert unavailable["reason"] == "roster_unavailable"


def test_ecosystem_registry_is_ordered_bounded_and_rejects_tampering():
    decision = ai_club_recruitment_decision(
        _archive(), None, team="B", next_season_index=2, allowance=8,
    )
    event = {
        "schema_version": 1,
        "from_season_id": "season-0001",
        "to_season_id": "season-0002",
        "archive_identity": "a" * 64,
        "manager_controlled_team": "A",
        "participating_teams": ["A", "B"],
        "ai_clubs": [{
            "team": "B", "settlement_identity": "b" * 64,
            "decision": decision, "squad_transaction_identity": None,
        }],
        "claim_boundary": "bounded test event",
    }
    registry = append_league_transition(None, event)
    assert validate_league_ecosystem(registry)["ai_club_count"] == 1
    assert league_ecosystem_view(registry)["latest_transition"] == event
    tampered = copy.deepcopy(registry)
    tampered["transitions"][0]["ai_clubs"][0]["team"] = "C"
    with pytest.raises(ValueError, match="club evidence"):
        validate_league_ecosystem(tampered)
