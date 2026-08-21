import copy

import pytest

from src.product.club_finance import evidence_identity
from src.product.sporting_director import (
    SportingDirective, build_sporting_brief, validate_sporting_brief,
    validate_sporting_choices,
)


def _player(index, role, quality=0.6, age=25):
    return {
        "player_id": f"p{index:02d}", "name": f"Player {index}",
        "team_id": "A", "role": role, "age": age,
        "abilities": {key: quality for key in (
            "tech", "pass_skill", "vision", "spatial", "pace", "press",
            "curve", "shot", "power", "aerial", "heading", "mental",
            "gk_reflex", "gk_aerial",
        )},
    }


def _brief():
    roles = ["GK", "RB", "CB", "CB", "LB", "DM", "CM", "AM", "RW", "ST", "LW"]
    players = [_player(index, role) for index, role in enumerate(roles)]
    roster = {"team_id": "A", "players": players, "formation": "4-3-3"}
    outgoing = [{
        "player_id": player["player_id"], "name": player["name"],
        "role": player["role"], "age": player["age"], "quality": 0.6,
        "wage_tier": 1,
    } for player in players]
    recruitment = {
        "team": "A", "market_id": "market-0002-a", "club_balance": 6,
        "available_budget": 6, "outgoing_players": outgoing,
        "candidates": [{
            "player_id": "candidate-cm", "name": "CM Prospect", "role": "CM",
            "age": 23, "quality": 0.66, "recruitment_cost": 3,
        }, {
            "player_id": "candidate-st", "name": "ST Veteran", "role": "ST",
            "age": 27, "quality": 0.67, "recruitment_cost": 5,
        }],
    }
    lifecycle_players = [{
        "player_id": player["player_id"], "role": player["role"],
        "contract_expires": player["role"] == "CM", "retires": False,
    } for player in players]
    lifecycle = {
        "team": "A", "source_season_id": "season-0001",
        "target_season_id": "season-0002", "cycle_id": "cycle-a",
        "players": lifecycle_players, "expiring_player_ids": ["p06"],
        "retiring_player_ids": [],
    }
    free_agent = {
        "team": "A", "market_id": "free-market-0002-A",
        "scouting_budget": {"limit": 2, "used": 1, "remaining": 1},
        "candidates": [{
            "player_id": "free-cm", "name": "Free CM", "role": "CM",
            "age": 24, "scouted": True,
            "observation": {
                "estimated_quality": 0.68, "quality_low": 0.64,
                "quality_high": 0.71,
            },
        }],
    }
    outcomes = {
        "manager_outcome_count": 1,
        "manager_outcomes": [{
            "realized": {"absolute_estimation_error": 0.02},
        }],
    }
    return build_sporting_brief(
        team="A", source_season_id="season-0001",
        target_season_id="season-0002", roster=roster,
        recruitment_market=recruitment, lifecycle_preview=lifecycle,
        free_agent_market=free_agent, scouting_outcomes=outcomes,
    )


def test_brief_binds_all_management_sources_and_rejects_tampering():
    brief = _brief()
    validate_sporting_brief(brief)
    assert brief["source_roster_identity"]
    assert brief["mean_historical_scouting_error"] == 0.02
    assert "CM" in brief["recommendation"]["priority_roles"]
    tampered = copy.deepcopy(brief)
    tampered["available_budget"] = 99
    with pytest.raises(ValueError, match="identity"):
        validate_sporting_brief(tampered)
    self_consistent_hash = copy.deepcopy(brief)
    self_consistent_hash["role_diagnostics"][0]["depth"] += 1
    self_consistent_hash.pop("planning_id")
    self_consistent_hash["planning_id"] = evidence_identity(self_consistent_hash)
    with pytest.raises(ValueError, match="diagnostic replay mismatch"):
        validate_sporting_brief(self_consistent_hash)

    invalid_budget = copy.deepcopy(brief)
    invalid_budget["available_budget"] = 9
    invalid_budget.pop("planning_id")
    invalid_budget["planning_id"] = evidence_identity(invalid_budget)
    with pytest.raises(ValueError, match="evidence summary"):
        validate_sporting_brief(invalid_budget)

    extra_claim = copy.deepcopy(brief)
    extra_claim["causal_grade"] = "proven"
    extra_claim.pop("planning_id")
    extra_claim["planning_id"] = evidence_identity(extra_claim)
    with pytest.raises(ValueError, match="identity"):
        validate_sporting_brief(extra_claim)


def test_one_directive_constrains_recruitment_retention_and_free_agents():
    brief = _brief()
    directive = SportingDirective(
        brief["planning_id"], philosophy="win_now", risk_level="low",
        priority_roles=("CM",),
    )
    recruitment = {
        "moves": [{"candidate_id": "candidate-cm", "outgoing_player_id": "p06"}],
    }
    free_agent = {
        "free_agent_id": "free-cm", "outgoing_player_id": "p06",
    }
    evaluation = validate_sporting_choices(
        directive, brief, recruitment=recruitment,
        retention={"renew_player_ids": ["p06"]}, free_agent=free_agent,
    )
    assert evaluation["selected_recruitment_moves"] == 1
    assert evaluation["selected_free_agent"] is True
    assert evaluation["selected_renewals"] == 1

    outside_priority = copy.deepcopy(recruitment)
    outside_priority["moves"][0] = {
        "candidate_id": "candidate-st", "outgoing_player_id": "p09",
    }
    with pytest.raises(ValueError, match="priority roles"):
        validate_sporting_choices(
            directive, brief, recruitment=outside_priority,
            retention=None, free_agent=None,
        )

    unscouted = copy.deepcopy(brief)
    unscouted["free_agent_candidates"][0]["scouted"] = False
    unscouted.pop("planning_id")
    unscouted["planning_id"] = evidence_identity(unscouted)
    risky = SportingDirective(
        unscouted["planning_id"], philosophy="balanced", risk_level="balanced",
        priority_roles=("CM",),
    )
    with pytest.raises(ValueError, match="requires a scouted"):
        validate_sporting_choices(
            risky, unscouted, recruitment=None, retention=None,
            free_agent=free_agent,
        )

    financial = SportingDirective(
        brief["planning_id"], philosophy="financial_control",
        risk_level="high", priority_roles=(),
    )
    with pytest.raises(ValueError, match="renewal limit"):
        validate_sporting_choices(
            financial, brief, recruitment=None,
            retention={"renew_player_ids": ["p01", "p02", "p03"]},
            free_agent=None,
        )
