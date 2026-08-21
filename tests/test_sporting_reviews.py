import copy

import pytest

from src.product.club_finance import evidence_identity
from src.product.sporting_reviews import (
    append_sporting_review, build_sporting_review, sporting_review_view,
    validate_sporting_review,
)
from tests.test_sporting_director import _brief


def _sources(*, coverage="complete"):
    brief = _brief()
    directive = {
        "schema_version": 1, "planning_id": brief["planning_id"],
        "philosophy": "balanced", "risk_level": "balanced",
        "priority_roles": ["CM"],
    }
    plan = {
        "schema_version": 1, "teams": ["A", "B", "C", "D"], "legs": 1,
        "fast": False, "manager_team": "A", "manager_objective": "top_half",
        "manager_points_target": None, "manager_resources": {
            "schema_version": 1, "recovery": 2, "medical": 2,
            "sports_science": 2,
        },
        "manager_recruitment": None,
        "manager_retention": None, "manager_free_agent": None,
        "manager_sporting_directive": directive,
    }
    from src.product.season import SeasonPlan
    parsed = SeasonPlan.from_payload(plan)
    from src.product.sporting_director import validate_sporting_choices
    evaluation = validate_sporting_choices(
        parsed.manager_sporting_directive, brief,
        recruitment=None, retention=None, free_agent=None,
    )
    archive = {
        "schema_version": 1, "season_id": "season-0002", "plan": plan,
        "sporting_brief": brief, "sporting_evaluation": evaluation,
        "recruitment_transaction": None,
        "manager_profile": {"objective": {
            "status": "missed", "current_position": 2, "current_points": 1,
        }},
        "board_review": {
            "grade": "negative", "confidence_delta": -5,
            "reputation_delta": -2,
        },
    }
    players = [{
        "player_id": row["player_id"], "name": row["name"],
        "team_id": "A", "role": row["role"], "age": row["age"],
        "abilities": {key: row["quality"] for key in (
            "tech", "pass_skill", "vision", "spatial", "pace", "press",
            "curve", "shot", "power", "aerial", "heading", "mental",
            "gk_reflex", "gk_aerial",
        )},
    } for row in brief["outgoing_players"]]
    roster = {"team_id": "A", "formation": "4-3-3", "players": players}
    participation = {
        "schema_version": 1, "season_id": "season-0002", "team": "A",
        "scheduled_fixture_ids": ["f1"], "reports": [],
        "missing_fixture_ids": ["f1"], "coverage": coverage,
        "player_totals": {},
        "claim_boundary": "exact minutes from available persisted simulation reports; missing reports do not receive invented participation",
    }
    if coverage == "complete":
        participation.update({
            "reports": [{
                "fixture_id": "f1", "match_id": "m1", "report": "m1.json",
                "sha256": "0" * 64, "minutes": {},
            }], "missing_fixture_ids": [],
        })
    settlement = {
        "schema_version": 1, "type": "season_settlement",
        "season_id": "season-0002", "team": "A",
        "archive_identity": evidence_identity(archive), "delta": -1,
        "revenue_total": 5, "expense_total": 6,
    }
    return archive, roster, participation, settlement


def test_review_is_noncausal_replayable_and_privacy_minimized():
    archive, roster, participation, settlement = _sources()
    review = build_sporting_review(
        archive, roster, participation=participation, settlement=settlement,
    )
    validate_sporting_review(review)
    assert [row["id"] for row in review["learning_signals"]] == [
        "priority_unaddressed", "objective_missed", "negative_season_balance",
    ]
    registry = append_sporting_review(None, review)
    public = sporting_review_view(registry)
    assert public["latest_review"]["review_id"] == review["review_id"]
    assert "post_squad" not in public["latest_review"]
    assert "archive_identity" not in public["latest_review"]
    assert "causal" in review["claim_boundary"]


def test_review_rejects_self_consistent_signal_and_numeric_tampering():
    archive, roster, participation, settlement = _sources(coverage="unavailable")
    review = build_sporting_review(
        archive, roster, participation=participation, settlement=settlement,
    )
    tampered = copy.deepcopy(review)
    tampered["learning_signals"] = []
    tampered.pop("review_id")
    tampered["review_id"] = evidence_identity(tampered)
    with pytest.raises(ValueError, match="learning signals"):
        validate_sporting_review(tampered)
    invalid = copy.deepcopy(review)
    invalid["season_result"]["finance_delta"] = 99
    invalid.pop("review_id")
    invalid["review_id"] = evidence_identity(invalid)
    with pytest.raises(ValueError, match="season result"):
        validate_sporting_review(invalid)
