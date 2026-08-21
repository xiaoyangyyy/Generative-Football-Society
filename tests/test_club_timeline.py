import copy

import pytest

from src.product.club_timeline import (
    ClubEventChoice, build_timeline_resolution, club_timeline_view,
    compatible_choice, derive_club_situation, validate_club_timeline,
)


def _state(*, rotations=("strongest", "strongest"), scores=((1, 0), (1, 0))):
    fixtures = []
    opponents = ("B", "C", "D", "B", "C", "D")
    tactics = ("balanced", "balanced")
    for index, opponent in enumerate(opponents, 1):
        completed = index <= 2
        fixture = {
            "fixture_id": f"md{index:02d}-fx01", "matchday": index,
            "order": 1, "home": "A", "away": opponent,
            "state": "completed" if completed else "scheduled",
            "score": (
                {"home": scores[index - 1][0], "away": scores[index - 1][1]}
                if completed else None
            ),
            "manager_decision": (
                {"team": "A", "tactic": tactics[index - 1],
                 "rotation": rotations[index - 1]}
                if completed else None
            ),
        }
        fixtures.append(fixture)
    return {
        "schema_version": 1, "season_id": "season-0001",
        "plan": {
            "teams": ["A", "B", "C", "D"], "legs": 2,
            "manager_team": "A", "manager_objective": "top_half",
            "manager_points_target": None,
        },
        "fixtures": fixtures,
        "club_timeline": {"schema_version": 1, "events": []},
    }


def test_repeated_strongest_lineups_create_a_real_rotation_fork():
    state = _state()
    event = derive_club_situation(state)
    assert event["kind"] == "squad_load"
    assert event["recommended_choice"] == "protect_squad"
    protect = ClubEventChoice(
        event["event_id"], event["event_identity"], "protect_squad",
    )
    with pytest.raises(ValueError, match="does not resolve"):
        compatible_choice(event, tactic="balanced", rotation="invalid")
    resolution = build_timeline_resolution(
        event, protect, tactic="balanced", rotation="rotate", control="manager",
    )
    target = state["fixtures"][2]
    target["manager_decision"] = {
        "team": "A", "tactic": "balanced", "rotation": "rotate",
        "club_event_choice": protect.as_dict(),
    }
    state["club_timeline"]["events"].append(resolution)
    validate_club_timeline(state)
    view = club_timeline_view(state)
    assert view["current_situation"] is None
    assert view["resolved_count"] == 1


def test_poor_form_event_replays_and_rejects_self_consistent_tampering():
    state = _state(rotations=("balanced", "rotate"), scores=((0, 1), (0, 2)))
    event = derive_club_situation(state)
    assert event["kind"] == "form_response"
    choice = ClubEventChoice(
        event["event_id"], event["event_identity"], "reset_approach",
    )
    resolution = build_timeline_resolution(
        event, choice, tactic="gegenpress", rotation="balanced",
        control="manager",
    )
    state["fixtures"][2]["manager_decision"] = {
        "team": "A", "tactic": "gegenpress", "rotation": "balanced",
        "club_event_choice": choice.as_dict(),
    }
    state["club_timeline"]["events"] = [resolution]
    validate_club_timeline(state)

    tampered = copy.deepcopy(state)
    tampered["club_timeline"]["events"][0]["event"]["trigger"]["recent_points"] = 6
    with pytest.raises(ValueError, match="resolution identity"):
        validate_club_timeline(tampered)


def test_momentum_and_midpoint_objective_pressure_are_distinct_branches():
    momentum = _state(rotations=("balanced", "rotate"), scores=((2, 0), (1, 0)))
    event = derive_club_situation(momentum)
    assert event["kind"] == "momentum_management"
    assert event["trigger"]["recent_points"] == 6
    assert compatible_choice(
        event, tactic="balanced", rotation="balanced",
    ).choice_id == "keep_identity"

    pressure = _state(rotations=("balanced", "rotate"), scores=((0, 1), (0, 1)))
    third = pressure["fixtures"][2]
    third.update({
        "state": "completed", "score": {"home": 0, "away": 2},
        "manager_decision": {
            "team": "A", "tactic": "balanced", "rotation": "balanced",
        },
    })
    event = derive_club_situation(pressure)
    assert event["kind"] == "objective_pressure"
    assert event["trigger"]["completed_managed_matches"] == 3
    assert event["trigger"]["midpoint"] == 3
    assert compatible_choice(
        event, tactic="gegenpress", rotation="balanced",
    ).choice_id == "change_approach"


def test_event_choice_cannot_survive_without_its_timeline_ledger():
    state = _state()
    event = derive_club_situation(state)
    choice = ClubEventChoice(
        event["event_id"], event["event_identity"], "push_starters",
    )
    state["fixtures"][2]["manager_decision"] = {
        "team": "A", "tactic": "balanced", "rotation": "strongest",
        "club_event_choice": choice.as_dict(),
    }
    state.pop("club_timeline")
    with pytest.raises(ValueError, match="missing from timeline"):
        validate_club_timeline(state)
