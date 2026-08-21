import copy

import pytest

from src.product.player_promises import (
    PlayerPromisePlan, PlayerRolePromise, build_player_promise_contract,
    player_promise_progress, settle_player_promise_outcomes,
    validate_player_promise_contract,
)


def _roster():
    return {
        "team_id": "A", "formation": "4-3-3", "source": "test",
        "players": [
            {"player_id": "core", "name": "Core", "role": "CM", "age": 28},
            {"player_id": "young", "name": "Young", "role": "AM", "age": 20},
            {"player_id": "old", "name": "Old", "role": "ST", "age": 31},
        ],
    }


def _contract():
    return build_player_promise_contract(
        season_id="season-0001", team="A", total_managed_matches=3,
        roster=_roster(),
        plan=PlayerPromisePlan((
            PlayerRolePromise("core", "core"),
            PlayerRolePromise("young", "development"),
        )), control="manager",
    )


def _fixtures():
    rows = []
    for index in range(3):
        starters = ["core"] + (["young"] if index == 0 else [f"other-{index}"])
        starters.extend(f"filler-{index}-{slot}" for slot in range(9))
        rows.append({
            "fixture_id": f"fx-{index}", "matchday": index + 1,
            "home": "A", "away": f"B{index}", "state": "completed",
            "manager_decision": {"lineup": {
                "starters": starters, "bench": ["young"] if index else [],
            }},
            "player_promise_availability": {"core": True, "young": True},
        })
    return rows


def _participation(*, partial=False):
    scheduled = ["fx-0", "fx-1", "fx-2"]
    count = 2 if partial else 3
    reports = []
    for index in range(count):
        minutes = {"core": 90.0}
        if index == 0:
            minutes["young"] = 90.0
        reports.append({
            "fixture_id": f"fx-{index}", "match_id": f"m-{index}",
            "report": f"outputs/m-{index}.json", "sha256": "a" * 64,
            "minutes": minutes,
        })
    totals = {
        "core": {"minutes": 90.0 * count, "appearances": count},
        "young": {"minutes": 90.0, "appearances": 1},
    }
    return {
        "schema_version": 1, "season_id": "season-0001", "team": "A",
        "scheduled_fixture_ids": scheduled, "reports": reports,
        "missing_fixture_ids": ["fx-2"] if partial else [],
        "coverage": "partial" if partial else "complete",
        "player_totals": totals,
        "claim_boundary": "exact minutes from available persisted simulation reports",
    }


def test_player_promise_plan_is_small_unique_and_age_eligible():
    with pytest.raises(ValueError, match="unique players"):
        PlayerPromisePlan((
            PlayerRolePromise("core", "core"),
            PlayerRolePromise("core", "rotation"),
        ))
    with pytest.raises(ValueError, match="roles must be unique"):
        PlayerPromisePlan((
            PlayerRolePromise("core", "core"),
            PlayerRolePromise("young", "core"),
        ))
    with pytest.raises(ValueError, match="age 23 or younger"):
        build_player_promise_contract(
            season_id="season-0001", team="A", total_managed_matches=3,
            roster=_roster(),
            plan=PlayerPromisePlan((PlayerRolePromise("old", "development"),)),
            control="manager",
        )


def test_selection_progress_and_actual_minutes_remain_separate():
    contract = _contract()
    progress = player_promise_progress(contract, fixtures=_fixtures(), final=True)
    assert [row["status"] for row in progress["entries"]] == [
        "fulfilled", "fulfilled",
    ]
    assert progress["entries"][0]["start_share"] == 1.0
    assert progress["entries"][1]["start_share"] == 0.333333

    outcome = settle_player_promise_outcomes(
        contract, progress, _participation(),
    )
    assert [row["observed_status"] for row in outcome["players"]] == [
        "fulfilled", "fulfilled",
    ]
    assert outcome["players"][1]["minute_share"] == 0.333333
    assert outcome["players"][1]["selection_status"] == "fulfilled"


def test_partial_minutes_degrade_without_negative_player_judgment():
    contract = _contract()
    progress = player_promise_progress(contract, fixtures=_fixtures(), final=True)
    outcome = settle_player_promise_outcomes(
        contract, progress, _participation(partial=True),
    )
    assert outcome["coverage"] == "partial"
    assert {row["observed_status"] for row in outcome["players"]} == {
        "evidence_partial"
    }
    assert "no causal player grade" in outcome["claim_boundary"]


def test_injury_or_suspension_is_excused_from_manager_start_denominator():
    fixtures = _fixtures()
    fixtures[1]["player_promise_availability"]["young"] = False
    progress = player_promise_progress(_contract(), fixtures=fixtures, final=True)
    young = progress["entries"][1]
    assert young["known_lineups"] == 2
    assert young["excused_unavailable"] == 1
    assert young["excused_fixture_ids"] == ["fx-1"]
    assert young["start_share"] == 0.5
    assert young["status"] == "fulfilled"

    for fixture in fixtures:
        fixture["player_promise_availability"]["young"] = False
    all_excused = player_promise_progress(
        _contract(), fixtures=fixtures, final=True,
    )["entries"][1]
    assert all_excused["known_lineups"] == 0
    assert all_excused["excused_unavailable"] == 3
    assert all_excused["status"] == "excused"


def test_closed_contract_rejects_snapshot_or_threshold_tampering():
    contract = _contract()
    tampered = copy.deepcopy(contract)
    tampered["players"][0]["minimum_start_share"] = 0.1
    with pytest.raises(ValueError, match="contract"):
        validate_player_promise_contract(tampered)


def test_compatibility_contract_is_explicit_and_empty():
    contract = build_player_promise_contract(
        season_id="season-0001", team="A", total_managed_matches=3,
        roster=_roster(), plan=None, control="deterministic_compatibility",
    )
    assert contract["players"] == []
    assert player_promise_progress(
        contract, fixtures=_fixtures(), final=True,
    )["entries"] == []
