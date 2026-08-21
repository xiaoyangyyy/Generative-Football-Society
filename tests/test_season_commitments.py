import copy
import hashlib
import json

import pytest

from src.product.season import (
    SeasonPlan, manager_board_review, manager_season_profile,
    new_season_state, validate_season_state,
)
from src.product.season_commitments import (
    SeasonCommitmentPlan, commitment_progress_from_evidence,
)


def _complete_managed(
    state, *, tactics=("balanced", "gegenpress", "balanced"),
    rotations=("strongest", "strongest", "balanced"),
):
    index = 0
    for fixture in state["fixtures"]:
        managed = "A" in {fixture["home"], fixture["away"]}
        fixture.update({
            "state": "completed", "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json",
            "score": (
                {"home": 2, "away": 0} if fixture["home"] == "A"
                else {"home": 0, "away": 2} if fixture["away"] == "A"
                else {"home": 0, "away": 0}
            ),
        })
        if managed:
            fixture["manager_decision"] = {
                "team": "A", "tactic": tactics[index],
                "rotation": rotations[index],
            }
            index += 1
    state["state"] = "complete"


def test_commitment_plan_is_explicit_bounded_and_requires_manager_control():
    plan = SeasonCommitmentPlan("adaptive", "trust_core")
    assert SeasonCommitmentPlan.from_payload(plan.as_dict()) == plan
    with pytest.raises(ValueError, match="tactic commitment"):
        SeasonCommitmentPlan("invented", "trust_core")
    with pytest.raises(ValueError, match="rotation commitment"):
        SeasonCommitmentPlan("adaptive", "invented")
    with pytest.raises(ValueError, match="requires manager_team"):
        SeasonPlan(
            teams=("A", "B", "C", "D"), manager_commitments=plan,
        )


def test_commitments_replay_completed_decisions_and_settle_transparently():
    state = new_season_state(
        SeasonPlan(
            teams=("A", "B", "C", "D"), manager_team="A",
            manager_objective="champion",
            manager_commitments=SeasonCommitmentPlan("adaptive", "trust_core"),
        ),
        season_id="season-commitments", seed=9,
        created_at="2026-01-01T00:00:00Z",
    )
    contract = state["season_commitments"]
    assert contract["tactic"]["minimum_distinct"] == 2
    assert contract["rotation"]["counted_rotations"] == ["strongest"]

    first = next(row for row in state["fixtures"] if "A" in {row["home"], row["away"]})
    first.update({
        "state": "completed", "match_id": first["fixture_id"],
        "report": f"outputs/{first['fixture_id']}.json",
        "score": {"home": 1, "away": 0},
        "manager_decision": {
            "team": "A", "tactic": "balanced", "rotation": "strongest",
        },
    })
    progress = manager_season_profile(state)["commitments"]
    assert progress["final"] is False
    assert [row["status"] for row in progress["entries"][1:]] == [
        "at_risk", "on_track",
    ]

    _complete_managed(state)
    profile = manager_season_profile(state)
    progress = profile["commitments"]
    assert progress["final"] is True
    assert [row["status"] for row in progress["entries"]] == [
        "fulfilled", "fulfilled", "fulfilled",
    ]
    assert progress["entries"][1]["metric"]["distinct_tactics"] == 2
    assert progress["entries"][2]["metric"]["counted_share"] == 0.666667
    review = manager_board_review(profile, league_size=4)
    assert review["components"]["season_commitments"] == 6
    assert review["evidence"]["commitment_statuses"] == {
        "board_objective": "fulfilled",
        "tactical_identity": "fulfilled",
        "squad_stewardship": "fulfilled",
    }


def test_self_consistent_commitment_contract_tamper_fails_source_replay():
    state = new_season_state(
        SeasonPlan(teams=("A", "B", "C", "D"), manager_team="A"),
        season_id="season-tamper", seed=4,
        created_at="2026-01-01T00:00:00Z",
    )
    tampered = copy.deepcopy(state)
    tampered["season_commitments"]["tactic"]["identity_tactic"] = "balanced"
    frozen = copy.deepcopy(tampered["season_commitments"])
    frozen.pop("contract_id")
    tampered["season_commitments"]["contract_id"] = hashlib.sha256(
        json.dumps(
            frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="source replay mismatch"):
        validate_season_state(tampered)


def test_commitment_evidence_rejects_duplicates_and_early_final_settlement():
    state = new_season_state(
        SeasonPlan(teams=("A", "B", "C", "D"), manager_team="A"),
        season_id="season-evidence", seed=5,
        created_at="2026-01-01T00:00:00Z",
    )
    objective = manager_season_profile(state)["objective"]
    row = {
        "fixture_id": "fixture-1", "matchday": 1,
        "tactic": "team_identity", "rotation": "balanced",
    }
    with pytest.raises(ValueError, match="entry is incomplete"):
        commitment_progress_from_evidence(
            state["season_commitments"], journal=[row, row],
            objective=objective, final=False,
        )
    with pytest.raises(ValueError, match="final commitment evidence is incomplete"):
        commitment_progress_from_evidence(
            state["season_commitments"], journal=[],
            objective={**objective, "status": "missed"}, final=True,
        )


def test_commitments_do_not_change_match_state_or_claim_causality():
    state = new_season_state(
        SeasonPlan(
            teams=("A", "B", "C", "D"), manager_team="A",
            manager_commitments=SeasonCommitmentPlan("club_identity", "share_load"),
        ),
        season_id="season-boundary", seed=2,
        created_at="2026-01-01T00:00:00Z",
    )
    assert all(fixture["score"] is None for fixture in state["fixtures"])
    assert "never modify match physics" in (
        state["plan"]["manager_commitments"]["claim_boundary"]
    )
    assert "not a performance forecast" in (
        state["season_commitments"]["claim_boundary"]
    )
