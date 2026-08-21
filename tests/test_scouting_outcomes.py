import copy

import pytest

from src.product.club_finance import evidence_identity
from src.product.scouting_outcomes import (
    append_scouting_outcome, build_scouting_outcome,
    resolve_signing_observation, scouting_outcome_view,
    validate_scouting_outcome_registry,
)
from src.simulation.scouting import scouting_observation


def _player(player_id="free-1", quality=0.7):
    return {
        "player_id": player_id, "name": "Prospect", "role": "CM", "age": 25,
        "abilities": {key: quality for key in (
            "tech", "pass_skill", "vision", "spatial", "pace", "press",
            "curve", "shot", "power", "aerial", "heading", "mental",
            "gk_reflex", "gk_aerial",
        )},
    }


def _sources(control="manager"):
    market_id = "free-market-0003-B"
    incoming = _player()
    signing = {
        "schema_version": 1, "type": "free_agent_signing",
        "target_season_id": "season-0003", "team": "B", "control": control,
        "market_id": market_id, "free_agent_id": "free-1",
        "free_agent_entry": {"origin_team": "A", "player": copy.deepcopy(incoming)},
        "incoming": incoming,
    }
    archive = {
        "schema_version": 1, "season_id": "season-0003",
        "final_standings": [
            {"team": "B", "position": 2, "points": 15},
            {"team": "A", "position": 1, "points": 18},
        ],
    }
    participation = {
        "schema_version": 1, "season_id": "season-0003", "team": "B",
        "scheduled_fixture_ids": ["f1", "f2"],
        "reports": [
            {"fixture_id": "f1", "match_id": "m1", "report": "one.json",
             "sha256": "1" * 64, "minutes": {"free-1": 90.0}},
            {"fixture_id": "f2", "match_id": "m2", "report": "two.json",
             "sha256": "2" * 64, "minutes": {"free-1": 45.0}},
        ],
        "missing_fixture_ids": [], "coverage": "complete",
        "player_totals": {"free-1": {"minutes": 135.0, "appearances": 2}},
        "claim_boundary": "test evidence",
    }
    settlement = {
        "schema_version": 1, "type": "season_settlement",
        "season_id": "season-0003", "team": "B",
        "archive_identity": evidence_identity(archive), "expense_total": 6,
    }
    observation = scouting_observation(
        market_id=market_id, team="B", player_id="free-1",
        true_quality=0.7, level="scouted",
    )
    return signing, archive, participation, settlement, observation


def test_outcome_replays_observation_usage_wage_and_tamper_detection():
    signing, archive, participation, settlement, observation = _sources()
    transition = {
        "signings": [signing], "ai_decisions": [],
    }
    registry = {"schema_version": 1, "reports": [{
        "market_id": signing["market_id"], "team": "B",
        "player_id": "free-1", "observation": observation,
    }]}
    resolved, source = resolve_signing_observation(signing, transition, registry)
    outcome = build_scouting_outcome(
        signing, observation=resolved, observation_source=source,
        archive=archive, participation=participation, settlement=settlement,
    )
    assert outcome["realized"]["usage_band"] == "core"
    assert outcome["realized"]["utilization_share"] == 0.75
    assert outcome["realized"]["wage_tier"] == 3
    assert outcome["realized"]["interval_contained_truth"] is True

    stored = append_scouting_outcome(None, outcome)
    assert validate_scouting_outcome_registry(stored)["outcome_count"] == 1
    tampered = copy.deepcopy(stored)
    tampered["outcomes"][0]["realized"]["minutes"] = 1.0
    with pytest.raises(ValueError, match="outcome replay mismatch"):
        validate_scouting_outcome_registry(tampered)


def test_public_view_reveals_manager_details_but_only_aggregates_ai_truth():
    signing, archive, participation, settlement, observation = _sources()
    manager = build_scouting_outcome(
        signing, observation=observation,
        observation_source="manager_scouting_report", archive=archive,
        participation=participation, settlement=settlement,
    )
    ai_signing = copy.deepcopy(signing)
    ai_signing["control"] = "ai"
    ai_signing["team"] = "A"
    ai_signing["market_id"] = "free-market-0003-A"
    ai_signing["free_agent_id"] = "free-2"
    ai_signing["incoming"] = _player("free-2", 0.66)
    ai_signing["free_agent_entry"] = {
        "origin_team": "C", "player": copy.deepcopy(ai_signing["incoming"]),
    }
    ai_archive = copy.deepcopy(archive)
    ai_participation = copy.deepcopy(participation)
    ai_participation["team"] = "A"
    ai_participation["reports"] = []
    ai_participation["missing_fixture_ids"] = ["f1", "f2"]
    ai_participation["player_totals"] = {}
    ai_participation["coverage"] = "unavailable"
    ai_settlement = copy.deepcopy(settlement)
    ai_settlement["team"] = "A"
    ai_settlement["archive_identity"] = evidence_identity(ai_archive)
    ai_observation = scouting_observation(
        market_id=ai_signing["market_id"], team="A", player_id="free-2",
        true_quality=0.66, level="scouted",
    )
    ai = build_scouting_outcome(
        ai_signing, observation=ai_observation,
        observation_source="ai_scouting_report", archive=ai_archive,
        participation=ai_participation, settlement=ai_settlement,
    )
    registry = append_scouting_outcome(None, manager)
    registry = append_scouting_outcome(registry, ai)

    public = scouting_outcome_view(registry, manager_team="B")
    assert public["manager_outcome_count"] == 1
    assert public["ai_outcome_count"] == 1
    assert public["manager_outcomes"][0]["player_id"] == "free-1"
    assert "signing_identity" not in public["manager_outcomes"][0]
    assert "archive_identity" not in public["manager_outcomes"][0]
    assert public["coverage_counts"] == {
        "complete": 1, "partial": 0, "unavailable": 1,
    }


def test_partial_evidence_never_claims_a_usage_band_and_ai_report_replays():
    signing, archive, participation, settlement, observation = _sources("ai")
    transition = {
        "signings": [signing],
        "ai_decisions": [{
            "team": "B", "market_id": signing["market_id"],
            "scouting_reports": [{
                "player_id": signing["free_agent_id"],
                "observation": observation,
            }],
        }],
    }
    resolved, source = resolve_signing_observation(signing, transition, None)
    assert source == "ai_scouting_report"
    assert resolved == observation

    participation["reports"] = participation["reports"][:1]
    participation["missing_fixture_ids"] = ["f2"]
    participation["player_totals"] = {
        "free-1": {"minutes": 90.0, "appearances": 1},
    }
    participation["coverage"] = "partial"
    outcome = build_scouting_outcome(
        signing, observation=resolved, observation_source=source,
        archive=archive, participation=participation, settlement=settlement,
    )
    assert outcome["realized"]["usage_band"] == "evidence_partial"

    wrong_source = copy.deepcopy(outcome)
    wrong_source["observation_source"] = "manager_scouting_report"
    with pytest.raises(ValueError, match="observation source mismatch"):
        validate_scouting_outcome_registry({
            "schema_version": 1, "outcomes": [wrong_source],
        })

    extra_field = copy.deepcopy(outcome)
    extra_field["realized"]["causal_grade"] = "excellent"
    with pytest.raises(ValueError, match="invalid scouting outcome"):
        validate_scouting_outcome_registry({
            "schema_version": 1, "outcomes": [extra_field],
        })
