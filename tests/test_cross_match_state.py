import json
from types import SimpleNamespace

import pytest

from src.simulation.cross_match_state import (
    PlayerCarryover,
    TeamSquadCarryover,
    ingest_match_result,
    recover_carryover,
    save_persistence,
    recover_persisted_teams,
)


def _agent(team: str, carry: TeamSquadCarryover | None = None):
    return SimpleNamespace(
        team_name=team,
        squad_carryover=carry or TeamSquadCarryover(team_id=team),
        fatigue=0.0,
        injury_load=0.2,
        readiness=0.5,
        semantic_memory={},
        team_dynamics={},
    )


def _settle(agent, *, stats=None, injuries=None):
    ingest_match_result(
        agent,
        roster=None,
        result="draw",
        score_diff=0,
        xg_for=1.0,
        xg_against=1.0,
        prof_score=0.0,
        social_chaos=0.0,
        stage_name="md-01",
        micro_player_stats=stats or {},
        new_injuries=injuries or [],
    )


def test_team_carryover_is_backward_compatible_without_fatigue():
    carry = TeamSquadCarryover.from_dict({"team_id": "Brazil", "players": {}})
    assert carry.team_fatigue_ema == 0.0
    assert carry.to_dict()["team_fatigue_ema"] == 0.0


def test_persistence_merges_teams_instead_of_erasing_previous_matches(tmp_path):
    save_persistence(str(tmp_path), {"Brazil": _agent("Brazil")})
    save_persistence(str(tmp_path), {"Argentina": _agent("Argentina")})
    path = tmp_path / "data" / "persistence" / "squad_carryover.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"Brazil", "Argentina"}
    assert not list(path.parent.glob(".*.tmp"))


def test_persistence_fails_closed_on_corrupt_existing_payload(tmp_path):
    path = tmp_path / "data" / "persistence" / "squad_carryover.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid squad carryover"):
        save_persistence(str(tmp_path), {"Brazil": _agent("Brazil")})
    assert path.read_text(encoding="utf-8") == "[]"


def test_match_load_raises_fatigue_and_recovery_decays_it():
    agent = _agent("Brazil")
    _settle(agent, stats={f"p{i}": {"minutes": 90} for i in range(11)})
    loaded = agent.squad_carryover.team_fatigue_ema
    assert 0.39 <= loaded <= 0.40
    recover_carryover(agent, rest_units=1.0)
    assert 0.0 < agent.squad_carryover.team_fatigue_ema < loaded


def test_only_players_with_minutes_get_an_appearance_and_new_injury_survives():
    carry = TeamSquadCarryover(
        team_id="Brazil",
        players={
            "starter": PlayerCarryover(player_id="starter"),
            "bench": PlayerCarryover(player_id="bench"),
        },
    )
    agent = _agent("Brazil", carry)
    _settle(agent, stats={"starter": {"minutes": 90}}, injuries=["starter"])
    assert carry.players["starter"].matches_played == 1
    assert carry.players["bench"].matches_played == 0
    assert carry.players["starter"].injury_matches_left >= 1


def test_pre_existing_absence_consumes_fixture_before_new_sanction_is_assigned():
    player = PlayerCarryover(player_id="p", suspension_matches_left=1)
    carry = TeamSquadCarryover(team_id="Brazil", players={"p": player})
    agent = _agent("Brazil", carry)
    _settle(agent, stats={"p": {"minutes": 0, "red_cards": 1}})
    assert player.suspension_matches_left == 1


def test_settlement_and_matchday_recovery_are_idempotent(tmp_path):
    agent = _agent("Brazil")
    stats = {f"p{i}": {"minutes": 90} for i in range(11)}
    ingest_match_result(
        agent, roster=None, result="win", score_diff=1,
        xg_for=1.5, xg_against=0.8, prof_score=0.0, social_chaos=0.0,
        stage_name="md-01", micro_player_stats=stats,
        transaction_id="season-1:fixture-1",
    )
    first_fatigue = agent.squad_carryover.team_fatigue_ema
    ingest_match_result(
        agent, roster=None, result="win", score_diff=1,
        xg_for=1.5, xg_against=0.8, prof_score=0.0, social_chaos=0.0,
        stage_name="md-01", micro_player_stats=stats,
        transaction_id="season-1:fixture-1",
    )
    assert agent.squad_carryover.team_fatigue_ema == first_fatigue
    save_persistence(str(tmp_path), {"Brazil": agent})
    for _ in range(2):
        recover_persisted_teams(
            str(tmp_path), ["Brazil"], rest_units=1.0,
            transaction_id="season-1:recovery:md01",
        )
    payload = json.loads((
        tmp_path / "data" / "persistence" / "squad_carryover.json"
    ).read_text(encoding="utf-8"))["Brazil"]
    expected = first_fatigue * __import__("math").exp(-0.42)
    assert payload["team_fatigue_ema"] == pytest.approx(expected)
    assert payload["recovery_ids"] == ["season-1:recovery:md01"]


def test_medical_recovery_is_injury_only_fractional_and_idempotent():
    agent = _agent("Brazil")
    carry = agent.squad_carryover
    carry.players["injured"] = PlayerCarryover(
        player_id="injured", injury_matches_left=3, injury_severity=0.8,
    )
    carry.players["suspended"] = PlayerCarryover(
        player_id="suspended", suspension_matches_left=2,
    )
    recover_carryover(
        agent, medical_recovery_credit=0.6, transaction_id="medical-md01",
    )
    first = carry.players["injured"]
    assert first.injury_matches_left == 3
    assert first.medical_recovery_credit == pytest.approx(0.6)
    severity = first.injury_severity
    recover_carryover(
        agent, medical_recovery_credit=0.6, transaction_id="medical-md01",
    )
    assert first.medical_recovery_credit == pytest.approx(0.6)
    assert first.injury_severity == pytest.approx(severity)

    recover_carryover(
        agent, medical_recovery_credit=0.6, transaction_id="medical-md02",
    )
    assert first.injury_matches_left == 2
    assert first.medical_recovery_credit == pytest.approx(0.2)
    assert carry.players["suspended"].suspension_matches_left == 2
    assert carry.players["suspended"].medical_recovery_credit == 0.0
