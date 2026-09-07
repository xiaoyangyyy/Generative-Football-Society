import copy
import hashlib
import json

import pytest

from src.product.world_state_evidence import (
    WORLD_STATE_SCHEMA_VERSION,
    _build_fixture_world_state_transition,
    build_fixture_world_state_transition,
    capture_world_state,
    validate_fixture_world_state_transition,
    validate_team_state_snapshot,
)
from src.simulation.agent import SocietyAgent
from src.simulation.cross_match_state import PlayerCarryover, TeamSquadCarryover
from src.simulation.society_continuity import capture_society_continuity


def _roster(root, team):
    path = root / f"data/rosters/{team}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "team_id": team, "formation": "4-3-3",
        "players": [{
            "player_id": f"{team.lower()}-01", "name": f"{team} One",
            "role": "GK", "squad_role": "starter", "availability": 1.0,
            "condition": {"composure": 0.6},
            "abilities": {key: 0.6 for key in (
                "tech", "pass_skill", "vision", "spatial", "pace", "press",
                "shot", "power", "aerial", "mental", "gk_reflex", "gk_aerial",
            )},
        }],
    }), encoding="utf-8")


def _write_carryover(root, *, fatigue, injured, recovery=False):
    path = root / "data/persistence/squad_carryover.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for team in ("Brazil", "Argentina"):
        player = PlayerCarryover(
            player_id=f"{team.lower()}-01", name=f"{team} One",
            matches_played=1, minutes_ema=13.5,
            injury_matches_left=injured if team == "Brazil" else 0,
            injury_severity=0.6 if injured and team == "Brazil" else 0.0,
            form_ema=0.61,
        )
        carry = TeamSquadCarryover(
            team_id=team, players={player.player_id: player},
            team_fatigue_ema=fatigue if team == "Brazil" else fatigue / 2,
            squad_morale_ema=0.62,
            last_match_stage="md01-fx01",
        )
        if recovery:
            carry.team_fatigue_ema *= 0.6
            carry.players[player.player_id].injury_matches_left = 0
            carry.players[player.player_id].injury_severity = 0.3
        rows[team] = carry.to_dict()
    path.write_text(json.dumps(rows), encoding="utf-8")


def _rehash(payload):
    frozen = copy.deepcopy(payload)
    frozen.pop("transition_identity", None)
    payload["transition_identity"] = hashlib.sha256(json.dumps(
        frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _rehash_snapshot(payload):
    frozen = copy.deepcopy(payload)
    frozen.pop("source_identity", None)
    payload["source_identity"] = hashlib.sha256(json.dumps(
        frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def test_three_phase_world_state_transition_replays_and_exposes_recovery(tmp_path):
    for team in ("Brazil", "Argentina"):
        _roster(tmp_path, team)
    before = capture_world_state(tmp_path, ("Brazil", "Argentina"))
    assert before["Brazil"]["source"] == "deterministic_roster_baseline"

    _write_carryover(tmp_path, fatigue=0.8, injured=2)
    after_match = capture_world_state(tmp_path, ("Brazil", "Argentina"))
    _write_carryover(tmp_path, fatigue=0.8, injured=2, recovery=True)
    after_recovery = capture_world_state(tmp_path, ("Brazil", "Argentina"))
    transition = build_fixture_world_state_transition(
        season_id="season-0001", fixture_id="md01-fx01", match_id="0001-match",
        home="Brazil", away="Argentina", before_match=before,
        after_match=after_match, after_recovery=after_recovery,
    )

    validate_fixture_world_state_transition(
        transition, season_id="season-0001", fixture_id="md01-fx01",
        match_id="0001-match", home="Brazil", away="Argentina",
    )
    assert transition["match_delta"]["Brazil"]["metrics_delta"][
        "team_fatigue_ema"
    ] == 0.8
    assert transition["match_delta"]["Brazil"]["summary"]["new_injuries"] == 1
    assert transition["recovery_delta"]["Brazil"]["metrics_delta"][
        "team_fatigue_ema"
    ] == -0.32
    assert transition["recovery_delta"]["Brazil"]["summary"][
        "injuries_cleared"
    ] == 1


def test_world_state_transition_rejects_rehashed_derived_delta_tampering(tmp_path):
    for team in ("Brazil", "Argentina"):
        _roster(tmp_path, team)
    before = capture_world_state(tmp_path, ("Brazil", "Argentina"))
    _write_carryover(tmp_path, fatigue=0.4, injured=0)
    after = capture_world_state(tmp_path, ("Brazil", "Argentina"))
    transition = build_fixture_world_state_transition(
        season_id="season-0001", fixture_id="md01-fx01", match_id="0001-match",
        home="Brazil", away="Argentina", before_match=before, after_match=after,
    )
    transition["match_delta"]["Brazil"]["metrics_delta"][
        "team_fatigue_ema"
    ] = 99.0
    _rehash(transition)

    with pytest.raises(ValueError, match="replay mismatch"):
        validate_fixture_world_state_transition(
            transition, season_id="season-0001", fixture_id="md01-fx01",
            match_id="0001-match", home="Brazil", away="Argentina",
        )


def test_world_state_snapshot_rejects_rehashed_availability_summary(tmp_path):
    _roster(tmp_path, "Brazil")
    snapshot = capture_world_state(tmp_path, ("Brazil",))["Brazil"]
    snapshot["metrics"]["unavailable_players"] = 1
    _rehash_snapshot(snapshot)

    with pytest.raises(ValueError, match="availability summary mismatch"):
        validate_team_state_snapshot(snapshot, team="Brazil")


def test_world_state_snapshot_rejects_malformed_player_as_validation_error(tmp_path):
    _roster(tmp_path, "Brazil")
    snapshot = capture_world_state(tmp_path, ("Brazil",))["Brazil"]
    snapshot["players"] = ["not-a-player-object"]
    _rehash_snapshot(snapshot)

    with pytest.raises(ValueError, match="player world-state snapshot is invalid"):
        validate_team_state_snapshot(snapshot, team="Brazil")


def test_world_state_v3_exposes_replayable_content_free_society_transition(tmp_path):
    for team in ("Brazil", "Argentina"):
        _roster(tmp_path, team)
    before = capture_world_state(tmp_path, ("Brazil", "Argentina"))
    agent = SocietyAgent(
        "Brazil", {"final_status_score": 55.0}, random_root_seed=73,
    )
    agent._register_memory_event(
        "A private prior-world narrative",
        event_type="micro_cognitive",
    )
    agent.apply_reflection_payload({
        "reflection": "A private shadow adjustment proposal",
        "confidence": 0.8,
        "suggested_adjustments": {"risk_budget": -0.04},
    }, operation_id="reflection:md01")
    society_state = capture_society_continuity(
        agent, source_transaction_id="season-0001:md01-fx01",
    )
    carry = TeamSquadCarryover(
        team_id="Brazil", society_state=society_state,
    )
    path = tmp_path / "data/persistence/squad_carryover.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"Brazil": carry.to_dict()}), encoding="utf-8")
    after = capture_world_state(tmp_path, ("Brazil", "Argentina"))

    transition = build_fixture_world_state_transition(
        season_id="season-0001",
        fixture_id="md01-fx01",
        match_id="0001-match",
        home="Brazil",
        away="Argentina",
        before_match=before,
        after_match=after,
    )

    assert transition["schema_version"] == WORLD_STATE_SCHEMA_VERSION == 3
    society = transition["match_delta"]["Brazil"]["society_transition"]
    assert society["available"] is True
    assert society["before_available"] is False
    assert society["after_available"] is True
    assert society["memory_record_delta"] >= 1
    assert society["cognitive_memory_delta"] == 1
    assert society["meta_learning_after"]["shadow"] == 1
    assert society["meta_learning_delta"]["shadow"] == 1
    assert "private prior-world narrative" not in json.dumps(
        transition, ensure_ascii=False,
    ).lower()
    validate_fixture_world_state_transition(
        transition,
        season_id="season-0001",
        fixture_id="md01-fx01",
        match_id="0001-match",
        home="Brazil",
        away="Argentina",
    )

    tampered = copy.deepcopy(transition)
    tampered["match_delta"]["Brazil"]["society_transition"][
        "memory_record_delta"
    ] = 99
    _rehash(tampered)
    with pytest.raises(ValueError, match="replay mismatch"):
        validate_fixture_world_state_transition(
            tampered,
            season_id="season-0001",
            fixture_id="md01-fx01",
            match_id="0001-match",
            home="Brazil",
            away="Argentina",
        )


def test_legacy_world_state_v2_without_meta_summary_remains_replayable(tmp_path):
    for team in ("Brazil", "Argentina"):
        _roster(tmp_path, team)
    before = capture_world_state(tmp_path, ("Brazil", "Argentina"))
    _write_carryover(tmp_path, fatigue=0.2, injured=0)
    after = capture_world_state(tmp_path, ("Brazil", "Argentina"))
    for phase in (before, after):
        for snapshot in phase.values():
            snapshot["schema_version"] = 2
            snapshot["society"].pop("meta_learning")
            _rehash_snapshot(snapshot)

    transition = _build_fixture_world_state_transition(
        season_id="season-legacy",
        fixture_id="md01-fx01",
        match_id="legacy-match",
        home="Brazil",
        away="Argentina",
        before_match=before,
        after_match=after,
        schema_version=2,
    )

    assert "meta_learning_delta" not in (
        transition["match_delta"]["Brazil"]["society_transition"]
    )
    validate_fixture_world_state_transition(
        transition,
        season_id="season-legacy",
        fixture_id="md01-fx01",
        match_id="legacy-match",
        home="Brazil",
        away="Argentina",
    )
