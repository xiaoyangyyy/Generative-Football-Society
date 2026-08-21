import copy
import json

import pytest

from src.product.player_development import (
    apply_development_transaction, append_development_transaction,
    build_development_transaction, collect_season_participation,
    development_view, validate_development_registry,
)


def _player(player_id, age, quality=0.6, role="CM"):
    player = {
        "player_id": player_id, "name": player_id.upper(), "team_id": "A",
        "role": role,
        "abilities": {
            key: quality for key in (
                "tech", "pass_skill", "vision", "spatial", "pace", "press",
                "curve", "shot", "power", "aerial", "heading", "mental",
                "gk_reflex", "gk_aerial",
            )
        },
    }
    if age is not None:
        player["age"] = age
    return player


def _roster(players):
    return {
        "team_id": "A", "formation": "4-3-3", "source": "test",
        "players": players,
    }


def _season(report="outputs/m1.json"):
    return {
        "season_id": "season-0001",
        "fixtures": [{
            "fixture_id": "md01-fx01", "home": "A", "away": "B",
            "match_id": "m1", "report": report,
        }],
    }


def _archive():
    return {
        "schema_version": 1, "season_id": "season-0001",
        "plan": {
            "teams": ["A", "B"], "manager_team": "A",
            "manager_resources": {
                "recovery": 1, "medical": 1, "sports_science": 4,
            },
        },
    }


def test_participation_uses_exact_persisted_minutes_and_report_identity(tmp_path):
    report = tmp_path / "outputs/m1.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "match_id": "m1",
        "match_plan": {"competition": {
            "season_id": "season-0001", "fixture_id": "md01-fx01",
        }},
        "raw_summary": {"player_stats": {"A": {
            "young": {"minutes": 90.0}, "veteran": {"minutes": 45.0},
        }}},
    }), encoding="utf-8")

    evidence = collect_season_participation(tmp_path, _season(), team="A")

    assert evidence["coverage"] == "complete"
    assert evidence["player_totals"] == {
        "veteran": {"minutes": 45.0, "appearances": 1},
        "young": {"minutes": 90.0, "appearances": 1},
    }
    assert len(evidence["reports"][0]["sha256"]) == 64


def test_development_applies_age_minutes_science_and_role_weights_replayably(tmp_path):
    report = tmp_path / "outputs/m1.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({
        "match_id": "m1",
        "match_plan": {"competition": {
            "season_id": "season-0001", "fixture_id": "md01-fx01",
        }},
        "raw_summary": {"player_stats": {"A": {
            "young": {"minutes": 90.0}, "veteran": {"minutes": 45.0},
        }}},
    }), encoding="utf-8")
    evidence = collect_season_participation(tmp_path, _season(), team="A")
    source = _roster([
        _player("young", 20), _player("veteran", 31, role="CB"),
        _player("unknown", None),
    ])
    target = _roster([
        _player("young", 20), _player("veteran", 31, role="CB"),
        _player("unknown", None), _player("signing", 23, 0.7, "ST"),
    ])

    transaction, result = build_development_transaction(
        _archive(), source, target, evidence,
        team="A", target_season_id="season-0002",
    )
    records = {record["player_id"]: record for record in transaction["players"]}

    assert records["young"]["season_delta"] == 0.01
    assert records["young"]["age_after"] == 21
    assert records["young"]["ability_changes"]["pass_skill"]["after"] == 0.6115
    assert records["veteran"]["season_delta"] < 0
    assert records["veteran"]["age_after"] == 32
    assert records["unknown"]["status"] == "age_unknown"
    assert records["signing"]["status"] == "new_signing"
    assert next(p for p in result["players"] if p["player_id"] == "signing")[
        "abilities"
    ]["shot"] == 0.7
    assert apply_development_transaction(target, transaction) == result


def test_missing_reports_never_invent_minutes_and_registry_rejects_tampering(tmp_path):
    evidence = collect_season_participation(
        tmp_path, _season("outputs/missing.json"), team="A",
    )
    assert evidence["coverage"] == "unavailable"
    assert evidence["player_totals"] == {}
    source = _roster([_player("young", 20)])
    transaction, _ = build_development_transaction(
        _archive(), source, copy.deepcopy(source), evidence,
        team="A", target_season_id="season-0002",
    )
    record = transaction["players"][0]
    assert record["minutes"] == 0
    assert record["minutes_bonus"] == 0
    assert record["base_delta"] == 0.004

    registry = append_development_transaction(
        None, team="A", transaction=transaction,
    )
    assert validate_development_registry(registry)["transaction_count"] == 1
    assert development_view(registry, team="A")["available"]
    tampered = copy.deepcopy(registry)
    tampered["clubs"]["A"]["transactions"][0]["target_season_id"] = "bad"
    with pytest.raises(ValueError, match="invalid development transaction"):
        validate_development_registry(tampered)
    tampered_summary = copy.deepcopy(registry)
    tampered_summary["clubs"]["A"]["transactions"][0]["summary"][
        "total_minutes_observed"
    ] = 90.0
    with pytest.raises(ValueError, match="transaction summary mismatch"):
        validate_development_registry(tampered_summary)
