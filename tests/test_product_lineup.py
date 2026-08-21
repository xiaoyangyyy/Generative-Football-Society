import copy
import json
from types import SimpleNamespace

import pytest

from src.simulation.lineup import (
    LineupSelection,
    apply_lineup_selection,
    automatic_lineup,
    build_squad_catalog,
    validate_and_freeze_lineup,
)


ROLES = (
    "GK", "RB", "CB", "CB", "LB", "CM", "CM", "CM", "RW", "ST", "LW",
    "GK", "CB", "DM", "AM", "ST",
)


def _roster(team="Test Team"):
    players = []
    for index, role in enumerate(ROLES):
        quality = 0.50 + 0.02 * index
        players.append({
            "player_id": f"p{index:02d}",
            "name": f"Player {index}",
            "role": role,
            "squad_role": "starter" if index < 11 else "bench",
            "availability": 1.0,
            "condition": {"composure": quality},
            "abilities": {
                "tech": quality, "pass_skill": quality, "vision": quality,
                "spatial": quality, "pace": quality, "press": quality,
                "shot": quality, "power": quality, "aerial": quality,
                "mental": quality, "gk_reflex": quality,
                "gk_aerial": quality,
            },
        })
    return {
        "team_id": team, "formation": "4-3-3", "players": players,
    }


def _write_roster(root, roster):
    path = root / "data/rosters/Test_Team.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(roster), encoding="utf-8")


def test_lineup_contract_rejects_wrong_size_duplicates_and_invalid_fingerprint():
    with pytest.raises(ValueError, match="exactly 11"):
        LineupSelection(starters=("p",) * 10)
    with pytest.raises(ValueError, match="unique"):
        LineupSelection(starters=tuple(f"p{i}" for i in range(10)) + ("p0",))
    with pytest.raises(ValueError, match="fingerprint"):
        LineupSelection(
            starters=tuple(f"p{i}" for i in range(11)),
            roster_fingerprint="bad",
        )


def test_catalog_applies_injury_state_and_automatic_lineup_is_frozen(tmp_path):
    roster = _roster()
    _write_roster(tmp_path, roster)
    persistence = tmp_path / "data/persistence/squad_carryover.json"
    persistence.parent.mkdir(parents=True)
    persistence.write_text(json.dumps({
        "Test Team": {
            "team_id": "Test Team",
            "players": {
                "p00": {
                    "player_id": "p00", "injury_matches_left": 2,
                    "injury_severity": 0.8,
                },
            },
        },
    }), encoding="utf-8")

    catalog = build_squad_catalog(tmp_path, "Test Team")
    assert catalog["available"]
    assert catalog["team_condition"] == {
        "fatigue": 0.0, "morale": 0.55, "media_pressure": 0.0,
        "injured_players": 1, "suspended_players": 0,
        "matches_settled": 0, "source": "persisted_squad_carryover",
        "unavailable_players": 1,
    }
    injured = next(row for row in catalog["players"] if row["player_id"] == "p00")
    assert injured["injured"] and not injured["selectable"]
    selection = automatic_lineup(catalog, "rotate")
    assert len(selection.starters) == 11
    assert "p00" not in selection.starters + selection.bench
    assert selection.roster_fingerprint == catalog["roster_fingerprint"]
    assert sum(
        1 for pid in selection.starters
        if next(row for row in catalog["players"] if row["player_id"] == pid)["role"] == "GK"
    ) == 1


def test_manual_lineup_is_validated_and_unselected_players_are_not_bench(tmp_path):
    roster = _roster()
    _write_roster(tmp_path, roster)
    catalog = build_squad_catalog(tmp_path, "Test Team")
    automatic = automatic_lineup(catalog, "strongest")
    manual = LineupSelection(
        starters=automatic.starters,
        bench=automatic.bench[:2],
        source="manual",
    )
    frozen = validate_and_freeze_lineup(catalog, manual)
    raw = _roster()
    from src.simulation.cross_match_state import (
        TeamSquadCarryover, apply_carryover_to_roster_for_agent,
    )
    from src.simulation.lineup import roster_fingerprint

    carryover_applied = apply_carryover_to_roster_for_agent(
        SimpleNamespace(
            team_name="Test Team",
            squad_carryover=TeamSquadCarryover(team_id="Test Team"),
        ),
        copy.deepcopy(raw),
    )
    assert roster_fingerprint(carryover_applied) == frozen.roster_fingerprint
    applied = apply_lineup_selection(carryover_applied, frozen)
    roles = {row["player_id"]: row["squad_role"] for row in applied["players"]}
    assert all(roles[pid] == "starter" for pid in frozen.starters)
    assert all(roles[pid] == "bench" for pid in frozen.bench)
    assert all(
        role == "reserve" for pid, role in roles.items()
        if pid not in frozen.starters + frozen.bench
    )

    import numpy as np
    from src.data_engine.roster_loader import build_team_squad_from_roster
    from tests.test_affective_phase1b import _FakeAgent

    agent = _FakeAgent("Test Team")
    team = build_team_squad_from_roster(
        agent, applied, np.random.default_rng(7),
    )
    assert team is not None
    assert len(team.players) == 13
    assert {
        player.player_id for player in team.players if not player.on_pitch
    } == set(frozen.bench)

    tampered = copy.deepcopy(carryover_applied)
    tampered["players"][0]["availability"] = 0.2
    with pytest.raises(ValueError, match="roster state has changed"):
        apply_lineup_selection(tampered, frozen)


def test_manual_lineup_rejects_unknown_unavailable_or_missing_goalkeeper(tmp_path):
    roster = _roster()
    _write_roster(tmp_path, roster)
    catalog = build_squad_catalog(tmp_path, "Test Team")
    valid = automatic_lineup(catalog, "balanced")
    with pytest.raises(ValueError, match="outside"):
        validate_and_freeze_lineup(catalog, LineupSelection(
            starters=valid.starters[:-1] + ("unknown",), bench=(),
        ))
    non_gk = tuple(
        row["player_id"] for row in catalog["players"] if row["role"] != "GK"
    )[:11]
    with pytest.raises(ValueError, match="goalkeeper"):
        validate_and_freeze_lineup(catalog, LineupSelection(starters=non_gk))


def test_catalog_explicitly_degrades_when_source_roster_has_no_goalkeeper(tmp_path):
    roster = _roster()
    for player in roster["players"]:
        if player["role"] == "GK":
            player["role"] = "CB"
    _write_roster(tmp_path, roster)
    catalog = build_squad_catalog(tmp_path, "Test Team")
    assert not catalog["available"]
    assert catalog["reason"] == "roster_missing_goalkeeper"
    assert catalog["automatic"] == {}
