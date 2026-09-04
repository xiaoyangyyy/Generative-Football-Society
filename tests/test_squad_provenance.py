import json

import numpy as np
import pytest

from src.match_engine.macro_bridge import build_match_affective_state
from src.simulation.squad_registry import roster_identity
from tests.test_affective_phase1b import _FakeAgent


ROLES = ("GK", "LB", "CB", "CB", "RB", "DM", "CM", "AM", "LW", "RW", "ST")


def _roster(team: str, *, players: int = 11) -> dict:
    return {
        "team_id": team,
        "source": "test-portable-root",
        "formation": "4-3-3",
        "players": [
            {
                "player_id": f"root-player-{index}",
                "name": f"Root Player {index}",
                "team_id": team,
                "role": ROLES[index % len(ROLES)],
                "squad_role": "starter",
                "availability": 1.0,
                "market_value_eur": 1_000_000 + index,
                "abilities": {
                    "tech": 0.6,
                    "phys": 0.6,
                    "vision": 0.6,
                    "aerial": 0.5,
                    "gk": 0.7 if index == 0 else 0.0,
                },
            }
            for index in range(players)
        ],
    }


def _write_roster(root, team: str, payload) -> None:
    path = root / "data" / "rosters" / f"{team}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_match_state_uses_roster_from_explicit_portable_root(tmp_path):
    payload = _roster("Portable")
    _write_roster(tmp_path, "Portable", payload)

    state = build_match_affective_state(
        _FakeAgent("Portable"),
        _FakeAgent("Missing"),
        rng=np.random.default_rng(7),
        base_dir=tmp_path,
    )

    assert state.home.players[0].name.startswith("Root Player")
    assert state.home.squad_provenance == {
        "schema_version": 1,
        "team_id": "Portable",
        "source": "effective_roster",
        "roster_source": "test-portable-root",
        "roster_identity": roster_identity(payload),
        "roster_player_count": 11,
        "simulation_player_count": 11,
        "fallback_used": False,
        "fallback_reason": None,
    }
    assert state.away.squad_provenance["source"] == "synthetic_status_fallback"
    assert state.away.squad_provenance["fallback_reason"] == "roster_unavailable"


def test_existing_corrupt_roster_is_not_silently_replaced(tmp_path):
    path = tmp_path / "data" / "rosters" / "Corrupt.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        build_match_affective_state(
            _FakeAgent("Corrupt"),
            _FakeAgent("Missing"),
            rng=np.random.default_rng(9),
            base_dir=tmp_path,
        )


def test_existing_roster_without_available_xi_fails_closed(tmp_path):
    _write_roster(tmp_path, "Short", _roster("Short", players=10))

    with pytest.raises(ValueError, match="cannot form an available XI"):
        build_match_affective_state(
            _FakeAgent("Short"),
            _FakeAgent("Missing"),
            rng=np.random.default_rng(11),
            base_dir=tmp_path,
        )
