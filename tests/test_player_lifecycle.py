import copy

import pytest

from src.simulation.player_lifecycle import (
    RetentionPlan, append_lifecycle_transaction, apply_lifecycle_transaction,
    build_lifecycle_preview, build_lifecycle_transaction,
    validate_lifecycle_registry,
)


def _player(player_id, role, age, quality, years=None):
    player = {
        "player_id": player_id, "name": player_id.upper(), "team_id": "A",
        "role": role, "age": age,
        "abilities": {key: quality for key in (
            "tech", "pass_skill", "vision", "spatial", "pace", "press",
            "curve", "shot", "power", "aerial", "heading", "mental",
            "gk_reflex", "gk_aerial",
        )},
    }
    if years is not None:
        player["career"] = {
            "schema_version": 1, "contract_years_remaining": years,
            "contract_source": "renewed",
        }
    return player


def _roster():
    roles = ["GK", "RB", "CB", "CB", "LB", "DM", "CM", "AM", "RW", "ST", "LW"]
    return {
        "team_id": "A", "formation": "4-3-3", "source": "test",
        "players": [
            _player(f"p{index:02d}", role, 37 if index == 2 else 24,
                    0.55 + index * 0.01, 1 if index in {0, 1, 3} else 2)
            for index, role in enumerate(roles)
        ],
    }


def test_preview_is_stable_bounded_and_prioritizes_expiring_goalkeeper():
    roster = _roster()
    left = build_lifecycle_preview(roster, team="A", target_season_id="season-0002")
    right = build_lifecycle_preview(roster, team="A", target_season_id="season-0002")
    assert left == right
    assert left["renewal_limit"] == 4
    assert left["recommended_renew_player_ids"][0] == "p00"
    assert "p02" in left["retiring_player_ids"]


def test_manager_plan_releases_contracts_promotes_same_role_and_replays():
    source = _roster()
    preview = build_lifecycle_preview(source, team="A", target_season_id="season-0002")
    plan = RetentionPlan(preview["cycle_id"], ("p00",))
    transaction, result = build_lifecycle_transaction(
        source, copy.deepcopy(source), team="A", target_season_id="season-0002",
        plan=plan, control="manager",
    )
    assert transaction["summary"] == {
        "renewed": 1, "continued": 7, "new_signings": 0,
        "contract_releases": 2, "retirements": 1, "academy_promotions": 3,
    }
    roles_by_id = {player["player_id"]: player["role"] for player in result["players"]}
    assert len(roles_by_id) == 11
    assert "p00" in roles_by_id
    assert "p01" not in roles_by_id and "p02" not in roles_by_id
    assert sorted(player["role"] for player in transaction["academy_promotions"]) == ["CB", "CB", "RB"]
    assert apply_lifecycle_transaction(copy.deepcopy(source), transaction) == result


def test_plan_rejects_non_expiring_player_and_tampered_replay():
    source = _roster()
    preview = build_lifecycle_preview(source, team="A", target_season_id="season-0002")
    with pytest.raises(ValueError, match="non-expiring"):
        build_lifecycle_transaction(
            source, copy.deepcopy(source), team="A", target_season_id="season-0002",
            plan=RetentionPlan(preview["cycle_id"], ("p04",)), control="manager",
        )
    transaction, _ = build_lifecycle_transaction(
        source, copy.deepcopy(source), team="A", target_season_id="season-0002",
    )
    tampered = copy.deepcopy(transaction)
    tampered["academy_promotions"][0]["abilities"]["tech"] = 0.9
    with pytest.raises(ValueError, match="academy source replay mismatch"):
        apply_lifecycle_transaction(copy.deepcopy(source), tampered)
    registry = append_lifecycle_transaction(None, team="A", transaction=transaction)
    assert validate_lifecycle_registry(registry)["transaction_count"] == 1
    bad_summary = copy.deepcopy(registry)
    bad_summary["clubs"]["A"]["transactions"][0]["summary"]["retirements"] += 1
    with pytest.raises(ValueError, match="summary mismatch"):
        validate_lifecycle_registry(bad_summary)
