import copy
import json
from types import SimpleNamespace

import pytest

from src.simulation.squad_registry import (
    RecruitmentMove, RecruitmentPlan, append_recruitment_transaction,
    apply_recruitment_plan, generate_recruitment_market,
    load_effective_roster, recruitment_market_for_workspace,
    recruitment_market_id, replay_squad_registry, roster_identity,
)
from src.simulation.cross_match_state import PlayerCarryover, TeamSquadCarryover
from src.simulation.match_pipeline import prepare_match_agents


def _player(index, role, quality=0.58):
    abilities = {
        key: quality for key in (
            "tech", "pass_skill", "vision", "spatial", "pace", "press",
            "curve", "shot", "power", "aerial", "heading", "mental",
            "gk_reflex", "gk_aerial",
        )
    }
    return {
        "player_id": f"p{index}", "name": f"Player {index}",
        "team_id": "A", "role": role, "squad_role": "starter",
        "availability": 1.0, "condition": {"composure": quality},
        "abilities": abilities,
    }


def _roster():
    roles = ["GK", "CB", "CB", "LB", "RB", "CM", "CM", "LM", "RM", "ST", "ST", "DM"]
    return {
        "team_id": "A", "formation": "4-4-2", "source": "test",
        "squad_size": len(roles),
        "players": [_player(index, role, 0.52 + index * 0.01) for index, role in enumerate(roles)],
    }


def _plan(market, *pairs):
    return RecruitmentPlan(
        market_id=market["market_id"],
        moves=tuple(
            RecruitmentMove(market["candidates"][candidate]["player_id"], outgoing)
            for candidate, outgoing in pairs
        ),
    )


def test_market_is_deterministic_fictional_and_bound_to_current_roster():
    roster = _roster()
    market_id = recruitment_market_id(season_index=2, team="A")
    first = generate_recruitment_market(roster, team="A", market_id=market_id)
    second = generate_recruitment_market(copy.deepcopy(roster), team="A", market_id=market_id)

    assert first == second
    assert first["budget"] == 8
    assert first["maximum_moves"] == 2
    assert len(first["candidates"]) == 5
    assert [row["recruitment_cost"] for row in first["candidates"]] == [3, 3, 4, 5, 6]
    assert first["candidates"][0]["role"] == "GK"
    assert all(row["source"] == "gfs_fictional_recruitment_v1" for row in first["candidates"])
    assert all(row["name"].startswith("GFS ") for row in first["candidates"])
    assert "real player" in first["claim_boundary"]

    roster["players"][1]["abilities"]["tech"] += 0.01
    changed = generate_recruitment_market(roster, team="A", market_id=market_id)
    assert changed["roster_identity"] != first["roster_identity"]
    assert changed["candidates"] != first["candidates"]


def test_plan_replaces_real_squad_players_with_exact_budget_and_audited_result():
    roster = _roster()
    market = generate_recruitment_market(
        roster, team="A", market_id=recruitment_market_id(season_index=2, team="A"),
    )
    plan = _plan(market, (0, "p10"), (2, "p11"))

    result, transaction = apply_recruitment_plan(roster, team="A", plan=plan)

    result_ids = {row["player_id"] for row in result["players"]}
    assert len(result["players"]) == len(roster["players"])
    assert {"p10", "p11"}.isdisjoint(result_ids)
    assert {move.candidate_id for move in plan.moves} <= result_ids
    assert transaction["spent"] == 7
    assert transaction["remaining"] == 1
    assert transaction["prior_roster_identity"] == roster_identity(roster)
    assert transaction["result_roster_identity"] == roster_identity(result)
    assert result["source"].endswith("+gfs_squad_registry_v1")


def test_plan_rejects_budget_market_outgoing_and_goalkeeper_abuse():
    roster = _roster()
    market = generate_recruitment_market(
        roster, team="A", market_id=recruitment_market_id(season_index=2, team="A"),
    )
    with pytest.raises(ValueError, match="fixed budget"):
        apply_recruitment_plan(roster, team="A", plan=_plan(market, (3, "p10"), (4, "p11")))
    with pytest.raises(ValueError, match="outside the frozen market"):
        apply_recruitment_plan(roster, team="A", plan=RecruitmentPlan(
            market["market_id"], (RecruitmentMove("invented", "p10"),),
        ))
    with pytest.raises(ValueError, match="outside the current squad"):
        apply_recruitment_plan(roster, team="A", plan=RecruitmentPlan(
            market["market_id"], (
                RecruitmentMove(market["candidates"][0]["player_id"], "unknown"),
            ),
        ))
    non_gk = next(index for index, row in enumerate(market["candidates"]) if row["role"] != "GK")
    with pytest.raises(ValueError, match="playable squad"):
        apply_recruitment_plan(roster, team="A", plan=_plan(market, (non_gk, "p0")))


def test_registry_replay_is_multi_season_and_fails_closed_on_tampering():
    base = _roster()
    market1 = generate_recruitment_market(
        base, team="A", market_id=recruitment_market_id(season_index=2, team="A"),
    )
    registry, roster1, tx1 = append_recruitment_transaction(
        base, team="A", plan=_plan(market1, (0, "p10")),
        season_id="season-0002", registry=None,
    )
    market2 = generate_recruitment_market(
        roster1, team="A", market_id=recruitment_market_id(season_index=3, team="A"),
    )
    registry, roster2, tx2 = append_recruitment_transaction(
        base, team="A", plan=_plan(market2, (1, "p11")),
        season_id="season-0003", registry=registry,
    )
    replayed, transactions = replay_squad_registry(base, team="A", registry=registry)

    assert roster_identity(replayed) == roster_identity(roster2)
    assert [row["season_id"] for row in transactions] == ["season-0002", "season-0003"]
    assert tx1["season_id"] == "season-0002"
    assert tx2["season_id"] == "season-0003"

    tampered = copy.deepcopy(registry)
    tampered["clubs"]["A"]["transactions"][0]["spent"] = 0
    with pytest.raises(ValueError, match="identity mismatch"):
        replay_squad_registry(base, team="A", registry=tampered)


def test_effective_roster_reads_only_the_atomic_product_session_registry(tmp_path):
    base = _roster()
    roster_path = tmp_path / "data/rosters/A.json"
    roster_path.parent.mkdir(parents=True)
    roster_path.write_text(json.dumps(base), encoding="utf-8")
    market = recruitment_market_for_workspace(
        tmp_path, team="A", season_index=2, registry=None,
    )
    registry, expected, _ = append_recruitment_transaction(
        base, team="A", plan=_plan(market, (0, "p10")),
        season_id="season-0002", registry=None,
    )
    session = tmp_path / "data/persistence/product_session.json"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({"squad_registry": registry}), encoding="utf-8")

    observed = load_effective_roster(tmp_path, "A")
    assert roster_identity(observed) == roster_identity(expected)
    assert json.loads(roster_path.read_text(encoding="utf-8")) == base


def test_match_preparation_consumes_effective_roster_and_prunes_released_state(tmp_path):
    base = _roster()
    roster_path = tmp_path / "data/rosters/A.json"
    roster_path.parent.mkdir(parents=True)
    roster_path.write_text(json.dumps(base), encoding="utf-8")
    opponent = copy.deepcopy(base)
    opponent["team_id"] = "B"
    for player in opponent["players"]:
        player["team_id"] = "B"
        player["player_id"] = "b-" + player["player_id"]
    (tmp_path / "data/rosters/B.json").write_text(
        json.dumps(opponent), encoding="utf-8",
    )
    market = recruitment_market_for_workspace(
        tmp_path, team="A", season_index=2, registry=None,
    )
    outgoing = "p10"
    registry, _, transaction = append_recruitment_transaction(
        base, team="A", plan=_plan(market, (0, outgoing)),
        season_id="season-0002", registry=None,
    )
    session = tmp_path / "data/persistence/product_session.json"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({"squad_registry": registry}), encoding="utf-8")

    def agent(team, carry=None):
        return SimpleNamespace(
            team_name=team,
            squad_carryover=carry or TeamSquadCarryover(team_id=team),
            fatigue=0.0, injury_load=0.0, readiness=0.5,
            semantic_memory={}, team_dynamics={},
        )

    home = agent("A", TeamSquadCarryover(team_id="A", players={
        outgoing: PlayerCarryover(
            player_id=outgoing, injury_matches_left=4, injury_severity=0.9,
        ),
    }))
    away = agent("B")
    home_roster, _ = prepare_match_agents(home, away, str(tmp_path))

    ids = {player["player_id"] for player in home_roster["players"]}
    incoming = transaction["moves"][0]["candidate_id"]
    assert incoming in ids
    assert outgoing not in ids
    assert incoming in home.squad_carryover.players
    assert outgoing not in home.squad_carryover.players
    assert home.semantic_memory["unavailable_players"] == 0
