import copy
import json

import pytest

from src.simulation.player_market import (
    FreeAgentPlan, ai_free_agent_decision, execute_free_agent_signing,
    finalize_market_transition,
    market_preview_from_window, market_view, open_market_window,
    pool_entry_from_player, replay_market_pool, validate_ai_free_agent_decision,
    validate_market_registry,
)


def _player(player_id, role="CM", age=25, quality=0.6):
    return {
        "player_id": player_id, "name": player_id.upper(), "team_id": "A",
        "role": role, "age": age,
        "abilities": {key: quality for key in (
            "tech", "pass_skill", "vision", "spatial", "pace", "press",
            "curve", "shot", "power", "aerial", "heading", "mental",
            "gk_reflex", "gk_aerial",
        )},
    }


def _roster():
    roles = ["GK", "RB", "CB", "CB", "LB", "DM", "CM", "AM", "RW", "ST", "LW"]
    return {
        "team_id": "B", "formation": "4-3-3", "source": "test",
        "players": [_player(f"b{index:02d}", role, quality=0.5) for index, role in enumerate(roles)],
    }


def _registry_with_agent():
    first = open_market_window(None, target_season_id="season-0002")
    entry = pool_entry_from_player(
        _player("free-cm", "CM", 24, 0.7), origin_team="A",
        entered_season_id="season-0002", retirement_age=38,
        reason="contract_released",
    )
    registry, _ = finalize_market_transition(None, first, entries=[entry])
    return registry


def test_new_release_waits_one_window_then_ages_and_becomes_available():
    registry = _registry_with_agent()
    assert replay_market_pool(registry)[0]["available_from_season_id"] == "season-0003"
    window = open_market_window(registry, target_season_id="season-0003")
    preview = market_preview_from_window(window, team="B")
    assert preview["candidate_count"] == 1
    assert preview["candidates"][0]["age"] == 25
    assert "quality" not in preview["candidates"][0]
    assert preview["candidates"][0]["observation"]["level"] == "baseline"


def test_signing_preserves_identity_replaces_same_role_and_replays_registry():
    registry = _registry_with_agent()
    window = open_market_window(registry, target_season_id="season-0003")
    preview = market_preview_from_window(window, team="B")
    plan = FreeAgentPlan(preview["market_id"], "free-cm", "b06")
    window, result, transaction = execute_free_agent_signing(
        window, _roster(), team="B", plan=plan, control="manager",
    )
    ids = {player["player_id"] for player in result["players"]}
    assert "free-cm" in ids and "b06" not in ids and len(ids) == 11
    outgoing_entry = pool_entry_from_player(
        transaction["outgoing"], origin_team="B",
        entered_season_id="season-0003", retirement_age=38,
        reason="squad_replacement",
    )
    registry, transition = finalize_market_transition(
        registry, window, entries=[outgoing_entry],
    )
    assert transition["summary"]["signings"] == 1
    assert replay_market_pool(registry)[0]["player_id"] == "b06"
    decision = ai_free_agent_decision(
        open_market_window(_registry_with_agent(), target_season_id="season-0003"),
        _roster(), team="B",
    )
    assert decision["plan"]["free_agent_id"] == "free-cm"
    assert decision["selected"]["quality_improvement"] >= 0.015
    assert len(decision["scouting_reports"]) <= 2

    tampered = copy.deepcopy(decision)
    tampered["scouting_reports"][0]["observation"]["estimated_quality"] += 0.001
    with pytest.raises(ValueError, match="scouting report evidence mismatch"):
        validate_ai_free_agent_decision(tampered)


def test_public_market_view_redacts_player_truth_from_pool_and_transactions():
    registry = _registry_with_agent()
    window = open_market_window(registry, target_season_id="season-0003")
    window, _result, transaction = execute_free_agent_signing(
        window, _roster(), team="B",
        plan=FreeAgentPlan(
            market_preview_from_window(window, team="B")["market_id"],
            "free-cm", "b06",
        ), control="manager",
    )
    outgoing_entry = pool_entry_from_player(
        transaction["outgoing"], origin_team="B",
        entered_season_id="season-0003", retirement_age=38,
        reason="squad_replacement",
    )
    registry, _ = finalize_market_transition(
        registry, window, entries=[outgoing_entry],
    )

    public = market_view(registry)
    serialized = json.dumps(public, sort_keys=True)
    assert '"abilities"' not in serialized
    assert '"quality"' not in serialized
    assert '"prior_roster_identity"' not in serialized
    assert public["latest_transition"]["signings"][0]["origin_team"] == "A"


def test_market_rejects_same_window_reentry_and_tampered_aging():
    registry = _registry_with_agent()
    window = open_market_window(registry, target_season_id="season-0003")
    with pytest.raises(ValueError, match="same-role"):
        execute_free_agent_signing(
            window, _roster(), team="B",
            plan=FreeAgentPlan(
                market_preview_from_window(window, team="B")["market_id"],
                "free-cm", "b00",
            ), control="manager",
        )
    registry, _ = finalize_market_transition(registry, window, entries=[])
    tampered = copy.deepcopy(registry)
    tampered["transitions"][1]["aging"][0]["age_after"] += 1
    with pytest.raises(ValueError, match="aging replay mismatch"):
        validate_market_registry(tampered)
    decision_window = open_market_window(_registry_with_agent(), target_season_id="season-0003")
    decision = ai_free_agent_decision(decision_window, _roster(), team="B")
    decision_window["ai_decisions"] = [copy.deepcopy(decision)]
    decision_window, _result, _transaction = execute_free_agent_signing(
        decision_window, _roster(), team="B",
        plan=FreeAgentPlan.from_payload(decision["plan"]), control="ai",
    )
    decision_registry, _ = finalize_market_transition(
        _registry_with_agent(), decision_window, entries=[],
    )
    decision_registry["transitions"][-1]["ai_decisions"][0][
        "selected"
    ]["quality_improvement"] = 0.1
    with pytest.raises(ValueError, match="decision replay mismatch"):
        validate_market_registry(decision_registry)
