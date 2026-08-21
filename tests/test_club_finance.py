import copy

import pytest

from src.product.club_finance import (
    OPENING_BALANCE, append_recruitment_charge, append_season_settlement,
    build_season_finance_settlement, club_finance_view, evidence_identity,
    ensure_finance_club, player_wage_tier, recruitment_allowance,
    seasonal_wage_expense, validate_finance_registry,
)


def _player(index, quality):
    return {
        "player_id": f"p{index}", "name": f"Player {index}",
        "team_id": "A", "role": "GK" if index == 0 else "CM",
        "abilities": {
            key: quality for key in (
                "tech", "pass_skill", "vision", "spatial", "pace", "press",
                "shot", "power", "aerial", "mental", "gk_reflex", "gk_aerial",
            )
        },
    }


def _roster(quality=0.66):
    return {
        "team_id": "A", "formation": "4-4-2",
        "players": [_player(index, quality) for index in range(18)],
    }


def _archive(*, season_id="season-0001", position=1, points=9, achieved=True):
    standings = [
        {"position": rank, "team": team, "points": points if team == "A" else 0}
        for rank, team in enumerate(("A", "B", "C", "D"), start=1)
    ]
    if position != 1:
        a = standings.pop(0)
        standings.insert(position - 1, a)
        for rank, row in enumerate(standings, start=1):
            row["position"] = rank
    return {
        "schema_version": 1, "season_id": season_id,
        "plan": {"teams": ["A", "B", "C", "D"], "manager_team": "A"},
        "final_standings": standings,
        "manager_profile": {
            "objective": {"status": "achieved" if achieved else "missed"},
        },
    }


def test_wage_tiers_and_expense_are_bounded_and_roster_derived():
    assert [player_wage_tier(_player(1, quality)) for quality in (0.58, 0.65, 0.73, 0.82)] == [1, 2, 3, 4]
    low = seasonal_wage_expense(_roster(0.58))
    high = seasonal_wage_expense(_roster(0.82))
    assert low["seasonal_wage_expense"] == 3
    assert high["seasonal_wage_expense"] == 12
    assert high["roster_identity"] != low["roster_identity"]


def test_season_settlement_uses_results_objective_and_current_wage_bill():
    archive = _archive(position=1, points=9, achieved=True)
    settlement = build_season_finance_settlement(archive, _roster(0.66), team="A")

    assert settlement["revenue"] == {
        "participation": 6, "points": 3,
        "league_position": 3, "objective": 3,
    }
    assert settlement["revenue_total"] == 15
    assert settlement["expense_total"] == 6
    assert settlement["delta"] == 9
    assert settlement["archive_identity"] == evidence_identity(archive)
    assert "not real club revenue" in settlement["claim_boundary"]


def test_non_player_club_uses_same_settlement_without_manager_objective_bonus():
    archive = _archive(position=1, points=9, achieved=True)
    settlement = build_season_finance_settlement(archive, None, team="B")

    assert settlement["control"] == "ai_club"
    assert settlement["revenue"]["objective"] == 0
    assert settlement["wage"]["source"] == "team_level_baseline_without_roster"


def test_ledger_replays_settlement_then_limits_and_charges_recruitment():
    settlement = build_season_finance_settlement(
        _archive(), _roster(0.66), team="A",
    )
    registry, settled = append_season_settlement(
        None, team="A", settlement=settlement,
    )
    assert settled["balance_before"] == OPENING_BALANCE
    assert settled["balance_after"] == 17
    transaction = {
        "team": "A", "season_id": "season-0002", "spent": 6,
        "moves": [{"candidate_id": "c1", "outgoing_player_id": "p1"}],
    }
    registry, charged = append_recruitment_charge(
        registry, team="A", season_id="season-0002",
        squad_transaction=transaction,
    )
    assert charged["evidence"]["allowance"] == 8
    assert charged["balance_after"] == 11
    view = club_finance_view(registry, team="A")
    assert view["balance"] == 11
    assert view["recruitment_allowance"] == 8
    assert validate_finance_registry(registry)["entry_count"] == 2


def test_low_balance_reduces_allowance_and_rejects_unaffordable_move():
    registry = ensure_finance_club(None, "A")
    registry["clubs"]["A"]["opening_balance"] = 3
    assert recruitment_allowance(3) == 3
    transaction = {
        "team": "A", "season_id": "season-0002", "spent": 4,
        "moves": [{"candidate_id": "c1", "outgoing_player_id": "p1"}],
    }
    with pytest.raises(ValueError, match="available club funds"):
        append_recruitment_charge(
            registry, team="A", season_id="season-0002",
            squad_transaction=transaction,
        )


def test_ledger_rejects_balance_cost_and_evidence_tampering():
    settlement = build_season_finance_settlement(
        _archive(), _roster(0.66), team="A",
    )
    registry, _ = append_season_settlement(None, team="A", settlement=settlement)
    for mutate in (
        lambda value: value["clubs"]["A"]["entries"][0].__setitem__("balance_after", 999),
        lambda value: value["clubs"]["A"]["entries"][0]["evidence"].__setitem__("delta", 1),
        lambda value: value["clubs"]["A"]["entries"][0].__setitem__("unexpected", True),
    ):
        tampered = copy.deepcopy(registry)
        mutate(tampered)
        with pytest.raises(ValueError, match="replay mismatch"):
            validate_finance_registry(tampered)
