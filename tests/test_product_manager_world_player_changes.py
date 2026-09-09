import copy

import pytest

from scripts.verify_product_recovery import CODE_IDENTITY_FILES
from src.product.manager_world_player_changes import (
    MAX_VISIBLE_FIELDS,
    MAX_VISIBLE_PLAYERS,
    project_manager_world_player_changes,
)


def _updated(index: int) -> dict:
    return {
        "player_id": f"p{index:03d}",
        "name": f"Player {index}",
        "change": "updated",
        "fields": {
            "matches_played": {"before": 0, "after": 1},
        },
    }


def test_player_changes_project_scalar_and_condition_evidence_without_mutation():
    source = [_updated(1)]
    source[0]["fields"].update({
        "injury_matches_left": {"before": 0, "after": 2},
        "condition_delta": {
            "before": {"fatigue": 0.1, "morale": 0.0},
            "after": {"fatigue": 0.3, "morale": 0.0},
        },
    })
    original = copy.deepcopy(source)

    projection = project_manager_world_player_changes(source, expected_total=1)

    assert source == original
    assert projection["available"] is True
    assert projection["total_changed"] == 1
    assert projection["visible_count"] == 1
    assert projection["players_truncated"] is False
    assert projection["players"][0] == {
        "player_id": "p001",
        "name": "Player 1",
        "change": "updated",
        "fields": [{
            "field": "matches_played", "before": 0, "after": 1,
        }, {
            "field": "injury_matches_left", "before": 0, "after": 2,
        }, {
            "field": "condition_delta.fatigue", "before": 0.1, "after": 0.3,
        }],
        "fields_truncated": False,
    }
    assert projection["descriptive_persisted_state_only"] is True
    assert projection["manager_or_action_effect_authorized"] is False
    assert projection["outcome_attribution_authorized"] is False


def test_player_changes_validate_full_source_before_bounding_public_rows():
    source = [_updated(index) for index in range(1, 14)]
    source[0]["fields"] = {
        field: {"before": 0, "after": 1}
        for field in (
            "matches_played", "minutes_ema", "goals", "assists",
            "yellow_cards", "red_cards", "suspension_matches_left",
            "injury_matches_left", "injury_severity",
            "medical_recovery_credit", "form_ema", "media_sentiment",
            "last_rating",
        )
    }

    projection = project_manager_world_player_changes(source, expected_total=13)

    assert projection["available"] is True
    assert projection["visible_count"] == MAX_VISIBLE_PLAYERS == 12
    assert projection["players_truncated"] is True
    assert len(projection["players"][0]["fields"]) == MAX_VISIBLE_FIELDS == 12
    assert projection["players"][0]["fields_truncated"] is True


@pytest.mark.parametrize(("mutate", "reason"), (
    (
        lambda rows: rows[0]["fields"].update({
            "last_rating": {"before": float("nan"), "after": 7.0},
        }),
        "invalid_player_change_evidence",
    ),
    (
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        "player_change_total_mismatch",
    ),
    (
        lambda rows: rows.reverse(),
        "noncanonical_player_change_identities",
    ),
))
def test_player_changes_fail_closed_on_invalid_full_source(mutate, reason):
    source = [_updated(1), _updated(2)]
    expected_total = 2
    mutate(source)
    if reason == "player_change_total_mismatch":
        expected_total = 2

    projection = project_manager_world_player_changes(
        source, expected_total=expected_total,
    )

    assert projection["available"] is False
    assert projection["reason"] == reason
    assert projection["players"] == []
    assert projection["manager_or_action_effect_authorized"] is False


def test_player_changes_keep_legacy_absence_distinct_from_zero_change():
    projection = project_manager_world_player_changes(None, expected_total=0)

    assert projection["available"] is False
    assert projection["reason"] == "legacy_or_missing_player_change_evidence"
    assert projection["total_changed"] == 0


def test_player_changes_reject_invalid_expected_total():
    with pytest.raises(ValueError, match="expected total is invalid"):
        project_manager_world_player_changes([], expected_total=True)


def test_player_changes_are_bound_into_product_recovery_identity():
    assert "src/product/manager_world_player_changes.py" in CODE_IDENTITY_FILES
