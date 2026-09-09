import copy

import pytest

from scripts.verify_product_recovery import CODE_IDENTITY_FILES
from src.product.manager_world_society_changes import (
    MAX_SOURCE_STATE_CHANGES,
    MAX_VISIBLE_STATE_CHANGES,
    project_manager_world_society_changes,
)


def _transition() -> dict:
    return {
        "available": True,
        "before_available": True,
        "after_available": True,
        "changed_state_fields": [
            "tactical_controls", "psychological_state",
        ],
        "state_changes": [{
            "scope": "tactical_controls",
            "field": "line_height",
            "before": 0.4,
            "after": 0.55,
        }, {
            "scope": "psychological_state",
            "field": "pressure",
            "before": 0.2,
            "after": 0.35,
        }],
    }


def test_society_changes_project_exact_values_without_mutation():
    source = _transition()
    original = copy.deepcopy(source)

    projection = project_manager_world_society_changes(source)

    assert source == original
    assert projection["available"] is True
    assert projection["changed_state_fields"] == [
        "tactical_controls", "psychological_state",
    ]
    assert projection["total_changes"] == 2
    assert projection["visible_count"] == 2
    assert projection["changes"] == source["state_changes"]
    assert projection["descriptive_persisted_state_only"] is True
    assert projection["manager_or_action_effect_authorized"] is False
    assert projection["outcome_attribution_authorized"] is False
    assert MAX_SOURCE_STATE_CHANGES == MAX_VISIBLE_STATE_CHANGES == 19


@pytest.mark.parametrize(("before_available", "after_available", "side"), (
    (False, True, "before"),
    (True, False, "after"),
))
def test_society_changes_support_state_creation_and_removal(
    before_available, after_available, side,
):
    source = {
        "available": True,
        "before_available": before_available,
        "after_available": after_available,
        "changed_state_fields": ["referee_grievance"],
        "state_changes": [{
            "scope": "referee_grievance",
            "field": "value",
            "before": 0.2 if before_available else None,
            "after": 0.3 if after_available else None,
        }],
    }

    projection = project_manager_world_society_changes(source)

    assert projection["available"] is True
    assert projection["changes"][0][side] is None


def test_society_changes_distinguish_valid_zero_change_from_missing_detail():
    source = {
        "available": True,
        "before_available": True,
        "after_available": True,
        "changed_state_fields": [],
        "state_changes": [],
    }

    projection = project_manager_world_society_changes(source)

    assert projection["available"] is True
    assert projection["reason"] is None
    assert projection["total_changes"] == 0
    assert projection["changes"] == []


@pytest.mark.parametrize(("mutate", "reason"), (
    (
        lambda value: value["state_changes"].reverse(),
        "noncanonical_society_state_changes",
    ),
    (
        lambda value: value["state_changes"][0].update({"after": 0.4}),
        "invalid_society_state_change_evidence",
    ),
    (
        lambda value: value["state_changes"][0].update({"after": float("nan")}),
        "invalid_society_state_change_evidence",
    ),
    (
        lambda value: value["state_changes"][0].update({"before": None}),
        "invalid_society_state_change_evidence",
    ),
    (
        lambda value: value["state_changes"][0].update({"after": 1.01}),
        "invalid_society_state_change_evidence",
    ),
    (
        lambda value: value.update({
            "changed_state_fields": ["psychological_state"],
        }),
        "society_state_scope_mismatch",
    ),
))
def test_society_changes_fail_closed_on_invalid_full_source(mutate, reason):
    source = _transition()
    mutate(source)

    projection = project_manager_world_society_changes(source)

    assert projection["available"] is False
    assert projection["reason"] == reason
    assert projection["changes"] == []
    assert projection["manager_or_action_effect_authorized"] is False


def test_society_changes_keep_legacy_absence_distinct_from_zero_change():
    source = _transition()
    source.pop("state_changes")

    projection = project_manager_world_society_changes(source)

    assert projection["available"] is False
    assert projection["reason"] == (
        "legacy_or_missing_society_state_change_evidence"
    )
    assert projection["changed_state_fields"] == [
        "tactical_controls", "psychological_state",
    ]
    assert projection["total_changes"] == 0


def test_society_changes_are_bound_into_product_recovery_identity():
    assert "src/product/manager_world_society_changes.py" in CODE_IDENTITY_FILES
