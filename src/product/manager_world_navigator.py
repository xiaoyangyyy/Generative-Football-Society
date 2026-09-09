"""Season-level navigation across current intervention and completed worlds."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping

from src.product.manager_world_player_changes import (
    project_manager_world_player_changes,
)
from src.product.manager_world_society_changes import (
    project_manager_world_society_changes,
)


SCHEMA_VERSION = 1
MAX_HISTORY_CHAPTERS = 64
_WORKFLOW_STATES = {
    "decision_required",
    "ready_to_explore",
    "future_generation_in_progress",
    "future_generation_interrupted",
    "future_generation_failed",
    "evidence_ready_for_review",
    "review_recorded_decision_refrozen",
    "season_complete",
}
_LIFECYCLE_STATES = {
    "frozen_awaiting_execution",
    "executed_with_direct_evidence",
    "executed_evidence_unavailable",
}
_ACTION_EVIDENCE_STATES = {
    "locally_attributable_action_changes_observed",
    "probability_influence_without_realized_action_change",
    "no_nonzero_action_influence_observed",
}
_COMPLETED_ACTION_STATES = _ACTION_EVIDENCE_STATES | {
    "not_applicable_stable_mode",
    "legacy_evidence_unavailable",
    "action_evidence_unavailable",
}
_ACTION_ADOPTION_STATE_ORDER = (
    "realized_action_change",
    "probability_influence_only",
    "no_nonzero_influence",
    "not_applicable_stable_mode",
    "evidence_unavailable",
)
_WORLD_METRICS = (
    "team_fatigue_ema",
    "squad_morale_ema",
    "team_media_pressure",
    "injured_players",
    "suspended_players",
    "unavailable_players",
)
_WORLD_SUMMARY_FIELDS = (
    "changed_players",
    "new_injuries",
    "injuries_cleared",
    "new_suspensions",
    "suspensions_cleared",
)
_FUTURE_REVIEW_COUNT_FIELDS = (
    "fixed_scenario_budget",
    "eligible_scenarios",
    "verified_anchor_scenarios",
    "action_divergence_scenarios",
    "local_attribution_scenarios",
    "descriptive_future_difference_scenarios",
)
_FUTURE_MECHANISM_SEMANTIC_FIELDS = (
    "cross_action_mechanism_examples",
    "direct_preference_mechanism_examples",
    "suppression_only_mechanism_examples",
    "hold_reference_redistribution_examples",
    "nonzero_cross_descriptive_windows",
)
_OFFICIAL_ACTION_SEMANTIC_COUNT_FIELDS = (
    "retained_semantic_examples",
    "semantic_cross_action_examples",
    "semantic_direct_preference_examples",
    "semantic_suppression_only_examples",
    "semantic_no_signal_examples",
    "semantic_legacy_unclassified_examples",
    "semantic_hold_reference_redistribution_examples",
    "semantic_direct_cross_ball_event_examples",
)
_REVIEWED_SCENARIO_FIELDS = {
    "schema_version", "source_scenario_identity", "branch_at_sec",
    "branch_minute", "future_status", "eligible", "anchor_verified",
    "branch_state_identity", "changed_actions",
    "locally_attributable_changes", "descriptive_future_difference_count",
    "simulator_local_action_attribution", "outcome_causality_authorized",
    "real_football_causality_authorized", "archive_identity",
}


def _player_counter_transitions(
    rows: list[Mapping[str, Any]], field: str,
) -> tuple[int, int]:
    started = 0
    cleared = 0
    for row in rows:
        fields = row.get("fields")
        fields = fields if isinstance(fields, Mapping) else {}
        change = fields.get(field)
        change = change if isinstance(change, Mapping) else {}
        before = change.get("before", 0)
        after = change.get("after", 0)
        started += before == 0 and after > 0
        cleared += before > 0 and after == 0
    return started, cleared


def _validated_player_changes(
    source_match_delta: Mapping[str, Any],
    transition_summary: Mapping[str, Any],
    *,
    persistent_state_available: bool,
) -> list[dict[str, Any]] | None:
    if not persistent_state_available:
        return None
    raw = source_match_delta.get("players_changed")
    projection = project_manager_world_player_changes(
        raw, expected_total=transition_summary["changed_players"],
    )
    injuries = (
        _player_counter_transitions(raw, "injury_matches_left")
        if projection["available"] else (-1, -1)
    )
    suspensions = (
        _player_counter_transitions(raw, "suspension_matches_left")
        if projection["available"] else (-1, -1)
    )
    if (
        projection["available"] is not True
        or injuries != (
            transition_summary["new_injuries"],
            transition_summary["injuries_cleared"],
        )
        or suspensions != (
            transition_summary["new_suspensions"],
            transition_summary["suspensions_cleared"],
        )
    ):
        raise ValueError(
            "manager world navigator player transition facts are invalid"
        )
    return copy.deepcopy(raw)


def _validated_society_transition(
    source_match_delta: Mapping[str, Any], *,
    persistent_state_available: bool,
) -> dict[str, Any] | None:
    if not persistent_state_available:
        return None
    raw = source_match_delta.get("society_transition")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(
            "manager world navigator society transition facts are invalid"
        )
    projection = project_manager_world_society_changes(raw)
    if "state_changes" in raw and projection["available"] is not True:
        raise ValueError(
            "manager world navigator society transition facts are invalid"
        )
    return copy.deepcopy(dict(raw))
_REVIEWED_SCENARIO_STATUSES = {
    "descriptive_only_ineligible", "no_realized_action_divergence",
    "action_divergence_without_local_attribution",
    "local_action_divergence_with_descriptive_future_difference",
    "local_action_divergence_without_measured_future_difference",
}
_REVIEW_WORLD_CERTIFICATE_FIELDS = {
    "schema_version", "status", "review_identity", "review_trace_identity",
    "decision_identity", "tactical_binding_identity",
    "official_action_evidence_identity", "official_match_id",
    "scenario_archive_count", "scenario_evidence_retained",
    "final_decision_identity_bound", "runtime_tactic_verified",
    "official_action_same_match",
    "scenario_to_runtime_opportunity_matching_performed",
    "outcome_comparison_performed", "causal_effect_authorized",
    "claim_boundary", "certificate_identity",
}
_BOUNDARY = (
    "navigation over the current manager intervention session and replayable "
    "completed simulator-world chapters only; navigation state is not an "
    "outcome-effect estimate, intervention ranking or causal conclusion"
)


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _is_identity(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _identity_matches(payload: Mapping[str, Any], field: str) -> bool:
    return (
        _is_identity(payload.get(field))
        and payload[field] == _identity({
            key: value for key, value in payload.items() if key != field
        })
    )


def _normalize_retained_record_semantics(
    raw: Any, *, action_counts: Mapping[str, int],
    source_records_truncated: Any,
) -> dict[str, Any] | None:
    """Validate versioned retained semantics; keep legacy absence explicit."""
    if raw is None:
        return None
    required = {
        "schema_version", "records", "actual_action_counts",
        "primary_signal_action_counts", "signal_mode_counts",
        "hold_reference_redistribution_records",
        "direct_cross_ball_event_links",
        "locally_attributable_cross_changes",
        "retained_record_coverage_complete",
        "source_manager_record_coverage_complete",
        "full_source_distribution_authorized",
        "outcome_attribution_authorized",
    }
    actions = {"hold", "pass", "cross", "shot", "none"}
    modes = {
        "direct_preference", "suppression_only", "none",
        "legacy_unclassified",
    }
    semantic_schema_version = raw.get("schema_version") if isinstance(
        raw, Mapping
    ) else None
    if semantic_schema_version in {2, 3}:
        required.update({
            "counterfactual_action_transition_counts",
            "locally_attributable_action_transition_counts",
        })
    if semantic_schema_version == 3:
        required.update({
            "expected_change_estimator",
            "records_with_exact_change_probability",
            "attribution_eligible_records_with_exact_change_probability",
            "expected_counterfactual_action_changes",
            "exact_retained_expectation_complete",
            "full_source_expectation_authorized",
        })
    if (
        semantic_schema_version not in {1, 2, 3}
        or not isinstance(raw, Mapping)
        or set(raw) != required
    ):
        raise ValueError(
            "manager world navigator retained action semantics shape is invalid"
        )
    actual = raw.get("actual_action_counts")
    primary = raw.get("primary_signal_action_counts")
    signals = raw.get("signal_mode_counts")
    if (
        not isinstance(actual, Mapping) or set(actual) != actions
        or not isinstance(primary, Mapping) or set(primary) != actions
        or not isinstance(signals, Mapping) or set(signals) != modes
    ):
        raise ValueError(
            "manager world navigator retained action semantics partition is invalid"
        )
    counts = [
        *actual.values(), *primary.values(), *signals.values(),
        raw.get("records"),
        raw.get("hold_reference_redistribution_records"),
        raw.get("direct_cross_ball_event_links"),
        raw.get("locally_attributable_cross_changes"),
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError(
            "manager world navigator retained action semantics counts are invalid"
        )
    if not isinstance(source_records_truncated, bool):
        raise ValueError(
            "manager world navigator retained action semantics source "
            "coverage is invalid"
        )
    records = action_counts["retained_records"]
    source_complete = not source_records_truncated
    if semantic_schema_version in {2, 3}:
        transitions = raw.get("counterfactual_action_transition_counts")
        local_transitions = raw.get(
            "locally_attributable_action_transition_counts"
        )
        if any(
            not isinstance(matrix, Mapping)
            or set(matrix) != actions
            or any(
                not isinstance(row, Mapping) or set(row) != actions
                for row in matrix.values()
            )
            for matrix in (transitions, local_transitions)
        ):
            raise ValueError(
                "manager world navigator retained action transition shape "
                "is invalid"
            )
        transition_values = [
            value for matrix in (transitions, local_transitions)
            for row in matrix.values() for value in row.values()
        ]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in transition_values
        ):
            raise ValueError(
                "manager world navigator retained action transition counts "
                "are invalid"
            )
        if (
            sum(sum(row.values()) for row in transitions.values()) != records
            or any(
                sum(transitions[baseline][action] for baseline in actions)
                != actual[action]
                for action in actions
            )
            or sum(
                sum(row.values()) for row in local_transitions.values()
            ) != action_counts["locally_attributable_action_changes"]
            or sum(
                local_transitions[baseline]["cross"] for baseline in actions
            ) != raw["locally_attributable_cross_changes"]
            or any(
                local_transitions[baseline][action]
                > transitions[baseline][action]
                for baseline in actions for action in actions
            )
            or any(local_transitions[action][action] != 0 for action in actions)
            or any(
                local_transitions["none"][action] != 0
                or local_transitions[action]["none"] != 0
                for action in actions
            )
        ):
            raise ValueError(
                "manager world navigator retained action transitions are invalid"
            )
    if semantic_schema_version == 3:
        exact_records = raw.get("records_with_exact_change_probability")
        eligible_exact = raw.get(
            "attribution_eligible_records_with_exact_change_probability"
        )
        expected_changes = raw.get(
            "expected_counterfactual_action_changes"
        )
        if (
            raw.get("expected_change_estimator")
            != "shared_uniform_inverse_cdf_overlap_v1"
            or isinstance(exact_records, bool)
            or not isinstance(exact_records, int)
            or not 0 <= exact_records <= records
            or isinstance(eligible_exact, bool)
            or not isinstance(eligible_exact, int)
            or not 0 <= eligible_exact <= min(
                exact_records,
                action_counts["attribution_eligible_decisions"],
            )
            or isinstance(expected_changes, bool)
            or not isinstance(expected_changes, (int, float))
            or not math.isfinite(float(expected_changes))
            or not 0.0 <= float(expected_changes) <= float(eligible_exact)
            or raw.get("exact_retained_expectation_complete") is not (
                exact_records == records
            )
            or raw.get("full_source_expectation_authorized") is not (
                exact_records == records and source_complete
            )
        ):
            raise ValueError(
                "manager world navigator retained action expectation is invalid"
            )
    if (
        raw["records"] != records
        or sum(actual.values()) != records
        or sum(primary.values()) != records
        or sum(signals.values()) != records
        or raw["hold_reference_redistribution_records"] > records
        or raw["direct_cross_ball_event_links"] > min(
            actual["cross"], action_counts["direct_ball_event_links"],
        )
        or raw["locally_attributable_cross_changes"] > min(
            actual["cross"],
            action_counts["locally_attributable_action_changes"],
        )
        or raw.get("retained_record_coverage_complete") is not True
        or raw.get("source_manager_record_coverage_complete") is not (
            source_complete
        )
        or raw.get("full_source_distribution_authorized") is not (
            source_complete
        )
        or raw.get("outcome_attribution_authorized") is not False
    ):
        raise ValueError(
            "manager world navigator retained action semantics are invalid"
        )
    return copy.deepcopy(dict(raw))


def _validate_reviewed_scenarios(
    scenarios: Any, *, evidence_level: Any,
    fixed_budget: Any, expected_counts: Mapping[str, Any],
    expected_timing_sensitivity: Any,
) -> list[dict[str, Any]]:
    """Validate the bounded, non-ranked historical fork archive."""
    if evidence_level == "aggregate_only":
        if scenarios != []:
            raise ValueError("aggregate-only future review has scenario archive")
        return []
    if (
        evidence_level != "scenario_evidence"
        or isinstance(fixed_budget, bool)
        or not isinstance(fixed_budget, int)
        or not 2 <= fixed_budget <= 4
        or not isinstance(scenarios, list)
        or len(scenarios) != fixed_budget
    ):
        raise ValueError("reviewed future scenario archive budget is invalid")
    prior_time: float | None = None
    seen_source: set[str] = set()
    seen_archive: set[str] = set()
    normalized = []
    for raw in scenarios:
        if not isinstance(raw, Mapping) or set(raw) != _REVIEWED_SCENARIO_FIELDS:
            raise ValueError("reviewed future scenario archive fields are invalid")
        branch = raw.get("branch_at_sec")
        minute = raw.get("branch_minute")
        state_identity = raw.get("branch_state_identity")
        changed = raw.get("changed_actions")
        local = raw.get("locally_attributable_changes")
        differences = raw.get("descriptive_future_difference_count")
        if (
            raw.get("schema_version") != 1
            or not _is_identity(raw.get("source_scenario_identity"))
            or not _identity_matches(raw, "archive_identity")
            or raw["source_scenario_identity"] in seen_source
            or raw["archive_identity"] in seen_archive
            or isinstance(branch, bool)
            or not isinstance(branch, (int, float))
            or not math.isfinite(float(branch))
            or not 0 <= float(branch) <= 5400
            or (prior_time is not None and float(branch) <= prior_time)
            or isinstance(minute, bool)
            or not isinstance(minute, (int, float))
            or not math.isfinite(float(minute))
            or not math.isclose(
                float(minute), float(branch) / 60.0,
                rel_tol=0.0, abs_tol=1e-9,
            )
            or raw.get("future_status") not in _REVIEWED_SCENARIO_STATUSES
            or not isinstance(raw.get("eligible"), bool)
            or not isinstance(raw.get("anchor_verified"), bool)
            or (raw.get("eligible") and not raw.get("anchor_verified"))
            or (state_identity is not None and not _is_identity(state_identity))
            or (raw.get("anchor_verified") and state_identity is None)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                or not 0 <= value <= 100_000
                for value in (changed, local, differences)
            )
            or local > changed
            or not isinstance(
                raw.get("simulator_local_action_attribution"), bool
            )
            or (
                raw.get("simulator_local_action_attribution")
                and (local == 0 or not raw.get("eligible"))
            )
            or raw.get("outcome_causality_authorized") is not False
            or raw.get("real_football_causality_authorized") is not False
        ):
            raise ValueError("reviewed future scenario archive values are invalid")
        semantic_status = (
            "descriptive_only_ineligible" if not raw["eligible"] else
            "no_realized_action_divergence" if changed == 0 else
            "action_divergence_without_local_attribution"
            if not raw["simulator_local_action_attribution"] else
            "local_action_divergence_with_descriptive_future_difference"
            if differences > 0 else
            "local_action_divergence_without_measured_future_difference"
        )
        if raw["future_status"] != semantic_status:
            raise ValueError("reviewed future scenario archive semantics are invalid")
        prior_time = float(branch)
        seen_source.add(raw["source_scenario_identity"])
        seen_archive.add(raw["archive_identity"])
        normalized.append(dict(raw))
    rebuilt = {
        "eligible_scenarios": sum(row["eligible"] for row in normalized),
        "verified_anchor_scenarios": sum(
            row["anchor_verified"] for row in normalized
        ),
        "action_divergence_scenarios": sum(
            row["changed_actions"] > 0 for row in normalized
        ),
        "local_attribution_scenarios": sum(
            row["simulator_local_action_attribution"] for row in normalized
        ),
        "descriptive_future_difference_scenarios": sum(
            row["descriptive_future_difference_count"] > 0
            for row in normalized
        ),
    }
    signatures = {
        (
            row["future_status"], row["changed_actions"],
            row["locally_attributable_changes"],
            row["descriptive_future_difference_count"],
        )
        for row in normalized
    }
    if (
        any(rebuilt[field] != expected_counts.get(field) for field in rebuilt)
        or expected_timing_sensitivity is not (len(signatures) > 1)
    ):
        raise ValueError("reviewed future scenario archive aggregate mismatch")
    return normalized


def _descriptive_world_propagation(
    chapters: list[dict[str, Any]],
) -> dict[str, Any]:
    result_rows = [
        row["descriptive_world_after"] for row in chapters
        if row["descriptive_world_after"]["result_available"]
    ]
    persistent_rows = [
        row["descriptive_world_after"] for row in chapters
        if row["descriptive_world_after"]["persistent_state_available"]
    ]
    return {
        "chapters": len(chapters),
        "results_available": len(result_rows),
        "outcomes": {
            outcome: sum(row["outcome"] == outcome for row in result_rows)
            for outcome in ("win", "draw", "loss")
        },
        "points_earned": sum(row["points_earned"] for row in result_rows),
        "persistent_state_chapters": len(persistent_rows),
        "recovery_complete_chapters": sum(
            row["recovery_complete"] for row in persistent_rows
        ),
        "match_metrics_delta_totals": {
            field: round(sum(
                row["metrics_delta"][field] for row in persistent_rows
            ), 6)
            for field in _WORLD_METRICS
        },
        "transition_summary_totals": {
            field: sum(
                row["transition_summary"][field] for row in persistent_rows
            )
            for field in _WORLD_SUMMARY_FIELDS
        },
        "descriptive_cooccurrence_only": True,
        "state_effect_comparison_authorized": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
    }


def _local_transition_descriptive_propagation(
    chapters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stratify same-chapter facts by V3-compatible local action transition."""
    actions = ("hold", "pass", "cross", "shot", "none")
    v3_chapters = [
        chapter for chapter in chapters
        if isinstance(
            chapter["action_adoption"].get("retained_record_semantics"),
            Mapping,
        )
        and chapter["action_adoption"]["retained_record_semantics"].get(
            "schema_version"
        ) in {2, 3}
    ]
    memberships: dict[tuple[str, str], list[tuple[dict[str, Any], int]]] = {}
    chapter_transition_counts: dict[str, int] = {}
    for chapter in v3_chapters:
        matrix = chapter["action_adoption"]["retained_record_semantics"][
            "locally_attributable_action_transition_counts"
        ]
        present = 0
        for baseline in actions:
            for actual in actions:
                occurrences = matrix[baseline][actual]
                if occurrences:
                    memberships.setdefault((baseline, actual), []).append(
                        (chapter, occurrences)
                    )
                    present += 1
        chapter_transition_counts[chapter["fixture_id"]] = present
    strata = []
    for (baseline, actual), rows in memberships.items():
        source_chapters = [chapter for chapter, _ in rows]
        latest = max(
            source_chapters,
            key=lambda row: (row["matchday"], row["fixture_id"]),
        )
        strata.append({
            "transition_id": f"{baseline}_to_{actual}",
            "counterfactual_baseline_action": baseline,
            "actual_action": actual,
            "transition_occurrences": sum(count for _, count in rows),
            "chapters": len(source_chapters),
            "chapters_with_other_local_transitions": sum(
                chapter_transition_counts[chapter["fixture_id"]] > 1
                for chapter in source_chapters
            ),
            "latest_fixture_id": latest["fixture_id"],
            "latest_matchday": latest["matchday"],
            "latest_chapter_identity": latest["chapter_identity"],
            "full_source_transition_distribution_authorized": all(
                chapter["action_adoption"]["retained_record_semantics"][
                    "full_source_distribution_authorized"
                ] is True
                for chapter in source_chapters
            ),
            "descriptive_world_after": _descriptive_world_propagation(
                source_chapters
            ),
            "chapter_membership_mutually_exclusive": False,
            "cross_stratum_comparison_authorized": False,
            "outcome_attribution_authorized": False,
        })
    strata.sort(key=lambda row: (
        -row["transition_occurrences"],
        row["counterfactual_baseline_action"], row["actual_action"],
    ))
    return {
        "schema_version": 1,
        "fixtures_with_v3_transition_semantics": len(v3_chapters),
        "fixtures_without_v3_transition_semantics": (
            len(chapters) - len(v3_chapters)
        ),
        "strata_count": len(strata),
        "strata": strata,
        "chapter_membership_mutually_exclusive": False,
        "cross_stratum_comparison_authorized": False,
        "outcome_attribution_authorized": False,
        "claim_boundary": (
            "same-chapter descriptive cooccurrence after simulator-local action "
            "transitions; chapters may enter multiple strata; no transition "
            "effect, ranking, score causality or real-football claim"
        ),
    }


def _season_world_trajectory(
    chapters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a bounded chronology with whole-season descriptive cumulatives."""
    ordered = sorted(
        chapters, key=lambda row: (row["matchday"], row["fixture_id"]),
    )
    matchdays = [row["matchday"] for row in ordered]
    if len(matchdays) != len(set(matchdays)):
        raise ValueError(
            "manager world navigator completed matchdays are duplicate"
        )
    cumulative_metrics = {field: 0.0 for field in _WORLD_METRICS}
    cumulative_transitions = {
        field: 0 for field in _WORLD_SUMMARY_FIELDS
    }
    cumulative_points = 0
    cumulative_results = 0
    cumulative_persistent = 0
    cumulative_action_changes = 0
    points: list[dict[str, Any]] = []
    for sequence, chapter in enumerate(ordered, start=1):
        world = chapter["descriptive_world_after"]
        if world["result_available"]:
            cumulative_results += 1
            cumulative_points += world["points_earned"]
        if world["persistent_state_available"]:
            cumulative_persistent += 1
            for field in _WORLD_METRICS:
                cumulative_metrics[field] = round(
                    cumulative_metrics[field]
                    + world["metrics_delta"][field], 6,
                )
            for field in _WORLD_SUMMARY_FIELDS:
                cumulative_transitions[field] += (
                    world["transition_summary"][field]
                )
        local_changes = chapter["locally_attributable_action_changes"]
        cumulative_action_changes += local_changes
        markers = []
        if local_changes:
            markers.append("local_action_change")
        if not world["result_available"]:
            markers.append("result_evidence_unavailable")
        if not world["persistent_state_available"]:
            markers.append("persistent_state_evidence_unavailable")
        elif not world["recovery_complete"]:
            markers.append("recovery_pending")
        if world["persistent_state_available"]:
            if world["transition_summary"]["new_injuries"]:
                markers.append("new_injury")
            if world["transition_summary"]["new_suspensions"]:
                markers.append("new_suspension")
        if chapter["continuity_gaps"]:
            markers.append("continuity_gap")
        point = {
            "sequence": sequence,
            "fixture_id": chapter["fixture_id"],
            "matchday": chapter["matchday"],
            "chapter_identity": chapter["chapter_identity"],
            "reviewed_future_context": copy.deepcopy(
                chapter["reviewed_future_context"]
            ),
            "reviewed_world_model_chain_complete": chapter[
                "reviewed_world_model_chain_complete"
            ],
            "review_to_official_world": copy.deepcopy(
                chapter["review_to_official_world"]
            ),
            "action_adoption_state": chapter["action_adoption"]["state"],
            "bounded_official_action_semantics": {
                **{
                    field: chapter["action_adoption"][field]
                    for field in _OFFICIAL_ACTION_SEMANTIC_COUNT_FIELDS
                },
                "semantic_examples_truncated": chapter[
                    "action_adoption"
                ]["semantic_examples_truncated"],
                "semantic_example_coverage_complete": chapter[
                    "action_adoption"
                ]["semantic_example_coverage_complete"],
                "full_record_distribution_authorized": False,
            },
            "official_retained_record_semantics": copy.deepcopy(
                chapter["action_adoption"]["retained_record_semantics"]
            ),
            "locally_attributable_action_changes": local_changes,
            "result_available": world["result_available"],
            "outcome": world["outcome"],
            "points_earned": world["points_earned"],
            "persistent_state_available": world[
                "persistent_state_available"
            ],
            "recovery_complete": world["recovery_complete"],
            "match_metrics_delta": copy.deepcopy(world["metrics_delta"]),
            "match_transition_summary": copy.deepcopy(
                world["transition_summary"]
            ),
            "cumulative": {
                "results_available": cumulative_results,
                "points_earned": cumulative_points,
                "persistent_state_chapters": cumulative_persistent,
                "locally_attributable_action_changes": (
                    cumulative_action_changes
                ),
                "match_metrics_delta_totals": copy.deepcopy(
                    cumulative_metrics
                ),
                "transition_summary_totals": copy.deepcopy(
                    cumulative_transitions
                ),
            },
            "world_change_markers": markers,
            "descriptive_chronology_only": True,
            "turning_point_inference_authorized": False,
            "causal_effect_authorized": False,
        }
        point["trajectory_point_identity"] = _identity(point)
        points.append(point)
    visible = points[-MAX_HISTORY_CHAPTERS:]
    reviewed_points = [
        row for row in points
        if row["reviewed_future_context"]["available"]
    ]
    return {
        "total_points": len(points),
        "visible_points": len(visible),
        "points_truncated": len(points) > MAX_HISTORY_CHAPTERS,
        "results_available": cumulative_results,
        "persistent_state_chapters": cumulative_persistent,
        "reviewed_future_chapters": len(reviewed_points),
        "reviewed_future_selected_chapters": sum(
            row["reviewed_future_context"]["selected_for_fixture"]
            for row in reviewed_points
        ),
        "reviewed_world_model_chain_complete": sum(
            row["reviewed_world_model_chain_complete"] for row in points
        ),
        "scenario_evidence_chapters": sum(
            row["reviewed_future_context"]["evidence_level"]
            == "scenario_evidence"
            for row in reviewed_points
        ),
        "reviewed_scenario_archives": sum(
            len(row["reviewed_future_context"]["scenarios"])
            for row in reviewed_points
        ),
        "cross_action_mechanism_examples": sum(
            row["reviewed_future_context"][
                "cross_action_mechanism_examples"
            ]
            for row in reviewed_points
        ),
        "direct_preference_mechanism_examples": sum(
            row["reviewed_future_context"][
                "direct_preference_mechanism_examples"
            ]
            for row in reviewed_points
        ),
        "suppression_only_mechanism_examples": sum(
            row["reviewed_future_context"][
                "suppression_only_mechanism_examples"
            ]
            for row in reviewed_points
        ),
        "hold_reference_redistribution_examples": sum(
            row["reviewed_future_context"][
                "hold_reference_redistribution_examples"
            ]
            for row in reviewed_points
        ),
        "nonzero_cross_descriptive_windows": sum(
            row["reviewed_future_context"][
                "nonzero_cross_descriptive_windows"
            ]
            for row in reviewed_points
        ),
        "identity_verified_scenario_archives": sum(
            _identity_matches(scenario, "archive_identity")
            for row in reviewed_points
            for scenario in row["reviewed_future_context"]["scenarios"]
        ),
        "reviewed_action_divergence_chapters": sum(
            row["reviewed_future_context"][
                "action_divergence_scenarios"
            ] > 0
            for row in reviewed_points
        ),
        "reviewed_timing_sensitivity_chapters": sum(
            row["reviewed_future_context"][
                "timing_sensitivity_observed"
            ] is True
            for row in reviewed_points
        ),
        "points": visible,
        "descriptive_chronology_only": True,
        "missing_evidence_imputed": False,
        "future_to_outcome_comparison_performed": False,
        "turning_point_inference_authorized": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
    }


def _primary_action(workflow_state: str) -> dict[str, Any] | None:
    actions = {
        "decision_required": (
            "freeze_manager_decision", "冻结本场经理决策",
            "manager-decision-form",
        ),
        "ready_to_explore": (
            "request_bounded_futures", "生成身份绑定的有界未来",
            "request-manager-future-set",
        ),
        "future_generation_in_progress": (
            "wait_for_bounded_futures", "等待固定场景预算完成", None,
        ),
        "future_generation_interrupted": (
            "resume_bounded_futures", "从已验证进度恢复未来生成",
            "manager-future-set-list",
        ),
        "future_generation_failed": (
            "retry_bounded_futures", "检查失败后重新生成有界未来",
            "request-manager-future-set",
        ),
        "evidence_ready_for_review": (
            "review_future_evidence", "复核未来证据并保留或修改方案",
            "manager-future-set-list",
        ),
        "review_recorded_decision_refrozen": (
            "advance_official_world", "推进正式比赛世界",
            "play-matchday",
        ),
    }
    selected = actions.get(workflow_state)
    if selected is None:
        return None
    action_id, label, target = selected
    return {
        "action_id": action_id,
        "label": label,
        "target_element_id": target,
    }


def _secondary_actions(
    workflow_state: str,
    allowed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    primary = _primary_action(workflow_state)
    primary_id = primary.get("action_id") if primary else None
    candidates = []
    if allowed.get("request_future_set") is True:
        candidates.append({
            "action_id": "request_bounded_futures",
            "label": "再生成一组有界未来",
            "target_element_id": "request-manager-future-set",
        })
    if allowed.get("review_future_set") is True:
        candidates.append({
            "action_id": "review_future_evidence",
            "label": "复核当前未来证据",
            "target_element_id": "manager-future-set-list",
        })
    if allowed.get("advance_official_match") is True:
        candidates.append({
            "action_id": "advance_official_world",
            "label": "跳过可选探索并推进正式比赛",
            "target_element_id": "play-matchday",
        })
    return [
        row for row in candidates if row["action_id"] != primary_id
    ]


def _history_chapter(entry: Mapping[str, Any]) -> dict[str, Any]:
    thread = entry.get("world_evolution_thread")
    if not isinstance(thread, Mapping):
        raise ValueError("manager world navigator chapter thread is unavailable")
    if (
        not _identity_matches(entry, "entry_identity")
        or not _identity_matches(thread, "thread_identity")
        or entry.get("fixture_id") != thread.get("fixture_id")
        or not isinstance(entry.get("future_review_execution_trace"), Mapping)
        or thread.get("causal_effect_authorized") is not False
        or thread.get("outcome_effect_estimate") is not None
    ):
        raise ValueError("manager world navigator chapter identity is invalid")
    stages = thread.get("stages")
    if not isinstance(stages, list):
        raise ValueError("manager world navigator chapter stages are invalid")
    stage_by_id = {
        row.get("stage_id"): row
        for row in stages if isinstance(row, Mapping)
    }
    required = {
        "prematch_future_review",
        "frozen_manager_decision",
        "official_tactical_runtime",
        "official_world_model_actions",
        "observed_match_result",
        "persistent_world_state",
    }
    if set(stage_by_id) != required or any(
        not _identity_matches(row, "stage_identity")
        for row in stage_by_id.values()
    ):
        raise ValueError("manager world navigator chapter stages are incomplete")
    observed = entry.get("observed_result")
    observed = observed if isinstance(observed, Mapping) else {}
    result_stage = stage_by_id["observed_match_result"]
    review_stage = stage_by_id["prematch_future_review"]
    decision_stage = stage_by_id["frozen_manager_decision"]
    entry_decision = entry.get("decision")
    entry_decision = (
        entry_decision if isinstance(entry_decision, Mapping) else {}
    )
    tactical = stage_by_id["official_tactical_runtime"]
    action = stage_by_id["official_world_model_actions"]
    persistent = stage_by_id["persistent_world_state"]
    action_count_fields = (
        "retained_records",
        "resolved_action_decisions",
        "influenced_decisions",
        "attribution_eligible_decisions",
        "locally_attributable_action_changes",
        "direct_ball_event_links",
        "changed_direct_ball_event_links",
        "decision_only_no_trajectory",
        "unresolved_ball_event_links",
        "source_match_opportunities",
    )
    action_counts = {
        field: action.get(field, 0) for field in action_count_fields
    }
    action_semantic_counts = {
        field: action.get(field, 0)
        for field in _OFFICIAL_ACTION_SEMANTIC_COUNT_FIELDS
    }
    semantic_examples_truncated = action.get(
        "semantic_examples_truncated", False
    )
    semantic_example_coverage_complete = action.get(
        "semantic_example_coverage_complete", False
    )
    retained_record_semantics = _normalize_retained_record_semantics(
        action.get("retained_record_semantics"),
        action_counts=action_counts,
        source_records_truncated=action.get("source_records_truncated"),
    )
    local_changes = action_counts["locally_attributable_action_changes"]
    action_status = action.get("status")
    action_source_identity = action.get("source_identity")
    transition_identity = persistent.get("source_identity")
    gaps = thread.get("continuity_gaps")
    matchday = entry.get("matchday")
    result_available = bool(observed)
    score = observed.get("score")
    outcome = observed.get("outcome")
    points_earned = observed.get("points_earned")
    result_source_identity = result_stage.get("source_identity")
    metrics_delta = persistent.get("metrics_delta")
    transition_summary = persistent.get("transition_summary")
    society_transition = persistent.get("society_transition")
    persistent_state_available = transition_identity is not None
    accounting = entry.get("long_term_accounting")
    accounting = accounting if isinstance(accounting, Mapping) else {}
    persistent_source = accounting.get("persistent_team_state_delta")
    persistent_source = (
        persistent_source if isinstance(persistent_source, Mapping) else {}
    )
    source_match_delta = persistent_source.get("match_delta")
    source_match_delta = (
        source_match_delta if isinstance(source_match_delta, Mapping) else {}
    )
    review_trace = entry.get("future_review_execution_trace")
    review_trace = review_trace if isinstance(review_trace, Mapping) else {}
    terminal_review = review_trace.get("terminal_review")
    terminal_review = (
        terminal_review if isinstance(terminal_review, Mapping) else {}
    )
    review_available = review_trace.get("available") is True
    review_counts = {
        field: review_stage.get(field) for field in _FUTURE_REVIEW_COUNT_FIELDS
    }
    review_semantic_counts = {
        field: review_stage.get(field, 0)
        for field in _FUTURE_MECHANISM_SEMANTIC_FIELDS
    }
    terminal_semantic_counts = {
        field: terminal_review.get(field, 0)
        for field in _FUTURE_MECHANISM_SEMANTIC_FIELDS
    }
    reviewed_scenarios = review_stage.get("reviewed_scenarios")
    review_world = thread.get("review_to_official_world")
    expected_same_match = bool(
        tactical.get("status") == "runtime_verified"
        and action.get("status") in _ACTION_EVIDENCE_STATES
        and isinstance(tactical.get("match_id"), str)
        and bool(tactical.get("match_id"))
        and tactical.get("match_id") == action.get("match_id")
    )
    expected_review_status = (
        "selected_for_fixture"
        if terminal_review.get("linked_to_final_selection") is True
        else "superseded_before_execution"
        if review_available else "not_reviewed"
    )
    if (
        decision_stage.get("status") != "frozen"
        or decision_stage.get("source_identity") != entry.get("decision_identity")
        or not _is_identity(decision_stage.get("source_identity"))
        or not isinstance(decision_stage.get("tactic"), str)
        or not decision_stage.get("tactic")
        or not isinstance(decision_stage.get("rotation"), str)
        or not decision_stage.get("rotation")
        or decision_stage.get("tactic") != entry_decision.get("tactic")
        or decision_stage.get("rotation") != entry_decision.get("rotation")
        or isinstance(matchday, bool)
        or not isinstance(matchday, int)
        or matchday < 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in action_counts.values()
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in action_semantic_counts.values()
        )
        or not isinstance(semantic_examples_truncated, bool)
        or not isinstance(semantic_example_coverage_complete, bool)
        or action_semantic_counts["retained_semantic_examples"]
        > action_counts["retained_records"]
        or sum(
            action_semantic_counts[field] for field in (
                "semantic_direct_preference_examples",
                "semantic_suppression_only_examples",
                "semantic_no_signal_examples",
                "semantic_legacy_unclassified_examples",
            )
        ) != action_semantic_counts["retained_semantic_examples"]
        or action_semantic_counts["semantic_cross_action_examples"]
        > action_semantic_counts["retained_semantic_examples"]
        or action_semantic_counts[
            "semantic_hold_reference_redistribution_examples"
        ] > action_semantic_counts["retained_semantic_examples"]
        or action_semantic_counts[
            "semantic_direct_cross_ball_event_examples"
        ] > min(
            action_semantic_counts["semantic_cross_action_examples"],
            action_counts["direct_ball_event_links"],
        )
        or (
            semantic_examples_truncated
            and action_semantic_counts["retained_semantic_examples"]
            >= action_counts["retained_records"]
        )
        or semantic_example_coverage_complete is not bool(
            action_counts["retained_records"] > 0
            and action_semantic_counts["retained_semantic_examples"]
            == action_counts["retained_records"]
            and not semantic_examples_truncated
        )
        or not isinstance(action.get("manager_record_coverage_complete"), bool)
        or not isinstance(action.get("source_records_truncated"), bool)
        or (
            action_status in _ACTION_EVIDENCE_STATES
            and action.get("manager_record_coverage_complete") is not (
                not action.get("source_records_truncated")
            )
        )
        or action_counts["influenced_decisions"]
        > action_counts["retained_records"]
        or action_counts["locally_attributable_action_changes"]
        > action_counts["attribution_eligible_decisions"]
        or action_counts["locally_attributable_action_changes"]
        > action_counts["influenced_decisions"]
        or action_counts["attribution_eligible_decisions"]
        > action_counts["retained_records"]
        or action_counts["resolved_action_decisions"]
        > action_counts["retained_records"]
        or action_counts["direct_ball_event_links"]
        > action_counts["retained_records"]
        or action_counts["changed_direct_ball_event_links"] > min(
            action_counts["direct_ball_event_links"], local_changes,
        )
        or (
            action_counts["direct_ball_event_links"]
            + action_counts["decision_only_no_trajectory"]
            + action_counts["unresolved_ball_event_links"]
            != action_counts["retained_records"]
        )
        or action_counts["retained_records"]
        > action_counts["source_match_opportunities"]
        or (
            review_available
            and (
                not _identity_matches(review_trace, "trace_identity")
                or review_stage.get("status") != expected_review_status
                or review_stage.get("source_identity")
                != review_trace.get("trace_identity")
                or review_trace.get("outcome_comparison_performed") is not False
                or review_trace.get("outcome_effect_estimate") is not None
                or review_trace.get("causal_effect_authorized") is not False
                or review_stage.get("review_identity")
                != terminal_review.get("review_identity")
                or not _is_identity(terminal_review.get("review_identity"))
                or not isinstance(
                    terminal_review.get("linked_to_final_selection"), bool,
                )
                or review_stage.get("review_intent")
                != terminal_review.get("intent")
                or review_stage.get("evidence_level")
                != terminal_review.get("evidence_level")
                or review_stage.get("retained_mechanism_examples")
                != terminal_review.get("retained_mechanism_examples")
                or review_semantic_counts != terminal_semantic_counts
                or reviewed_scenarios
                != terminal_review.get("reviewed_scenarios")
                or any(
                    review_counts[field] != terminal_review.get(field)
                    for field in _FUTURE_REVIEW_COUNT_FIELDS
                )
                or review_stage.get("timing_sensitivity_observed")
                is not terminal_review.get("timing_sensitivity_observed")
                or review_stage.get("ranking_performed") is not False
                or terminal_review.get("ranking_performed") is not False
                or review_stage.get("best_branch_time") is not None
                or terminal_review.get("best_branch_time") is not None
                or terminal_review.get("intent") not in {
                    "keep_after_review", "revise_after_review",
                }
                or terminal_review.get("evidence_level") not in {
                    "scenario_evidence", "aggregate_only",
                }
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in (
                        *review_counts.values(),
                        *(
                            terminal_review.get(field)
                            for field in _FUTURE_REVIEW_COUNT_FIELDS
                        ),
                        review_stage.get("retained_mechanism_examples"),
                        terminal_review.get("retained_mechanism_examples"),
                        *review_semantic_counts.values(),
                        *terminal_semantic_counts.values(),
                    )
                )
                or review_semantic_counts[
                    "cross_action_mechanism_examples"
                ] > review_stage.get("retained_mechanism_examples")
                or (
                    review_semantic_counts[
                        "direct_preference_mechanism_examples"
                    ]
                    + review_semantic_counts[
                        "suppression_only_mechanism_examples"
                    ]
                    > review_stage.get("retained_mechanism_examples")
                )
                or review_semantic_counts[
                    "hold_reference_redistribution_examples"
                ] > review_stage.get("retained_mechanism_examples")
                or review_semantic_counts[
                    "nonzero_cross_descriptive_windows"
                ] > 2 * review_stage.get("retained_mechanism_examples")
                or review_counts["eligible_scenarios"]
                > review_counts["fixed_scenario_budget"]
                or review_counts["verified_anchor_scenarios"]
                > review_counts["fixed_scenario_budget"]
                or review_counts["eligible_scenarios"]
                > review_counts["verified_anchor_scenarios"]
                or review_counts["local_attribution_scenarios"]
                > review_counts["action_divergence_scenarios"]
                or not isinstance(
                    review_stage.get("timing_sensitivity_observed"), bool,
                )
            )
        )
        or (
            not review_available
            and (
                review_trace.get("available") is not False
                or not isinstance(review_trace.get("reason"), str)
                or not review_trace.get("reason")
                or review_stage.get("status") != "not_reviewed"
                or review_stage.get("source_identity") is not None
                or review_stage.get("review_identity") is not None
                or review_stage.get("review_intent") is not None
                or review_stage.get("evidence_level") is not None
                or review_stage.get("retained_mechanism_examples") != 0
                or any(
                    value != 0 for value in review_semantic_counts.values()
                )
                or any(value != 0 for value in review_counts.values())
                or review_stage.get("timing_sensitivity_observed") is not None
                or review_stage.get("ranking_performed") is not None
                or review_stage.get("best_branch_time") is not None
                or reviewed_scenarios != []
            )
        )
        or (
            result_available
            and (
                not isinstance(score, Mapping)
                or set(score) != {"home", "away"}
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in score.values()
                )
                or outcome not in {"win", "draw", "loss"}
                or isinstance(points_earned, bool)
                or not isinstance(points_earned, int)
                or points_earned != {
                    "win": 3, "draw": 1, "loss": 0,
                }[outcome]
                or observed.get("descriptive_only") is not True
                or result_stage.get("status") != "observed_descriptive"
                or result_source_identity != _identity(observed)
                or result_stage.get("score") != score
                or result_stage.get("outcome") != outcome
                or result_stage.get("points_earned") != points_earned
                or result_stage.get("descriptive_only") is not True
            )
        )
        or (
            not result_available
            and (
                result_stage.get("status") != "result_evidence_unavailable"
                or result_source_identity is not None
            )
        )
        or (
            persistent_state_available
            and (
                persistent.get("status") not in {
                    "after_recovery_recorded",
                    "after_match_recorded_recovery_pending",
                }
                or not isinstance(metrics_delta, Mapping)
                or set(metrics_delta) != set(_WORLD_METRICS)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in metrics_delta.values()
                )
                or not isinstance(transition_summary, Mapping)
                or set(transition_summary) != set(_WORLD_SUMMARY_FIELDS)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in transition_summary.values()
                )
                or persistent.get("recovery_complete") is not (
                    persistent.get("status") == "after_recovery_recorded"
                )
                or persistent_source.get("available") is not True
                or transition_identity
                != persistent_source.get("transition_identity")
                or persistent.get("recovery_complete") is not (
                    persistent_source.get("recovery_complete") is True
                )
                or metrics_delta != source_match_delta.get("metrics_delta")
                or transition_summary != source_match_delta.get("summary")
                or society_transition
                != source_match_delta.get("society_transition")
            )
        )
        or (
            not persistent_state_available
            and (
                persistent.get("status")
                != "persistent_state_evidence_unavailable"
                or metrics_delta is not None
                or transition_summary is not None
                or society_transition is not None
                or persistent.get("recovery_complete") is not False
                or persistent_source.get("available") is True
            )
        )
        or (
            action_source_identity is not None
            and not _is_identity(action_source_identity)
        )
        or (
            transition_identity is not None
            and not _is_identity(transition_identity)
        )
        or not isinstance(gaps, list)
        or any(not isinstance(gap, str) or not gap for gap in gaps)
        or len(gaps) != len(set(gaps))
    ):
        raise ValueError("manager world navigator chapter facts are invalid")
    player_changes = _validated_player_changes(
        source_match_delta,
        transition_summary,
        persistent_state_available=persistent_state_available,
    )
    society_transition = _validated_society_transition(
        source_match_delta,
        persistent_state_available=persistent_state_available,
    )
    if review_available:
        _validate_reviewed_scenarios(
            reviewed_scenarios,
            evidence_level=review_stage.get("evidence_level"),
            fixed_budget=review_counts["fixed_scenario_budget"],
            expected_counts=review_counts,
            expected_timing_sensitivity=review_stage.get(
                "timing_sensitivity_observed"
            ),
        )
    if (
        not isinstance(review_world, Mapping)
        or set(review_world) != _REVIEW_WORLD_CERTIFICATE_FIELDS
        or review_world.get("schema_version") != 1
        or not _identity_matches(review_world, "certificate_identity")
        or review_world.get("review_identity")
        != review_stage.get("review_identity")
        or review_world.get("review_trace_identity")
        != review_stage.get("source_identity")
        or review_world.get("decision_identity") != entry.get("decision_identity")
        or review_world.get("tactical_binding_identity")
        != tactical.get("source_identity")
        or review_world.get("official_action_evidence_identity")
        != action_source_identity
        or review_world.get("official_match_id") != action.get("match_id")
        or isinstance(review_world.get("scenario_archive_count"), bool)
        or not isinstance(review_world.get("scenario_archive_count"), int)
        or review_world.get("scenario_archive_count")
        != len(reviewed_scenarios or [])
        or review_world.get("scenario_evidence_retained") is not bool(
            reviewed_scenarios
        )
        or review_world.get("final_decision_identity_bound") is not (
            review_stage.get("status") == "selected_for_fixture"
        )
        or review_world.get("runtime_tactic_verified") is not (
            tactical.get("status") == "runtime_verified"
        )
        or review_world.get("official_action_same_match") is not (
            expected_same_match
        )
        or review_world.get(
            "scenario_to_runtime_opportunity_matching_performed"
        ) is not False
        or review_world.get("outcome_comparison_performed") is not False
        or review_world.get("causal_effect_authorized") is not False
        or not isinstance(review_world.get("claim_boundary"), str)
        or not review_world.get("claim_boundary")
    ):
        raise ValueError(
            "manager world navigator review-world certificate is invalid"
        )
    expected_review_world_status = (
        "not_reviewed" if not review_available else
        "review_superseded"
        if review_stage.get("status") != "selected_for_fixture" else
        "tactical_runtime_binding_unavailable"
        if tactical.get("status") != "runtime_verified" else
        "official_action_evidence_unavailable"
        if not expected_same_match else "complete"
    )
    if (
        review_world.get("status") != expected_review_world_status
        or thread.get("reviewed_world_model_chain_complete") is not (
            expected_review_world_status == "complete"
        )
    ):
        raise ValueError(
            "manager world navigator review-world continuity is invalid"
        )
    if action_status == "locally_attributable_action_changes_observed":
        adoption_state = "realized_action_change"
    elif action_status == "probability_influence_without_realized_action_change":
        adoption_state = "probability_influence_only"
    elif action_status == "no_nonzero_action_influence_observed":
        adoption_state = "no_nonzero_influence"
    elif action_status == "not_applicable_stable_mode":
        adoption_state = "not_applicable_stable_mode"
    else:
        adoption_state = "evidence_unavailable"
    if (
        action_status not in _COMPLETED_ACTION_STATES
        or (action_status in _ACTION_EVIDENCE_STATES)
        is not (action_source_identity is not None)
        or (
            action_status not in _ACTION_EVIDENCE_STATES
            and (
                any(action_counts.values())
                or action.get("manager_record_coverage_complete") is not False
                or action.get("source_records_truncated") is not False
            )
        )
        or (
            action_status == "locally_attributable_action_changes_observed"
            and local_changes < 1
        )
        or (
            action_status == "probability_influence_without_realized_action_change"
            and (
                action_counts["influenced_decisions"] < 1
                or local_changes != 0
            )
        )
        or (
            action_status == "no_nonzero_action_influence_observed"
            and action_counts["influenced_decisions"] != 0
        )
    ):
        raise ValueError("manager world navigator action adoption is invalid")
    payload = {
        "fixture_id": entry.get("fixture_id"),
        "matchday": matchday,
        "lifecycle_state": entry.get("lifecycle_state"),
        "source_entry_identity": entry.get("entry_identity"),
        "source_thread_identity": thread.get("thread_identity"),
        "thread_state": thread.get("thread_state"),
        "official_runtime_chain_complete": (
            thread.get("official_runtime_chain_complete") is True
        ),
        "world_model_runtime_chain_complete": (
            thread.get("world_model_runtime_chain_complete") is True
        ),
        "reviewed_world_model_chain_complete": (
            thread.get("reviewed_world_model_chain_complete") is True
        ),
        "review_to_official_world": copy.deepcopy(review_world),
        "stage_statuses": {
            stage_id: stage_by_id[stage_id].get("status")
            for stage_id in (
                "prematch_future_review",
                "frozen_manager_decision",
                "official_tactical_runtime",
                "official_world_model_actions",
                "observed_match_result",
                "persistent_world_state",
            )
        },
        "score": copy.deepcopy(score),
        "outcome": outcome,
        "manager_choice": {
            "decision_identity": decision_stage.get("source_identity"),
            "selected_tactic": decision_stage.get("tactic"),
            "rotation": decision_stage.get("rotation"),
            "applied_tactic": tactical.get("applied_tactic"),
            "runtime_verified": tactical.get("status") == "runtime_verified",
            "runtime_matches_selection": bool(
                tactical.get("status") == "runtime_verified"
                and tactical.get("applied_tactic") == decision_stage.get("tactic")
            ),
        },
        "reviewed_future_context": {
            "available": review_available,
            "selected_for_fixture": (
                review_stage.get("status") == "selected_for_fixture"
            ),
            "review_identity": review_stage.get("review_identity"),
            "intent": review_stage.get("review_intent"),
            "evidence_level": review_stage.get("evidence_level"),
            **copy.deepcopy(review_counts),
            "retained_mechanism_examples": review_stage.get(
                "retained_mechanism_examples"
            ),
            **copy.deepcopy(review_semantic_counts),
            "timing_sensitivity_observed": review_stage.get(
                "timing_sensitivity_observed"
            ),
            "ranking_performed": review_stage.get("ranking_performed"),
            "best_branch_time": review_stage.get("best_branch_time"),
            "scenarios": copy.deepcopy(reviewed_scenarios),
            "outcome_comparison_performed": False,
            "causal_effect_authorized": False,
        },
        "descriptive_world_after": {
            "result_available": result_available,
            "outcome": outcome if result_available else None,
            "points_earned": points_earned if result_available else None,
            "persistent_state_available": persistent_state_available,
            "recovery_complete": (
                persistent.get("recovery_complete") is True
                if persistent_state_available else False
            ),
            "metrics_delta": (
                copy.deepcopy(dict(metrics_delta))
                if persistent_state_available else None
            ),
            "transition_summary": (
                copy.deepcopy(dict(transition_summary))
                if persistent_state_available else None
            ),
            "players_changed": player_changes,
            "society_transition": (
                copy.deepcopy(dict(society_transition))
                if persistent_state_available
                and isinstance(society_transition, Mapping)
                else None
            ),
            "descriptive_cooccurrence_only": True,
            "causal_effect_authorized": False,
        },
        "locally_attributable_action_changes": local_changes,
        "action_adoption": {
            "state": adoption_state,
            **copy.deepcopy(action_counts),
            **copy.deepcopy(action_semantic_counts),
            "semantic_examples_truncated": semantic_examples_truncated,
            "semantic_example_coverage_complete": (
                semantic_example_coverage_complete
            ),
            "retained_record_semantics": retained_record_semantics,
            "manager_record_coverage_complete": action.get(
                "manager_record_coverage_complete"
            ),
            "source_records_truncated": action.get(
                "source_records_truncated"
            ),
        },
        "official_action_evidence_identity": action_source_identity,
        "persistent_transition_identity": transition_identity,
        "continuity_gaps": copy.deepcopy(gaps),
        "causal_effect_authorized": False,
    }
    payload["chapter_identity"] = _identity(payload)
    return payload


def build_manager_world_navigator(
    season: Mapping[str, Any],
) -> dict[str, Any]:
    """Join the current manager session and completed ledger chapters."""
    season_id = str(season.get("season_id") or "")
    revision = season.get("revision")
    workspace = season.get("manager_intervention_workspace")
    ledger = season.get("manager_decision_ledger")
    if (
        not season_id
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(workspace, Mapping)
        or not _identity_matches(workspace, "workspace_identity")
        or workspace.get("season_id") != season_id
        or workspace.get("season_revision") != revision
        or not isinstance(ledger, Mapping)
        or not isinstance(ledger.get("entries"), list)
    ):
        raise ValueError("manager world navigator sources are invalid")

    if any(
        not isinstance(row, Mapping)
        or row.get("lifecycle_state") not in _LIFECYCLE_STATES
        for row in ledger["entries"]
    ):
        raise ValueError("manager world navigator ledger entries are invalid")
    fixture_ids = [row.get("fixture_id") for row in ledger["entries"]]
    if (
        any(not isinstance(fixture_id, str) or not fixture_id
            for fixture_id in fixture_ids)
        or len(fixture_ids) != len(set(fixture_ids))
    ):
        raise ValueError(
            "manager world navigator fixture identities are invalid or duplicate"
        )
    completed_entries = [
        row for row in ledger["entries"]
        if row.get("lifecycle_state") != "frozen_awaiting_execution"
    ]
    all_chapters = sorted(
        (_history_chapter(row) for row in completed_entries),
        key=lambda row: (row["matchday"], row["fixture_id"]),
    )
    chapters = list(reversed(all_chapters[-MAX_HISTORY_CHAPTERS:]))
    world_trajectory = _season_world_trajectory(all_chapters)
    workflow_state = str(workspace.get("workflow_state") or "")
    if workflow_state not in _WORKFLOW_STATES:
        raise ValueError("manager world navigator workflow state is invalid")
    allowed = workspace.get("allowed_actions")
    allowed = allowed if isinstance(allowed, Mapping) else {}
    current_gaps = workspace.get("continuity_gaps")
    if (
        not isinstance(current_gaps, list)
        or any(not isinstance(gap, str) or not gap for gap in current_gaps)
        or len(current_gaps) != len(set(current_gaps))
    ):
        raise ValueError(
            "manager world navigator current continuity gaps are invalid"
        )
    current = {
        "workflow_state": workflow_state,
        "workspace_identity": workspace.get("workspace_identity"),
        "fixture": copy.deepcopy(workspace.get("fixture")),
        "frozen_decision_identity": (
            (workspace.get("frozen_intervention") or {}).get(
                "decision_identity"
            )
            if isinstance(
                workspace.get("frozen_intervention"), Mapping,
            ) else None
        ),
        "stages": copy.deepcopy(workspace.get("stages") or []),
        "evidence_summary": copy.deepcopy(
            workspace.get("evidence_summary")
        ),
        "continuity_gaps": copy.deepcopy(current_gaps),
    }
    current["current_chapter_identity"] = _identity(current)
    primary = _primary_action(workflow_state)
    influence_path = {
        "review_selected": sum(
            row["stage_statuses"]["prematch_future_review"]
            == "selected_for_fixture"
            for row in all_chapters
        ),
        "reviewed_future_scenario_evidence": sum(
            row["reviewed_future_context"]["evidence_level"]
            == "scenario_evidence"
            for row in all_chapters
        ),
        "reviewed_future_scenario_archives": sum(
            len(row["reviewed_future_context"]["scenarios"])
            for row in all_chapters
        ),
        "reviewed_future_action_divergence": sum(
            row["reviewed_future_context"][
                "action_divergence_scenarios"
            ] > 0
            for row in all_chapters
        ),
        "reviewed_future_local_attribution": sum(
            row["reviewed_future_context"][
                "local_attribution_scenarios"
            ] > 0
            for row in all_chapters
        ),
        "reviewed_future_timing_sensitivity": sum(
            row["reviewed_future_context"][
                "timing_sensitivity_observed"
            ] is True
            for row in all_chapters
        ),
        "reviewed_future_cross_actions": sum(
            row["reviewed_future_context"][
                "cross_action_mechanism_examples"
            ]
            for row in all_chapters
        ),
        "reviewed_future_direct_preferences": sum(
            row["reviewed_future_context"][
                "direct_preference_mechanism_examples"
            ]
            for row in all_chapters
        ),
        "reviewed_future_suppression_only_signals": sum(
            row["reviewed_future_context"][
                "suppression_only_mechanism_examples"
            ]
            for row in all_chapters
        ),
        "reviewed_future_hold_reference_redistributions": sum(
            row["reviewed_future_context"][
                "hold_reference_redistribution_examples"
            ]
            for row in all_chapters
        ),
        "reviewed_future_nonzero_cross_windows": sum(
            row["reviewed_future_context"][
                "nonzero_cross_descriptive_windows"
            ]
            for row in all_chapters
        ),
        "runtime_tactic_verified": sum(
            row["stage_statuses"]["official_tactical_runtime"]
            == "runtime_verified"
            for row in all_chapters
        ),
        "official_action_evidence": sum(
            row["official_action_evidence_identity"] is not None
            for row in all_chapters
        ),
        "chapters_with_local_action_changes": sum(
            row["locally_attributable_action_changes"] > 0
            for row in all_chapters
        ),
        "persistent_world_transitions": sum(
            row["persistent_transition_identity"] is not None
            for row in all_chapters
        ),
        "complete_world_model_runtime_chains": sum(
            row["world_model_runtime_chain_complete"]
            for row in all_chapters
        ),
        "complete_reviewed_world_model_chains": sum(
            row["reviewed_world_model_chain_complete"]
            for row in all_chapters
        ),
    }
    adoption_totals = {
        field: sum(
            row["action_adoption"][field] for row in all_chapters
        )
        for field in (
            "retained_records",
            "resolved_action_decisions",
            "influenced_decisions",
            "attribution_eligible_decisions",
            "locally_attributable_action_changes",
            "direct_ball_event_links",
            "changed_direct_ball_event_links",
            "decision_only_no_trajectory",
            "unresolved_ball_event_links",
            "source_match_opportunities",
        )
    }
    semantic_sample_totals = {
        field: sum(
            row["action_adoption"][field] for row in all_chapters
        )
        for field in _OFFICIAL_ACTION_SEMANTIC_COUNT_FIELDS
    }
    full_semantic_rows = [
        row["action_adoption"]["retained_record_semantics"]
        for row in all_chapters
        if isinstance(
            row["action_adoption"]["retained_record_semantics"], Mapping,
        )
    ]
    transition_semantic_rows = [
        row for row in full_semantic_rows
        if row.get("schema_version") in {2, 3}
    ]
    expectation_semantic_rows = [
        row for row in full_semantic_rows if row.get("schema_version") == 3
    ]
    semantic_actions = ("hold", "pass", "cross", "shot", "none")
    semantic_modes = (
        "direct_preference", "suppression_only", "none",
        "legacy_unclassified",
    )
    adoption_state_facts: dict[str, dict[str, Any]] = {}
    for chapter in all_chapters:
        state = chapter["action_adoption"]["state"]
        facts = adoption_state_facts.setdefault(state, {
            "chapters": 0,
            "latest_fixture_id": None,
            "latest_matchday": 0,
            "latest_chapter_identity": None,
        })
        facts["chapters"] += 1
        current_key = (
            facts["latest_matchday"], str(facts["latest_fixture_id"] or ""),
        )
        chapter_key = (chapter["matchday"], chapter["fixture_id"])
        if chapter_key > current_key:
            facts["latest_fixture_id"] = chapter["fixture_id"]
            facts["latest_matchday"] = chapter["matchday"]
            facts["latest_chapter_identity"] = chapter["chapter_identity"]
    action_adoption_ledger = {
        "fixtures_with_action_evidence": sum(
            row["official_action_evidence_identity"] is not None
            for row in all_chapters
        ),
        "fixtures_with_complete_record_coverage": sum(
            row["official_action_evidence_identity"] is not None
            and row["action_adoption"]["manager_record_coverage_complete"]
            for row in all_chapters
        ),
        "fixtures_with_incomplete_record_coverage": sum(
            row["official_action_evidence_identity"] is not None
            and not row["action_adoption"]["manager_record_coverage_complete"]
            for row in all_chapters
        ),
        **adoption_totals,
        "bounded_semantic_examples": {
            "fixtures_with_examples": sum(
                row["action_adoption"]["retained_semantic_examples"] > 0
                for row in all_chapters
            ),
            "fixtures_with_complete_coverage": sum(
                row["action_adoption"][
                    "semantic_example_coverage_complete"
                ]
                for row in all_chapters
            ),
            "fixtures_with_truncated_examples": sum(
                row["action_adoption"]["semantic_examples_truncated"]
                for row in all_chapters
            ),
            **semantic_sample_totals,
            "full_record_distribution_authorized": False,
            "claim_boundary": (
                "retained bounded official-action examples only; counts are "
                "not a full-record action or signal distribution when any "
                "fixture is truncated or lacks semantic examples"
            ),
        },
        "retained_record_semantics": {
            "fixtures_with_v2_semantics": len(full_semantic_rows),
            "fixtures_without_v2_semantics": (
                len(all_chapters) - len(full_semantic_rows)
            ),
            "records": sum(row["records"] for row in full_semantic_rows),
            "actual_action_counts": {
                action_name: sum(
                    row["actual_action_counts"][action_name]
                    for row in full_semantic_rows
                )
                for action_name in semantic_actions
            },
            "primary_signal_action_counts": {
                action_name: sum(
                    row["primary_signal_action_counts"][action_name]
                    for row in full_semantic_rows
                )
                for action_name in semantic_actions
            },
            "signal_mode_counts": {
                mode: sum(
                    row["signal_mode_counts"][mode]
                    for row in full_semantic_rows
                )
                for mode in semantic_modes
            },
            "hold_reference_redistribution_records": sum(
                row["hold_reference_redistribution_records"]
                for row in full_semantic_rows
            ),
            "direct_cross_ball_event_links": sum(
                row["direct_cross_ball_event_links"]
                for row in full_semantic_rows
            ),
            "locally_attributable_cross_changes": sum(
                row["locally_attributable_cross_changes"]
                for row in full_semantic_rows
            ),
            "fixtures_with_full_source_distribution": sum(
                row["full_source_distribution_authorized"] is True
                for row in full_semantic_rows
            ),
            "all_chapters_have_v2_semantics": bool(
                all_chapters and len(full_semantic_rows) == len(all_chapters)
            ),
            "full_source_distribution_authorized": bool(
                all_chapters
                and len(full_semantic_rows) == len(all_chapters)
                and all(
                    row["full_source_distribution_authorized"] is True
                    for row in full_semantic_rows
                )
            ),
            "outcome_attribution_authorized": False,
            "fixtures_with_v3_transition_semantics": len(
                transition_semantic_rows
            ),
            "fixtures_without_v3_transition_semantics": (
                len(all_chapters) - len(transition_semantic_rows)
            ),
            "counterfactual_action_transition_counts": {
                baseline: {
                    actual: sum(
                        row["counterfactual_action_transition_counts"]
                        [baseline][actual]
                        for row in transition_semantic_rows
                    )
                    for actual in semantic_actions
                }
                for baseline in semantic_actions
            },
            "locally_attributable_action_transition_counts": {
                baseline: {
                    actual: sum(
                        row[
                            "locally_attributable_action_transition_counts"
                        ][baseline][actual]
                        for row in transition_semantic_rows
                    )
                    for actual in semantic_actions
                }
                for baseline in semantic_actions
            },
            "all_chapters_have_v3_transition_semantics": bool(
                all_chapters
                and len(transition_semantic_rows) == len(all_chapters)
            ),
            "full_source_transition_distribution_authorized": bool(
                all_chapters
                and len(transition_semantic_rows) == len(all_chapters)
                and all(
                    row["full_source_distribution_authorized"] is True
                    for row in transition_semantic_rows
                )
            ),
            "fixtures_with_v4_expectation_semantics": len(
                expectation_semantic_rows
            ),
            "fixtures_without_v4_expectation_semantics": (
                len(all_chapters) - len(expectation_semantic_rows)
            ),
            "records_with_exact_change_probability": sum(
                row["records_with_exact_change_probability"]
                for row in expectation_semantic_rows
            ),
            "attribution_eligible_records_with_exact_change_probability": sum(
                row[
                    "attribution_eligible_records_with_exact_change_probability"
                ]
                for row in expectation_semantic_rows
            ),
            "expected_counterfactual_action_changes": round(sum(
                row["expected_counterfactual_action_changes"]
                for row in expectation_semantic_rows
            ), 9),
            "all_chapters_have_v4_expectation_semantics": bool(
                all_chapters
                and len(expectation_semantic_rows) == len(all_chapters)
            ),
            "full_source_expectation_authorized": bool(
                all_chapters
                and len(expectation_semantic_rows) == len(all_chapters)
                and all(
                    row["full_source_expectation_authorized"] is True
                    for row in expectation_semantic_rows
                )
            ),
        },
        "influence_rate": (
            round(
                adoption_totals["influenced_decisions"]
                / adoption_totals["retained_records"], 6,
            )
            if adoption_totals["retained_records"] else None
        ),
        "realized_change_rate_among_influenced": (
            round(
                adoption_totals["locally_attributable_action_changes"]
                / adoption_totals["influenced_decisions"], 6,
            )
            if adoption_totals["influenced_decisions"] else None
        ),
        "direct_ball_event_link_coverage": (
            round(
                adoption_totals["direct_ball_event_links"] / (
                    adoption_totals["direct_ball_event_links"]
                    + adoption_totals["unresolved_ball_event_links"]
                ), 6,
            )
            if (
                adoption_totals["direct_ball_event_links"]
                + adoption_totals["unresolved_ball_event_links"]
            ) else None
        ),
        "state_counts": [
            {
                "state": state,
                **adoption_state_facts[state],
                "descriptive_world_after": _descriptive_world_propagation([
                    row for row in all_chapters
                    if row["action_adoption"]["state"] == state
                ]),
            }
            for state in _ACTION_ADOPTION_STATE_ORDER
            if state in adoption_state_facts
        ],
        "descriptive_world_after": _descriptive_world_propagation(
            all_chapters
        ),
        "local_transition_descriptive_propagation": (
            _local_transition_descriptive_propagation(all_chapters)
        ),
    }
    gap_facts: dict[str, dict[str, Any]] = {}
    for chapter in all_chapters:
        for gap in chapter["continuity_gaps"]:
            facts = gap_facts.setdefault(gap, {
                "chapters": 0,
                "latest_fixture_id": None,
                "latest_matchday": 0,
                "latest_chapter_identity": None,
            })
            facts["chapters"] += 1
            current_key = (
                facts["latest_matchday"],
                str(facts["latest_fixture_id"] or ""),
            )
            chapter_key = (chapter["matchday"], chapter["fixture_id"])
            if chapter_key > current_key:
                facts["latest_fixture_id"] = chapter["fixture_id"]
                facts["latest_matchday"] = chapter["matchday"]
                facts["latest_chapter_identity"] = chapter["chapter_identity"]
    continuity_gap_counts = [
        {"gap": gap, **facts}
        for gap, facts in sorted(
            gap_facts.items(),
            key=lambda item: (-item[1]["chapters"], item[0]),
        )
    ]
    summary = {
        "completed_world_chapters": len(all_chapters),
        "visible_world_chapters": len(chapters),
        "chapters_truncated": (
            len(all_chapters) > MAX_HISTORY_CHAPTERS
        ),
        "official_runtime_chapters": sum(
            row["official_runtime_chain_complete"] for row in all_chapters
        ),
        "world_model_runtime_chapters": sum(
            row["world_model_runtime_chain_complete"] for row in all_chapters
        ),
        "reviewed_world_model_runtime_chapters": sum(
            row["reviewed_world_model_chain_complete"] for row in all_chapters
        ),
        "local_action_changes": sum(
            row["locally_attributable_action_changes"] for row in all_chapters
        ),
        "chapters_with_continuity_gaps": sum(
            bool(row["continuity_gaps"]) for row in all_chapters
        ),
        "chapters_without_continuity_gaps": sum(
            not row["continuity_gaps"] for row in all_chapters
        ),
        "continuity_gap_counts": continuity_gap_counts,
        "visible_official_runtime_chapters": sum(
            row["official_runtime_chain_complete"] for row in chapters
        ),
        "visible_world_model_runtime_chapters": sum(
            row["world_model_runtime_chain_complete"] for row in chapters
        ),
        "visible_reviewed_world_model_runtime_chapters": sum(
            row["reviewed_world_model_chain_complete"] for row in chapters
        ),
        "visible_local_action_changes": sum(
            row["locally_attributable_action_changes"] for row in chapters
        ),
        "visible_chapters_with_continuity_gaps": sum(
            bool(row["continuity_gaps"]) for row in chapters
        ),
        "world_model_influence_path": influence_path,
        "world_model_action_adoption_ledger": action_adoption_ledger,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "season_id": season_id,
        "season_revision": revision,
        "navigation_state": (
            "season_complete"
            if workflow_state == "season_complete" else "active_manager_world"
        ),
        "current_chapter": current,
        "primary_action": primary,
        "secondary_actions": _secondary_actions(
            workflow_state, allowed,
        ),
        "history_chapters": chapters,
        "world_trajectory": world_trajectory,
        "summary": summary,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": _BOUNDARY,
    }
    payload["navigator_identity"] = _identity(payload)
    return payload


def validate_manager_world_navigator(
    navigator: Mapping[str, Any],
    *,
    season: Mapping[str, Any],
) -> None:
    """Replay the navigator from the current session and decision ledger."""
    expected = build_manager_world_navigator(season)
    if dict(navigator) != expected:
        raise ValueError("manager world navigator replay mismatch")


__all__ = [
    "MAX_HISTORY_CHAPTERS",
    "SCHEMA_VERSION",
    "build_manager_world_navigator",
    "validate_manager_world_navigator",
]
