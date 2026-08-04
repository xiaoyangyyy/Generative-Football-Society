"""Cross-decision evaluation for LLM-selected belief-space compute routes."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.belief_space_meta_planner import (
    BELIEF_SPACE_META_PLANNER_VERSION,
    belief_space_meta_plan_audit_is_valid,
)


def belief_space_meta_planner_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    audits = malformed = applied = route_matches = 0
    expired_without_use = unsafe = budget_violations = 0
    answer_conditioned = broad_coverage_failures = 0
    matches = 0
    clustered_match_rates = []
    for cluster in record_clusters:
        records = sorted(
            list(cluster), key=lambda row: float(row.get("created_t_sec", 0.0)),
        )
        match_applied = match_matches = 0
        for index, record in enumerate(records):
            audit = record.get("llm_belief_space_meta_plan_context") or {}
            if not audit:
                continue
            if not belief_space_meta_plan_audit_is_valid(audit):
                malformed += 1
                continue
            audits += 1
            preference = audit["compute_preference"]
            unsafe += int(
                audit.get("selected_after_action_freeze") is not True
                or audit.get("can_change_current_action") is not False
                or audit.get("can_change_tactical_controls") is not False
                or audit.get("can_schedule_future_action") is not False
                or audit.get("causal_interpretation") is not False
            )
            following = next((
                candidate for candidate in records[index + 1:]
                if str(candidate.get("team_id")) == str(record.get("team_id"))
                and float(candidate.get("created_t_sec", 0.0))
                <= float(preference["expires_t_sec"]) + 1e-9
            ), None)
            if following is None:
                expired_without_use += 1
                continue
            trajectory = (
                following.get("opponent_response_context") or {}
            ).get("trajectory_rollout") or {}
            if not trajectory:
                continue
            applied += 1
            match_applied += 1
            option = preference["selected_option"]
            expected_route = option.get("route") or {}
            actual_route = trajectory.get("belief_space_route") or {}
            budget_ok = int(trajectory.get(
                "branch_evaluation_budget", -1,
            )) == int(option["branch_evaluation_budget"])
            route_ok = actual_route == expected_route
            matched = bool(budget_ok and route_ok)
            route_matches += int(matched)
            match_matches += int(matched)
            budget_violations += int(
                int(trajectory.get("branch_evaluations", 0))
                > int(trajectory.get("branch_evaluation_budget", 0))
                or not budget_ok
            )
            answer_conditioned += int(
                option["mode"] == "answer_conditioned_route"
                and trajectory.get("belief_space_route_applied") is True
            )
            broad_coverage_failures += int(
                int(trajectory.get("broad_coverage_pairs", 0)) < 16
            )
        if match_applied:
            matches += 1
            clustered_match_rates.append(match_matches / match_applied)
    return {
        "version": BELIEF_SPACE_META_PLANNER_VERSION,
        "evaluation_kind": "cross_decision_belief_space_compute_routing",
        "accepted_meta_plan_audits": audits,
        "malformed_meta_plan_audits": malformed,
        "applied_next_decision_preferences": applied,
        "matches": matches,
        "expired_or_unobserved_preferences": expired_without_use,
        "matching_route_and_budget_applications": route_matches,
        "match_clustered_route_application_rate": float(np.mean(
            clustered_match_rates
        )) if clustered_match_rates else 0.0,
        "answer_conditioned_routes_applied": answer_conditioned,
        "compute_budget_violations": budget_violations,
        "broad_coverage_failures": broad_coverage_failures,
        "unsafe_action_or_tactical_authority_claims": unsafe,
        "all_compute_preferences_post_action_non_controlling": bool(
            audits > 0 and unsafe == 0
        ),
    }
