"""Replayable manager-decision lifecycle ledger from authoritative evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.product.manager_future_review import validate_manager_future_review
from src.product.manager_world_thread import (
    build_manager_world_evolution_thread,
    manager_world_evolution_summary,
    validate_manager_world_evolution_thread,
)
from src.product.player_promises import player_promise_progress
from src.product.season import ManagerDecision, SeasonPlan, manager_season_profile
from src.product.season_commitments import commitment_progress_from_evidence
from src.product.world_model_action_execution import (
    validate_world_model_action_execution,
)


LEDGER_SCHEMA_VERSION = 1
_BOUNDARY = (
    "frozen decisions, direct runtime execution and long-term policy accounting are "
    "kept separate; scores are descriptive and do not establish causal effectiveness"
)


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _journal(fixtures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fixture in fixtures:
        decision = fixture.get("manager_decision")
        if not isinstance(decision, Mapping):
            tactic, rotation = "unrecorded", "unrecorded"
        else:
            parsed = ManagerDecision.from_payload(decision)
            tactic, rotation = parsed.tactic, parsed.rotation
        rows.append({
            "fixture_id": fixture["fixture_id"],
            "matchday": fixture["matchday"],
            "tactic": tactic, "rotation": rotation,
        })
    return rows


def _policy_progress(
    contract: Mapping[str, Any] | None, *, fixtures: Sequence[Mapping[str, Any]],
    objective: Mapping[str, Any], final: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(contract, Mapping):
        return None
    progress = commitment_progress_from_evidence(
        contract, journal=_journal(fixtures), objective=objective, final=final,
    )
    return {
        "recorded_matches": progress["recorded_matches"],
        "entries": copy.deepcopy(progress["entries"][1:]),
    }


def _player_progress(
    contract: Mapping[str, Any] | None, *, fixtures: Sequence[Mapping[str, Any]],
    final: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(contract, Mapping):
        return None
    progress = player_promise_progress(contract, fixtures=fixtures, final=final)
    return {
        "completed_matches": progress["completed_matches"],
        "entries": [
            {
                "player_id": row["player_id"], "name": row["name"],
                "promised_role": row["promised_role"], "status": row["status"],
                "starts": row["starts"], "known_lineups": row["known_lineups"],
                "excused_unavailable": row["excused_unavailable"],
                "start_share": row["start_share"],
                "minimum_start_share": row["minimum_start_share"],
            }
            for row in progress["entries"]
        ],
    }


def _execution_view(
    debrief: Mapping[str, Any] | None, *, fixture: Mapping[str, Any],
    decision: ManagerDecision, team: str,
) -> dict[str, Any]:
    if not isinstance(debrief, Mapping):
        return {
            "schema_version": 1, "available": False,
            "reason": (
                "fixture_not_completed" if fixture.get("state") != "completed"
                else "report_outside_live_window_or_unavailable"
            ),
        }
    if debrief.get("available") is not True:
        return copy.deepcopy(dict(debrief))
    recorded = debrief.get("decision")
    result = debrief.get("observed_result")
    if (
        debrief.get("match_id") != fixture.get("match_id")
        or debrief.get("team") != team
        or not isinstance(recorded, Mapping)
        or recorded.get("tactic") != decision.tactic
        or recorded.get("rotation") != decision.rotation
        or debrief.get("evidence_grade") != "direct_runtime_execution"
        or debrief.get("causal_outcome_attribution") is not False
        or not isinstance(result, Mapping)
        or result.get("descriptive_only") is not True
    ):
        return {
            "schema_version": 1, "available": False,
            "reason": "execution_identity_or_boundary_mismatch",
        }
    action_execution = debrief.get("world_model_action_execution")
    if isinstance(action_execution, Mapping):
        try:
            validate_world_model_action_execution(action_execution)
        except ValueError:
            return {
                "schema_version": 1,
                "available": False,
                "reason": "world_model_action_execution_invalid",
            }
        if action_execution.get("available") is True and (
            action_execution.get("match_id") != fixture.get("match_id")
            or action_execution.get("team") != team
        ):
            return {
                "schema_version": 1,
                "available": False,
                "reason": "world_model_action_execution_identity_mismatch",
            }
    return copy.deepcopy(dict(debrief))


def _advisor_execution_trace(
    support: Mapping[str, Any], *, decision_identity: str,
    lifecycle_state: str, execution: Mapping[str, Any],
) -> dict[str, Any]:
    if support.get("available") is not True:
        return {
            "schema_version": 1, "available": False,
            "reason": "world_model_advice_not_available",
        }
    adoption = support.get("adoption")
    if isinstance(adoption, Mapping):
        interaction = {
            "status": "explicit",
            "intent": adoption.get("intent"),
            "adoption_identity": adoption.get("adoption_identity"),
        }
    else:
        interaction = {
            "status": "unlinked", "intent": None,
            "adoption_identity": None,
        }
    binding = execution.get("tactical_binding")
    binding_available = bool(
        execution.get("available") is True
        and isinstance(binding, Mapping)
        and binding.get("available") is True
    )
    if binding_available:
        runtime = {
            "status": "verified",
            "match_id": execution.get("match_id"),
            "applied_tactic": binding.get("applied_tactic"),
            "binding_identity": binding.get("binding_identity"),
            "initial_vector_identity": binding.get("initial_vector_identity"),
            "changed_controls": copy.deepcopy(binding.get("changed_controls")),
            "final_delta_l1": binding.get("final_delta_l1"),
        }
    else:
        runtime = {
            "status": (
                "awaiting_execution"
                if lifecycle_state == "frozen_awaiting_execution"
                else "evidence_unavailable"
            ),
            "reason": (
                "fixture_not_completed"
                if lifecycle_state == "frozen_awaiting_execution"
                else (
                    binding.get("reason")
                    if isinstance(binding, Mapping) else
                    execution.get("reason", "tactical_binding_unavailable")
                )
            ),
        }
    selected = str(support.get("selected_tactic") or "")
    recommended = str(support.get("recommended_tactic") or "")
    intent = interaction["intent"]
    direct_recommendation_execution = bool(
        binding_available
        and intent == "adopt_recommendation"
        and runtime["applied_tactic"] == selected == recommended
    )
    if not binding_available:
        end_to_end_state = runtime["status"]
    elif direct_recommendation_execution:
        end_to_end_state = "recommendation_executed"
    elif intent == "reviewed_then_selected":
        end_to_end_state = "reviewed_alternative_executed"
    else:
        end_to_end_state = "advised_selection_executed_unlinked"
    payload = {
        "schema_version": 1, "available": True,
        "advice_identity": support.get("advice_identity"),
        "recommendation": {
            "tactic": recommended,
            "confidence": support.get("recommendation_confidence"),
            "margin": support.get("recommendation_margin"),
        },
        "interaction": interaction,
        "selection": {
            "decision_identity": decision_identity,
            "tactic": selected,
            "aligned_with_recommendation": support.get(
                "aligned_with_recommendation"
            ),
        },
        "runtime_binding": runtime,
        "end_to_end_state": end_to_end_state,
        "direct_recommendation_execution": direct_recommendation_execution,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "identity-bound advice, explicit interaction, frozen selection and "
            "direct tactical runtime binding only; no score, win-probability or "
            "causal-effect claim"
        ),
    }
    payload["trace_identity"] = _identity(payload)
    return payload


def _future_review_execution_trace(
    reviews: Sequence[Mapping[str, Any]], *, decision_identity: str,
    selected_tactic: str, lifecycle_state: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Link the terminal future review to runtime selection, never outcomes."""
    if not reviews:
        return {
            "schema_version": 1, "available": False,
            "reason": "world_model_future_review_not_available",
        }
    chain = []
    for index, review in enumerate(reviews):
        terminal = index == len(reviews) - 1
        final_identity = str(review.get("final_decision_identity") or "")
        linked = terminal and final_identity == decision_identity
        chain.append({
            "review_identity": review.get("review_identity"),
            "task_id": review.get("task_id"),
            "intent": review.get("intent"),
            "final_decision_identity": final_identity,
            "relation_to_final_selection": (
                "selected_for_fixture" if linked else
                "superseded_by_unreviewed_edit" if terminal else
                "followed_by_later_review"
            ),
        })
    terminal_review = reviews[-1]
    terminal_linked = (
        terminal_review.get("final_decision_identity") == decision_identity
    )
    binding = execution.get("tactical_binding")
    binding_available = bool(
        terminal_linked
        and execution.get("available") is True
        and isinstance(binding, Mapping)
        and binding.get("available") is True
        and binding.get("applied_tactic") == selected_tactic
    )
    if not terminal_linked:
        end_to_end_state = "terminal_review_superseded_before_execution"
    elif lifecycle_state == "frozen_awaiting_execution":
        end_to_end_state = "reviewed_selection_awaiting_execution"
    elif binding_available:
        end_to_end_state = "reviewed_selection_runtime_verified"
    else:
        end_to_end_state = "reviewed_selection_execution_evidence_unavailable"
    if binding_available:
        runtime = {
            "status": "verified",
            "match_id": execution.get("match_id"),
            "applied_tactic": binding.get("applied_tactic"),
            "binding_identity": binding.get("binding_identity"),
            "initial_vector_identity": binding.get("initial_vector_identity"),
        }
    else:
        runtime = {
            "status": (
                "not_applicable" if not terminal_linked else
                "awaiting_execution"
                if lifecycle_state == "frozen_awaiting_execution" else
                "evidence_unavailable"
            ),
            "reason": (
                "terminal_review_was_superseded"
                if not terminal_linked else
                "fixture_not_completed"
                if lifecycle_state == "frozen_awaiting_execution" else
                (
                    binding.get("reason")
                    or (
                        "applied_tactic_mismatch"
                        if binding.get("available") is True
                        and binding.get("applied_tactic") != selected_tactic
                        else "tactical_binding_unavailable"
                    )
                    if isinstance(binding, Mapping) else
                    execution.get("reason", "tactical_binding_unavailable")
                )
            ),
        }
    terminal_scenarios = terminal_review.get("scenario_evidence") or []
    mechanism_examples = [
        example
        for scenario in terminal_scenarios
        if isinstance(scenario, Mapping)
        for example in (scenario.get("mechanism_examples") or [])
        if isinstance(example, Mapping)
    ]
    payload = {
        "schema_version": 1,
        "available": True,
        "review_chain": chain,
        "terminal_review": {
            "review_identity": terminal_review.get("review_identity"),
            "task_id": terminal_review.get("task_id"),
            "intent": terminal_review.get("intent"),
            "final_decision_identity": terminal_review.get(
                "final_decision_identity"
            ),
            "linked_to_final_selection": terminal_linked,
            "retained_mechanism_examples": len(mechanism_examples),
        },
        "final_selection": {
            "decision_identity": decision_identity,
            "tactic": selected_tactic,
        },
        "runtime_binding": runtime,
        "end_to_end_state": end_to_end_state,
        "outcome_comparison_performed": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "future-set review history, final frozen decision identity and "
            "direct tactical runtime binding only; pre-match simulator paths "
            "are not matched to the observed score or treated as forecasts"
        ),
    }
    payload["trace_identity"] = _identity(payload)
    return payload


def world_model_advisor_summary(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive bounded adoption evidence without treating results as outcomes."""
    advised = [
        row for row in entries
        if (row.get("world_model_decision_support") or {}).get("available") is True
    ]
    interactions = [
        row for row in advised
        if isinstance(
            (row.get("world_model_decision_support") or {}).get("adoption"),
            Mapping,
        )
    ]
    adopted = [
        row for row in interactions
        if row["world_model_decision_support"]["adoption"].get("intent")
        == "adopt_recommendation"
    ]
    reviewed = [
        row for row in interactions
        if row["world_model_decision_support"]["adoption"].get("intent")
        == "reviewed_then_selected"
    ]
    aligned = [
        row for row in advised
        if row["world_model_decision_support"].get(
            "aligned_with_recommendation"
        ) is True
    ]
    executed = [
        row for row in advised
        if str(row.get("lifecycle_state") or "").startswith("executed_")
    ]
    direct = [
        row for row in advised
        if row.get("lifecycle_state") == "executed_with_direct_evidence"
    ]
    verified_bindings = [
        row for row in advised
        if (row.get("advisor_execution_trace") or {}).get(
            "runtime_binding", {}
        ).get("status") == "verified"
    ]
    recommendation_executions = [
        row for row in advised
        if (row.get("advisor_execution_trace") or {}).get(
            "direct_recommendation_execution"
        ) is True
    ]
    reviewed_executions = [
        row for row in advised
        if (row.get("advisor_execution_trace") or {}).get(
            "end_to_end_state"
        ) == "reviewed_alternative_executed"
    ]
    low_authority = [
        row for row in advised
        if (
            row["world_model_decision_support"].get("reliability") or {}
        ).get("guidance") == "low_authority"
    ]
    tactics: dict[str, int] = {}
    for row in advised:
        tactic = str(
            row["world_model_decision_support"].get("recommended_tactic") or ""
        )
        tactics[tactic] = tactics.get(tactic, 0) + 1
    confidences = [
        float(row["world_model_decision_support"]["recommendation_confidence"])
        for row in advised
    ]
    margins = [
        float(row["world_model_decision_support"]["recommendation_margin"])
        for row in advised
    ]
    evidence = {
        "schema_version": 1,
        "decisions": len(entries),
        "advised_decisions": len(advised),
        "unadvised_decisions": len(entries) - len(advised),
        "advice_coverage": round(len(advised) / max(1, len(entries)), 6),
        "explicit_interactions": len(interactions),
        "explicit_interaction_coverage": round(
            len(interactions) / max(1, len(advised)), 6,
        ),
        "adopted_recommendation": len(adopted),
        "reviewed_then_selected": len(reviewed),
        "unlinked_advice": len(advised) - len(interactions),
        "aligned_selections": len(aligned),
        "descriptive_alignment_rate": round(
            len(aligned) / max(1, len(advised)), 6,
        ),
        "executed_advised_decisions": len(executed),
        "direct_execution_evidence": len(direct),
        "direct_execution_coverage": round(
            len(direct) / max(1, len(executed)), 6,
        ),
        "verified_tactical_bindings": len(verified_bindings),
        "verified_tactical_binding_coverage": round(
            len(verified_bindings) / max(1, len(executed)), 6,
        ),
        "direct_recommendation_executions": len(recommendation_executions),
        "reviewed_alternative_executions": len(reviewed_executions),
        "low_authority_advice": len(low_authority),
        "mean_recommendation_confidence": round(
            sum(confidences) / max(1, len(confidences)), 9,
        ),
        "mean_recommendation_margin": round(
            sum(margins) / max(1, len(margins)), 9,
        ),
        "recommendation_counts": [
            {"tactic": tactic, "count": count}
            for tactic, count in sorted(tactics.items())
        ],
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "descriptive request, explicit interaction, selection-alignment and "
            "direct-execution coverage only; no score, win-probability, causal "
            "effectiveness or real-football claim"
        ),
    }
    evidence["evidence_identity"] = _identity(evidence)
    return evidence


def world_model_future_review_summary(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize explicit future-set interactions without outcome attribution."""
    reviews = [
        review
        for entry in entries
        for review in (entry.get("world_model_future_reviews") or [])
        if isinstance(review, Mapping)
    ]
    kept = sum(review.get("intent") == "keep_after_review" for review in reviews)
    revised = sum(
        review.get("intent") == "revise_after_review" for review in reviews
    )
    scenarios = [
        scenario
        for review in reviews
        for scenario in (review.get("scenario_evidence") or [])
        if isinstance(scenario, Mapping)
    ]
    mechanism_examples = [
        example
        for scenario in scenarios
        for example in (scenario.get("mechanism_examples") or [])
        if isinstance(example, Mapping)
    ]
    evidence = {
        "schema_version": 1,
        "reviewed_future_sets": len(reviews),
        "fixtures_with_review": sum(
            bool(entry.get("world_model_future_reviews")) for entry in entries
        ),
        "kept_after_review": kept,
        "revised_after_review": revised,
        "decision_change_rate": round(revised / max(1, len(reviews)), 6),
        "sets_with_action_divergence": sum(
            int(
                (review.get("evidence_summary") or {}).get(
                    "action_divergence_scenarios", 0,
                )
            ) > 0
            for review in reviews
        ),
        "sets_with_timing_sensitivity": sum(
            (review.get("evidence_summary") or {}).get(
                "timing_sensitivity_observed"
            ) is True
            for review in reviews
        ),
        "retained_mechanism_examples": len(mechanism_examples),
        "locally_attributable_mechanism_examples": sum(
            example.get("local_policy_attribution_eligible") is True
            for example in mechanism_examples
        ),
        "examples_with_descriptive_windows": sum(
            bool(example.get("downstream_windows"))
            for example in mechanism_examples
        ),
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "explicit simulator-evidence review and subsequent selection counts "
            "only; no recommendation quality, score or causal-effect claim"
        ),
    }
    evidence["evidence_identity"] = _identity(evidence)
    return evidence


def world_model_future_review_execution_summary(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize review-to-runtime closure without outcome evaluation."""
    traces = [
        row.get("future_review_execution_trace")
        for row in entries
        if isinstance(row.get("future_review_execution_trace"), Mapping)
        and row["future_review_execution_trace"].get("available") is True
    ]
    states = (
        "terminal_review_superseded_before_execution",
        "reviewed_selection_awaiting_execution",
        "reviewed_selection_runtime_verified",
        "reviewed_selection_execution_evidence_unavailable",
    )
    counts = {
        state: sum(trace.get("end_to_end_state") == state for trace in traces)
        for state in states
    }
    executed_linked = (
        counts["reviewed_selection_runtime_verified"]
        + counts["reviewed_selection_execution_evidence_unavailable"]
    )
    evidence = {
        "schema_version": 1,
        "fixtures_with_review_execution_trace": len(traces),
        **counts,
        "direct_runtime_binding_coverage": round(
            counts["reviewed_selection_runtime_verified"]
            / max(1, executed_linked),
            6,
        ),
        "outcome_comparison_performed": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "review-to-final-selection and tactical runtime binding coverage "
            "only; no forecast accuracy, score attribution or causal effect"
        ),
    }
    evidence["evidence_identity"] = _identity(evidence)
    return evidence


def world_model_official_action_execution_summary(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize retained official action execution without outcome claims."""
    evidence_rows = []
    for row in entries:
        execution = row.get("execution")
        evidence = (
            execution.get("world_model_action_execution")
            if isinstance(execution, Mapping) else None
        )
        if isinstance(evidence, Mapping) and evidence.get("available") is True:
            evidence_rows.append(evidence)
    counts = [
        evidence["retained_record_evidence"]
        for evidence in evidence_rows
    ]
    direct = sum(int(row["direct_ball_event_links"]) for row in counts)
    unresolved = sum(
        int(row["unresolved_or_missing_ball_event_links"])
        for row in counts
    )
    payload = {
        "schema_version": 1,
        "fixtures_with_official_action_evidence": len(evidence_rows),
        "fixtures_with_complete_manager_record_coverage": sum(
            row.get("manager_record_coverage_complete") is True
            for row in counts
        ),
        "fixtures_with_local_action_changes": sum(
            int(row["locally_attributable_action_changes"]) > 0
            for row in counts
        ),
        "manager_action_records_retained": sum(
            int(row["records"]) for row in counts
        ),
        "influenced_action_decisions": sum(
            int(row["influenced_decisions"]) for row in counts
        ),
        "locally_attributable_action_changes": sum(
            int(row["locally_attributable_action_changes"])
            for row in counts
        ),
        "direct_ball_event_links": direct,
        "changed_direct_ball_event_links": sum(
            int(row["changed_direct_ball_event_links"])
            for row in counts
        ),
        "unresolved_or_missing_ball_event_links": unresolved,
        "direct_ball_event_link_coverage": round(
            direct / max(1, direct + unresolved), 6,
        ),
        "outcome_comparison_performed": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "retained official-match manager-team action decisions and exact "
            "ball-event identity links only; no score, outcome or real-football "
            "causal effect"
        ),
    }
    payload["evidence_identity"] = _identity(payload)
    return payload


def validate_manager_decision_ledger(ledger: Mapping[str, Any]) -> None:
    """Replay identities and all derived summaries in an exported ledger."""
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ValueError("manager decision ledger schema is invalid")
    if ledger.get("available") is not True:
        if ledger.get("entries") != [] or not ledger.get("reason"):
            raise ValueError("unavailable manager decision ledger is invalid")
        return
    entries = ledger.get("entries")
    summary = ledger.get("summary")
    if not isinstance(entries, list) or not isinstance(summary, Mapping):
        raise ValueError("manager decision ledger structure is invalid")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("manager decision ledger entry is invalid")
        frozen = copy.deepcopy(dict(entry))
        observed = frozen.pop("entry_identity", None)
        if not isinstance(observed, str) or observed != _identity(frozen):
            raise ValueError("manager decision ledger entry identity mismatch")
        support = entry.get("world_model_decision_support")
        execution = entry.get("execution")
        lifecycle_state = str(entry.get("lifecycle_state") or "")
        execution_available = (
            isinstance(execution, Mapping)
            and execution.get("available") is True
        )
        if (
            lifecycle_state == "executed_with_direct_evidence"
            and not execution_available
        ):
            raise ValueError("manager decision lifecycle/execution mismatch")
        if (
            lifecycle_state in {
                "frozen_awaiting_execution", "executed_evidence_unavailable",
            }
            and execution_available
        ):
            raise ValueError("manager decision lifecycle/execution mismatch")
        action_execution = (
            execution.get("world_model_action_execution")
            if isinstance(execution, Mapping) else None
        )
        if action_execution is not None:
            validate_world_model_action_execution(action_execution)
        expected_trace = _advisor_execution_trace(
            support if isinstance(support, Mapping) else {},
            decision_identity=str(entry.get("decision_identity") or ""),
            lifecycle_state=lifecycle_state,
            execution=execution if isinstance(execution, Mapping) else {},
        )
        if entry.get("advisor_execution_trace") != expected_trace:
            raise ValueError("manager advisor execution trace replay mismatch")
        reviews = entry.get("world_model_future_reviews")
        if not isinstance(reviews, list):
            raise ValueError("manager future review ledger evidence is invalid")
        for review in reviews:
            validate_manager_future_review(
                review,
                season_id=str(ledger.get("season_id") or ""),
                fixture_id=str(entry.get("fixture_id") or ""),
                matchday=int(entry.get("matchday") or 0),
                manager_team=str(ledger.get("team") or ""),
            )
        decision = entry.get("decision")
        selected_tactic = (
            str(decision.get("tactic") or "")
            if isinstance(decision, Mapping) else ""
        )
        expected_future_trace = _future_review_execution_trace(
            reviews,
            decision_identity=str(entry.get("decision_identity") or ""),
            selected_tactic=selected_tactic,
            lifecycle_state=lifecycle_state,
            execution=execution if isinstance(execution, Mapping) else {},
        )
        if entry.get("future_review_execution_trace") != expected_future_trace:
            raise ValueError(
                "manager future review execution trace replay mismatch"
            )
        thread = entry.get("world_evolution_thread")
        if not isinstance(thread, Mapping):
            raise ValueError("manager world evolution thread is unavailable")
        validate_manager_world_evolution_thread(thread, entry=entry)
    lifecycle_names = (
        "frozen_awaiting_execution", "executed_with_direct_evidence",
        "executed_evidence_unavailable",
    )
    lifecycle_counts = {
        name: sum(row.get("lifecycle_state") == name for row in entries)
        for name in lifecycle_names
    }
    expected_summary = {
        "decisions": len(entries), **lifecycle_counts,
        "direct_execution_coverage": round(
            lifecycle_counts["executed_with_direct_evidence"]
            / max(
                1,
                sum(
                    value for key, value in lifecycle_counts.items()
                    if key.startswith("executed_")
                ),
            ),
            6,
        ),
        "world_model_advisor": world_model_advisor_summary(entries),
        "world_model_future_reviews": world_model_future_review_summary(entries),
        "world_model_future_review_execution": (
            world_model_future_review_execution_summary(entries)
        ),
        "world_model_official_action_execution": (
            world_model_official_action_execution_summary(entries)
        ),
        "manager_world_evolution": manager_world_evolution_summary(entries),
    }
    if dict(summary) != expected_summary:
        raise ValueError("manager decision ledger summary replay mismatch")
    frozen_ledger = copy.deepcopy(dict(ledger))
    observed_ledger_identity = frozen_ledger.pop("ledger_identity", None)
    if (
        not isinstance(observed_ledger_identity, str)
        or observed_ledger_identity != _identity(frozen_ledger)
    ):
        raise ValueError("manager decision ledger identity mismatch")


def build_manager_decision_ledger(
    state: Mapping[str, Any], *,
    execution_by_fixture: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive one immutable-looking view without persisting a second journal."""
    profile = manager_season_profile(state)
    if profile.get("available") is not True:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION, "available": False,
            "reason": "season_has_no_manager_team", "entries": [],
            "claim_boundary": _BOUNDARY,
        }
    plan = SeasonPlan.from_payload(state["plan"])
    team = str(plan.manager_team)
    managed = [
        row for row in state["fixtures"]
        if team in {row.get("home"), row.get("away")}
    ]
    execution = execution_by_fixture or {}
    if not isinstance(execution, Mapping):
        raise ValueError("decision ledger execution evidence is invalid")
    completed_prefix: list[Mapping[str, Any]] = []
    entries = []
    commitment_contract = state.get("season_commitments")
    player_contract = state.get("player_role_promises")
    objective = profile["objective"]

    for fixture in managed:
        raw_decision = fixture.get("manager_decision")
        completed = fixture.get("state") == "completed"
        if not isinstance(raw_decision, Mapping):
            if completed:
                completed_prefix.append(fixture)
            continue
        decision = ManagerDecision.from_payload(raw_decision)
        before_policy = _policy_progress(
            commitment_contract, fixtures=completed_prefix, objective=objective,
        )
        before_players = _player_progress(
            player_contract, fixtures=completed_prefix,
        )
        projected_fixture = fixture
        evidence_state = "completed_evidence"
        if completed:
            after_fixtures = [*completed_prefix, fixture]
        else:
            projected_fixture = copy.deepcopy(fixture)
            projected_fixture["state"] = "completed"
            after_fixtures = [*completed_prefix, projected_fixture]
            evidence_state = "hypothetical_if_counted"
        final_accounting = bool(
            completed
            and len(after_fixtures) == len(managed)
            and all(row.get("state") == "completed" for row in managed)
        )
        after_policy = _policy_progress(
            commitment_contract, fixtures=after_fixtures, objective=objective,
            final=final_accounting,
        )
        after_players = _player_progress(
            player_contract, fixtures=after_fixtures, final=final_accounting,
        )
        debrief = execution.get(str(fixture["fixture_id"]))
        if debrief is not None and not isinstance(debrief, Mapping):
            raise ValueError("decision ledger debrief evidence is invalid")
        execution_view = _execution_view(
            debrief, fixture=fixture, decision=decision, team=team,
        )
        direct_available = execution_view.get("available") is True
        lifecycle_state = (
            "frozen_awaiting_execution" if not completed
            else "executed_with_direct_evidence" if direct_available
            else "executed_evidence_unavailable"
        )
        score = fixture.get("score") if completed else None
        home_side = fixture["home"] == team
        result = None
        if isinstance(score, Mapping):
            goals_for = int(score["home"] if home_side else score["away"])
            goals_against = int(score["away"] if home_side else score["home"])
            result = {
                "score": copy.deepcopy(dict(score)),
                "outcome": (
                    "win" if goals_for > goals_against
                    else "draw" if goals_for == goals_against else "loss"
                ),
                "points_earned": 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0,
                "descriptive_only": True,
            }
        world_transition = fixture.get("world_state_transition")
        persistent_state = {
            "available": False,
            "reason": "per_fixture_persisted_state_snapshot_not_retained",
        }
        if isinstance(world_transition, Mapping):
            phases = world_transition["phases"]
            persistent_state = {
                "available": True,
                "transition_identity": world_transition["transition_identity"],
                "recovery_complete": world_transition["recovery_complete"],
                "before_match": copy.deepcopy(phases["before_match"][team]["metrics"]),
                "after_match": copy.deepcopy(phases["after_match"][team]["metrics"]),
                "after_recovery": (
                    copy.deepcopy(phases["after_recovery"][team]["metrics"])
                    if isinstance(phases.get("after_recovery"), Mapping) else None
                ),
                "match_delta": copy.deepcopy(world_transition["match_delta"][team]),
                "recovery_delta": (
                    copy.deepcopy(world_transition["recovery_delta"][team])
                    if isinstance(world_transition.get("recovery_delta"), Mapping)
                    else None
                ),
                "claim_boundary": world_transition["claim_boundary"],
            }
        advice = fixture.get("manager_decision_advice")
        adoption = fixture.get("manager_advice_adoption")
        decision_support = {
            "available": False,
            "reason": "world_model_advice_not_requested",
        }
        if isinstance(advice, Mapping):
            decision_support = {
                "available": True,
                "advice_identity": advice["advice_identity"],
                "mode": advice["mode"],
                "recommended_tactic": advice["recommended_tactic"],
                "recommendation_confidence": advice[
                    "recommendation_confidence"
                ],
                "recommendation_margin": advice["recommendation_margin"],
                "reliability": copy.deepcopy(advice["reliability"]),
                "selected_tactic": decision.tactic,
                "aligned_with_recommendation": (
                    decision.tactic == advice["recommended_tactic"]
                ),
                "adoption": copy.deepcopy(adoption),
                "claim_boundary": advice["claim_boundary"],
            }
        future_reviews = copy.deepcopy(
            fixture.get("manager_future_reviews") or []
        )
        decision_identity = _identity(decision.as_dict())
        payload = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "fixture_id": fixture["fixture_id"], "matchday": fixture["matchday"],
            "opponent": fixture["away"] if home_side else fixture["home"],
            "venue": "home" if home_side else "away",
            "lifecycle_state": lifecycle_state,
            "decision": copy.deepcopy(decision.as_dict()),
            "decision_identity": decision_identity,
            "opponent_preparation": copy.deepcopy(fixture.get("opponent_preparation")),
            "world_model_decision_support": decision_support,
            "world_model_future_reviews": future_reviews,
            "advisor_execution_trace": _advisor_execution_trace(
                decision_support,
                decision_identity=decision_identity,
                lifecycle_state=lifecycle_state,
                execution=execution_view,
            ),
            "future_review_execution_trace": _future_review_execution_trace(
                future_reviews,
                decision_identity=decision_identity,
                selected_tactic=decision.tactic,
                lifecycle_state=lifecycle_state,
                execution=execution_view,
            ),
            "execution": execution_view,
            "observed_result": result,
            "long_term_accounting": {
                "evidence_state": evidence_state,
                "season_commitments": {
                    "before": before_policy, "after": after_policy,
                },
                "player_role_promises": {
                    "before": before_players, "after": after_players,
                },
                "persistent_team_state_delta": persistent_state,
            },
            "report": fixture.get("report"), "dashboard": fixture.get("dashboard"),
            "claim_boundary": _BOUNDARY,
        }
        payload["world_evolution_thread"] = (
            build_manager_world_evolution_thread(payload)
        )
        payload["entry_identity"] = _identity(payload)
        entries.append(payload)
        if completed:
            completed_prefix.append(fixture)

    lifecycle_counts = {
        name: sum(row["lifecycle_state"] == name for row in entries)
        for name in (
            "frozen_awaiting_execution", "executed_with_direct_evidence",
            "executed_evidence_unavailable",
        )
    }
    ledger = {
        "schema_version": LEDGER_SCHEMA_VERSION, "available": True,
        "season_id": state["season_id"], "team": team,
        "entries": list(reversed(entries)), "summary": {
            "decisions": len(entries), **lifecycle_counts,
            "direct_execution_coverage": round(
                lifecycle_counts["executed_with_direct_evidence"]
                / max(1, sum(value for key, value in lifecycle_counts.items() if key.startswith("executed_"))),
                6,
            ),
            "world_model_advisor": world_model_advisor_summary(entries),
            "world_model_future_reviews": world_model_future_review_summary(
                entries
            ),
            "world_model_future_review_execution": (
                world_model_future_review_execution_summary(entries)
            ),
            "world_model_official_action_execution": (
                world_model_official_action_execution_summary(entries)
            ),
            "manager_world_evolution": manager_world_evolution_summary(entries),
        },
        "claim_boundary": _BOUNDARY,
    }
    ledger["ledger_identity"] = _identity(ledger)
    validate_manager_decision_ledger(ledger)
    return ledger


__all__ = [
    "LEDGER_SCHEMA_VERSION", "build_manager_decision_ledger",
    "validate_manager_decision_ledger", "world_model_advisor_summary",
    "world_model_future_review_summary",
    "world_model_future_review_execution_summary",
    "world_model_official_action_execution_summary",
    "manager_world_evolution_summary",
]
