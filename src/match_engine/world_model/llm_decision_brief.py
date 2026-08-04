"""Compact, provenance-linked world-model evidence for the coach LLM."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np


LLM_DECISION_BRIEF_VERSION = 10


TASK_REASONING_DOMAINS = {
    "opponent_information_adaptation": "opponent_information",
    "opponent_information_policy": "opponent_information",
    "opponent_information_query": "opponent_information",
    "belief_space_meta_planning": "compute_planning",
    "active_probe_design": "experimental_design",
    "opponent_response_hypothesis": "opponent_game",
    "opponent_hypothesis": "opponent_belief",
    "opponent_change_claim": "opponent_belief",
    "world_model_risk_preference": "temporal_risk",
    "world_model_risk_constraint": "trajectory_risk",
    "world_model_distributional_claim": "distributional_risk",
    "world_model_contrastive_claim": "model_explanation",
    "world_model_event_option": "contingent_planning",
    "world_model_event_hypothesis": "semantic_forecast",
    "world_model_critique": "model_calibration",
}


def deliberation_portfolio_objective(
    task_names: list[str] | tuple[str, ...],
    task_rows: dict[str, dict[str, Any]],
) -> float:
    rows = [task_rows[task] for task in task_names]
    domains = {str(row["reasoning_domain"]) for row in rows}
    return float(
        sum(float(row["portfolio_score"]) for row in rows)
        + 0.04 * max(0, len(domains) - 1)
    )


def _best_deliberation_portfolio(
    tasks: list[dict[str, Any]], size: int,
) -> dict[str, Any] | None:
    task_rows = {str(row["task"]): row for row in tasks}
    eligible = sorted(
        task for task, row in task_rows.items() if row.get("eligible")
    )
    candidates = []
    for names in itertools.combinations(eligible, size):
        compute = sum(
            int(task_rows[task]["expected_compute_credits"])
            for task in names
        )
        if compute > 6:
            continue
        candidates.append({
            "tasks": list(names),
            "objective": deliberation_portfolio_objective(names, task_rows),
            "raw_priority_sum": sum(
                float(task_rows[task]["priority"]) for task in names
            ),
            "expected_compute_credits": compute,
            "reasoning_domains": sorted({
                str(task_rows[task]["reasoning_domain"]) for task in names
            }),
        })
    if not candidates:
        return None
    maximum_raw_priority = max(
        row["raw_priority_sum"] for row in candidates
    )
    evidence_safe = [
        row for row in candidates
        if row["raw_priority_sum"]
        >= 0.80 * maximum_raw_priority - 1e-12
    ]
    return sorted(evidence_safe, key=lambda row: (
        -row["objective"], -row["raw_priority_sum"], row["tasks"],
    ))[0]


def _json_hash(value: Any, prefix: str) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (KeyError, TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _distribution_brief(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return _pick(raw, (
        "version", "available", "reason", "ensemble_members",
        "predictive_distribution_available",
        "predictive_distribution_reason", "distribution_scope",
        "residual_calibration_samples", "probability_source",
        "causal_interpretation",
    ))


def _uncertainty_components_brief(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return _pick(raw, (
        "outcome_epistemic", "transition_output_epistemic",
        "transition_epistemic", "validation_quality_epistemic",
        "progress_residual_aleatoric", "transition_ensemble_trained",
        "transition_ensemble_members",
    ))


def _trajectory_modes_brief(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return _pick(raw, (
        "version", "available", "reason", "ensemble_members", "mode_count",
        "multimodal", "probability_source",
        "downside_event_probabilities", "path_downside_event_probabilities",
        "temporal_path_available", "trajectory_steps", "causal_interpretation",
    ))


def _state_scales_brief(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    fusion = {}
    for event, evidence in (raw.get("semantic_event_fusion") or {}).items():
        gate = evidence.get("gate") or {}
        fusion[str(event)] = {
            **_pick(evidence, (
                "projection_probability", "fused_probability",
            )),
            "gate": _pick(gate, (
                "active", "authority", "reason", "validation_scope",
            )),
        }
    return {
        **_pick(raw, (
            "version", "source", "primary_scale", "horizon_s",
            "ensemble_members", "short", "tactical", "strategic",
            "semantic_event_probabilities", "claims",
        )),
        "semantic_event_fusion": fusion,
        "trajectory_modes": _trajectory_modes_brief(
            raw.get("trajectory_modes")
        ),
    }


def _prediction_brief(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    output = _pick(raw, (
        "horizon_s", "rollout_steps", "segment_horizon_s", "prediction_source",
        "policy_utility", "raw_policy_utility", "effective_confidence",
        "retention_probability", "progress", "xg_net_delta", "goal_diff_delta",
        "uncertainty", "epistemic_uncertainty", "aleatoric_uncertainty",
        "uncertainty_source",
        "semantic_event_probabilities", "residual_memory",
    ))
    output["uncertainty_components"] = _uncertainty_components_brief(
        raw.get("uncertainty_components")
    )
    output["distributional_policy_utility"] = _distribution_brief(
        raw.get("distributional_policy_utility")
    )
    output["state_scales"] = _state_scales_brief(raw.get("state_scales"))
    return output


def _temporal_brief(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return _pick(raw, (
        "version", "available", "reason", "distribution_scope",
        "ensemble_members", "residual_scenarios", "scenario_count",
        "horizon_keys", "horizon_seconds", "mean_path_minimum_utility",
        "lower_tail_path_minimum_cvar_25", "mean_maximum_drawdown",
        "worst_maximum_drawdown", "ever_downside_scenario_rate",
        "positive_to_negative_reversal_scenario_rate",
        "downside_recovery_scenario_rate", "terminal_upside_scenario_rate",
        "temporal_coupling_source", "temporal_dependence_learned",
        "temporal_dependence_validation", "marginals_split_calibrated",
        "member_identity_preserved_across_horizons",
        "residual_quantile_rank_preserved_across_horizons",
        "scenario_rates_are_temporal_probabilities", "temporal_joint_calibrated",
        "causal_interpretation",
    ))


def _candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    output = _pick(candidate, (
        "action", "risk_adjusted_value", "single_ply_risk_adjusted_value",
        "two_ply_risk_adjusted_value", "state_conditioned_risk_adjusted_value",
        "expected_value", "physics_prior", "effective_confidence", "confidence",
        "uncertainty", "epistemic_uncertainty", "aleatoric_uncertainty",
        "learnable_uncertainty", "uncertainty_source",
        "opponent_belief_entropy", "opponent_belief_expected_value",
        "opponent_belief_tail_value", "opponent_belief_value_std",
        "opponent_hypothesis_discrimination", "opponent_hypothesis_values",
        "opponent_response_information",
        "best_continuation_action", "best_continuation_value",
        "multi_horizon_forecast_summary", "multi_horizon_forecast_adjustment",
    ))
    output["uncertainty_components"] = _uncertainty_components_brief(
        candidate.get("uncertainty_components")
    )
    response = candidate.get("opponent_response_prediction") or {}
    output["opponent_response_prediction"] = _pick(response, (
        "version", "action", "source", "learned_active", "trust",
        "response_posterior", "normalized_entropy", "profile",
        "causal_interpretation",
    ))
    output["multi_horizon_predictions"] = {
        str(horizon): _prediction_brief(prediction)
        for horizon, prediction in (
            candidate.get("multi_horizon_predictions") or {}
        ).items()
    }
    output["temporal_utility_paths"] = _temporal_brief(
        candidate.get("temporal_utility_paths")
    )
    return output


def _action_margin(candidates: list[dict[str, Any]]) -> float:
    values = sorted((
        _finite(candidate.get("risk_adjusted_value"), -10.0)
        for candidate in candidates
    ), reverse=True)
    return max(0.0, values[0] - values[1]) if len(values) >= 2 else 0.0


def _max_temporal(candidates: list[dict[str, Any]], key: str) -> float:
    return max((
        _finite((candidate.get("temporal_utility_paths") or {}).get(key))
        for candidate in candidates
    ), default=0.0)


def _max_downside(candidates: list[dict[str, Any]]) -> float:
    values = []
    for candidate in candidates:
        for prediction in (
            candidate.get("multi_horizon_predictions") or {}
        ).values():
            modes = ((prediction.get("state_scales") or {}).get(
                "trajectory_modes"
            ) or {})
            values.extend(map(_finite, (
                modes.get("path_downside_event_probabilities") or {}
            ).values()))
            values.extend(map(_finite, (
                modes.get("downside_event_probabilities") or {}
            ).values()))
    return max(values, default=0.0)


def _deliberation_agenda(packet: dict[str, Any]) -> dict[str, Any]:
    candidates = list(packet.get("candidates") or [])
    margin = _action_margin(candidates)
    ambiguity = float(np.clip(1.0 - margin / 0.20, 0.0, 1.0))
    entropy = _finite((packet.get("opponent_belief") or {}).get(
        "normalized_entropy"
    ))
    feedback = packet.get("opponent_information_feedback") or {}
    question_policy = packet.get(
        "opponent_information_cognitive_policy"
    ) or {}
    policy_available = bool(question_policy.get("available"))
    meta_plan = packet.get("belief_space_meta_plan") or {}
    active_probe = packet.get("active_probe_design") or {}
    active_probe_priority = max((
        _finite(option.get("experiment_priority"))
        for option in active_probe.get("options") or []
        if isinstance(option, dict)
    ), default=0.0)
    reversal = _max_temporal(
        candidates, "positive_to_negative_reversal_scenario_rate",
    )
    drawdown = _max_temporal(candidates, "mean_maximum_drawdown")
    downside = _max_downside(candidates)
    frontiers = (packet.get("distributional_action_frontiers") or {}).get(
        "horizons"
    ) or {}
    pareto_tradeoff = max((
        len(frontier.get("pareto_actions") or [])
        for frontier in frontiers.values() if isinstance(frontier, dict)
    ), default=0) > 1
    trajectory_active = bool(
        ((packet.get("second_order_game") or {}).get("trajectory_rollout") or {})
        .get("active")
    )
    memory_active = int((packet.get("contextual_residual_memory") or {}).get(
        "active_groups", 0
    )) > 0
    change_status = str(
        ((packet.get("opponent_belief") or {}).get("change_point") or {}).get(
            "status", "stable"
        )
    )
    task_specs = [
        ("opponent_information_adaptation", bool(feedback.get("available")),
         1.00, "resolved_query_feedback_available"),
        ("opponent_information_policy", policy_available,
         0.45 + 0.35 * entropy + 0.15 * ambiguity,
         "finite_horizon_ask_or_stop_policy"),
        ("opponent_information_query", not policy_available and entropy >= 0.25,
         0.35 + 0.40 * entropy + 0.20 * ambiguity,
         "legacy_opponent_belief_uncertainty"),
        ("belief_space_meta_planning", bool(meta_plan.get("available")),
         0.50 + 0.25 * ambiguity + (
             0.20 if meta_plan.get("answer_conditioned_route_available")
             else 0.0
         ), "answer_conditioned_world_model_compute_routing"),
        ("active_probe_design", bool(active_probe.get("available")),
         0.55 + 0.40 * active_probe_priority,
         "pre_registered_discovery_memory_frontier"),
        ("world_model_risk_preference", reversal >= 0.10 or drawdown >= 0.08,
         0.30 + 0.35 * reversal + 0.35 * min(1.0, drawdown),
         "temporal_reversal_or_drawdown"),
        ("world_model_risk_constraint", downside >= 0.15,
         0.30 + 0.60 * downside, "trajectory_downside_risk"),
        ("world_model_distributional_claim", pareto_tradeoff,
         0.45 + 0.35 * ambiguity, "distributional_pareto_tradeoff"),
        ("world_model_contrastive_claim", bool(candidates),
         0.25 + 0.55 * ambiguity, "small_action_value_margin"),
        ("world_model_event_option", trajectory_active,
         0.45 + 0.30 * ambiguity, "validated_predicted_state_search"),
        ("world_model_event_hypothesis", bool(candidates),
         0.25 + 0.35 * ambiguity, "falsifiable_semantic_event"),
        ("world_model_critique", memory_active or ambiguity >= 0.50,
         0.25 + 0.35 * ambiguity + (0.15 if memory_active else 0.0),
         "forecast_ambiguity_or_residual_memory"),
        ("opponent_response_hypothesis", entropy >= 0.35,
         0.25 + 0.45 * entropy, "uncertain_action_conditioned_response"),
        ("opponent_hypothesis", entropy >= 0.35,
         0.20 + 0.40 * entropy, "uncertain_current_opponent_tactic"),
        ("opponent_change_claim", change_status in {"watch", "confirmed"},
         0.80 if change_status == "confirmed" else 0.60,
         "numeric_opponent_change_detector"),
    ]
    compute_value_memory = packet.get(
        "deliberation_compute_value_memory"
    ) or {}
    from src.match_engine.world_model.llm_deliberation_compute_value import (
        deliberation_compute_value_memory_is_valid,
    )

    value_memory_valid = deliberation_compute_value_memory_is_valid(
        compute_value_memory
    )
    value_rows = compute_value_memory.get("task_values") or {}
    task_selection_memory = packet.get("task_selection_value_memory") or {}
    from src.match_engine.world_model.llm_deliberation_encouragement import (
        task_selection_value_memory_is_valid,
    )

    selection_memory_valid = task_selection_value_memory_is_valid(
        task_selection_memory
    )
    selection_status = str(task_selection_memory.get("status", ""))
    selection_effect = float(task_selection_memory.get(
        "event_option_minus_contrastive_itt", 0.0,
    )) if selection_memory_valid else 0.0
    tasks = [
        {
            "task": task,
            "eligible": bool(eligible),
            "priority": float(np.clip(priority, 0.0, 1.0)),
            "reason": reason,
            "compute_cost_class": (
                "trajectory_rollout"
                if task in {
                    "world_model_event_option",
                    "world_model_contrastive_claim",
                }
                else "existing_evidence_audit"
            ),
            "reasoning_domain": TASK_REASONING_DOMAINS[task],
            "expected_compute_credits": (
                2 if task in {
                    "world_model_event_option",
                    "world_model_contrastive_claim",
                } else 1
            ),
            "learned_compute_value_adjustment": float(np.clip(
                (
                    0.10 * float((value_rows.get(task) or {}).get(
                        "posterior_marginal_value", 0.0,
                    ))
                    if value_memory_valid and str((
                        value_rows.get(task) or {}
                    ).get("status", "")).startswith("validated_")
                    else 0.0
                ),
                -0.08, 0.08,
            )),
            "learned_task_selection_adjustment": float(np.clip(
                (
                    0.08 * selection_effect
                    if task == "world_model_event_option"
                    and selection_status.startswith("validated_")
                    else -0.08 * selection_effect
                    if task == "world_model_contrastive_claim"
                    and selection_status.startswith("validated_")
                    else 0.0
                ),
                -0.06, 0.06,
            )),
        }
        for task, eligible, priority, reason in task_specs
    ]
    for task in tasks:
        task["portfolio_score"] = float(np.clip(
            task["priority"]
            + task["learned_compute_value_adjustment"]
            + task["learned_task_selection_adjustment"]
            - (0.03 if task["compute_cost_class"] == "trajectory_rollout"
               else 0.0),
            0.0, 1.0,
        ))
    portfolios = {
        str(size): portfolio
        for size in range(1, 4)
        if (portfolio := _best_deliberation_portfolio(tasks, size)) is not None
    }
    largest = max(map(int, portfolios), default=0)
    recommended = list((portfolios.get(str(largest)) or {}).get("tasks") or [])
    return {
        "version": 2,
        "maximum_recommended_focus_tasks": 3,
        "total_post_llm_compute_credits": 6,
        "maximum_compute_credits_per_task": 3,
        "recommended_focus": recommended,
        "recommended_portfolios_by_size": portfolios,
        "portfolio_policy": {
            "objective": (
                "priority_plus_validated_compute_value_minus_real_compute_cost_"
                "plus_reasoning_domain_diversity"
            ),
            "reasoning_domain_diversity_bonus": 0.04,
            "trajectory_compute_cost_penalty": 0.03,
            "maximum_expected_compute_credits": 6,
            "learned_value_adjustment_bounds": [-0.08, 0.08],
            "learned_task_selection_adjustment_bounds": [-0.06, 0.06],
            "minimum_raw_priority_fraction": 0.80,
        },
        "tasks": tasks,
        "signals": {
            "top_action_margin": margin,
            "action_ambiguity": ambiguity,
            "opponent_belief_entropy": entropy,
            "maximum_temporal_reversal_rate": reversal,
            "maximum_mean_drawdown": drawdown,
            "maximum_trajectory_downside_rate": downside,
            "distributional_pareto_tradeoff": pareto_tradeoff,
            "trajectory_search_active": trajectory_active,
            "resolved_query_feedback_available": bool(feedback.get("available")),
            "opponent_change_point_status": change_status,
        },
        "policy": (
            "Focus on at most three recommended tasks; omit unsupported optional "
            "contracts. The world model allocates fixed compute credits after "
            "selection; the full packet remains authoritative for engine audits."
        ),
    }


def _belief_space_meta_plan_brief(plan: Any) -> dict[str, Any]:
    """Keep exact selectable options without duplicating audit-only estimates."""
    if not isinstance(plan, dict):
        return {}

    def compact_option(option: Any) -> dict[str, Any]:
        if not isinstance(option, dict):
            return {}
        route = option.get("route") or {}
        compact_route = (
            _pick(route, (
                "route_digest", "source_answer_feature",
                "source_answer_branch", "target_first_actions",
                "target_continuation_actions", "hypothesis_priority",
                "posterior_normalized_entropy", "top_action_robust_margin",
            ))
            if isinstance(route, dict) else {}
        )
        return {
            **_pick(option, (
                "option_id", "mode", "branch_evaluation_budget",
                "expected_compute_role",
            )),
            "route": compact_route,
        }

    return {
        **_pick(plan, (
            "version", "available", "answer_conditioned_route_available",
            "answer_evidence_digest", "answer_metrics",
            "recommended_option_id", "preference_status", "selection_scope",
        )),
        "options": [
            compact_option(option) for option in plan.get("options") or []
        ],
        "applied_option_id": str(
            (plan.get("applied_option") or {}).get("option_id", "")
        ),
    }


def _active_probe_design_brief(design: Any) -> dict[str, Any]:
    if not isinstance(design, dict):
        return {}
    def compact_option(option: dict[str, Any]) -> dict[str, Any]:
        memory = option.get("discovery_memory") or {}
        return {
            **_pick(option, (
                "probe_id", "action", "horizon", "endpoint",
                "raw_alternative_probability", "alternative_probability",
                "null_probability", "expected_discrimination",
                "experiment_priority",
            )),
            "discovery_memory": _pick(memory, (
                "active", "status", "calibration_adjustment", "authority",
                "validation_skill", "profile_matches", "profile_key",
            )),
        }
    return {
        **_pick(design, (
            "available", "recommended_probe_id",
        )),
        "options": [
            compact_option(option)
            for option in design.get("options") or []
            if isinstance(option, dict)
        ],
    }


def build_llm_decision_brief(packet: dict[str, Any]) -> dict[str, Any]:
    """Project a full packet into compact exact evidence for language reasoning."""
    source_packet = {
        key: value for key, value in packet.items()
        if key != "llm_decision_brief"
    }
    candidates = [
        _candidate_brief(candidate)
        for candidate in source_packet.get("candidates") or []
        if isinstance(candidate, dict)
    ]
    payload = {
        "version": LLM_DECISION_BRIEF_VERSION,
        "source_packet_version": source_packet.get("version"),
        "source_packet_fingerprint": _json_hash(
            source_packet, "world-model-decision-packet:"
        ),
        **_pick(source_packet, (
            "available", "reason", "team_id", "checkpoint_signature",
            "environment_signature", "horizon_s", "observation_coverage",
            "recommended_action", "recommendation_confidence",
        )),
        "deliberation_agenda": _deliberation_agenda(source_packet),
        "candidates": candidates,
        "distributional_action_frontiers": source_packet.get(
            "distributional_action_frontiers", {}
        ),
        "opponent_belief": source_packet.get("opponent_belief", {}),
        "second_order_game": source_packet.get("second_order_game", {}),
        "active_learning": source_packet.get("active_learning", {}),
        "active_probe_design": _active_probe_design_brief(
            source_packet.get("active_probe_design", {})
        ),
        "active_probe_discovery_memory": _pick(
            source_packet.get("active_probe_discovery_memory", {}),
            (
                "version", "active_profiles", "quarantined_profiles",
                "maximum_authority", "compatible_matches", "memory_digest",
                "reason",
            ),
        ),
        "opponent_information_feedback": source_packet.get(
            "opponent_information_feedback", {}
        ),
        "opponent_information_question_menu": source_packet.get(
            "opponent_information_question_menu", {}
        ),
        "opponent_information_cognitive_policy": source_packet.get(
            "opponent_information_cognitive_policy", {}
        ),
        "belief_space_meta_plan": _belief_space_meta_plan_brief(
            source_packet.get("belief_space_meta_plan", {})
        ),
        "deliberation_compute_value_memory": source_packet.get(
            "deliberation_compute_value_memory", {}
        ),
        "task_selection_value_memory": source_packet.get(
            "task_selection_value_memory", {}
        ),
        "online_calibration": source_packet.get("online_calibration", {}),
        "policy_outcome_calibration": source_packet.get(
            "policy_outcome_calibration", {}
        ),
        "contextual_residual_memory": source_packet.get(
            "contextual_residual_memory", {}
        ),
        "llm_semantic_critic_memory": source_packet.get(
            "llm_semantic_critic_memory", {}
        ),
        "evidence_projection_policy": (
            "Exact identifiers and decision summaries are exposed to the LLM; "
            "member/scenario arrays stay in the full packet and are recomputed "
            "by the engine for every audit."
        ),
        "can_replace_full_packet_for_audit": False,
    }
    return {**payload, "brief_digest": _json_hash(
        payload, "world-model-llm-brief:"
    )}


def llm_decision_brief_is_valid(
    brief: Any, packet: dict[str, Any],
) -> bool:
    if not llm_decision_brief_self_is_valid(brief):
        return False
    try:
        return brief == build_llm_decision_brief(packet)
    except (TypeError, ValueError, OverflowError):
        return False


def llm_decision_brief_self_is_valid(brief: Any) -> bool:
    if not isinstance(brief, dict):
        return False
    payload = dict(brief)
    digest = payload.pop("brief_digest", None)
    agenda = payload.get("deliberation_agenda") or {}
    focus = agenda.get("recommended_focus") or []
    try:
        expected = _json_hash(payload, "world-model-llm-brief:")
        maximum = int(agenda.get("maximum_recommended_focus_tasks"))
        compute_budget = int(agenda.get("total_post_llm_compute_credits"))
        per_task_budget = int(agenda.get(
            "maximum_compute_credits_per_task"
        ))
        task_rows = list(agenda["tasks"])
        task_names = [str(row["task"]) for row in task_rows]
        portfolios = dict(agenda["recommended_portfolios_by_size"])
        expected_portfolios = {
            str(size): portfolio
            for size in range(1, 4)
            if (portfolio := _best_deliberation_portfolio(
                task_rows, size,
            )) is not None
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        digest == expected
        and payload.get("version") == LLM_DECISION_BRIEF_VERSION
        and str(payload.get("source_packet_fingerprint", "")).startswith(
            "world-model-decision-packet:"
        )
        and payload.get("can_replace_full_packet_for_audit") is False
        and 1 <= maximum <= 3
        and compute_budget == 6
        and per_task_budget == 3
        and agenda.get("version") == 2
        and len(task_names) == len(set(task_names))
        and set(task_names) == set(TASK_REASONING_DOMAINS)
        and portfolios == expected_portfolios
        and focus == list((
            portfolios.get(str(max(map(int, portfolios), default=0))) or {}
        ).get("tasks") or [])
        and isinstance(focus, list)
        and len(focus) <= maximum
    )


def compact_world_model_facts_for_llm(
    facts: dict[str, Any],
) -> dict[str, Any]:
    """Swap only the serialized LLM view; callers retain the full packet."""
    shadow = facts.get("world_model_shadow_deliberation_brief") or {}
    authoritative = facts.get("world_model_llm_decision_brief") or {}
    if facts.get("world_model_shadow_deliberation_phase"):
        from src.match_engine.world_model.llm_deliberation_encouragement import (
            shadow_deliberation_brief_is_valid,
        )

        if shadow_deliberation_brief_is_valid(shadow, authoritative):
            output = dict(facts)
            output["world_model_decision_support"] = shadow
            output.pop("world_model_llm_decision_brief", None)
            output.pop("world_model_shadow_deliberation_brief", None)
            output.pop("world_model_shadow_deliberation_phase", None)
            return output
    output = dict(facts)
    packet = facts.get("world_model_decision_support") or {}
    if not isinstance(packet, dict) or not packet:
        output.pop("world_model_llm_decision_brief", None)
        return output
    brief = facts.get("world_model_llm_decision_brief") or {}
    if not llm_decision_brief_is_valid(brief, packet):
        brief = build_llm_decision_brief(packet)
    output["world_model_decision_support"] = brief
    output.pop("world_model_llm_decision_brief", None)
    return output


def llm_decision_brief_metadata(brief: dict[str, Any]) -> dict[str, Any]:
    agenda = brief.get("deliberation_agenda") or {}
    return {
        "version": LLM_DECISION_BRIEF_VERSION,
        "brief_digest": str(brief.get("brief_digest", "")),
        "source_packet_version": brief.get("source_packet_version"),
        "source_packet_fingerprint": str(
            brief.get("source_packet_fingerprint", "")
        ),
        "recommended_focus": list(agenda.get("recommended_focus") or []),
        "maximum_recommended_focus_tasks": int(
            agenda.get("maximum_recommended_focus_tasks", 0)
        ),
        "total_post_llm_compute_credits": int(
            agenda.get("total_post_llm_compute_credits", 0)
        ),
        "maximum_compute_credits_per_task": int(
            agenda.get("maximum_compute_credits_per_task", 0)
        ),
        "compute_value_memory_digest": str((brief.get(
            "deliberation_compute_value_memory"
        ) or {}).get("memory_digest", "")),
        "task_selection_value_memory_digest": str((brief.get(
            "task_selection_value_memory"
        ) or {}).get("memory_digest", "")),
        "full_packet_retained_for_engine_audit": True,
        "brief_used_for_llm_serialization": True,
    }


def llm_decision_brief_metadata_is_valid(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    focus = metadata.get("recommended_focus")
    try:
        maximum = int(metadata.get("maximum_recommended_focus_tasks"))
        compute_budget = int(metadata.get("total_post_llm_compute_credits"))
        per_task_budget = int(metadata.get(
            "maximum_compute_credits_per_task"
        ))
        memory_digest = str(metadata.get("compute_value_memory_digest", ""))
        selection_memory_digest = str(metadata.get(
            "task_selection_value_memory_digest", ""
        ))
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        metadata.get("version") == LLM_DECISION_BRIEF_VERSION
        and str(metadata.get("brief_digest", "")).startswith(
            "world-model-llm-brief:"
        )
        and str(metadata.get("source_packet_fingerprint", "")).startswith(
            "world-model-decision-packet:"
        )
        and isinstance(focus, list)
        and 1 <= maximum <= 3
        and compute_budget == 6
        and per_task_budget == 3
        and (
            not memory_digest
            or memory_digest.startswith("llm-deliberation-compute-value:")
        )
        and (
            not selection_memory_digest
            or selection_memory_digest.startswith(
                "llm-deliberation-encouragement:"
            )
        )
        and len(focus) <= maximum
        and len(focus) == len(set(map(str, focus)))
        and metadata.get("full_packet_retained_for_engine_audit") is True
        and metadata.get("brief_used_for_llm_serialization") is True
    )


def llm_decision_brief_diagnostics(
    record_clusters: Any,
) -> dict[str, Any]:
    contexts = malformed = 0
    focus_counts: dict[str, int] = {}
    source_versions = set()
    per_match_focus = []
    for records in record_clusters:
        match_focus = []
        for record in records:
            metadata = record.get("llm_decision_brief_context") or {}
            if not metadata:
                continue
            contexts += 1
            if not llm_decision_brief_metadata_is_valid(metadata):
                malformed += 1
                continue
            source_versions.add(str(metadata.get("source_packet_version")))
            focus = list(metadata["recommended_focus"])
            match_focus.extend(focus)
            for task in focus:
                focus_counts[str(task)] = focus_counts.get(str(task), 0) + 1
        if match_focus:
            per_match_focus.append(len(match_focus))
    return {
        "version": LLM_DECISION_BRIEF_VERSION,
        "brief_contexts": contexts,
        "malformed_brief_contexts": malformed,
        "focus_task_counts": dict(sorted(focus_counts.items())),
        "source_packet_versions": sorted(source_versions),
        "match_clustered_mean_recommended_focus_tasks": float(np.mean(
            per_match_focus
        )) if per_match_focus else 0.0,
        "all_briefs_digest_linked_and_bounded": malformed == 0,
        "full_packet_retained_for_all_engine_audits": malformed == 0,
        "brief_used_for_all_llm_serialization": malformed == 0,
    }
