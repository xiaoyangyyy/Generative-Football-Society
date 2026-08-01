"""Shadow chance constraints over transparent ensemble trajectory modes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS
from src.match_engine.world_model.state_scales import (
    FALSIFIABLE_DOWNSIDE_EVENTS,
)


RISK_CERTIFICATE_VERSION = 2
_HORIZON_PATTERN = re.compile(r"^(transition|\d+(?:\.\d+)?s)$")
_WILSON_Z_90 = 1.6448536269514722
_RISK_SCOPES = {"terminal", "within_horizon"}


def llm_risk_certificate_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": RISK_CERTIFICATE_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "shadow_ensemble_mode_chance_constraint",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-risk-certificate:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_risk_constraint(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("selected_action", "")).strip().lower()
    horizon = str(raw.get("horizon", "")).strip().lower()
    event = str(raw.get("downside_event", "")).strip().lower()
    risk_scope = str(raw.get("risk_scope", "terminal")).strip().lower()
    if (
        action not in RESPONSE_ACTIONS
        or not _HORIZON_PATTERN.fullmatch(horizon)
        or event not in FALSIFIABLE_DOWNSIDE_EVENTS
        or risk_scope not in _RISK_SCOPES
    ):
        return None
    try:
        maximum = float(raw.get("max_violation_probability"))
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if (
        not np.isfinite(maximum) or not 0.05 <= maximum <= 0.95
        or not np.isfinite(confidence) or not 0.5 <= confidence <= 1.0
    ):
        return None
    return {
        "selected_action": action,
        "horizon": horizon,
        "downside_event": event,
        "risk_scope": risk_scope,
        "max_violation_probability": maximum,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:280],
    }


def wilson_upper_bound(
    violations: int,
    samples: int,
    *,
    z: float = _WILSON_Z_90,
) -> float:
    n = int(samples)
    k = int(violations)
    if n <= 0 or k < 0 or k > n:
        raise ValueError("Wilson counts must satisfy 0 <= violations <= samples")
    p = k / n
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    radius = float(z) * math.sqrt(
        p * (1.0 - p) / n + z2 / (4.0 * n * n)
    )
    return float(np.clip((centre + radius) / denominator, 0.0, 1.0))


def _two_step_gate(runtime, rollout_steps: int) -> dict[str, Any]:
    if rollout_steps == 1:
        return {
            "active": True,
            "authority": 1.0,
            "reason": "trained_one_step_member_modes",
        }
    provider = getattr(runtime, "two_step_planning_gate", None)
    if not callable(provider):
        return {
            "active": False,
            "authority": 0.0,
            "reason": "runtime_has_no_two_step_validation_contract",
        }
    try:
        gate = dict(provider())
        authority = finite_float(gate.get("authority") or 0.0, 0.0)
    except (RuntimeError, TypeError, ValueError):
        return {
            "active": False,
            "authority": 0.0,
            "reason": "two_step_validation_contract_error",
        }
    gate["authority"] = float(np.clip(authority, 0.0, 0.50))
    gate["active"] = bool(gate.get("active") and gate["authority"] > 0.0)
    return gate


def apply_llm_risk_constraint(
    runtime,
    packet: dict[str, Any],
    raw_constraint: Any,
    *,
    selected_action: str,
    certificate_signature: str = "risk-certificate-contract-unspecified",
) -> dict[str, Any]:
    """Certify one LLM-declared downside limit without changing policy."""
    constraint = validate_llm_risk_constraint(raw_constraint)
    base = {
        "version": RISK_CERTIFICATE_VERSION,
        "accepted": False,
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "world_model_prediction_mutated": False,
        "causal_interpretation": False,
        "can_change_selected_action": False,
        "can_update_world_model": False,
        "certificate_signature": str(certificate_signature),
    }
    if constraint is None:
        return {**base, "reason": "missing_or_invalid_risk_constraint"}
    if constraint["selected_action"] != str(selected_action).lower():
        return {
            **base,
            "reason": "risk_constraint_action_must_equal_selected_action",
            "constraint": constraint,
        }
    if not packet.get("available"):
        return {
            **base,
            "reason": "risk_certificate_world_model_quality_gate_closed",
            "constraint": constraint,
        }
    candidate = next((
        item for item in packet.get("candidates") or []
        if str(item.get("action", "")).lower()
        == constraint["selected_action"]
    ), None)
    prediction = (
        (candidate.get("multi_horizon_predictions") or {}).get(
            constraint["horizon"]
        ) if candidate else None
    )
    if not isinstance(prediction, dict):
        return {
            **base,
            "reason": "risk_constraint_horizon_not_evaluated",
            "constraint": constraint,
        }
    rollout_steps = int(prediction.get("rollout_steps", 1))
    if rollout_steps not in {1, 2}:
        return {
            **base,
            "reason": "risk_constraint_rollout_depth_not_validated",
            "constraint": constraint,
        }
    planning_gate = _two_step_gate(runtime, rollout_steps)
    if not planning_gate.get("active"):
        return {
            **base,
            "reason": "risk_certificate_planning_gate_closed",
            "constraint": constraint,
            "planning_gate": planning_gate,
        }
    modes = ((prediction.get("state_scales") or {}).get(
        "trajectory_modes"
    ) or {})
    if not (
        modes.get("available")
        and modes.get("counts_trusted")
        and not modes.get("causal_interpretation")
    ):
        return {
            **base,
            "reason": "trained_trajectory_modes_unavailable",
            "constraint": constraint,
        }
    event = constraint["downside_event"]
    risk_scope = constraint["risk_scope"]
    if risk_scope == "within_horizon" and (
        rollout_steps != 2
        or not modes.get("temporal_path_available")
        or not modes.get("path_counts_trusted")
        or int(modes.get("trajectory_steps", 0)) != rollout_steps
    ):
        return {
            **base,
            "reason": "member_consistent_temporal_path_unavailable",
            "constraint": constraint,
            "planning_gate": planning_gate,
        }
    count_key = (
        "path_downside_event_counts"
        if risk_scope == "within_horizon"
        else "downside_event_counts"
    )
    probability_key = (
        "path_downside_event_probabilities"
        if risk_scope == "within_horizon"
        else "downside_event_probabilities"
    )
    mode_event_key = (
        "path_downside_events"
        if risk_scope == "within_horizon"
        else "terminal_downside_events"
    )
    try:
        members = int(modes["ensemble_members"])
        violations = int(modes[count_key][event])
        probability = float(modes[probability_key][event])
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            **base,
            "reason": "trajectory_mode_risk_contract_invalid",
            "constraint": constraint,
        }
    if (
        members < 2 or violations < 0 or violations > members
        or not np.isfinite(probability) or not 0.0 <= probability <= 1.0
    ):
        return {
            **base,
            "reason": "trajectory_mode_risk_contract_invalid",
            "constraint": constraint,
        }
    expected_probability = (violations + 0.5) / (members + 1.0)
    if abs(probability - expected_probability) > 1e-6:
        return {
            **base,
            "reason": "trajectory_mode_risk_contract_invalid",
            "constraint": constraint,
        }
    probability = float(expected_probability)
    upper = wilson_upper_bound(violations, members)
    maximum = float(constraint["max_violation_probability"])
    try:
        violating_modes = []
        for mode in modes.get("modes") or []:
            mode_probability = float(mode.get("probability", 0.0))
            violation_rate = float(
                (mode.get(mode_event_key) or mode.get("downside_events") or {})
                .get(event, 0.0)
            )
            if (
                not np.isfinite(mode_probability)
                or not 0.0 <= mode_probability <= 1.0
                or not np.isfinite(violation_rate)
                or not 0.0 <= violation_rate <= 1.0
            ):
                raise ValueError("invalid trajectory mode probability")
            if violation_rate > 0.0:
                violating_modes.append({
                    "mode_id": str(mode.get("mode_id", "unknown")),
                    "mode_probability": mode_probability,
                    "within_mode_violation_rate": violation_rate,
                })
    except (AttributeError, TypeError, ValueError, OverflowError):
        return {
            **base,
            "reason": "trajectory_mode_risk_contract_invalid",
            "constraint": constraint,
        }
    audit = {
        **base,
        "accepted": True,
        "reason": "shadow_ensemble_mode_risk_certificate",
        "constraint": constraint,
        "checkpoint_signature": str(packet.get(
            "checkpoint_signature", "runtime_unspecified",
        )),
        "environment_signature": str(packet.get(
            "environment_signature", "environment_unspecified",
        )),
        "rollout_steps": rollout_steps,
        "risk_scope": risk_scope,
        "member_trajectory_identity_preserved": bool(
            risk_scope == "terminal"
            or modes.get("temporal_path_available")
        ),
        "planning_gate": planning_gate,
        "trajectory_mode_version": int(modes.get("version", 0)),
        "trajectory_mode_count": int(modes.get("mode_count", 0)),
        "trajectory_mode_entropy": float(modes.get(
            "normalized_mode_entropy", 0.0,
        )),
        "ensemble_members": members,
        "member_violations": violations,
        "projected_violation_probability": probability,
        "wilson_upper_violation_probability_90": upper,
        "nominal_constraint_satisfied": probability <= maximum,
        "conservatively_certified": upper <= maximum,
        "certificate_margin": maximum - upper,
        "violating_modes": violating_modes,
        "probability_source": "jeffreys_smoothed_member_frequency",
        "upper_bound_source": "one_sided_90_percent_wilson_member_bound",
        "certificate_can_veto_action": False,
    }
    packet["llm_risk_certificate_audit"] = audit
    return audit


def observed_downside_event(
    event: str,
    outcome: dict[str, Any],
    baseline: dict[str, Any],
    *,
    attacking_home: bool,
) -> bool:
    if event == "lose_possession":
        return not bool(outcome["retained_possession"])
    if event == "negative_territorial_shift":
        return float(outcome["progress"]) <= -0.03
    if event == "worsen_scoreline":
        return float(outcome["goal_diff_delta"]) <= -0.5
    if event == "fail_enter_final_third":
        initial = float(baseline["ball_x"])
        if not attacking_home:
            initial = 1.0 - initial
        return initial + float(outcome["progress"]) < 0.67
    raise ValueError("unsupported downside event")


def score_llm_risk_certificate(
    context: dict[str, Any],
    outcome: dict[str, Any],
    baseline: dict[str, Any],
    *,
    attacking_home: bool,
    checkpoint_signature: str,
    environment_signature: str,
    interval_monitor: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not context.get("accepted"):
        return None
    constraint = validate_llm_risk_constraint(context.get("constraint"))
    if constraint is None:
        return None
    risk_scope = constraint["risk_scope"]
    monitor = dict(interval_monitor or {})
    if risk_scope == "within_horizon":
        if not monitor.get("complete"):
            return None
        observed = bool(monitor.get("downside_observed"))
    else:
        observed = observed_downside_event(
            constraint["downside_event"],
            outcome,
            baseline,
            attacking_home=attacking_home,
        )
    probability = float(np.clip(float(
        context["projected_violation_probability"]
    ), 0.0, 1.0))
    target = float(observed)
    certified = bool(context.get("conservatively_certified"))
    checkpoint_compatible = str(context.get("checkpoint_signature", "")) == str(
        checkpoint_signature
    )
    environment_compatible = str(
        context.get("environment_signature", "")
    ) == str(environment_signature)
    return {
        "version": RISK_CERTIFICATE_VERSION,
        "constraint": constraint,
        "downside_observed": bool(observed),
        "predicted_violation_probability": probability,
        "violation_brier": float((probability - target) ** 2),
        "wilson_upper_violation_probability_90": float(
            context.get("wilson_upper_violation_probability_90", 1.0)
        ),
        "conservatively_certified": certified,
        "false_safe_certificate": bool(certified and observed),
        "paired_same_horizon_outcome": True,
        "risk_scope": risk_scope,
        "interval_monitor_complete": bool(
            risk_scope == "terminal" or monitor.get("complete")
        ),
        "interval_observation_count": int(
            monitor.get("observations", 0)
        ),
        "interval_max_gap_s": float(monitor.get("max_gap_s", 0.0)),
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "certificate_can_veto_action": False,
        "causal_interpretation": False,
        "member_trajectory_identity_preserved": bool(context.get(
            "member_trajectory_identity_preserved", False,
        )),
        "issuance_evaluation_provenance_compatible": bool(
            checkpoint_compatible and environment_compatible
        ),
        "certificate_signature": str(context.get(
            "certificate_signature", "",
        )),
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
    }


def risk_certificate_diagnostics(
    match_logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    match_rows = []
    eligible_certificates = 0
    unscored_eligible_certificates = 0
    for payload in match_logs:
        rows = []
        for record in (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        ):
            outcomes = record.get("multi_horizon_regime_outcomes") or {}
            context = record.get("llm_risk_certificate_context") or {}
            constraint = validate_llm_risk_constraint(
                context.get("constraint")
            )
            if (
                context.get("accepted")
                and constraint is not None
                and constraint["selected_action"]
                == str(record.get("intervention_actual_action", "")).lower()
            ):
                horizon_key = constraint["horizon"]
                if horizon_key in outcomes:
                    eligible_certificates += 1
                    candidate = outcomes[horizon_key]
                    if not isinstance(candidate, dict) or not isinstance(
                        candidate.get("llm_risk_certificate_evaluation"), dict,
                    ):
                        unscored_eligible_certificates += 1
            for outcome in outcomes.values():
                evaluation = (
                    outcome.get("llm_risk_certificate_evaluation")
                    if isinstance(outcome, dict) else None
                )
                if isinstance(evaluation, dict):
                    rows.append(evaluation)
        if rows:
            match_rows.append(rows)
    valid = []
    malformed = 0
    for rows in match_rows:
        for row in rows:
            constraint = validate_llm_risk_constraint(row.get("constraint"))
            try:
                version = int(row["version"])
                observed = float(bool(row["downside_observed"]))
                probability = float(row["predicted_violation_probability"])
                brier = float(row["violation_brier"])
                upper = float(row["wilson_upper_violation_probability_90"])
            except (KeyError, TypeError, ValueError, OverflowError):
                malformed += 1
                continue
            if (
                version != RISK_CERTIFICATE_VERSION
                or constraint is None
                or not all(np.isfinite(value) for value in (
                    probability, brier, upper,
                ))
                or not 0.0 <= probability <= 1.0
                or not 0.0 <= upper <= 1.0
                or abs(brier - (probability - observed) ** 2) > 1e-6
                or not row.get("paired_same_horizon_outcome")
                or not row.get("issuance_evaluation_provenance_compatible")
                or not row.get("interval_monitor_complete")
                or not row.get("member_trajectory_identity_preserved")
            ):
                malformed += 1
                continue
            valid.append(row)
    valid_ids = {id(row) for row in valid}
    valid_match_rows = [
        [row for row in rows if id(row) in valid_ids]
        for rows in match_rows
    ]
    valid_match_rows = [rows for rows in valid_match_rows if rows]
    match_brier = [
        float(np.mean([float(row["violation_brier"]) for row in rows]))
        for rows in valid_match_rows
    ]
    certified_groups = [
        [row for row in rows if row.get("conservatively_certified")]
        for rows in valid_match_rows
    ]
    certified_groups = [rows for rows in certified_groups if rows]
    match_certified_violation = [
        float(np.mean([
            float(bool(row["downside_observed"])) for row in rows
        ]))
        for rows in certified_groups
    ]
    match_certified_threshold = [
        float(np.mean([
            float(row["constraint"]["max_violation_probability"])
            for row in rows
        ]))
        for rows in certified_groups
    ]
    signatures = sorted({
        str(row.get("certificate_signature", "")) for row in valid
    })
    scopes = sorted({(
        str(row.get("checkpoint_signature", "")),
        str(row.get("environment_signature", "")),
    ) for row in valid})
    unspecified = {
        "", "risk-certificate-contract-unspecified",
        "runtime_unspecified", "environment_unspecified",
    }
    return {
        "version": RISK_CERTIFICATE_VERSION,
        "evaluation_kind": "realized_shadow_ensemble_mode_risk_certificate",
        "realized_certificates": len(valid),
        "eligible_certificates": eligible_certificates,
        "unscored_eligible_certificates": unscored_eligible_certificates,
        "matches": len(valid_match_rows),
        "malformed_certificate_evaluations": malformed,
        "certified_outcomes": sum(len(rows) for rows in certified_groups),
        "certified_matches": len(certified_groups),
        "match_clustered_violation_brier": (
            float(np.mean(match_brier)) if match_brier else 0.0
        ),
        "match_clustered_certified_violation_rate": (
            float(np.mean(match_certified_violation))
            if match_certified_violation else 0.0
        ),
        "match_clustered_mean_certified_threshold": (
            float(np.mean(match_certified_threshold))
            if match_certified_threshold else 0.0
        ),
        "false_safe_certificates": sum(
            bool(row.get("false_safe_certificate")) for row in valid
        ),
        "all_shadow_only": all(bool(row.get("shadow_only")) for row in valid),
        "all_non_controlling": all(
            not bool(row.get("authority_active"))
            and not bool(row.get("policy_mutated"))
            and not bool(row.get("certificate_can_veto_action"))
            for row in valid
        ),
        "all_non_causal": all(
            not bool(row.get("causal_interpretation")) for row in valid
        ),
        "all_interval_monitors_complete": all(
            bool(row.get("interval_monitor_complete")) for row in valid
        ),
        "all_member_trajectory_identity_preserved": all(
            bool(row.get("member_trajectory_identity_preserved"))
            for row in valid
        ),
        "risk_scope_counts": {
            scope: sum(
                str(row.get("risk_scope", "terminal")) == scope
                for row in valid
            )
            for scope in sorted(_RISK_SCOPES)
        },
        "provenance_compatible": bool(
            len(signatures) == 1 and signatures[0] not in unspecified
            and len(scopes) == 1
            and all(value not in unspecified for value in scopes[0])
        ),
        "certificate_signatures": signatures,
        "model_policy_scopes": [list(scope) for scope in scopes],
    }
