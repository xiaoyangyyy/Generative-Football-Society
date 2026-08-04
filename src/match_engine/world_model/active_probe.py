"""Pre-registered, falsifiable probes for randomized safe exploration."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


ACTIVE_PROBE_VERSION = 1


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _bernoulli_kl(left: float, right: float) -> float:
    epsilon = 1e-6
    left = min(1.0 - epsilon, max(epsilon, left))
    right = min(1.0 - epsilon, max(epsilon, right))
    return (
        left * math.log(left / right)
        + (1.0 - left) * math.log((1.0 - left) / (1.0 - right))
    )


def _discrimination(alternative: float, null: float) -> float:
    midpoint = 0.5 * (alternative + null)
    return min(1.0, max(0.0, (
        0.5 * _bernoulli_kl(alternative, midpoint)
        + 0.5 * _bernoulli_kl(null, midpoint)
    ) / math.log(2.0)))


def _option_is_valid(option: Any) -> bool:
    if not isinstance(option, dict):
        return False
    payload = dict(option)
    probe_id = payload.pop("probe_id", None)
    alternative = _probability(option.get("alternative_probability"))
    null = _probability(option.get("null_probability"))
    try:
        regret = float(option["estimated_regret"])
        max_regret = float(option["maximum_allowed_regret"])
        information = float(option["expected_discrimination"])
        falsification_threshold = float(option["falsification_threshold"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        probe_id == _digest(payload, "active-world-model-probe:")
        and option.get("action") in {"hold", "pass", "cross", "shot"}
        and option.get("null_action") in {"hold", "pass", "cross", "shot"}
        and option.get("action") != option.get("null_action")
        and option.get("endpoint") == "retained_possession"
        and isinstance(option.get("horizon"), str)
        and option.get("horizon")
        and alternative is not None and null is not None
        and abs(alternative - null) >= 0.02 - 1e-9
        and math.isfinite(regret) and math.isfinite(max_regret)
        and 0.0 <= regret <= max_regret <= 0.30
        and math.isfinite(information) and 0.0 <= information <= 1.0
        and abs(information - _discrimination(alternative, null)) <= 1e-9
        and option.get("falsification_statistic")
        == "bernoulli_log_likelihood_ratio_vs_exploit_forecast"
        and falsification_threshold == 0.0
        and option.get("requires_realized_probe_action") is True
        and option.get("can_change_current_action") is False
        and option.get("can_change_tactical_controls") is False
        and option.get("causal_interpretation") is False
    )


def build_active_probe_design(packet: dict[str, Any]) -> dict[str, Any]:
    learning = packet.get("active_learning") or {}
    exploration_action = str(learning.get("exploration_action", "none"))
    exploit_action = str(learning.get("exploit_action", "none"))
    candidates = {
        str(row.get("action")): row
        for row in packet.get("candidates") or []
        if isinstance(row, dict)
    }
    exploration = candidates.get(exploration_action) or {}
    exploit = candidates.get(exploit_action) or {}
    options = []
    if learning.get("eligible") and exploration and exploit:
        exploration_predictions = exploration.get(
            "multi_horizon_predictions"
        ) or {}
        exploit_predictions = exploit.get("multi_horizon_predictions") or {}
        for horizon in sorted(
            set(exploration_predictions) & set(exploit_predictions),
            key=lambda key: (
                key != "transition",
                float(str(key).removesuffix("s"))
                if key != "transition" else 0.0,
            ),
        )[:3]:
            alternative = _probability(
                exploration_predictions[horizon].get(
                    "retention_probability"
                )
            )
            null = _probability(
                exploit_predictions[horizon].get("retention_probability")
            )
            if alternative is None or null is None or abs(alternative - null) < 0.02:
                continue
            option_payload = {
                "version": ACTIVE_PROBE_VERSION,
                "action": exploration_action,
                "null_action": exploit_action,
                "horizon": str(horizon),
                "endpoint": "retained_possession",
                "alternative_probability": alternative,
                "null_probability": null,
                "expected_discrimination": _discrimination(alternative, null),
                "estimated_regret": float(learning.get(
                    "estimated_regret", 0.0
                )),
                "maximum_allowed_regret": float((
                    learning.get("constraints") or {}
                ).get("max_regret", 0.0)),
                "falsification_statistic": (
                    "bernoulli_log_likelihood_ratio_vs_exploit_forecast"
                ),
                "falsification_threshold": 0.0,
                "requires_realized_probe_action": True,
                "can_change_current_action": False,
                "can_change_tactical_controls": False,
                "causal_interpretation": False,
            }
            options.append({
                **option_payload,
                "probe_id": _digest(
                    option_payload, "active-world-model-probe:"
                ),
            })
    options.sort(key=lambda row: (
        -float(row["expected_discrimination"]), str(row["horizon"]),
    ))
    reason = (
        "falsifiable_safe_exploration_available" if options
        else "active_learning_ineligible"
        if not learning.get("eligible")
        else "no_discriminating_observable_endpoint"
    )
    context = packet.get("decision_context") or {}
    payload = {
        "version": ACTIVE_PROBE_VERSION,
        "available": bool(options),
        "reason": reason,
        "team_id": str(packet.get("team_id", "")),
        "checkpoint_signature": str(packet.get("checkpoint_signature", "")),
        "environment_signature": str(packet.get("environment_signature", "")),
        "as_of_t_sec": float(context.get("clock_seconds", 0.0)),
        "options": options,
        "recommended_probe_id": options[0]["probe_id"] if options else "none",
        "selection_scope": "post_action_interpretation_only",
        "execution_scope": "existing_randomized_policy_bridge_only",
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "design_digest": _digest(payload, "active-probe-design:"),
    }


def active_probe_design_is_valid(design: Any) -> bool:
    if not isinstance(design, dict):
        return False
    payload = dict(design)
    digest = payload.pop("design_digest", None)
    options = design.get("options") or []
    return bool(
        digest == _digest(payload, "active-probe-design:")
        and design.get("version") == ACTIVE_PROBE_VERSION
        and bool(design.get("available")) == bool(options)
        and all(_option_is_valid(option) for option in options)
        and len({option.get("probe_id") for option in options}) == len(options)
        and design.get("recommended_probe_id")
        == (options[0]["probe_id"] if options else "none")
        and design.get("selection_scope") == "post_action_interpretation_only"
        and design.get("execution_scope")
        == "existing_randomized_policy_bridge_only"
        and design.get("can_change_current_action") is False
        and design.get("can_change_tactical_controls") is False
        and design.get("can_schedule_future_action") is False
        and design.get("causal_interpretation") is False
    )


def validate_llm_active_probe(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    probe_id = str(raw.get("probe_id", ""))[:96]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not probe_id or not math.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
        return None
    return {
        "probe_id": probe_id,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def evaluate_llm_active_probe(
    packet: dict[str, Any],
    raw: Any,
    *,
    selected_action: str,
    decision_mode: str,
    selected_after_action_freeze: bool,
) -> dict[str, Any]:
    selection = validate_llm_active_probe(raw)
    design = packet.get("active_probe_design") or {}
    base = {
        "version": ACTIVE_PROBE_VERSION,
        "accepted": False,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    if selection is None:
        return {**base, "reason": "missing_or_invalid_probe_selection"}
    if not selected_after_action_freeze or not active_probe_design_is_valid(design):
        return {**base, "reason": "post_action_valid_probe_design_required"}
    option = next((
        row for row in design["options"]
        if row["probe_id"] == selection["probe_id"]
    ), None)
    if option is None:
        return {**base, "reason": "unknown_active_probe"}
    if (
        str(selected_action) != str(option["action"])
        or str(decision_mode) != "explore"
    ):
        return {**base, "reason": "frozen_exploration_action_mismatch"}
    payload = {
        **base,
        "accepted": True,
        "reason": "falsifiable_probe_pre_registered_after_action_freeze",
        "selection": selection,
        "probe": option,
        "source_design_digest": design["design_digest"],
        "team_id": design["team_id"],
        "checkpoint_signature": design["checkpoint_signature"],
        "environment_signature": design["environment_signature"],
        "selected_after_action_freeze": True,
    }
    return {
        **payload,
        "audit_digest": _digest(payload, "active-probe-audit:"),
    }


def active_probe_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    selection = validate_llm_active_probe(audit.get("selection"))
    probe = audit.get("probe") or {}
    return bool(
        digest == _digest(payload, "active-probe-audit:")
        and selection is not None
        and selection["probe_id"] == probe.get("probe_id")
        and _option_is_valid(probe)
        and audit.get("selected_after_action_freeze") is True
        and audit.get("can_change_current_action") is False
        and audit.get("can_change_tactical_controls") is False
        and audit.get("can_schedule_future_action") is False
        and audit.get("causal_interpretation") is False
    )


def score_active_probe(
    audit: dict[str, Any],
    outcome: dict[str, Any],
    *,
    horizon: str,
    realized_action: str,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, Any] | None:
    if not active_probe_audit_is_valid(audit):
        return None
    probe = audit["probe"]
    if (
        str(realized_action) != str(probe["action"])
        or str(horizon) != str(probe["horizon"])
        or str(checkpoint_signature) != str(audit["checkpoint_signature"])
        or str(environment_signature) != str(audit["environment_signature"])
        or not isinstance(outcome.get("retained_possession"), bool)
    ):
        return None
    observed = 1.0 if outcome["retained_possession"] else 0.0
    alternative = float(probe["alternative_probability"])
    null = float(probe["null_probability"])
    epsilon = 1e-6
    alternative_likelihood = (
        alternative if observed else 1.0 - alternative
    )
    null_likelihood = null if observed else 1.0 - null
    log_ratio = math.log(
        max(epsilon, alternative_likelihood)
        / max(epsilon, null_likelihood)
    )
    payload = {
        "version": ACTIVE_PROBE_VERSION,
        "probe_id": probe["probe_id"],
        "endpoint": probe["endpoint"],
        "horizon": probe["horizon"],
        "observed_value": bool(outcome["retained_possession"]),
        "alternative_probability": alternative,
        "null_probability": null,
        "alternative_brier_score": (alternative - observed) ** 2,
        "null_brier_score": (null - observed) ** 2,
        "brier_skill_vs_null": (
            (null - observed) ** 2 - (alternative - observed) ** 2
        ),
        "log_likelihood_ratio_vs_null": log_ratio,
        "falsification_signal": bool(
            log_ratio < float(probe["falsification_threshold"])
        ),
        "interpretation": (
            "single_observation_updates_pre_registered_evidence; "
            "cross-match aggregation is required for promotion"
        ),
        "causal_interpretation": False,
    }
    return {
        **payload,
        "evaluation_digest": _digest(payload, "active-probe-evaluation:"),
    }


def active_probe_evaluation_is_valid(
    evaluation: Any,
    audit: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(evaluation, dict):
        return False
    payload = dict(evaluation)
    digest = payload.pop("evaluation_digest", None)
    alternative = _probability(evaluation.get("alternative_probability"))
    null = _probability(evaluation.get("null_probability"))
    observed_raw = evaluation.get("observed_value")
    if alternative is None or null is None or not isinstance(observed_raw, bool):
        return False
    observed = 1.0 if observed_raw else 0.0
    epsilon = 1e-6
    expected_log_ratio = math.log(
        max(epsilon, alternative if observed else 1.0 - alternative)
        / max(epsilon, null if observed else 1.0 - null)
    )
    try:
        alternative_brier = float(evaluation["alternative_brier_score"])
        null_brier = float(evaluation["null_brier_score"])
        skill = float(evaluation["brier_skill_vs_null"])
        log_ratio = float(evaluation["log_likelihood_ratio_vs_null"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (
        alternative_brier, null_brier, skill, log_ratio,
    )):
        return False
    probe = (audit or {}).get("probe") or {}
    audit_ok = bool(
        not audit
        or (
            active_probe_audit_is_valid(audit)
            and evaluation.get("probe_id") == probe.get("probe_id")
            and evaluation.get("endpoint") == probe.get("endpoint")
            and evaluation.get("horizon") == probe.get("horizon")
            and abs(alternative - float(
                probe.get("alternative_probability", -1.0)
            )) <= 1e-12
            and abs(null - float(probe.get("null_probability", -1.0)))
            <= 1e-12
        )
    )
    return bool(
        digest == _digest(payload, "active-probe-evaluation:")
        and evaluation.get("version") == ACTIVE_PROBE_VERSION
        and abs(alternative_brier - (alternative - observed) ** 2) <= 1e-12
        and abs(null_brier - (null - observed) ** 2) <= 1e-12
        and abs(skill - (null_brier - alternative_brier)) <= 1e-12
        and abs(log_ratio - expected_log_ratio) <= 1e-12
        and evaluation.get("falsification_signal")
        == bool(log_ratio < 0.0)
        and evaluation.get("causal_interpretation") is False
        and audit_ok
    )
