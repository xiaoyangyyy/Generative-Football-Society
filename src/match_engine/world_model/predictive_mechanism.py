"""Engine-grounded, LLM-articulated predictive mechanism hypotheses."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from src.match_engine.world_model.llm_event_hypothesis import (
    observed_semantic_event,
)


PREDICTIVE_MECHANISM_VERSION = 1
MAXIMUM_MECHANISM_OPTIONS = 12
MINIMUM_ABSOLUTE_CONDITIONAL_LIFT = 0.05
MECHANISM_LOG_EVIDENCE_BOUNDARY = math.log(20.0)
MECHANISM_MAXIMUM_MATCHES = 20
_CELLS = (
    "driver_false_outcome_false",
    "driver_false_outcome_true",
    "driver_true_outcome_false",
    "driver_true_outcome_true",
)


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _distribution(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != set(_CELLS):
        return None
    try:
        out = {key: float(value[key]) for key in _CELLS}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        not all(math.isfinite(number) and 0.0 < number < 1.0
                for number in out.values())
        or abs(sum(out.values()) - 1.0) > 1e-9
    ):
        return None
    return out


def _jensen_shannon(
    alternative: dict[str, float], null: dict[str, float],
) -> float:
    midpoint = {
        key: 0.5 * (alternative[key] + null[key]) for key in _CELLS
    }
    divergence = 0.0
    for distribution in (alternative, null):
        divergence += 0.5 * sum(
            distribution[key] * math.log(
                distribution[key] / midpoint[key]
            ) for key in _CELLS
        )
    return float(min(1.0, max(0.0, divergence / math.log(2.0))))


def _inactive_evidence_contract(
    *, action: str, horizon: str, driver_event: str, outcome_event: str,
    relationship: str, checkpoint_signature: str, environment_signature: str,
) -> dict[str, Any]:
    payload = {
        "version": 1, "status": "start", "profile_matches": 0,
        "cumulative_log_likelihood_ratio": 0.0,
        "mean_log_likelihood_ratio": 0.0,
        "positive_log_evidence_boundary": MECHANISM_LOG_EVIDENCE_BOUNDARY,
        "negative_log_evidence_boundary": -MECHANISM_LOG_EVIDENCE_BOUNDARY,
        "maximum_matches": MECHANISM_MAXIMUM_MATCHES,
        "profile_key": "|".join((
            action, horizon, driver_event, outcome_event, relationship,
        )),
        "checkpoint_signature": checkpoint_signature,
        "environment_signature": environment_signature,
        "can_change_current_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "contract_digest": _digest(
            payload, "predictive-mechanism-memory-contract:",
        ),
    }


def _evidence_contract_is_valid(contract: Any) -> bool:
    if not isinstance(contract, dict):
        return False
    payload = dict(contract)
    digest = payload.pop("contract_digest", None)
    try:
        matches = int(contract["profile_matches"])
        cumulative = float(contract["cumulative_log_likelihood_ratio"])
        mean = float(contract["mean_log_likelihood_ratio"])
        positive = float(contract["positive_log_evidence_boundary"])
        negative = float(contract["negative_log_evidence_boundary"])
        maximum = int(contract["maximum_matches"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    expected_status = (
        "retained_supported" if cumulative >= positive
        else "eliminated_falsified" if cumulative <= negative
        else "retired_inconclusive" if matches >= maximum
        else "start" if matches == 0 else "continue"
    )
    status = contract.get("status")
    return bool(
        digest == _digest(payload, "predictive-mechanism-memory-contract:")
        and contract.get("version") == 1
        and matches >= 0
        and all(math.isfinite(value) for value in (
            cumulative, mean, positive, negative,
        ))
        and abs(positive - MECHANISM_LOG_EVIDENCE_BOUNDARY) <= 1e-12
        and abs(negative + MECHANISM_LOG_EVIDENCE_BOUNDARY) <= 1e-12
        and maximum == MECHANISM_MAXIMUM_MATCHES
        and abs(mean - (cumulative / matches if matches else 0.0)) <= 1e-9
        and status in {expected_status, "incompatible_checkpoint_or_environment"}
        and (status == expected_status or (matches == 0 and cumulative == 0.0))
        and bool(contract.get("profile_key"))
        and bool(contract.get("checkpoint_signature"))
        and bool(contract.get("environment_signature"))
        and contract.get("can_change_current_action") is False
        and contract.get("can_update_world_model") is False
        and contract.get("causal_interpretation") is False
    )


def _mechanism_option(
    *, action: str, horizon: str, edge: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    alternative = _distribution(edge.get("joint_probabilities"))
    null = _distribution(edge.get("independence_null_probabilities"))
    try:
        lift = float(edge["conditional_lift"])
        members = int(edge["ensemble_members"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        not edge.get("available") or alternative is None or null is None
        or not math.isfinite(lift) or members < 2
        or abs(lift) < MINIMUM_ABSOLUTE_CONDITIONAL_LIFT
    ):
        return None
    discrimination = _jensen_shannon(alternative, null)
    if discrimination <= 1e-9:
        return None
    payload = {
        "version": PREDICTIVE_MECHANISM_VERSION,
        "action": str(action),
        "horizon": str(horizon),
        "driver_event": str(edge["driver_event"]),
        "outcome_event": str(edge["outcome_event"]),
        "relationship": "enabling" if lift > 0.0 else "inhibiting",
        "conditional_lift": lift,
        "joint_probabilities": alternative,
        "independence_null_probabilities": null,
        "expected_discrimination": discrimination,
        "ensemble_members": members,
        "cross_match_evidence": evidence,
        "falsification_statistic": (
            "categorical_log_likelihood_ratio_vs_independence"
        ),
        "selected_after_action_freeze": True,
        "association_only": True,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "hypothesis_id": _digest(payload, "predictive-mechanism-option:"),
    }


def _option_is_valid(option: Any) -> bool:
    if not isinstance(option, dict):
        return False
    payload = dict(option)
    hypothesis_id = payload.pop("hypothesis_id", None)
    alternative = _distribution(option.get("joint_probabilities"))
    null = _distribution(option.get("independence_null_probabilities"))
    evidence = option.get("cross_match_evidence") or {}
    try:
        lift = float(option["conditional_lift"])
        discrimination = float(option["expected_discrimination"])
        members = int(option["ensemble_members"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        hypothesis_id == _digest(payload, "predictive-mechanism-option:")
        and option.get("action") in {"hold", "pass", "cross", "shot"}
        and bool(option.get("horizon"))
        and option.get("driver_event") != option.get("outcome_event")
        and option.get("relationship")
        == ("enabling" if lift > 0.0 else "inhibiting")
        and math.isfinite(lift)
        and abs(lift) >= MINIMUM_ABSOLUTE_CONDITIONAL_LIFT
        and alternative is not None and null is not None
        and _evidence_contract_is_valid(evidence)
        and math.isfinite(discrimination)
        and abs(discrimination - _jensen_shannon(alternative, null)) <= 1e-9
        and discrimination > 0.0 and members >= 2
        and option.get("falsification_statistic")
        == "categorical_log_likelihood_ratio_vs_independence"
        and option.get("selected_after_action_freeze") is True
        and option.get("association_only") is True
        and option.get("can_change_current_action") is False
        and option.get("can_change_tactical_controls") is False
        and option.get("can_schedule_future_action") is False
        and option.get("can_update_world_model") is False
        and option.get("causal_interpretation") is False
    )


def build_predictive_mechanism_design(
    packet: dict[str, Any], *, evidence_memory: Any = None,
) -> dict[str, Any]:
    candidates = []
    for candidate in packet.get("candidates") or []:
        action = str(candidate.get("action", ""))
        for horizon, prediction in (
            candidate.get("multi_horizon_predictions") or {}
        ).items():
            edges = ((prediction.get("state_scales") or {}).get(
                "predictive_mechanism_edges"
            ) or {})
            for edge in edges.values():
                if not isinstance(edge, dict):
                    continue
                driver_event = str(edge.get("driver_event", ""))
                outcome_event = str(edge.get("outcome_event", ""))
                try:
                    relationship = (
                        "enabling" if float(edge.get("conditional_lift", 0.0))
                        > 0.0 else "inhibiting"
                    )
                except (TypeError, ValueError, OverflowError):
                    relationship = "inhibiting"
                evidence = _inactive_evidence_contract(
                    action=action, horizon=str(horizon),
                    driver_event=driver_event, outcome_event=outcome_event,
                    relationship=relationship,
                    checkpoint_signature=str(packet.get(
                        "checkpoint_signature", "",
                    )),
                    environment_signature=str(packet.get(
                        "environment_signature", "",
                    )),
                )
                evidence_provider = getattr(evidence_memory, "evidence", None)
                if callable(evidence_provider):
                    try:
                        candidate_evidence = evidence_provider(
                            action=action, horizon=str(horizon),
                            driver_event=driver_event,
                            outcome_event=outcome_event,
                            relationship=relationship,
                            checkpoint_signature=str(packet.get(
                                "checkpoint_signature", "",
                            )),
                            environment_signature=str(packet.get(
                                "environment_signature", "",
                            )),
                        )
                    except (KeyError, TypeError, ValueError, OverflowError):
                        candidate_evidence = None
                    if _evidence_contract_is_valid(candidate_evidence):
                        evidence = candidate_evidence
                option = _mechanism_option(
                    action=action, horizon=str(horizon), edge=edge,
                    evidence=evidence,
                )
                if option is not None:
                    candidates.append(option)
    candidates.sort(key=lambda row: (
        -float(row["expected_discrimination"]),
        -abs(float(row["conditional_lift"])),
        str(row["action"]), str(row["horizon"]),
        str(row["hypothesis_id"]),
    ))
    options = [
        row for row in candidates
        if row["cross_match_evidence"]["status"] in {
            "start", "continue", "incompatible_checkpoint_or_environment",
        }
    ][:MAXIMUM_MECHANISM_OPTIONS]
    resolved = [
        row for row in candidates
        if row["cross_match_evidence"]["status"] in {
            "retained_supported", "eliminated_falsified",
            "retired_inconclusive",
        }
    ][:MAXIMUM_MECHANISM_OPTIONS]
    payload = {
        "version": PREDICTIVE_MECHANISM_VERSION,
        "available": bool(options),
        "reason": (
            "member_grounded_predictive_mechanisms_available" if options
            else "no_discriminating_member_joint"
        ),
        "team_id": str(packet.get("team_id", "")),
        "checkpoint_signature": str(packet.get("checkpoint_signature", "")),
        "environment_signature": str(packet.get("environment_signature", "")),
        "options": options,
        "resolved_hypotheses": resolved,
        "recommended_hypothesis_id": (
            options[0]["hypothesis_id"] if options else "none"
        ),
        "maximum_options": MAXIMUM_MECHANISM_OPTIONS,
        "maximum_resolved_hypotheses": MAXIMUM_MECHANISM_OPTIONS,
        "retained_hypotheses": sum(
            row["cross_match_evidence"]["status"] == "retained_supported"
            for row in resolved
        ),
        "eliminated_hypotheses": sum(
            row["cross_match_evidence"]["status"] == "eliminated_falsified"
            for row in resolved
        ),
        "selection_scope": "post_action_explanation_only",
        "evidence_scope": "same_member_joint_event_projection",
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "design_digest": _digest(payload, "predictive-mechanism-design:"),
    }


def predictive_mechanism_design_is_valid(design: Any) -> bool:
    if not isinstance(design, dict):
        return False
    payload = dict(design)
    digest = payload.pop("design_digest", None)
    options = design.get("options") or []
    resolved = design.get("resolved_hypotheses") or []
    return bool(
        digest == _digest(payload, "predictive-mechanism-design:")
        and design.get("version") == PREDICTIVE_MECHANISM_VERSION
        and bool(design.get("available")) == bool(options)
        and len(options) <= MAXIMUM_MECHANISM_OPTIONS
        and len(resolved) <= MAXIMUM_MECHANISM_OPTIONS
        and design.get("maximum_resolved_hypotheses")
        == MAXIMUM_MECHANISM_OPTIONS
        and all(_option_is_valid(option) for option in options + resolved)
        and len({row["hypothesis_id"] for row in options + resolved})
        == len(options) + len(resolved)
        and all(row["cross_match_evidence"]["status"] in {
            "start", "continue", "incompatible_checkpoint_or_environment",
        } for row in options)
        and all(row["cross_match_evidence"]["status"] in {
            "retained_supported", "eliminated_falsified",
            "retired_inconclusive",
        } for row in resolved)
        and design.get("retained_hypotheses") == sum(
            row["cross_match_evidence"]["status"] == "retained_supported"
            for row in resolved
        )
        and design.get("eliminated_hypotheses") == sum(
            row["cross_match_evidence"]["status"] == "eliminated_falsified"
            for row in resolved
        )
        and design.get("recommended_hypothesis_id")
        == (options[0]["hypothesis_id"] if options else "none")
        and design.get("selection_scope") == "post_action_explanation_only"
        and design.get("evidence_scope")
        == "same_member_joint_event_projection"
        and design.get("can_change_current_action") is False
        and design.get("can_change_tactical_controls") is False
        and design.get("can_schedule_future_action") is False
        and design.get("can_update_world_model") is False
        and design.get("causal_interpretation") is False
    )


def validate_llm_predictive_mechanism(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    hypothesis_id = str(raw.get("hypothesis_id", ""))[:96]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not hypothesis_id or not math.isfinite(confidence)
        or not 0.5 <= confidence <= 1.0
    ):
        return None
    return {
        "hypothesis_id": hypothesis_id,
        "confidence": confidence,
        "mechanism_statement": str(raw.get(
            "mechanism_statement", "",
        ))[:280],
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def evaluate_llm_predictive_mechanism(
    packet: dict[str, Any], raw: Any, *, selected_action: str,
    selected_after_action_freeze: bool,
) -> dict[str, Any]:
    selection = validate_llm_predictive_mechanism(raw)
    design = packet.get("predictive_mechanism_design") or {}
    base = {
        "version": PREDICTIVE_MECHANISM_VERSION,
        "accepted": False,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    if selection is None:
        return {**base, "reason": "missing_or_invalid_mechanism_selection"}
    if not selected_after_action_freeze or not predictive_mechanism_design_is_valid(design):
        return {**base, "reason": "post_action_valid_mechanism_design_required"}
    option = next((
        row for row in design["options"]
        if row["hypothesis_id"] == selection["hypothesis_id"]
    ), None)
    if option is None:
        return {**base, "reason": "unknown_predictive_mechanism"}
    if str(option["action"]) != str(selected_action).lower():
        return {**base, "reason": "frozen_action_mechanism_mismatch"}
    payload = {
        **base, "accepted": True,
        "reason": "engine_grounded_predictive_mechanism_articulated",
        "selection": selection, "hypothesis": option,
        "source_design_digest": design["design_digest"],
        "team_id": design["team_id"],
        "checkpoint_signature": design["checkpoint_signature"],
        "environment_signature": design["environment_signature"],
        "selected_after_action_freeze": True,
        "association_only": True,
    }
    return {
        **payload,
        "audit_digest": _digest(payload, "predictive-mechanism-audit:"),
    }


def predictive_mechanism_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    selection = validate_llm_predictive_mechanism(audit.get("selection"))
    hypothesis = audit.get("hypothesis") or {}
    return bool(
        digest == _digest(payload, "predictive-mechanism-audit:")
        and selection is not None
        and selection["hypothesis_id"] == hypothesis.get("hypothesis_id")
        and _option_is_valid(hypothesis)
        and audit.get("selected_after_action_freeze") is True
        and audit.get("association_only") is True
        and audit.get("can_change_current_action") is False
        and audit.get("can_change_tactical_controls") is False
        and audit.get("can_schedule_future_action") is False
        and audit.get("can_update_world_model") is False
        and audit.get("causal_interpretation") is False
    )


def score_predictive_mechanism(
    audit: dict[str, Any], outcome: dict[str, Any], baseline: dict[str, Any],
    *, attacking_home: bool, horizon: str, realized_action: str,
    checkpoint_signature: str, environment_signature: str,
) -> dict[str, Any] | None:
    if not predictive_mechanism_audit_is_valid(audit):
        return None
    hypothesis = audit["hypothesis"]
    if (
        str(horizon) != str(hypothesis["horizon"])
        or str(realized_action).lower() != str(hypothesis["action"])
        or str(checkpoint_signature) != str(audit["checkpoint_signature"])
        or str(environment_signature) != str(audit["environment_signature"])
    ):
        return None
    try:
        driver = observed_semantic_event(
            hypothesis["driver_event"], outcome, baseline,
            attacking_home=attacking_home,
        )
        consequence = observed_semantic_event(
            hypothesis["outcome_event"], outcome, baseline,
            attacking_home=attacking_home,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    cell = (
        f"driver_{'true' if driver else 'false'}_"
        f"outcome_{'true' if consequence else 'false'}"
    )
    alternative = hypothesis["joint_probabilities"]
    null = hypothesis["independence_null_probabilities"]
    log_ratio = math.log(alternative[cell] / null[cell])
    alternative_brier = sum(
        (alternative[key] - float(key == cell)) ** 2 for key in _CELLS
    )
    null_brier = sum(
        (null[key] - float(key == cell)) ** 2 for key in _CELLS
    )
    payload = {
        "version": PREDICTIVE_MECHANISM_VERSION,
        "hypothesis_id": hypothesis["hypothesis_id"],
        "horizon": hypothesis["horizon"],
        "driver_event": hypothesis["driver_event"],
        "outcome_event": hypothesis["outcome_event"],
        "driver_observed": driver,
        "outcome_observed": consequence,
        "observed_joint_cell": cell,
        "joint_probability": alternative[cell],
        "independence_null_probability": null[cell],
        "categorical_log_likelihood_ratio_vs_independence": log_ratio,
        "joint_brier_score": alternative_brier,
        "independence_null_brier_score": null_brier,
        "brier_skill_vs_independence": null_brier - alternative_brier,
        "association_only": True,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "evaluation_digest": _digest(
            payload, "predictive-mechanism-evaluation:",
        ),
    }


def predictive_mechanism_evaluation_is_valid(
    evaluation: Any, audit: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(evaluation, dict):
        return False
    payload = dict(evaluation)
    digest = payload.pop("evaluation_digest", None)
    if digest != _digest(payload, "predictive-mechanism-evaluation:"):
        return False
    if audit is None:
        return True
    if not predictive_mechanism_audit_is_valid(audit):
        return False
    hypothesis = audit["hypothesis"]
    try:
        cell = evaluation["observed_joint_cell"]
        alternative = hypothesis["joint_probabilities"]
        null = hypothesis["independence_null_probabilities"]
        expected_log = math.log(alternative[cell] / null[cell])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        cell in _CELLS
        and evaluation.get("hypothesis_id") == hypothesis["hypothesis_id"]
        and evaluation.get("horizon") == hypothesis["horizon"]
        and abs(float(evaluation[
            "categorical_log_likelihood_ratio_vs_independence"
        ]) - expected_log) <= 1e-12
        and evaluation.get("association_only") is True
        and evaluation.get("causal_interpretation") is False
    )
