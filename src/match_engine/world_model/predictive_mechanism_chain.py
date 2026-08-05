"""Higher-order predictive mechanism chains grounded in member trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


PREDICTIVE_MECHANISM_CHAIN_VERSION = 1
MAXIMUM_CHAIN_OPTIONS = 9
MINIMUM_CONDITIONAL_INFORMATION_BITS = 0.002
CHAIN_LOG_EVIDENCE_BOUNDARY = math.log(20.0)
CHAIN_MAXIMUM_MATCHES = 20
_CELLS = tuple(
    f"a{a}_b{b}_c{c}"
    for a in (0, 1) for b in (0, 1) for c in (0, 1)
)


def predictive_mechanism_chain_llm_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": PREDICTIVE_MECHANISM_CHAIN_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "bounded_three_event_chain_completion_forecast",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-predictive-chain:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _distribution(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != set(_CELLS):
        return None
    try:
        result = {key: float(value[key]) for key in _CELLS}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        not all(math.isfinite(number) and 0.0 < number < 1.0
                for number in result.values())
        or abs(sum(result.values()) - 1.0) > 1e-9
    ):
        return None
    return result


def _inactive_evidence(
    *, action: str, horizon: str, first_event: str, mediator_event: str,
    outcome_event: str, checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, Any]:
    key = tuple(map(str, (
        action, horizon, first_event, mediator_event, outcome_event,
    )))
    payload = {
        "version": 1, "status": "start", "profile_matches": 0,
        "cumulative_log_likelihood_ratio": 0.0,
        "mean_log_likelihood_ratio": 0.0,
        "positive_log_evidence_boundary": CHAIN_LOG_EVIDENCE_BOUNDARY,
        "negative_log_evidence_boundary": -CHAIN_LOG_EVIDENCE_BOUNDARY,
        "maximum_matches": CHAIN_MAXIMUM_MATCHES,
        "profile_key": "|".join(key),
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "can_change_current_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "contract_digest": _digest(
            payload, "predictive-chain-memory-contract:",
        ),
    }


def _evidence_is_valid(contract: Any) -> bool:
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
    expected = (
        "retained_higher_order_dependence" if cumulative >= positive
        else "eliminated_markov_sufficient" if cumulative <= negative
        else "retired_inconclusive" if matches >= maximum
        else "start" if matches == 0 else "continue"
    )
    status = contract.get("status")
    return bool(
        digest == _digest(payload, "predictive-chain-memory-contract:")
        and contract.get("version") == 1 and matches >= 0
        and all(math.isfinite(value) for value in (
            cumulative, mean, positive, negative,
        ))
        and abs(positive - CHAIN_LOG_EVIDENCE_BOUNDARY) <= 1e-12
        and abs(negative + CHAIN_LOG_EVIDENCE_BOUNDARY) <= 1e-12
        and maximum == CHAIN_MAXIMUM_MATCHES
        and abs(mean - (cumulative / matches if matches else 0.0)) <= 1e-9
        and status in {expected, "incompatible_checkpoint_or_environment"}
        and (status == expected or (matches == 0 and cumulative == 0.0))
        and bool(contract.get("profile_key"))
        and bool(contract.get("checkpoint_signature"))
        and bool(contract.get("environment_signature"))
        and contract.get("can_change_current_action") is False
        and contract.get("can_update_world_model") is False
        and contract.get("causal_interpretation") is False
    )


def _chain_priority(information: float, evidence: dict[str, Any]) -> float:
    matches = int(evidence["profile_matches"])
    cumulative = float(evidence["cumulative_log_likelihood_ratio"])
    remaining = 1.0 - min(
        1.0, abs(cumulative) / CHAIN_LOG_EVIDENCE_BOUNDARY,
    )
    novelty = 1.0 / (1.0 + matches / 4.0)
    return float(min(1.0, max(
        0.0, 0.75 * information + 0.15 * remaining + 0.10 * novelty,
    )))


def _chain_option(
    *, action: str, horizon: str, path: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    joint = _distribution(path.get("joint_probabilities"))
    null = _distribution(path.get("pairwise_markov_null_probabilities"))
    try:
        information = float(path["conditional_mutual_information_bits"])
        completion = float(path["chain_completion_probability"])
        null_completion = float(path["null_chain_completion_probability"])
        members = int(path["ensemble_members"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        not path.get("available") or joint is None or null is None
        or not all(math.isfinite(value) for value in (
            information, completion, null_completion,
        ))
        or information < MINIMUM_CONDITIONAL_INFORMATION_BITS
        or members < 2 or not _evidence_is_valid(evidence)
    ):
        return None
    payload = {
        "version": PREDICTIVE_MECHANISM_CHAIN_VERSION,
        "action": str(action), "horizon": str(horizon),
        "first_event": str(path["first_event"]),
        "mediator_event": str(path["mediator_event"]),
        "outcome_event": str(path["outcome_event"]),
        "joint_probabilities": joint,
        "pairwise_markov_null_probabilities": null,
        "conditional_mutual_information_bits": information,
        "chain_completion_probability": completion,
        "null_chain_completion_probability": null_completion,
        "ensemble_members": members,
        "cross_match_evidence": evidence,
        "chain_priority": _chain_priority(information, evidence),
        "falsification_statistic": (
            "categorical_log_likelihood_ratio_vs_pairwise_markov_null"
        ),
        "higher_order_dependence_only": True,
        "selected_after_action_freeze": True,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "chain_id": _digest(payload, "predictive-mechanism-chain-option:"),
    }


def _option_is_valid(option: Any) -> bool:
    if not isinstance(option, dict):
        return False
    payload = dict(option)
    chain_id = payload.pop("chain_id", None)
    joint = _distribution(option.get("joint_probabilities"))
    null = _distribution(option.get("pairwise_markov_null_probabilities"))
    evidence = option.get("cross_match_evidence") or {}
    try:
        information = float(option["conditional_mutual_information_bits"])
        completion = float(option["chain_completion_probability"])
        null_completion = float(option["null_chain_completion_probability"])
        priority = float(option["chain_priority"])
        members = int(option["ensemble_members"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    expected_information = (
        sum(joint[key] * math.log(joint[key] / null[key]) for key in _CELLS)
        / math.log(2.0)
        if joint is not None and null is not None else -1.0
    )
    return bool(
        chain_id == _digest(payload, "predictive-mechanism-chain-option:")
        and option.get("action") in {"hold", "pass", "cross", "shot"}
        and bool(option.get("horizon"))
        and len({
            option.get("first_event"), option.get("mediator_event"),
            option.get("outcome_event"),
        }) == 3
        and joint is not None and null is not None
        and math.isfinite(information)
        and information >= MINIMUM_CONDITIONAL_INFORMATION_BITS
        and abs(information - expected_information) <= 1e-9
        and abs(completion - joint["a1_b1_c1"]) <= 1e-12
        and abs(null_completion - null["a1_b1_c1"]) <= 1e-12
        and members >= 2 and _evidence_is_valid(evidence)
        and math.isfinite(priority)
        and abs(priority - _chain_priority(information, evidence)) <= 1e-9
        and option.get("falsification_statistic")
        == "categorical_log_likelihood_ratio_vs_pairwise_markov_null"
        and option.get("higher_order_dependence_only") is True
        and option.get("selected_after_action_freeze") is True
        and option.get("can_change_current_action") is False
        and option.get("can_change_tactical_controls") is False
        and option.get("can_schedule_future_action") is False
        and option.get("can_update_world_model") is False
        and option.get("causal_interpretation") is False
    )


def build_predictive_mechanism_chain_design(
    packet: dict[str, Any], *, evidence_memory: Any = None,
) -> dict[str, Any]:
    candidates = []
    for candidate in packet.get("candidates") or []:
        action = str(candidate.get("action", ""))
        for horizon, prediction in (
            candidate.get("multi_horizon_predictions") or {}
        ).items():
            paths = ((prediction.get("state_scales") or {}).get(
                "predictive_mechanism_paths"
            ) or {})
            for path in paths.values():
                if not isinstance(path, dict):
                    continue
                evidence = _inactive_evidence(
                    action=action, horizon=str(horizon),
                    first_event=str(path.get("first_event", "")),
                    mediator_event=str(path.get("mediator_event", "")),
                    outcome_event=str(path.get("outcome_event", "")),
                    checkpoint_signature=str(packet.get(
                        "checkpoint_signature", "",
                    )), environment_signature=str(packet.get(
                        "environment_signature", "",
                    )),
                )
                provider = getattr(evidence_memory, "evidence", None)
                if callable(provider):
                    try:
                        candidate_evidence = provider(
                            action=action, horizon=str(horizon),
                            first_event=str(path.get("first_event", "")),
                            mediator_event=str(path.get("mediator_event", "")),
                            outcome_event=str(path.get("outcome_event", "")),
                            checkpoint_signature=str(packet.get(
                                "checkpoint_signature", "",
                            )), environment_signature=str(packet.get(
                                "environment_signature", "",
                            )),
                        )
                    except (
                        AttributeError, KeyError, TypeError, ValueError,
                        OverflowError,
                    ):
                        candidate_evidence = None
                    if _evidence_is_valid(candidate_evidence):
                        evidence = candidate_evidence
                option = _chain_option(
                    action=action, horizon=str(horizon), path=path,
                    evidence=evidence,
                )
                if option is not None:
                    candidates.append(option)
    candidates.sort(key=lambda row: (
        -float(row["chain_priority"]),
        -float(row["conditional_mutual_information_bits"]),
        str(row["action"]), str(row["horizon"]), str(row["chain_id"]),
    ))
    active_statuses = {"start", "continue", "incompatible_checkpoint_or_environment"}
    options = [
        row for row in candidates
        if row["cross_match_evidence"]["status"] in active_statuses
    ][:MAXIMUM_CHAIN_OPTIONS]
    resolved = [
        row for row in candidates
        if row["cross_match_evidence"]["status"] not in active_statuses
    ][:MAXIMUM_CHAIN_OPTIONS]
    summary_builder = getattr(evidence_memory, "summary", None)
    raw_summary = summary_builder() if callable(summary_builder) else {}
    memory_summary = {
        key: raw_summary[key] for key in (
            "version", "compatible_matches", "continuing_profiles",
            "retained_profiles", "eliminated_profiles",
            "retired_inconclusive_profiles", "memory_digest", "reason",
        ) if key in raw_summary
    }
    payload = {
        "version": PREDICTIVE_MECHANISM_CHAIN_VERSION,
        "available": bool(options),
        "reason": (
            "higher_order_predictive_chains_available" if options
            else "no_unresolved_non_markov_chain"
        ),
        "team_id": str(packet.get("team_id", "")),
        "checkpoint_signature": str(packet.get("checkpoint_signature", "")),
        "environment_signature": str(packet.get("environment_signature", "")),
        "options": options, "resolved_chains": resolved,
        "recommended_chain_id": options[0]["chain_id"] if options else "none",
        "maximum_options": MAXIMUM_CHAIN_OPTIONS,
        "chain_memory_summary": memory_summary,
        "selection_scope": "post_action_explanation_only",
        "null_model": "pairwise_first_order_markov_factorization",
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "design_digest": _digest(payload, "predictive-mechanism-chain-design:"),
    }


def predictive_mechanism_chain_design_is_valid(design: Any) -> bool:
    if not isinstance(design, dict):
        return False
    payload = dict(design)
    digest = payload.pop("design_digest", None)
    options = design.get("options") or []
    resolved = design.get("resolved_chains") or []
    return bool(
        digest == _digest(payload, "predictive-mechanism-chain-design:")
        and design.get("version") == PREDICTIVE_MECHANISM_CHAIN_VERSION
        and bool(design.get("available")) == bool(options)
        and len(options) <= MAXIMUM_CHAIN_OPTIONS
        and len(resolved) <= MAXIMUM_CHAIN_OPTIONS
        and all(_option_is_valid(row) for row in options + resolved)
        and len({row["chain_id"] for row in options + resolved})
        == len(options) + len(resolved)
        and design.get("recommended_chain_id")
        == (options[0]["chain_id"] if options else "none")
        and design.get("maximum_options") == MAXIMUM_CHAIN_OPTIONS
        and design.get("selection_scope") == "post_action_explanation_only"
        and design.get("null_model")
        == "pairwise_first_order_markov_factorization"
        and design.get("can_change_current_action") is False
        and design.get("can_change_tactical_controls") is False
        and design.get("can_schedule_future_action") is False
        and design.get("can_update_world_model") is False
        and design.get("causal_interpretation") is False
    )


def validate_llm_predictive_mechanism_chain(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    chain_id = str(raw.get("chain_id", ""))[:96]
    try:
        confidence = float(raw.get("confidence", 0.0))
        completion_probability = float(raw.get(
            "chain_completion_probability", raw.get(
                "completion_probability", float("nan"),
            ),
        ))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not chain_id or not math.isfinite(confidence)
        or not 0.5 <= confidence <= 1.0
        or not math.isfinite(completion_probability)
        or not 0.01 <= completion_probability <= 0.99
    ):
        return None
    return {
        "chain_id": chain_id, "confidence": confidence,
        "chain_completion_probability": completion_probability,
        "mediator_statement": str(raw.get("mediator_statement", ""))[:280],
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def _inactive_fusion(
    *, chain: dict[str, Any], llm_signature: str,
    checkpoint_signature: str, environment_signature: str,
) -> dict[str, Any]:
    payload = {
        "version": 1, "active": False, "llm_weight": 0.0,
        "reason": "no_match_held_out_chain_forecast_gain",
        "profile_matches": 0, "training_matches": 0,
        "validation_matches": 0, "validation_skill": 0.0,
        "authority_cap": 0.35,
        "profile_key": "|".join(str(chain[key]) for key in (
            "action", "horizon", "first_event", "mediator_event",
            "outcome_event",
        )),
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "llm_signature": str(llm_signature),
        "can_change_current_action": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "contract_digest": _digest(payload, "predictive-chain-fusion-contract:"),
    }


def _fusion_is_valid(contract: Any) -> bool:
    if not isinstance(contract, dict):
        return False
    payload = dict(contract)
    digest = payload.pop("contract_digest", None)
    try:
        weight = float(contract["llm_weight"])
        matches = int(contract["profile_matches"])
        training = int(contract["training_matches"])
        validation = int(contract["validation_matches"])
        skill = float(contract["validation_skill"])
        cap = float(contract["authority_cap"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    active = bool(contract.get("active"))
    return bool(
        digest == _digest(payload, "predictive-chain-fusion-contract:")
        and contract.get("version") == 1
        and all(math.isfinite(value) for value in (weight, skill, cap))
        and matches >= 0 and training >= 0 and validation >= 0
        and 0.0 <= weight <= cap <= 0.35
        and (not active or (
            training >= 2 and validation >= 2 and skill >= 0.02
            and weight > 0.0
        ))
        and (active or weight == 0.0)
        and bool(contract.get("profile_key"))
        and bool(contract.get("checkpoint_signature"))
        and bool(contract.get("environment_signature"))
        and bool(contract.get("llm_signature"))
        and contract.get("can_change_current_action") is False
        and contract.get("can_update_world_model") is False
        and contract.get("causal_interpretation") is False
    )


def evaluate_llm_predictive_mechanism_chain(
    packet: dict[str, Any], raw: Any, *, selected_action: str,
    selected_after_action_freeze: bool, forecast_memory: Any = None,
    llm_signature: str = "llm-predictive-chain-unspecified",
) -> dict[str, Any]:
    selection = validate_llm_predictive_mechanism_chain(raw)
    design = packet.get("predictive_mechanism_chain_design") or {}
    base = {
        "version": PREDICTIVE_MECHANISM_CHAIN_VERSION, "accepted": False,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "can_update_world_model": False, "causal_interpretation": False,
    }
    if selection is None:
        return {**base, "reason": "missing_or_invalid_chain_selection"}
    if not selected_after_action_freeze or not predictive_mechanism_chain_design_is_valid(design):
        return {**base, "reason": "post_action_valid_chain_design_required"}
    option = next((
        row for row in design["options"] if row["chain_id"] == selection["chain_id"]
    ), None)
    if option is None:
        return {**base, "reason": "unknown_predictive_mechanism_chain"}
    if str(option["action"]) != str(selected_action).lower():
        return {**base, "reason": "frozen_action_chain_mismatch"}
    fusion = _inactive_fusion(
        chain=option, llm_signature=str(llm_signature),
        checkpoint_signature=str(design["checkpoint_signature"]),
        environment_signature=str(design["environment_signature"]),
    )
    provider = getattr(forecast_memory, "authority", None)
    if callable(provider):
        try:
            candidate_fusion = provider(
                action=str(option["action"]), horizon=str(option["horizon"]),
                first_event=str(option["first_event"]),
                mediator_event=str(option["mediator_event"]),
                outcome_event=str(option["outcome_event"]),
                checkpoint_signature=str(design["checkpoint_signature"]),
                environment_signature=str(design["environment_signature"]),
                llm_signature=str(llm_signature),
            )
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            candidate_fusion = None
        if _fusion_is_valid(candidate_fusion):
            fusion = candidate_fusion
    world_probability = float(option["chain_completion_probability"])
    llm_probability = float(selection["chain_completion_probability"])
    llm_weight = float(fusion["llm_weight"])
    fused_probability = float(
        world_probability + llm_weight * (llm_probability - world_probability)
    )
    payload = {
        **base, "accepted": True,
        "reason": "engine_grounded_higher_order_chain_articulated",
        "selection": selection, "chain": option,
        "source_design_digest": design["design_digest"],
        "team_id": design["team_id"],
        "checkpoint_signature": design["checkpoint_signature"],
        "environment_signature": design["environment_signature"],
        "selected_after_action_freeze": True,
        "higher_order_dependence_only": True,
        "llm_signature": str(llm_signature),
        "world_model_chain_completion_probability": world_probability,
        "llm_chain_completion_probability": llm_probability,
        "fused_chain_completion_probability": fused_probability,
        "chain_forecast_fusion": fusion,
        "world_model_prediction_mutated": False,
    }
    return {
        **payload,
        "audit_digest": _digest(payload, "predictive-mechanism-chain-audit:"),
    }


def predictive_mechanism_chain_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    selection = validate_llm_predictive_mechanism_chain(audit.get("selection"))
    chain = audit.get("chain") or {}
    return bool(
        digest == _digest(payload, "predictive-mechanism-chain-audit:")
        and selection is not None
        and selection["chain_id"] == chain.get("chain_id")
        and _option_is_valid(chain)
        and audit.get("selected_after_action_freeze") is True
        and audit.get("higher_order_dependence_only") is True
        and _fusion_is_valid(audit.get("chain_forecast_fusion"))
        and bool(audit.get("llm_signature"))
        and abs(float(audit.get(
            "world_model_chain_completion_probability", -1.0,
        )) - float(chain["chain_completion_probability"])) <= 1e-12
        and abs(float(audit.get(
            "llm_chain_completion_probability", -1.0,
        )) - float(selection["chain_completion_probability"])) <= 1e-12
        and abs(float(audit.get(
            "fused_chain_completion_probability", -1.0,
        )) - (
            float(chain["chain_completion_probability"])
            + float(audit["chain_forecast_fusion"]["llm_weight"])
            * (float(selection["chain_completion_probability"])
               - float(chain["chain_completion_probability"]))
        )) <= 1e-12
        and audit.get("world_model_prediction_mutated") is False
        and audit.get("can_change_current_action") is False
        and audit.get("can_change_tactical_controls") is False
        and audit.get("can_schedule_future_action") is False
        and audit.get("can_update_world_model") is False
        and audit.get("causal_interpretation") is False
    )
