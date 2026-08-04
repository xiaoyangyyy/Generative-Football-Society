"""Bounded, redundancy-aware portfolios of same-action active probes."""

from __future__ import annotations

import hashlib
from itertools import combinations
import json
import math
from typing import Any

from src.match_engine.world_model.active_probe import (
    active_probe_audit_is_valid,
    active_probe_design_is_valid,
    evaluate_llm_active_probe,
)
from src.match_engine.world_model.policy_experiment import (
    policy_horizon_seconds,
)


ACTIVE_PROBE_PORTFOLIO_VERSION = 1
MAXIMUM_PROBES_PER_PORTFOLIO = 2
ADDITIONAL_OBSERVATION_COST = 0.02


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def _horizon_seconds(horizon: str) -> float:
    return policy_horizon_seconds(str(horizon))


def _redundancy_penalty(probes: list[dict[str, Any]]) -> float:
    if len(probes) < 2:
        return 0.0
    first, second = probes[:2]
    distance = abs(
        _horizon_seconds(str(first["horizon"]))
        - _horizon_seconds(str(second["horizon"]))
    )
    proximity = 1.0 / (1.0 + distance / 60.0)
    endpoint_factor = 1.0 if first["endpoint"] == second["endpoint"] else 0.5
    return float(
        0.30
        * min(
            float(first["experiment_priority"]),
            float(second["experiment_priority"]),
        )
        * proximity
        * endpoint_factor
    )


def _portfolio_option(probes: list[dict[str, Any]]) -> dict[str, Any]:
    redundancy = _redundancy_penalty(probes)
    observation_cost = ADDITIONAL_OBSERVATION_COST * max(0, len(probes) - 1)
    joint_value = max(0.0, sum(
        float(probe["experiment_priority"]) for probe in probes
    ) - redundancy - observation_cost)
    payload = {
        "version": ACTIVE_PROBE_PORTFOLIO_VERSION,
        "action": str(probes[0]["action"]),
        "probe_ids": [str(probe["probe_id"]) for probe in probes],
        "horizons": [str(probe["horizon"]) for probe in probes],
        "probe_priorities": [
            float(probe["experiment_priority"]) for probe in probes
        ],
        "observation_count": len(probes),
        "observation_budget": MAXIMUM_PROBES_PER_PORTFOLIO,
        "redundancy_penalty": redundancy,
        "additional_observation_cost": observation_cost,
        "joint_information_value": joint_value,
        "incremental_action_regret": 0.0,
        "single_realized_action": True,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "portfolio_id": _digest(payload, "active-probe-portfolio-option:"),
    }


def _portfolio_option_is_valid(option: Any) -> bool:
    if not isinstance(option, dict):
        return False
    payload = dict(option)
    portfolio_id = payload.pop("portfolio_id", None)
    try:
        expected_digest = _digest(
            payload, "active-probe-portfolio-option:"
        )
        count = int(option["observation_count"])
        budget = int(option["observation_budget"])
        priorities = [float(value) for value in option["probe_priorities"]]
        redundancy = float(option["redundancy_penalty"])
        observation_cost = float(option["additional_observation_cost"])
        joint_value = float(option["joint_information_value"])
        incremental_regret = float(option["incremental_action_regret"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    probe_ids = option.get("probe_ids") or []
    horizons = option.get("horizons") or []
    if len(horizons) == 2 and len(priorities) == 2:
        try:
            distance = abs(
                _horizon_seconds(str(horizons[0]))
                - _horizon_seconds(str(horizons[1]))
            )
        except (TypeError, ValueError, OverflowError):
            return False
        expected_redundancy = (
            0.30 * min(priorities) / (1.0 + distance / 60.0)
        )
    else:
        expected_redundancy = 0.0
    expected_joint = max(
        0.0, sum(priorities) - redundancy - observation_cost,
    )
    return bool(
        portfolio_id == expected_digest
        and option.get("action") in {"hold", "pass", "cross", "shot"}
        and option.get("version") == ACTIVE_PROBE_PORTFOLIO_VERSION
        and 1 <= count <= budget == MAXIMUM_PROBES_PER_PORTFOLIO
        and len(probe_ids) == len(set(probe_ids)) == count
        and len(horizons) == len(set(horizons)) == count
        and len(priorities) == count
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in priorities)
        and math.isfinite(redundancy) and redundancy >= 0.0
        and abs(redundancy - expected_redundancy) <= 1e-9
        and abs(
            observation_cost
            - ADDITIONAL_OBSERVATION_COST * max(0, count - 1)
        ) <= 1e-12
        and math.isfinite(joint_value)
        and abs(joint_value - expected_joint) <= 1e-9
        and incremental_regret == 0.0
        and option.get("single_realized_action") is True
        and option.get("can_change_current_action") is False
        and option.get("can_change_tactical_controls") is False
        and option.get("can_schedule_future_action") is False
        and option.get("causal_interpretation") is False
    )


def build_active_probe_portfolio_design(
    packet: dict[str, Any],
) -> dict[str, Any]:
    probe_design = packet.get("active_probe_design") or {}
    probes = list(probe_design.get("options") or [])
    options = []
    if active_probe_design_is_valid(probe_design):
        for count in range(1, min(
            MAXIMUM_PROBES_PER_PORTFOLIO, len(probes),
        ) + 1):
            for selected in combinations(probes, count):
                if len({probe["action"] for probe in selected}) != 1:
                    continue
                option = _portfolio_option(list(selected))
                if count == 2:
                    second_marginal = (
                        option["joint_information_value"]
                        - max(float(probe["experiment_priority"]) for probe in selected)
                    )
                    if second_marginal < 0.02:
                        continue
                options.append(option)
    options.sort(key=lambda option: (
        -float(option["joint_information_value"]),
        int(option["observation_count"]),
        str(option["portfolio_id"]),
    ))
    payload = {
        "version": ACTIVE_PROBE_PORTFOLIO_VERSION,
        "available": bool(options),
        "reason": (
            "bounded_probe_portfolios_available" if options
            else "active_probe_design_unavailable"
        ),
        "team_id": str(packet.get("team_id", "")),
        "checkpoint_signature": str(packet.get("checkpoint_signature", "")),
        "environment_signature": str(packet.get("environment_signature", "")),
        "as_of_t_sec": float((packet.get("decision_context") or {}).get(
            "clock_seconds", 0.0,
        )),
        "source_probe_design_digest": str(probe_design.get(
            "design_digest", "",
        )),
        "options": options,
        "recommended_portfolio_id": (
            options[0]["portfolio_id"] if options else "none"
        ),
        "maximum_probes_per_portfolio": MAXIMUM_PROBES_PER_PORTFOLIO,
        "execution_scope": "one_existing_randomized_action_multiple_observations",
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "design_digest": _digest(payload, "active-probe-portfolio-design:"),
    }


def active_probe_portfolio_design_is_valid(
    design: Any,
    probe_design: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(design, dict):
        return False
    payload = dict(design)
    digest = payload.pop("design_digest", None)
    options = design.get("options") or []
    try:
        expected_digest = _digest(
            payload, "active-probe-portfolio-design:"
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if not bool(
        digest == expected_digest
        and design.get("version") == ACTIVE_PROBE_PORTFOLIO_VERSION
        and bool(design.get("available")) == bool(options)
        and all(_portfolio_option_is_valid(option) for option in options)
        and len({option.get("portfolio_id") for option in options}) == len(options)
        and design.get("recommended_portfolio_id")
        == (options[0]["portfolio_id"] if options else "none")
        and design.get("maximum_probes_per_portfolio")
        == MAXIMUM_PROBES_PER_PORTFOLIO
        and design.get("execution_scope")
        == "one_existing_randomized_action_multiple_observations"
        and design.get("can_change_current_action") is False
        and design.get("can_change_tactical_controls") is False
        and design.get("can_schedule_future_action") is False
        and design.get("causal_interpretation") is False
    ):
        return False
    if probe_design is None:
        return True
    if (
        not active_probe_design_is_valid(probe_design)
        or design.get("source_probe_design_digest")
        != probe_design.get("design_digest")
    ):
        return False
    probes = {
        probe["probe_id"]: probe for probe in probe_design["options"]
    }
    for option in options:
        try:
            selected = [probes[probe_id] for probe_id in option["probe_ids"]]
        except KeyError:
            return False
        expected = _portfolio_option(selected)
        if expected != option:
            return False
    return True


def validate_llm_active_probe_portfolio(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    portfolio_id = str(raw.get("portfolio_id", ""))[:96]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not portfolio_id or not math.isfinite(confidence)
        or not 0.5 <= confidence <= 1.0
    ):
        return None
    return {
        "portfolio_id": portfolio_id,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:240],
    }


def evaluate_llm_active_probe_portfolio(
    packet: dict[str, Any],
    raw: Any,
    *,
    selected_action: str,
    decision_mode: str,
    selected_after_action_freeze: bool,
) -> dict[str, Any]:
    selection = validate_llm_active_probe_portfolio(raw)
    design = packet.get("active_probe_portfolio_design") or {}
    probe_design = packet.get("active_probe_design") or {}
    base = {
        "version": ACTIVE_PROBE_PORTFOLIO_VERSION,
        "accepted": False,
        "can_change_current_action": False,
        "can_change_tactical_controls": False,
        "can_schedule_future_action": False,
        "causal_interpretation": False,
    }
    if selection is None:
        return {**base, "reason": "missing_or_invalid_probe_portfolio"}
    if (
        not selected_after_action_freeze
        or not active_probe_portfolio_design_is_valid(design, probe_design)
    ):
        return {**base, "reason": "post_action_valid_portfolio_required"}
    option = next((
        row for row in design["options"]
        if row["portfolio_id"] == selection["portfolio_id"]
    ), None)
    if option is None:
        return {**base, "reason": "unknown_probe_portfolio"}
    if (
        str(selected_action) != str(option["action"])
        or str(decision_mode) != "explore"
    ):
        return {**base, "reason": "frozen_exploration_action_mismatch"}
    probe_audits = []
    for probe_id in option["probe_ids"]:
        audit = evaluate_llm_active_probe(
            packet,
            {
                "probe_id": probe_id,
                "confidence": selection["confidence"],
                "rationale": selection["rationale"],
            },
            selected_action=selected_action,
            decision_mode=decision_mode,
            selected_after_action_freeze=True,
        )
        if not active_probe_audit_is_valid(audit):
            return {**base, "reason": "portfolio_probe_validation_failed"}
        probe_audits.append(audit)
    payload = {
        **base,
        "accepted": True,
        "reason": "bounded_multi_horizon_probe_portfolio_registered",
        "selection": selection,
        "portfolio": option,
        "probe_audits": probe_audits,
        "source_design_digest": design["design_digest"],
        "team_id": design["team_id"],
        "checkpoint_signature": design["checkpoint_signature"],
        "environment_signature": design["environment_signature"],
        "selected_after_action_freeze": True,
    }
    return {
        **payload,
        "audit_digest": _digest(payload, "active-probe-portfolio-audit:"),
    }


def active_probe_portfolio_audit_is_valid(audit: Any) -> bool:
    if not isinstance(audit, dict) or not audit.get("accepted"):
        return False
    payload = dict(audit)
    digest = payload.pop("audit_digest", None)
    try:
        expected_digest = _digest(
            payload, "active-probe-portfolio-audit:"
        )
    except (TypeError, ValueError, OverflowError):
        return False
    selection = validate_llm_active_probe_portfolio(audit.get("selection"))
    portfolio = audit.get("portfolio") or {}
    probe_audits = audit.get("probe_audits") or []
    return bool(
        digest == expected_digest
        and selection is not None
        and selection["portfolio_id"] == portfolio.get("portfolio_id")
        and _portfolio_option_is_valid(portfolio)
        and len(probe_audits) == int(portfolio["observation_count"])
        and all(active_probe_audit_is_valid(row) for row in probe_audits)
        and [row["probe"]["probe_id"] for row in probe_audits]
        == portfolio["probe_ids"]
        and [row["probe"]["horizon"] for row in probe_audits]
        == portfolio["horizons"]
        and all(
            row["probe"]["action"] == portfolio["action"]
            and row.get("team_id") == audit.get("team_id")
            and row.get("checkpoint_signature")
            == audit.get("checkpoint_signature")
            and row.get("environment_signature")
            == audit.get("environment_signature")
            for row in probe_audits
        )
        and audit.get("selected_after_action_freeze") is True
        and audit.get("can_change_current_action") is False
        and audit.get("can_change_tactical_controls") is False
        and audit.get("can_schedule_future_action") is False
        and audit.get("causal_interpretation") is False
    )
