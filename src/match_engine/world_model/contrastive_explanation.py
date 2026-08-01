"""Model-checked contrastive explanations for LLM coach decisions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.opponent_contract import tactic_feature_vector
from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS


CONTRASTIVE_EXPLANATION_VERSION = 1
CONTEXT_FACTORS = (
    "score_context",
    "match_phase",
    "own_tactics",
    "opponent_tactics",
    "crowd_context",
)
CLAIMED_EFFECTS = ("supports_selected", "opposes_selected")
_HORIZON_PATTERN = re.compile(r"^(transition|\d+(?:\.\d+)?s)$")


def llm_contrastive_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": CONTRASTIVE_EXPLANATION_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "model_checked_contrastive_action_explanation",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-contrastive:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_contrastive_claim(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    selected = str(raw.get("selected_action", "")).strip().lower()
    alternative = str(raw.get("alternative_action", "")).strip().lower()
    horizon = str(raw.get("horizon", "")).strip().lower()
    factor = str(raw.get("factor", "")).strip().lower()
    effect = str(raw.get("effect", "")).strip().lower()
    if (
        selected not in RESPONSE_ACTIONS
        or alternative not in RESPONSE_ACTIONS
        or alternative == selected
        or not _HORIZON_PATTERN.fullmatch(horizon)
        or factor not in CONTEXT_FACTORS
        or effect not in CLAIMED_EFFECTS
    ):
        return None
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
        return None
    return {
        "selected_action": selected,
        "alternative_action": alternative,
        "horizon": horizon,
        "factor": factor,
        "effect": effect,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:280],
    }


def _neutralize_context(
    observation: np.ndarray,
    *,
    factor: str,
    attacking_home: bool,
) -> tuple[np.ndarray, list[int], float]:
    neutral = np.asarray(observation, dtype=np.float32).copy()
    if factor == "score_context":
        indices = [204, 205, 206]
        neutral[indices] = [0.0, 0.0, 0.5]
    elif factor == "match_phase":
        indices = [207]
        neutral[207] = 0.5
    elif factor == "crowd_context":
        indices = [208]
        neutral[208] = 0.5
    elif factor in {"own_tactics", "opponent_tactics"}:
        own_start = 298 if attacking_home else 302
        opponent_start = 302 if attacking_home else 298
        start = own_start if factor == "own_tactics" else opponent_start
        indices = list(range(start, start + 4))
        neutral[indices] = tactic_feature_vector("balanced")
    else:
        raise ValueError("unsupported contrastive context factor")
    magnitude = float(np.max(np.abs(
        neutral[indices] - np.asarray(observation, dtype=np.float32)[indices]
    )))
    return neutral, indices, magnitude


def _planning_gate(runtime, *, rollout_steps: int) -> dict[str, Any]:
    if rollout_steps == 1:
        return {
            "active": True,
            "authority": 1.0,
            "reason": "one_step_runtime_quality_checked_per_action",
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


def evaluate_llm_contrastive_claim(
    runtime,
    state,
    packet: dict[str, Any],
    raw_claim: Any,
    *,
    team_id: str,
    selected_action: str,
    contrastive_signature: str = "contrastive-contract-unspecified",
    uncertainty_penalty: float = 0.25,
    minimum_directional_effect: float = 0.005,
    max_member_trajectory_paths: int = 64,
) -> dict[str, Any]:
    """Test one LLM explanation by neutralizing only its declared context."""
    claim = validate_llm_contrastive_claim(raw_claim)
    base = {
        "version": CONTRASTIVE_EXPLANATION_VERSION,
        "accepted": False,
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "world_model_prediction_mutated": False,
        "causal_interpretation": False,
        "contrastive_signature": str(contrastive_signature),
    }
    if claim is None:
        return {**base, "reason": "missing_or_invalid_contrastive_claim"}
    if claim["selected_action"] != str(selected_action).lower():
        return {
            **base,
            "reason": "contrastive_selected_action_must_equal_plan",
            "claim": claim,
        }
    if not packet.get("available"):
        return {
            **base,
            "reason": "contrastive_world_model_quality_gate_closed",
            "claim": claim,
        }
    candidates = {
        str(item.get("action", "")).lower(): item
        for item in packet.get("candidates") or []
    }
    selected = candidates.get(claim["selected_action"])
    alternative = candidates.get(claim["alternative_action"])
    if selected is None or alternative is None:
        return {
            **base,
            "reason": "contrastive_actions_not_evaluated",
            "claim": claim,
        }
    selected_prediction = (
        selected.get("multi_horizon_predictions") or {}
    ).get(claim["horizon"])
    alternative_prediction = (
        alternative.get("multi_horizon_predictions") or {}
    ).get(claim["horizon"])
    if not isinstance(selected_prediction, dict) or not isinstance(
        alternative_prediction, dict,
    ):
        return {
            **base,
            "reason": "contrastive_horizon_not_evaluated",
            "claim": claim,
        }
    selected_steps = int(selected_prediction.get("rollout_steps", 1))
    alternative_steps = int(alternative_prediction.get("rollout_steps", 1))
    if selected_steps != alternative_steps or selected_steps not in {1, 2}:
        return {
            **base,
            "reason": "contrastive_rollout_depth_not_validated",
            "claim": claim,
        }
    selected_horizon_s = max(0.1, float(selected_prediction.get(
        "horizon_s", packet.get("horizon_s", 10.0),
    )))
    alternative_horizon_s = max(0.1, float(alternative_prediction.get(
        "horizon_s", packet.get("horizon_s", 10.0),
    )))
    if abs(selected_horizon_s - alternative_horizon_s) > 1e-6:
        return {
            **base,
            "reason": "contrastive_horizon_contract_mismatch",
            "claim": claim,
        }
    planning_gate = _planning_gate(runtime, rollout_steps=selected_steps)
    if not planning_gate.get("active"):
        return {
            **base,
            "reason": "contrastive_planning_gate_closed",
            "claim": claim,
            "planning_gate": planning_gate,
        }
    predictor = getattr(runtime, "predict_policy_utility", None)
    if not callable(predictor):
        return {
            **base,
            "reason": "runtime_has_no_policy_utility_predictor",
            "claim": claim,
        }
    attacking_home = str(team_id) == str(state.home.team_id)
    try:
        observation = np.asarray(runtime.encode_state(
            state, attacking_home=attacking_home,
        ), dtype=np.float32)
        neutral, masked_indices, intervention_magnitude = _neutralize_context(
            observation,
            factor=claim["factor"],
            attacking_home=attacking_home,
        )
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return {
            **base,
            "reason": "contrastive_context_intervention_failed",
            "claim": claim,
        }
    if intervention_magnitude < 1e-6:
        return {
            **base,
            "reason": "contrastive_factor_already_neutral",
            "claim": claim,
        }
    transition_members = max(1, int(getattr(
        getattr(runtime, "model", None), "transition_member_count", 1,
    )))
    trajectory_paths = 4 * selected_steps * transition_members
    trajectory_budget = max(
        0, min(64, int(max_member_trajectory_paths)),
    )
    if trajectory_paths > trajectory_budget:
        return {
            **base,
            "reason": "contrastive_trajectory_budget_insufficient",
            "claim": claim,
            "trajectory_member_paths_required": trajectory_paths,
            "trajectory_member_path_budget": trajectory_budget,
        }
    horizon_s = selected_horizon_s
    segment_horizon_s = horizon_s / selected_steps
    values: dict[str, dict[str, float]] = {"observed": {}, "neutralized": {}}
    uncertainty: dict[str, dict[str, float]] = {
        "observed": {}, "neutralized": {},
    }
    quality: dict[str, dict[str, float]] = {
        "observed": {}, "neutralized": {},
    }
    try:
        for context_name, context_observation in (
            ("observed", observation), ("neutralized", neutral),
        ):
            for candidate in (selected, alternative):
                action_name = str(candidate["action"])
                confidence_kind = "shot" if action_name == "shot" else "pass"
                confidence = float(runtime.planner_confidence(
                    context_observation, kind=confidence_kind,
                ))
                quality[context_name][action_name] = confidence
                if not np.isfinite(confidence) or confidence <= 0.0:
                    raise RuntimeError("contrastive quality gate closed")
                action = encode_high_level_action(
                    action_name,
                    target=np.asarray(candidate["target"], dtype=np.float32),
                    horizon_s=segment_horizon_s,
                )
                action[13] = float(candidate.get("physics_prior", 0.5))
                forecast = predictor(
                    context_observation,
                    action,
                    action_kind=action_name,
                    attacking_home=attacking_home,
                    horizon_s=horizon_s,
                )
                utility = float(forecast["policy_utility"])
                forecast_uncertainty = float(forecast.get("uncertainty", 1.0))
                turnover = 1.0 - float(forecast.get(
                    "retention_probability", 0.0,
                ))
                score = (
                    utility
                    - float(uncertainty_penalty) * forecast_uncertainty
                    - 0.15 * turnover
                )
                if not all(np.isfinite(item) for item in (
                    utility, forecast_uncertainty, turnover, score,
                )):
                    raise ValueError("non-finite contrastive forecast")
                values[context_name][action_name] = float(score)
                uncertainty[context_name][action_name] = float(
                    np.clip(forecast_uncertainty, 0.0, 1.0)
                )
    except (KeyError, RuntimeError, TypeError, ValueError, AttributeError):
        return {
            **base,
            "reason": "contrastive_action_evaluation_failed",
            "claim": claim,
            "planning_gate": planning_gate,
            "trajectory_member_paths": trajectory_paths,
            "trajectory_member_path_budget": trajectory_budget,
        }
    observed_margin = (
        values["observed"][claim["selected_action"]]
        - values["observed"][claim["alternative_action"]]
    )
    neutralized_margin = (
        values["neutralized"][claim["selected_action"]]
        - values["neutralized"][claim["alternative_action"]]
    )
    factor_effect = observed_margin - neutralized_margin
    claimed_sign = 1.0 if claim["effect"] == "supports_selected" else -1.0
    directional_effect = claimed_sign * factor_effect
    faithful = directional_effect >= max(
        0.0, float(minimum_directional_effect),
    )
    audit = {
        **base,
        "accepted": True,
        "reason": "shadow_model_checked_contrastive_explanation",
        "claim": claim,
        "checkpoint_signature": str(packet.get(
            "checkpoint_signature", "runtime_unspecified",
        )),
        "environment_signature": str(packet.get(
            "environment_signature", "environment_unspecified",
        )),
        "planning_gate": planning_gate,
        "masked_feature_indices": masked_indices,
        "masked_feature_count": len(masked_indices),
        "neutral_reference": "simulator_schema_neutral_context",
        "context_intervention_linf": intervention_magnitude,
        "risk_adjusted_values": values,
        "probe_value_source": (
            "runtime_policy_utility_without_cross_match_residual_memory"
        ),
        "uncertainties": uncertainty,
        "quality_factors": quality,
        "observed_selected_vs_alternative_margin": observed_margin,
        "neutralized_selected_vs_alternative_margin": neutralized_margin,
        "declared_factor_effect_on_margin": factor_effect,
        "claimed_directional_effect": directional_effect,
        "minimum_directional_effect": float(minimum_directional_effect),
        "directionally_faithful": faithful,
        "model_calls": 4,
        "trajectory_member_paths": trajectory_paths,
        "trajectory_member_path_budget": trajectory_budget,
        "can_change_selected_action": False,
        "can_update_world_model": False,
    }
    packet["llm_contrastive_explanation_audit"] = audit
    return audit


def contrastive_explanation_diagnostics(
    match_logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    match_rows = []
    for payload in match_logs:
        rows = []
        records = (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        )
        for record in records:
            audit = record.get("llm_contrastive_explanation_context")
            if isinstance(audit, dict) and audit.get("accepted"):
                rows.append(audit)
        if rows:
            match_rows.append(rows)
    malformed = 0
    valid = []
    for rows in match_rows:
        for row in rows:
            try:
                version = int(row["version"])
                effect = float(row["declared_factor_effect_on_margin"])
                directional = float(row["claimed_directional_effect"])
                minimum_effect = float(row["minimum_directional_effect"])
                paths = int(row["trajectory_member_paths"])
                budget = int(row["trajectory_member_path_budget"])
            except (KeyError, TypeError, ValueError, OverflowError):
                malformed += 1
                continue
            claim = row.get("claim") or {}
            validated_claim = validate_llm_contrastive_claim(claim)
            sign = 1.0 if claim.get("effect") == "supports_selected" else -1.0
            if (
                version != CONTRASTIVE_EXPLANATION_VERSION
                or validated_claim is None
                or not np.isfinite(effect)
                or not np.isfinite(directional)
                or not np.isfinite(minimum_effect)
                or minimum_effect < 0.0
                or abs(directional - sign * effect) > 1e-6
                or bool(row.get("directionally_faithful"))
                != bool(directional >= minimum_effect)
                or row.get("probe_value_source") != (
                    "runtime_policy_utility_without_cross_match_residual_memory"
                )
                or paths <= 0 or budget <= 0 or paths > budget
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
    match_faithfulness = [
        float(np.mean([
            float(bool(row.get("directionally_faithful"))) for row in rows
        ]))
        for rows in valid_match_rows
    ]
    match_effects = [
        float(np.mean([
            abs(float(row["declared_factor_effect_on_margin"])) for row in rows
        ]))
        for rows in valid_match_rows
    ]
    signatures = sorted({
        str(row.get("contrastive_signature", "")) for row in valid
    })
    scopes = sorted({(
        str(row.get("checkpoint_signature", "")),
        str(row.get("environment_signature", "")),
    ) for row in valid})
    unspecified = {
        "", "contrastive-contract-unspecified",
        "runtime_unspecified", "environment_unspecified",
    }
    factor_profile = {
        factor: sum(
            1 for row in valid
            if (row.get("claim") or {}).get("factor") == factor
        )
        for factor in CONTEXT_FACTORS
    }
    return {
        "version": CONTRASTIVE_EXPLANATION_VERSION,
        "evaluation_kind": "model_checked_contrastive_explanation",
        "claims": len(valid),
        "matches": len(valid_match_rows),
        "malformed_claim_audits": malformed,
        "match_clustered_directional_faithfulness": (
            float(np.mean(match_faithfulness)) if match_faithfulness else 0.0
        ),
        "match_clustered_mean_absolute_factor_effect": (
            float(np.mean(match_effects)) if match_effects else 0.0
        ),
        "factor_profile": factor_profile,
        "budgets_respected": all(
            int(row["trajectory_member_paths"])
            <= int(row["trajectory_member_path_budget"])
            for row in valid
        ),
        "all_shadow_only": all(bool(row.get("shadow_only")) for row in valid),
        "all_non_controlling": all(
            not bool(row.get("authority_active"))
            and not bool(row.get("policy_mutated"))
            and not bool(row.get("world_model_prediction_mutated"))
            and not bool(row.get("can_change_selected_action"))
            for row in valid
        ),
        "all_non_causal": all(
            not bool(row.get("causal_interpretation")) for row in valid
        ),
        "provenance_compatible": bool(
            len(signatures) == 1 and signatures[0] not in unspecified
            and len(scopes) == 1
            and all(value not in unspecified for value in scopes[0])
        ),
        "contrastive_signatures": signatures,
        "model_policy_scopes": [list(scope) for scope in scopes],
    }
