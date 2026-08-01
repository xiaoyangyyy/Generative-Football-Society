"""Shadow-only contract for falsifiable LLM semantic event forecasts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS
from src.match_engine.world_model.state_scales import (
    FALSIFIABLE_SEMANTIC_EVENTS,
)


LLM_EVENT_HYPOTHESIS_VERSION = 1
EVENT_EXPECTATIONS = ("occur", "not_occur")
EVIDENCE_SCALES = ("short", "tactical", "strategic")
_HORIZON_PATTERN = re.compile(r"^(transition|\d+(?:\.\d+)?s)$")


def llm_event_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": LLM_EVENT_HYPOTHESIS_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "falsifiable_multiscale_semantic_event_forecast",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-event:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_event_hypothesis(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action", "")).strip().lower()
    horizon = str(raw.get("horizon", "")).strip().lower()
    event = str(raw.get("event", "")).strip().lower()
    expectation = str(raw.get("expectation", "")).strip().lower()
    if (
        action not in RESPONSE_ACTIONS
        or not _HORIZON_PATTERN.fullmatch(horizon)
        or event not in FALSIFIABLE_SEMANTIC_EVENTS
        or expectation not in EVENT_EXPECTATIONS
    ):
        return None
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
        return None
    evidence = raw.get("evidence_scales") or []
    if not isinstance(evidence, list):
        return None
    evidence = list(dict.fromkeys(
        str(item).strip().lower()
        for item in evidence
        if str(item).strip().lower() in EVIDENCE_SCALES
    ))[:3]
    if not evidence:
        return None
    return {
        "action": action,
        "horizon": horizon,
        "event": event,
        "expectation": expectation,
        "confidence": confidence,
        "evidence_scales": evidence,
        "rationale": str(raw.get("rationale", ""))[:280],
    }


def apply_llm_event_hypothesis(
    packet: dict[str, Any],
    raw_hypothesis: Any,
    *,
    selected_action: str,
    event_signature: str = "event-contract-unspecified",
) -> dict[str, Any]:
    """Ground an LLM event claim in a model forecast without changing policy."""
    hypothesis = validate_llm_event_hypothesis(raw_hypothesis)
    if hypothesis is None:
        return {
            "version": LLM_EVENT_HYPOTHESIS_VERSION,
            "accepted": False,
            "reason": "missing_or_invalid_semantic_event_hypothesis",
        }
    if hypothesis["action"] != str(selected_action).lower():
        return {
            "version": LLM_EVENT_HYPOTHESIS_VERSION,
            "accepted": False,
            "reason": "event_action_must_equal_selected_action",
            "hypothesis": hypothesis,
        }
    candidate = next((
        item for item in packet.get("candidates") or []
        if str(item.get("action", "")).lower() == hypothesis["action"]
    ), None)
    predictions = (
        candidate.get("multi_horizon_predictions") if candidate else None
    ) or {}
    prediction = predictions.get(hypothesis["horizon"])
    if not isinstance(prediction, dict):
        return {
            "version": LLM_EVENT_HYPOTHESIS_VERSION,
            "accepted": False,
            "reason": "event_horizon_not_evaluated",
            "hypothesis": hypothesis,
            "available_horizons": sorted(predictions),
        }
    state_scales = prediction.get("state_scales") or {}
    event_probabilities = prediction.get(
        "semantic_event_probabilities"
    ) or {}
    if hypothesis["event"] not in event_probabilities:
        return {
            "version": LLM_EVENT_HYPOTHESIS_VERSION,
            "accepted": False,
            "reason": "event_not_predicted_by_world_model",
            "hypothesis": hypothesis,
            "available_events": sorted(event_probabilities),
        }
    grounded_scales = [
        scale for scale in hypothesis["evidence_scales"]
        if isinstance(state_scales.get(scale), dict)
    ]
    if not grounded_scales:
        return {
            "version": LLM_EVENT_HYPOTHESIS_VERSION,
            "accepted": False,
            "reason": "event_evidence_scale_not_present",
            "hypothesis": hypothesis,
        }
    try:
        raw_model_probability = float(
            event_probabilities[hypothesis["event"]]
        )
    except (TypeError, ValueError):
        raw_model_probability = float("nan")
    if not np.isfinite(raw_model_probability):
        return {
            "version": LLM_EVENT_HYPOTHESIS_VERSION,
            "accepted": False,
            "reason": "event_probability_not_finite",
            "hypothesis": hypothesis,
        }
    model_probability = float(np.clip(raw_model_probability, 0.0, 1.0))
    fusion = (
        (state_scales.get("semantic_event_fusion") or {}).get(
            hypothesis["event"]
        ) or {}
    )
    model_components = {
        "projection_probability": float(np.clip(float(fusion.get(
            "projection_probability", model_probability,
        )), 0.0, 1.0)),
        "learned_probability": float(np.clip(float(fusion.get(
            "learned_probability", 0.5,
        )), 0.0, 1.0)),
        "fused_probability": model_probability,
        "gate": dict(fusion.get("gate") or {
            "active": False,
            "authority": 0.0,
            "reason": "projection_only",
        }),
    }
    llm_probability = (
        hypothesis["confidence"]
        if hypothesis["expectation"] == "occur"
        else 1.0 - hypothesis["confidence"]
    )
    audit = {
        "version": LLM_EVENT_HYPOTHESIS_VERSION,
        "accepted": True,
        "reason": "shadow_semantic_event_hypothesis",
        "hypothesis": hypothesis,
        "grounded_evidence_scales": grounded_scales,
        "world_model_event_probability": model_probability,
        "world_model_event_components": model_components,
        "llm_event_probability": float(llm_probability),
        "probability_disagreement": float(abs(
            llm_probability - model_probability
        )),
        "same_binary_expectation": bool(
            (llm_probability >= 0.5) == (model_probability >= 0.5)
        ),
        "authority_active": False,
        "policy_mutated": False,
        "world_model_prediction_mutated": False,
        "can_update_world_model": False,
        "causal_interpretation": False,
        "event_signature": str(event_signature),
    }
    packet["llm_semantic_event_hypothesis_audit"] = audit
    return audit


def observed_semantic_event(
    event: str,
    outcome: dict[str, Any],
    baseline: dict[str, Any],
    *,
    attacking_home: bool,
) -> bool:
    """Resolve a declared event from simulator-native outcome quantities."""
    if event == "retain_possession":
        return bool(outcome["retained_possession"])
    if event == "positive_territorial_shift":
        return float(outcome["progress"]) >= 0.03
    if event == "improve_scoreline":
        return float(outcome["goal_diff_delta"]) >= 0.5
    if event == "enter_final_third":
        initial = float(baseline["ball_x"])
        if not attacking_home:
            initial = 1.0 - initial
        return initial + float(outcome["progress"]) >= 0.67
    raise ValueError("unsupported semantic event")


def score_llm_event_hypothesis(
    context: dict[str, Any],
    outcome: dict[str, Any],
    baseline: dict[str, Any],
    *,
    attacking_home: bool,
    checkpoint_signature: str = "runtime_unspecified",
    environment_signature: str = "environment_unspecified",
) -> dict[str, Any] | None:
    """Compare LLM and neural event probabilities against the same outcome."""
    if not context.get("accepted"):
        return None
    hypothesis = context.get("hypothesis") or {}
    event = str(hypothesis.get("event", ""))
    if event not in FALSIFIABLE_SEMANTIC_EVENTS:
        return None
    observed = observed_semantic_event(
        event, outcome, baseline, attacking_home=attacking_home,
    )
    target = float(observed)
    llm_probability = float(np.clip(
        float(context.get("llm_event_probability", 0.5)), 0.0, 1.0,
    ))
    model_probability = float(np.clip(
        float(context.get("world_model_event_probability", 0.5)), 0.0, 1.0,
    ))
    llm_brier = (llm_probability - target) ** 2
    model_brier = (model_probability - target) ** 2
    components = context.get("world_model_event_components") or {}
    component_scores = {}
    for name in ("projection_probability", "learned_probability"):
        try:
            probability = float(components[name])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(probability):
            probability = float(np.clip(probability, 0.0, 1.0))
            component_scores[name] = probability
            component_scores[name.replace("probability", "brier")] = float(
                (probability - target) ** 2
            )
    component_scores["fused_probability"] = model_probability
    component_scores["fused_brier"] = float(model_brier)
    component_scores["gate"] = dict(components.get("gate") or {})
    return {
        "version": LLM_EVENT_HYPOTHESIS_VERSION,
        "event": event,
        "observed": observed,
        "llm_probability": llm_probability,
        "world_model_probability": model_probability,
        "llm_brier": float(llm_brier),
        "world_model_brier": float(model_brier),
        "world_model_event_components": component_scores,
        "llm_brier_gain_vs_world_model": float(model_brier - llm_brier),
        "shadow_only": True,
        "causal_interpretation": False,
        "event_signature": str(context.get(
            "event_signature", "event-contract-unspecified",
        )),
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
    }


def _calibration_error(
    rows: list[dict[str, Any]], probability_key: str, *, bins: int = 5,
) -> float:
    if not rows:
        return 0.0
    total = float(len(rows))
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            row for row in rows
            if low <= float(row[probability_key])
            and (
                float(row[probability_key]) < high
                or index == bins - 1
            )
        ]
        if not selected:
            continue
        confidence = float(np.mean([
            float(row[probability_key]) for row in selected
        ]))
        frequency = float(np.mean([
            float(bool(row["observed"])) for row in selected
        ]))
        error += len(selected) / total * abs(confidence - frequency)
    return float(error)


def _match_equal_calibration_error(
    match_rows: list[list[dict[str, Any]]], probability_key: str,
) -> float:
    if not match_rows:
        return 0.0
    return float(np.mean([
        _calibration_error(rows, probability_key) for rows in match_rows
    ]))


def _match_equal_field(
    match_rows: list[list[dict[str, Any]]], field: str,
) -> float:
    populated = [rows for rows in match_rows if rows]
    return float(np.mean([
        np.mean([float(row[field]) for row in rows]) for rows in populated
    ])) if populated else 0.0


def semantic_event_diagnostics(match_logs: list[dict[str, Any]]) -> dict[str, Any]:
    """Match-equal audit of paired LLM and neural semantic-event forecasts."""
    match_rows: list[list[dict[str, Any]]] = []
    profile_rows: dict[str, list[dict[str, Any]]] = {}
    malformed_evaluations = 0
    for payload in match_logs:
        records = (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        )
        rows = []
        for record in records:
            outcomes = record.get("multi_horizon_regime_outcomes") or {}
            for horizon, outcome in outcomes.items():
                evaluation = (
                    outcome.get("llm_semantic_event_evaluation")
                    if isinstance(outcome, dict) else None
                )
                if not isinstance(evaluation, dict):
                    continue
                try:
                    numeric = [float(evaluation[key]) for key in (
                        "llm_probability",
                        "world_model_probability",
                        "llm_brier",
                        "world_model_brier",
                    )]
                except (KeyError, TypeError, ValueError):
                    malformed_evaluations += 1
                    continue
                if (
                    not all(np.isfinite(value) for value in numeric)
                    or str(evaluation.get("event", ""))
                    not in FALSIFIABLE_SEMANTIC_EVENTS
                    or not 0.0 <= numeric[0] <= 1.0
                    or not 0.0 <= numeric[1] <= 1.0
                ):
                    malformed_evaluations += 1
                    continue
                row = dict(evaluation)
                row["horizon"] = str(horizon)
                components = row.get("world_model_event_components") or {}
                component_gate = components.get("gate") or {}
                if component_gate.get("active"):
                    required_components = (
                        "projection_brier", "learned_brier", "fused_brier",
                    )
                    try:
                        component_values = [
                            float(components[key]) for key in required_components
                        ]
                    except (KeyError, TypeError, ValueError):
                        malformed_evaluations += 1
                        continue
                    if not all(np.isfinite(value) for value in component_values):
                        malformed_evaluations += 1
                        continue
                    row["_projection_brier"] = component_values[0]
                    row["_learned_brier"] = component_values[1]
                    row["_fused_brier"] = component_values[2]
                rows.append(row)
                key = f"{row.get('event', 'unknown')}|{horizon}"
                profile_rows.setdefault(key, []).append(row)
        if rows:
            match_rows.append(rows)
    flat = [row for rows in match_rows for row in rows]
    learned_match_rows = [
        [
            row for row in rows
            if bool(((row.get("world_model_event_components") or {}).get(
                "gate"
            ) or {}).get("active"))
        ]
        for rows in match_rows
    ]
    learned_match_rows = [rows for rows in learned_match_rows if rows]
    clustered_llm_brier = float(np.mean([
        np.mean([float(row["llm_brier"]) for row in rows])
        for rows in match_rows
    ])) if match_rows else 0.0
    clustered_model_brier = float(np.mean([
        np.mean([float(row["world_model_brier"]) for row in rows])
        for rows in match_rows
    ])) if match_rows else 0.0
    profiles = {
        key: {
            "samples": len(rows),
            "llm_brier": float(np.mean([
                float(row["llm_brier"]) for row in rows
            ])),
            "world_model_brier": float(np.mean([
                float(row["world_model_brier"]) for row in rows
            ])),
        }
        for key, rows in sorted(profile_rows.items())
    }
    event_signatures = sorted({
        str(row.get("event_signature", "event-contract-unspecified"))
        for row in flat
    })
    model_scopes = sorted({
        (
            str(row.get("checkpoint_signature", "runtime_unspecified")),
            str(row.get("environment_signature", "environment_unspecified")),
        )
        for row in flat
    })
    provenance_compatible = bool(
        len(event_signatures) == 1
        and event_signatures[0] != "event-contract-unspecified"
        and len(model_scopes) == 1
        and model_scopes[0][0] != "runtime_unspecified"
        and model_scopes[0][1] != "environment_unspecified"
    )
    return {
        "version": LLM_EVENT_HYPOTHESIS_VERSION,
        "evaluation_kind": "paired_shadow_semantic_event_brier",
        "realized_predictions": len(flat),
        "matches": len(match_rows),
        "malformed_evaluations": malformed_evaluations,
        "match_clustered_llm_brier": clustered_llm_brier,
        "match_clustered_world_model_brier": clustered_model_brier,
        "match_clustered_llm_gain_vs_world_model": float(
            clustered_model_brier - clustered_llm_brier
        ),
        "llm_calibration_error": _match_equal_calibration_error(
            match_rows, "llm_probability",
        ),
        "world_model_calibration_error": _match_equal_calibration_error(
            match_rows, "world_model_probability",
        ),
        "realized_learned_head_predictions": sum(map(
            len, learned_match_rows,
        )),
        "learned_head_matches": len(learned_match_rows),
        "match_clustered_projection_brier": _match_equal_field(
            learned_match_rows, "_projection_brier",
        ),
        "match_clustered_learned_head_brier": _match_equal_field(
            learned_match_rows, "_learned_brier",
        ),
        "match_clustered_fused_event_brier": _match_equal_field(
            learned_match_rows, "_fused_brier",
        ),
        "all_learned_authority_bounded": all(
            0.0 < float(((
                row.get("world_model_event_components") or {}
            ).get("gate") or {}).get("authority", 0.0)) <= 0.50
            for rows in learned_match_rows for row in rows
        ),
        "all_paired_same_outcome": all(
            "llm_brier" in row and "world_model_brier" in row
            for row in flat
        ) and malformed_evaluations == 0,
        "all_shadow_only": all(bool(row.get("shadow_only")) for row in flat),
        "all_non_causal": all(
            not bool(row.get("causal_interpretation")) for row in flat
        ),
        "profiles": profiles,
        "event_signatures": event_signatures,
        "model_policy_scopes": [list(scope) for scope in model_scopes],
        "provenance_compatible": provenance_compatible,
        "authority_active": False,
    }
