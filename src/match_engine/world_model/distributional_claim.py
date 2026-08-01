"""LLM claims over member-utility distributions and realized calibration."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.distributional_utility import (
    DISTRIBUTIONAL_CRITERIA,
)
from src.match_engine.world_model.opponent_response import RESPONSE_ACTIONS


DISTRIBUTIONAL_CLAIM_VERSION = 1


def llm_distributional_claim_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": DISTRIBUTIONAL_CLAIM_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "shadow_distributional_action_rationale",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-distributional-claim:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def validate_llm_distributional_claim(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    selected = str(raw.get("selected_action", "")).strip().lower()
    alternative = str(raw.get("alternative_action", "")).strip().lower()
    horizon = str(raw.get("horizon", "")).strip().lower()
    criterion = str(raw.get("criterion", "")).strip().lower()
    relation = str(raw.get("relation", "")).strip().lower()
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if (
        selected not in RESPONSE_ACTIONS
        or alternative not in RESPONSE_ACTIONS
        or selected == alternative
        or not horizon
        or criterion not in DISTRIBUTIONAL_CRITERIA
        or relation not in {
            "selected_better", "alternative_better", "approximately_equal",
        }
        or not np.isfinite(confidence)
        or not 0.5 <= confidence <= 1.0
    ):
        return None
    return {
        "selected_action": selected,
        "alternative_action": alternative,
        "horizon": horizon,
        "criterion": criterion,
        "relation": relation,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", ""))[:280],
    }


def evaluate_llm_distributional_claim(
    packet: dict[str, Any],
    raw_claim: Any,
    *,
    selected_action: str,
    claim_signature: str = "distributional-claim-contract-unspecified",
) -> dict[str, Any]:
    claim = validate_llm_distributional_claim(raw_claim)
    base = {
        "version": DISTRIBUTIONAL_CLAIM_VERSION,
        "accepted": False,
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "world_model_prediction_mutated": False,
        "can_change_selected_action": False,
        "causal_interpretation": False,
        "claim_signature": str(claim_signature),
    }
    if claim is None:
        return {**base, "reason": "missing_or_invalid_distributional_claim"}
    if claim["selected_action"] != str(selected_action).lower():
        return {
            **base,
            "reason": "distributional_selected_action_must_equal_plan",
            "claim": claim,
        }
    candidates = {
        str(candidate.get("action", "")).lower(): candidate
        for candidate in packet.get("candidates") or []
    }
    if not packet.get("available") or not all(
        action in candidates
        for action in (claim["selected_action"], claim["alternative_action"])
    ):
        return {
            **base,
            "reason": "distributional_actions_not_evaluated",
            "claim": claim,
        }
    distributions = []
    for action in (claim["selected_action"], claim["alternative_action"]):
        prediction = (
            candidates[action].get("multi_horizon_predictions") or {}
        ).get(claim["horizon"]) or {}
        distribution = prediction.get("distributional_policy_utility") or {}
        if not (
            distribution.get("available")
            and distribution.get("counts_trusted")
            and distribution.get("member_identity_preserved")
        ):
            return {
                **base,
                "reason": "trained_distributional_evidence_unavailable",
                "claim": claim,
            }
        distributions.append(distribution)
    criterion = claim["criterion"]
    selected_value = float(distributions[0][criterion])
    alternative_value = float(distributions[1][criterion])
    delta = selected_value - alternative_value
    tolerance = 0.02 if criterion == "upside_probability" else 0.01
    measured_relation = (
        "selected_better" if delta > tolerance
        else "alternative_better" if delta < -tolerance
        else "approximately_equal"
    )
    audit = {
        **base,
        "accepted": True,
        "reason": "shadow_model_checked_distributional_claim",
        "claim": claim,
        "measured_relation": measured_relation,
        "directionally_faithful": measured_relation == claim["relation"],
        "criterion_tolerance": tolerance,
        "selected_criterion_value": selected_value,
        "alternative_criterion_value": alternative_value,
        "criterion_delta": delta,
        "selected_distribution": dict(distributions[0]),
        "checkpoint_signature": str(packet.get(
            "checkpoint_signature", "runtime_unspecified",
        )),
        "environment_signature": str(packet.get(
            "environment_signature", "environment_unspecified",
        )),
    }
    packet["llm_distributional_claim_audit"] = audit
    return audit


def _pinball_loss(observed: float, predicted: float, quantile: float) -> float:
    residual = observed - predicted
    return float(max(quantile * residual, (quantile - 1.0) * residual))


def _distribution_values(
    distribution: dict[str, Any],
) -> tuple[np.ndarray, float, float, float, float] | None:
    try:
        values = np.asarray(distribution["member_values"], dtype=np.float64)
        quantiles = distribution["quantiles"]
        q10, q50, q90 = map(
            float, (quantiles["q10"], quantiles["q50"], quantiles["q90"]),
        )
        mean = float(distribution["mean_utility"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        values.ndim != 1 or len(values) < 2
        or not np.all(np.isfinite(values))
        or not all(np.isfinite(value) for value in (q10, q50, q90, mean))
        or not q10 <= q50 <= q90
    ):
        return None
    expected_quantiles = np.quantile(
        values, [0.10, 0.50, 0.90], method="linear",
    )
    if (
        abs(mean - float(np.mean(values))) > 1e-6
        or any(abs(actual - expected) > 1e-6 for actual, expected in zip(
            (q10, q50, q90), expected_quantiles,
        ))
    ):
        return None
    return values, mean, q10, q50, q90


def _empirical_crps(values: np.ndarray, observed: float) -> float:
    return max(0.0, float(
        np.mean(np.abs(values - observed))
        - 0.5 * np.mean(np.abs(values[:, None] - values[None, :]))
    ))


def score_distributional_claim(
    context: dict[str, Any],
    outcome: dict[str, Any],
    *,
    checkpoint_signature: str,
    environment_signature: str,
) -> dict[str, Any] | None:
    if not context.get("accepted"):
        return None
    claim = validate_llm_distributional_claim(context.get("claim"))
    parsed = _distribution_values(context.get("selected_distribution") or {})
    try:
        observed = float(outcome["policy_utility"])
        selected_criterion = float(context["selected_criterion_value"])
        alternative_criterion = float(context["alternative_criterion_value"])
        criterion_tolerance = float(context["criterion_tolerance"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if claim is None or parsed is None or not np.isfinite(observed):
        return None
    values, mean, q10, q50, q90 = parsed
    distribution = context["selected_distribution"]
    return {
        "version": DISTRIBUTIONAL_CLAIM_VERSION,
        "claim": claim,
        "observed_policy_utility": observed,
        "member_values": list(map(float, values)),
        "quantiles": {"q10": q10, "q50": q50, "q90": q90},
        "predicted_mean_utility": mean,
        "mean_absolute_error": abs(mean - observed),
        "mean_squared_error": (mean - observed) ** 2,
        "empirical_crps": _empirical_crps(values, observed),
        "q10_pinball": _pinball_loss(observed, q10, 0.10),
        "q50_pinball": _pinball_loss(observed, q50, 0.50),
        "q90_pinball": _pinball_loss(observed, q90, 0.90),
        "central_80_covered": q10 <= observed <= q90,
        "below_median": observed <= q50,
        "directionally_faithful": bool(context.get("directionally_faithful")),
        "selected_criterion_value": selected_criterion,
        "alternative_criterion_value": alternative_criterion,
        "criterion_tolerance": criterion_tolerance,
        "measured_relation": str(context.get("measured_relation", "")),
        "paired_same_action_horizon": True,
        "member_identity_preserved": bool(
            distribution.get("member_identity_preserved")
        ),
        "shadow_only": True,
        "authority_active": False,
        "policy_mutated": False,
        "causal_interpretation": False,
        "claim_signature": str(context.get("claim_signature", "")),
        "checkpoint_signature": str(checkpoint_signature),
        "environment_signature": str(environment_signature),
        "issuance_evaluation_provenance_compatible": bool(
            str(context.get("checkpoint_signature", ""))
            == str(checkpoint_signature)
            and str(context.get("environment_signature", ""))
            == str(environment_signature)
        ),
    }


def _valid_evaluation(row: dict[str, Any]) -> bool:
    try:
        observed = float(row["observed_policy_utility"])
        mean = float(row["predicted_mean_utility"])
        mae = float(row["mean_absolute_error"])
        mse = float(row["mean_squared_error"])
        crps = float(row["empirical_crps"])
        parsed = _distribution_values({
            "member_values": row["member_values"],
            "mean_utility": mean,
            "quantiles": row["quantiles"],
        })
        selected = float(row["selected_criterion_value"])
        alternative = float(row["alternative_criterion_value"])
        tolerance = float(row["criterion_tolerance"])
        pinballs = [float(row[key]) for key in (
            "q10_pinball", "q50_pinball", "q90_pinball",
        )]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if parsed is None:
        return False
    values, _, q10, q50, q90 = parsed
    relation = (
        "selected_better" if selected - alternative > tolerance
        else "alternative_better" if selected - alternative < -tolerance
        else "approximately_equal"
    )
    return bool(
        int(row.get("version", 0)) == DISTRIBUTIONAL_CLAIM_VERSION
        and validate_llm_distributional_claim(row.get("claim")) is not None
        and all(np.isfinite(value) for value in (
            observed, mean, mae, mse, crps, selected, alternative,
            tolerance, *pinballs,
        ))
        and tolerance >= 0.0
        and abs(mae - abs(mean - observed)) <= 1e-6
        and abs(mse - (mean - observed) ** 2) <= 1e-6
        and abs(crps - _empirical_crps(values, observed)) <= 1e-6
        and all(abs(actual - _pinball_loss(
            observed, predicted, level,
        )) <= 1e-6 for actual, predicted, level in zip(
            pinballs, (q10, q50, q90), (0.10, 0.50, 0.90),
        ))
        and bool(row.get("central_80_covered")) == (q10 <= observed <= q90)
        and bool(row.get("below_median")) == (observed <= q50)
        and str(row.get("measured_relation", "")) == relation
        and bool(row.get("directionally_faithful")) == (
            relation == str((row.get("claim") or {}).get("relation", ""))
        )
        and row.get("paired_same_action_horizon")
        and row.get("member_identity_preserved")
        and row.get("issuance_evaluation_provenance_compatible")
    )


def distributional_claim_diagnostics(
    match_logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    matches = []
    eligible_claims = unscored_eligible_claims = 0
    for payload in match_logs:
        rows = []
        for record in (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        ):
            outcomes = record.get("multi_horizon_regime_outcomes") or {}
            context = record.get("llm_distributional_claim_context") or {}
            claim = validate_llm_distributional_claim(context.get("claim"))
            if (
                context.get("accepted") and claim is not None
                and claim["selected_action"]
                == str(record.get("intervention_actual_action", "")).lower()
                and claim["horizon"] in outcomes
            ):
                eligible_claims += 1
                candidate = outcomes[claim["horizon"]]
                if not isinstance(candidate, dict) or not isinstance(
                    candidate.get("llm_distributional_claim_evaluation"), dict,
                ):
                    unscored_eligible_claims += 1
            rows.extend(
                outcome["llm_distributional_claim_evaluation"]
                for outcome in outcomes.values()
                if isinstance(outcome, dict)
                and isinstance(
                    outcome.get("llm_distributional_claim_evaluation"), dict,
                )
            )
        if rows:
            matches.append(rows)
    valid = [row for rows in matches for row in rows if _valid_evaluation(row)]
    malformed = sum(len(rows) for rows in matches) - len(valid)
    valid_ids = {id(row) for row in valid}
    groups = [
        [row for row in rows if id(row) in valid_ids] for rows in matches
    ]
    groups = [rows for rows in groups if rows]

    def match_metric(key: str) -> list[float]:
        return [
            float(np.mean([float(row[key]) for row in rows])) for rows in groups
        ]

    def match_rate(key: str) -> list[float]:
        return [
            float(np.mean([bool(row[key]) for row in rows])) for rows in groups
        ]

    crps = match_metric("empirical_crps")
    mae = match_metric("mean_absolute_error")
    coverage = match_rate("central_80_covered")
    below_median = match_rate("below_median")
    faithfulness = match_rate("directionally_faithful")
    signatures = sorted({str(row.get("claim_signature", "")) for row in valid})
    scopes = sorted({(
        str(row.get("checkpoint_signature", "")),
        str(row.get("environment_signature", "")),
    ) for row in valid})
    unspecified = {
        "", "distributional-claim-contract-unspecified",
        "runtime_unspecified", "environment_unspecified",
    }
    return {
        "version": DISTRIBUTIONAL_CLAIM_VERSION,
        "evaluation_kind": "realized_member_distribution_policy_utility",
        "realized_distributions": len(valid),
        "eligible_claims": eligible_claims,
        "unscored_eligible_claims": unscored_eligible_claims,
        "matches": len(groups),
        "malformed_evaluations": malformed,
        "match_clustered_crps": float(np.mean(crps)) if crps else 0.0,
        "match_clustered_mean_absolute_error": float(np.mean(mae)) if mae else 0.0,
        "match_clustered_crps_skill_vs_mean_absolute_error": (
            float(np.mean(mae)) - float(np.mean(crps)) if crps else 0.0
        ),
        "match_clustered_central_80_coverage": (
            float(np.mean(coverage)) if coverage else 0.0
        ),
        "match_clustered_below_median_rate": (
            float(np.mean(below_median)) if below_median else 0.0
        ),
        "match_clustered_directional_faithfulness": (
            float(np.mean(faithfulness)) if faithfulness else 0.0
        ),
        "all_shadow_only": all(bool(row.get("shadow_only")) for row in valid),
        "all_non_controlling": all(
            not bool(row.get("authority_active"))
            and not bool(row.get("policy_mutated")) for row in valid
        ),
        "all_non_causal": all(
            not bool(row.get("causal_interpretation")) for row in valid
        ),
        "all_member_identity_preserved": all(
            bool(row.get("member_identity_preserved")) for row in valid
        ),
        "provenance_compatible": bool(
            len(signatures) == 1 and signatures[0] not in unspecified
            and len(scopes) == 1
            and all(value not in unspecified for value in scopes[0])
        ),
        "claim_signatures": signatures,
        "model_policy_scopes": [list(scope) for scope in scopes],
    }
