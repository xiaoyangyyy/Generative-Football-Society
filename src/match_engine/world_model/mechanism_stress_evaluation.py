"""Natural-outcome scoring and diagnostics for mechanism stress tests."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.llm_event_hypothesis import (
    observed_semantic_event,
)
from src.match_engine.world_model.mechanism_stress_test import (
    MECHANISM_STRESS_TEST_VERSION,
    mechanism_stress_test_audit_is_valid,
)


_CELLS = (
    "driver_false_outcome_false", "driver_false_outcome_true",
    "driver_true_outcome_false", "driver_true_outcome_true",
)


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def score_mechanism_stress_test(
    audit: dict[str, Any], outcome: dict[str, Any], baseline: dict[str, Any],
    *, attacking_home: bool, horizon: str, realized_action: str,
    checkpoint_signature: str, environment_signature: str,
) -> dict[str, Any] | None:
    if not mechanism_stress_test_audit_is_valid(audit):
        return None
    test = audit["stress_test"]
    if (
        str(horizon) != str(test["horizon"])
        or str(realized_action).lower() != str(test["action"])
        or str(checkpoint_signature) != str(audit["checkpoint_signature"])
        or str(environment_signature) != str(audit["environment_signature"])
    ):
        return None
    try:
        driver = observed_semantic_event(
            test["driver_event"], outcome, baseline,
            attacking_home=attacking_home,
        )
        consequence = observed_semantic_event(
            test["outcome_event"], outcome, baseline,
            attacking_home=attacking_home,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    cell = (
        f"driver_{'true' if driver else 'false'}_"
        f"outcome_{'true' if consequence else 'false'}"
    )
    observed_joint = test["observed_context_joint_probabilities"]
    neutralized_joint = test["neutralized_context_joint_probabilities"]
    log_ratio = math.log(observed_joint[cell] / neutralized_joint[cell])
    observed_brier = sum(
        (observed_joint[key] - float(key == cell)) ** 2 for key in _CELLS
    )
    neutralized_brier = sum(
        (neutralized_joint[key] - float(key == cell)) ** 2 for key in _CELLS
    )
    payload = {
        "version": MECHANISM_STRESS_TEST_VERSION,
        "stress_test_id": test["stress_test_id"],
        "source_hypothesis_id": test["source_hypothesis_id"],
        "horizon": test["horizon"],
        "context_factor": test["context_factor"],
        "classification": test["classification"],
        "driver_observed": driver, "outcome_observed": consequence,
        "observed_joint_cell": cell,
        "observed_context_probability": observed_joint[cell],
        "neutralized_context_probability": neutralized_joint[cell],
        "log_likelihood_ratio_observed_vs_neutralized": log_ratio,
        "observed_context_brier_score": observed_brier,
        "neutralized_context_brier_score": neutralized_brier,
        "brier_skill_observed_vs_neutralized": (
            neutralized_brier - observed_brier
        ),
        "validates_predictive_context_dependence_only": True,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "evaluation_digest": _digest(
            payload, "mechanism-stress-evaluation:",
        ),
    }


def mechanism_stress_evaluation_is_valid(
    evaluation: Any, audit: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(evaluation, dict):
        return False
    payload = dict(evaluation)
    digest = payload.pop("evaluation_digest", None)
    if digest != _digest(payload, "mechanism-stress-evaluation:"):
        return False
    if audit is None:
        return True
    if not mechanism_stress_test_audit_is_valid(audit):
        return False
    test = audit["stress_test"]
    try:
        cell = str(evaluation["observed_joint_cell"])
        observed = test["observed_context_joint_probabilities"]
        neutralized = test["neutralized_context_joint_probabilities"]
        expected = math.log(observed[cell] / neutralized[cell])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        cell in _CELLS
        and evaluation.get("stress_test_id") == test["stress_test_id"]
        and evaluation.get("source_hypothesis_id")
        == test["source_hypothesis_id"]
        and evaluation.get("horizon") == test["horizon"]
        and evaluation.get("context_factor") == test["context_factor"]
        and evaluation.get("classification") == test["classification"]
        and abs(float(evaluation[
            "log_likelihood_ratio_observed_vs_neutralized"
        ]) - expected) <= 1e-12
        and evaluation.get("validates_predictive_context_dependence_only")
        is True
        and evaluation.get("causal_interpretation") is False
    )


def mechanism_stress_test_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    accepted = executed = scored = malformed_audits = malformed_scores = 0
    fragile = sensitive = robust = unsafe = matches = 0
    factors = set()
    likelihood = []
    brier_skill = []
    for cluster in record_clusters:
        match_scores = 0
        for record in cluster:
            audit = record.get("llm_mechanism_stress_test_context") or {}
            if not audit:
                continue
            if not mechanism_stress_test_audit_is_valid(audit):
                malformed_audits += 1
                continue
            accepted += 1
            test = audit["stress_test"]
            factors.add(str(test["context_factor"]))
            fragile += int(test["classification"] == "fragile")
            sensitive += int(test["classification"] == "context_sensitive")
            robust += int(
                test["classification"]
                == "robust_within_tested_neutralization"
            )
            unsafe += int(
                audit.get("selected_after_action_freeze") is not True
                or audit.get("can_change_current_action") is not False
                or audit.get("can_change_tactical_controls") is not False
                or audit.get("can_schedule_future_action") is not False
                or audit.get("can_update_world_model") is not False
                or audit.get("causal_interpretation") is not False
            )
            action_ok = bool(
                str(record.get("intervention_actual_action", "")).lower()
                == str(test["action"])
            )
            executed += int(action_ok)
            if not action_ok:
                continue
            evaluation = (
                (record.get("multi_horizon_regime_outcomes") or {}).get(
                    str(test["horizon"]),
                ) or {}
            ).get("llm_mechanism_stress_test_evaluation") or {}
            if not evaluation:
                continue
            if not mechanism_stress_evaluation_is_valid(evaluation, audit):
                malformed_scores += 1
                continue
            try:
                lr = float(evaluation[
                    "log_likelihood_ratio_observed_vs_neutralized"
                ])
                skill = float(evaluation[
                    "brier_skill_observed_vs_neutralized"
                ])
            except (KeyError, TypeError, ValueError, OverflowError):
                malformed_scores += 1
                continue
            if not math.isfinite(lr) or not math.isfinite(skill):
                malformed_scores += 1
                continue
            likelihood.append(lr)
            brier_skill.append(skill)
            scored += 1
            match_scores += 1
        matches += int(match_scores > 0)
    return {
        "version": MECHANISM_STRESS_TEST_VERSION,
        "evaluation_kind": "post_action_context_mechanism_stress_test",
        "accepted_stress_audits": accepted,
        "executed_stress_actions": executed,
        "scored_natural_outcomes": scored,
        "distinct_context_factors": len(factors),
        "fragile_classifications": fragile,
        "context_sensitive_classifications": sensitive,
        "robust_classifications": robust,
        "matches": matches,
        "malformed_stress_audits": malformed_audits,
        "malformed_stress_scores": malformed_scores,
        "mean_log_likelihood_observed_vs_neutralized": (
            float(np.mean(likelihood)) if likelihood else 0.0
        ),
        "mean_brier_skill_observed_vs_neutralized": (
            float(np.mean(brier_skill)) if brier_skill else 0.0
        ),
        "unsafe_action_tactical_or_learning_authority_claims": unsafe,
        "all_stress_tests_post_action_shadow_only": bool(
            accepted > 0 and unsafe == 0
        ),
        "interpretation": (
            "Stress tests compare model sensitivity to schema-grounded "
            "neutralizations; natural outcomes validate predictive context "
            "dependence, not the counterfactual or a causal mechanism."
        ),
    }
