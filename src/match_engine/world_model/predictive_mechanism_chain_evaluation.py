"""Outcome scoring and diagnostics for three-event predictive chains."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.llm_event_hypothesis import (
    observed_semantic_event,
)
from src.match_engine.world_model.predictive_mechanism_chain import (
    PREDICTIVE_MECHANISM_CHAIN_VERSION,
    predictive_mechanism_chain_audit_is_valid,
)


_CELLS = tuple(
    f"a{a}_b{b}_c{c}"
    for a in (0, 1) for b in (0, 1) for c in (0, 1)
)


def _digest(payload: dict[str, Any], prefix: str) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:24]


def score_predictive_mechanism_chain(
    audit: dict[str, Any], outcome: dict[str, Any], baseline: dict[str, Any],
    *, attacking_home: bool, horizon: str, realized_action: str,
    checkpoint_signature: str, environment_signature: str,
) -> dict[str, Any] | None:
    """Score the immutable joint path against its adjacent-edge Markov null."""
    if not predictive_mechanism_chain_audit_is_valid(audit):
        return None
    chain = audit["chain"]
    if (
        str(horizon) != str(chain["horizon"])
        or str(realized_action).lower() != str(chain["action"])
        or str(checkpoint_signature) != str(audit["checkpoint_signature"])
        or str(environment_signature) != str(audit["environment_signature"])
    ):
        return None
    try:
        observed = tuple(bool(observed_semantic_event(
            chain[event_key], outcome, baseline,
            attacking_home=attacking_home,
        )) for event_key in (
            "first_event", "mediator_event", "outcome_event",
        ))
        cell = f"a{int(observed[0])}_b{int(observed[1])}_c{int(observed[2])}"
        joint = chain["joint_probabilities"]
        null = chain["pairwise_markov_null_probabilities"]
        log_ratio = math.log(float(joint[cell]) / float(null[cell]))
        joint_brier = sum(
            (float(joint[key]) - float(key == cell)) ** 2 for key in _CELLS
        )
        null_brier = sum(
            (float(null[key]) - float(key == cell)) ** 2 for key in _CELLS
        )
        completion_observed = bool(all(observed))
        completion_target = float(completion_observed)
        world_completion = float(
            audit["world_model_chain_completion_probability"]
        )
        llm_completion = float(audit["llm_chain_completion_probability"])
        fused_completion = float(audit["fused_chain_completion_probability"])
        markov_completion = float(null["a1_b1_c1"])
        probabilities = {
            "world_model": world_completion,
            "llm": llm_completion,
            "fused": fused_completion,
            "pairwise_markov_null": markov_completion,
        }
        completion_brier = {
            name: (probability - completion_target) ** 2
            for name, probability in probabilities.items()
        }
        completion_log_loss = {
            name: -math.log(max(1e-12, min(
                1.0 - 1e-12,
                probability if completion_observed else 1.0 - probability,
            )))
            for name, probability in probabilities.items()
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    payload = {
        "version": PREDICTIVE_MECHANISM_CHAIN_VERSION,
        "chain_id": chain["chain_id"],
        "horizon": chain["horizon"],
        "first_event": chain["first_event"],
        "mediator_event": chain["mediator_event"],
        "outcome_event": chain["outcome_event"],
        "first_event_observed": observed[0],
        "mediator_event_observed": observed[1],
        "outcome_event_observed": observed[2],
        "observed_joint_cell": cell,
        "joint_probability": float(joint[cell]),
        "pairwise_markov_null_probability": float(null[cell]),
        "categorical_log_likelihood_ratio_vs_pairwise_markov_null": log_ratio,
        "joint_brier_score": joint_brier,
        "pairwise_markov_null_brier_score": null_brier,
        "brier_skill_vs_pairwise_markov_null": null_brier - joint_brier,
        "chain_completion_observed": completion_observed,
        "chain_completion_probabilities": probabilities,
        "chain_completion_brier_scores": completion_brier,
        "chain_completion_log_losses": completion_log_loss,
        "fused_brier_skill_vs_world_model": (
            completion_brier["world_model"] - completion_brier["fused"]
        ),
        "fused_log_loss_gain_vs_world_model": (
            completion_log_loss["world_model"]
            - completion_log_loss["fused"]
        ),
        "chain_forecast_fusion": dict(audit["chain_forecast_fusion"]),
        "llm_signature": str(audit["llm_signature"]),
        "world_model_prediction_mutated": False,
        "higher_order_dependence_only": True,
        "causal_interpretation": False,
    }
    return {
        **payload,
        "evaluation_digest": _digest(
            payload, "predictive-mechanism-chain-evaluation:",
        ),
    }


def predictive_mechanism_chain_evaluation_is_valid(
    evaluation: Any, audit: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(evaluation, dict):
        return False
    payload = dict(evaluation)
    digest = payload.pop("evaluation_digest", None)
    if digest != _digest(payload, "predictive-mechanism-chain-evaluation:"):
        return False
    if audit is None:
        return True
    if not predictive_mechanism_chain_audit_is_valid(audit):
        return False
    chain = audit["chain"]
    try:
        cell = str(evaluation["observed_joint_cell"])
        expected_log = math.log(
            float(chain["joint_probabilities"][cell])
            / float(chain["pairwise_markov_null_probabilities"][cell])
        )
        expected_skill = (
            float(evaluation["pairwise_markov_null_brier_score"])
            - float(evaluation["joint_brier_score"])
        )
        observed = bool(evaluation["chain_completion_observed"])
        target = float(observed)
        probabilities = evaluation["chain_completion_probabilities"]
        briers = evaluation["chain_completion_brier_scores"]
        losses = evaluation["chain_completion_log_losses"]
        expected_probabilities = {
            "world_model": float(
                audit["world_model_chain_completion_probability"]
            ),
            "llm": float(audit["llm_chain_completion_probability"]),
            "fused": float(audit["fused_chain_completion_probability"]),
            "pairwise_markov_null": float(
                chain["pairwise_markov_null_probabilities"]["a1_b1_c1"]
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        cell in _CELLS
        and evaluation.get("chain_id") == chain["chain_id"]
        and evaluation.get("horizon") == chain["horizon"]
        and abs(float(evaluation[
            "categorical_log_likelihood_ratio_vs_pairwise_markov_null"
        ]) - expected_log) <= 1e-12
        and abs(float(evaluation[
            "brier_skill_vs_pairwise_markov_null"
        ]) - expected_skill) <= 1e-12
        and evaluation.get("higher_order_dependence_only") is True
        and observed == all(bool(evaluation[key]) for key in (
            "first_event_observed", "mediator_event_observed",
            "outcome_event_observed",
        ))
        and set(probabilities) == set(expected_probabilities)
        and all(
            abs(float(probabilities[name]) - probability) <= 1e-12
            and abs(float(briers[name]) - (probability - target) ** 2) <= 1e-12
            and abs(float(losses[name]) - (-math.log(max(
                1e-12, min(
                    1.0 - 1e-12,
                    probability if observed else 1.0 - probability,
                ),
            )))) <= 1e-12
            for name, probability in expected_probabilities.items()
        )
        and evaluation.get("chain_forecast_fusion")
        == audit.get("chain_forecast_fusion")
        and evaluation.get("llm_signature") == audit.get("llm_signature")
        and evaluation.get("world_model_prediction_mutated") is False
        and evaluation.get("causal_interpretation") is False
    )


def predictive_mechanism_chain_diagnostics(
    record_clusters: Iterable[Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    accepted = executed = scored = matches = 0
    malformed_audits = malformed_scores = unsafe = 0
    log_ratios: list[float] = []
    brier_skills: list[float] = []
    completion_briers: dict[str, list[float]] = {
        name: [] for name in (
            "world_model", "llm", "fused", "pairwise_markov_null",
        )
    }
    fused_brier_gains: list[float] = []
    active_fusion_scores = 0
    chains: set[str] = set()
    paths: set[tuple[str, str, str]] = set()
    for cluster in record_clusters:
        match_scores = 0
        for record in cluster:
            audit = record.get("llm_predictive_mechanism_chain_context") or {}
            if not audit:
                continue
            if not predictive_mechanism_chain_audit_is_valid(audit):
                malformed_audits += 1
                continue
            accepted += 1
            chain = audit["chain"]
            chains.add(str(chain["chain_id"]))
            paths.add(tuple(str(chain[key]) for key in (
                "first_event", "mediator_event", "outcome_event",
            )))
            unsafe += int(
                audit.get("selected_after_action_freeze") is not True
                or audit.get("can_change_current_action") is not False
                or audit.get("can_change_tactical_controls") is not False
                or audit.get("can_schedule_future_action") is not False
                or audit.get("can_update_world_model") is not False
                or audit.get("causal_interpretation") is not False
            )
            action_executed = (
                str(record.get("intervention_actual_action", "")).lower()
                == str(chain["action"])
            )
            executed += int(action_executed)
            if not action_executed:
                continue
            evaluation = (
                (record.get("multi_horizon_regime_outcomes") or {}).get(
                    str(chain["horizon"]),
                ) or {}
            ).get("llm_predictive_mechanism_chain_evaluation") or {}
            if not evaluation:
                continue
            if not predictive_mechanism_chain_evaluation_is_valid(
                evaluation, audit,
            ):
                malformed_scores += 1
                continue
            try:
                log_ratio = float(evaluation[
                    "categorical_log_likelihood_ratio_vs_pairwise_markov_null"
                ])
                brier_skill = float(evaluation[
                    "brier_skill_vs_pairwise_markov_null"
                ])
            except (KeyError, TypeError, ValueError, OverflowError):
                malformed_scores += 1
                continue
            if not math.isfinite(log_ratio) or not math.isfinite(brier_skill):
                malformed_scores += 1
                continue
            scored += 1
            match_scores += 1
            log_ratios.append(log_ratio)
            brier_skills.append(brier_skill)
            for name in completion_briers:
                completion_briers[name].append(float(
                    evaluation["chain_completion_brier_scores"][name]
                ))
            fused_brier_gains.append(float(
                evaluation["fused_brier_skill_vs_world_model"]
            ))
            active_fusion_scores += int(bool(
                (evaluation.get("chain_forecast_fusion") or {}).get("active")
            ))
        matches += int(match_scores > 0)
    return {
        "version": PREDICTIVE_MECHANISM_CHAIN_VERSION,
        "evaluation_kind": "three_event_joint_vs_pairwise_markov_tournament",
        "accepted_chain_audits": accepted,
        "executed_chain_actions": executed,
        "scored_joint_outcomes": scored,
        "distinct_chains": len(chains),
        "distinct_mechanism_paths": len(paths),
        "matches": matches,
        "malformed_chain_audits": malformed_audits,
        "malformed_chain_scores": malformed_scores,
        "mean_log_likelihood_ratio_vs_pairwise_markov_null": (
            float(np.mean(log_ratios)) if log_ratios else 0.0
        ),
        "mean_brier_skill_vs_pairwise_markov_null": (
            float(np.mean(brier_skills)) if brier_skills else 0.0
        ),
        "positive_likelihood_evidence_rate": (
            float(np.mean(np.asarray(log_ratios) > 0.0))
            if log_ratios else 0.0
        ),
        "active_llm_world_model_fusion_scores": active_fusion_scores,
        "mean_chain_completion_brier": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in completion_briers.items()
        },
        "mean_fused_brier_skill_vs_world_model": (
            float(np.mean(fused_brier_gains)) if fused_brier_gains else 0.0
        ),
        "positive_fused_brier_gain_rate": (
            float(np.mean(np.asarray(fused_brier_gains) > 0.0))
            if fused_brier_gains else 0.0
        ),
        "unsafe_action_tactical_or_learning_authority_claims": unsafe,
        "all_chains_post_action_noncausal": bool(accepted > 0 and unsafe == 0),
        "interpretation": (
            "The full three-event joint is compared with a null preserving "
            "adjacent pairs; gain measures residual first-to-third dependence "
            "conditional on the middle event, not causal mediation."
        ),
    }
