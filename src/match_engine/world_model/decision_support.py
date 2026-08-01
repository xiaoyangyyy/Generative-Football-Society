"""Auditable world-model decision packets for strategic LLM consumers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.action_codec import encode_high_level_action
from src.match_engine.world_model.schema import observation_coverage
from src.match_engine.world_model.policy_prediction import (
    build_multi_horizon_policy_predictions,
    summarize_multi_horizon_predictions,
)
from src.match_engine.tactical_catalog import TACTICAL_PRESETS, resolve_tactical_preset
from src.match_engine.world_model.opponent_belief import (
    OPPONENT_HYPOTHESES,
    reweight_counterfactual_candidates,
    tactic_feature_vector,
    update_opponent_belief,
)
from src.match_engine.world_model.opponent_game import attach_second_order_game
from src.match_engine.world_model.trajectory_game import (
    build_predicted_state_continuations,
)
from src.match_engine.world_model.distributional_utility import (
    build_distributional_action_frontiers,
)
from src.match_engine.world_model.temporal_utility import (
    build_temporal_utility_paths,
)


DECISION_PACKET_VERSION = 23
COACH_ACTIONS = ("hold", "pass", "cross", "shot")
PREMATCH_TACTICAL_CANDIDATES = (
    "balanced",
    "gegenpress",
    "possession_control",
    "counter_attack",
    "low_block",
    "wing_play",
    "direct_vertical",
)


def _online_calibration_diagnostics(runtime) -> dict[str, Any]:
    diagnostics = getattr(runtime, "online_calibration_diagnostics", None)
    if not callable(diagnostics):
        return {"available": False}
    try:
        return diagnostics()
    except (RuntimeError, TypeError, ValueError):
        return {"available": False, "reason": "diagnostics_error"}


def _candidate_target(state, action: str, attacking_home: bool) -> np.ndarray:
    ball = np.asarray(state.ball.position, dtype=np.float32).copy()
    direction = 1.0 if attacking_home else -1.0
    if action == "hold":
        return ball
    if action == "pass":
        return np.clip(ball + np.array([0.14 * direction, 0.0]), 0.02, 0.98)
    if action == "cross":
        return np.array([0.91 if attacking_home else 0.09, 0.50], dtype=np.float32)
    return np.array([0.995 if attacking_home else 0.005, 0.50], dtype=np.float32)


def _physics_prior(state, action: str, attacking_home: bool) -> float:
    if action == "hold":
        return 0.90
    if action == "pass":
        return 0.72
    if action == "cross":
        return 0.46
    goal_x = 0.995 if attacking_home else 0.005
    distance = abs(goal_x - float(state.ball.position[0]))
    return float(np.clip(0.35 * (1.0 - distance), 0.03, 0.45))


def _evaluate_action_candidates(
    runtime,
    state,
    team_id: str,
    *,
    horizon_s: float,
    uncertainty_penalty: float,
    prediction_horizons_s: tuple[float, ...],
    residual_memory=None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run the shared short-horizon action counterfactuals once."""
    attacking_home = team_id == state.home.team_id
    observation = runtime.encode_state(state, attacking_home=attacking_home)
    from src.match_engine.world_model.policy_outcomes import (
        capture_policy_outcome_baseline,
    )

    decision_context = capture_policy_outcome_baseline(
        state, team_id=team_id,
    )
    decision_context["team_id"] = str(team_id)
    candidates = []
    for action_name in COACH_ACTIONS:
        target = _candidate_target(state, action_name, attacking_home)
        action = encode_high_level_action(
            action_name, target=target, horizon_s=horizon_s,
        )
        prior = _physics_prior(state, action_name, attacking_home)
        action[13] = prior
        quality_kind = "shot" if action_name == "shot" else "pass"
        confidence = float(runtime.planner_confidence(
            observation, kind=quality_kind,
        ))
        future = runtime.predict_future(
            observation, action, action_kind=action_name,
            horizon_s=horizon_s,
        )
        if action_name == "shot":
            expected_value = runtime.score_shot_action(
                observation, action, attacking_home=attacking_home,
            )
        else:
            expected_value = runtime.score_action(observation, action)
        uncertainty = float(np.clip(future.state_uncertainty, 0.0, 1.0))
        epistemic_uncertainty = float(np.clip(
            future.epistemic_uncertainty, 0.0, 1.0,
        ))
        aleatoric_uncertainty = float(np.clip(
            future.aleatoric_uncertainty, 0.0, 1.0,
        ))
        learnable_uncertainty = max(
            epistemic_uncertainty, 1.0 - confidence,
        )
        effective_confidence = confidence * (1.0 - uncertainty)
        turnover = float(future.event_probabilities["turnover"])
        risk_adjusted = (
            finite_float(expected_value, 0.0)
            - uncertainty_penalty * uncertainty
            - 0.15 * turnover
        )
        multi_horizon_predictions = build_multi_horizon_policy_predictions(
            runtime,
            observation,
            action_name=action_name,
            target=target,
            physics_prior=prior,
            confidence=confidence,
            attacking_home=attacking_home,
            base_horizon_s=horizon_s,
            outcome_horizons_s=prediction_horizons_s,
            residual_memory=residual_memory,
            decision_context=decision_context,
        )
        temporal_utility_paths = build_temporal_utility_paths(
            multi_horizon_predictions,
        )
        forecast_summary = summarize_multi_horizon_predictions(
            multi_horizon_predictions,
        )
        forecast_adjustment = (
            0.20
            * float(np.clip(
                forecast_summary["policy_utility"], -0.5, 0.5,
            ))
            * float(np.clip(
                forecast_summary["effective_confidence"], 0.0, 1.0,
            ))
        )
        risk_adjusted += forecast_adjustment
        candidates.append({
            "action": action_name,
            "target": [float(target[0]), float(target[1])],
            "physics_prior": prior,
            "expected_value": finite_float(expected_value, 0.0),
            "risk_adjusted_value": finite_float(risk_adjusted, -1.0),
            "multi_horizon_forecast_adjustment": forecast_adjustment,
            "multi_horizon_forecast_summary": forecast_summary,
            "confidence": confidence,
            "effective_confidence": effective_confidence,
            "uncertainty": uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "learnable_uncertainty": learnable_uncertainty,
            "uncertainty_source": future.uncertainty_source,
            "uncertainty_components": future.uncertainty_components,
            "event_probabilities": {
                key: float(value)
                for key, value in future.event_probabilities.items()
            },
            "progress_quantiles": [
                float(value) for value in future.progress_quantiles
            ],
            "event_time_s": [float(value) for value in future.event_time_s],
            "multi_horizon_predictions": multi_horizon_predictions,
            "temporal_utility_paths": temporal_utility_paths,
        })
    return observation, candidates


def _tactical_action_weights(preset_name: str) -> dict[str, float]:
    """Translate a tactical vector into an auditable high-level policy mixture."""
    preset = TACTICAL_PRESETS[resolve_tactical_preset(preset_name)]
    raw = {
        "hold": (
            0.08
            + 0.24 * preset.get("possession_orientation", 0.5)
            + 0.12 * preset.get("low_block", 0.0)
        ),
        "pass": (
            0.18
            + 0.30 * preset.get("build_up_short", 0.5)
            + 0.18 * preset.get("through_ball_bias", 0.5)
        ),
        "cross": (
            0.05
            + 0.30 * preset.get("cross_frequency", 0.4)
            + 0.20 * preset.get("wing_focus", 0.5)
        ),
        "shot": (
            0.06
            + 0.18 * preset.get("verticality", 0.5)
            + 0.16 * preset.get("risk_budget", 0.5)
        ),
    }
    total = sum(raw.values())
    return {key: float(value / total) for key, value in raw.items()}


def _condition_on_opponent_hypothesis(
    observation: np.ndarray,
    *,
    team_is_home: bool,
    hypothesis: str,
) -> np.ndarray:
    conditioned = np.asarray(observation, dtype=np.float32).copy()
    # Observation layout reserves [298:302] for home controls and [302:306]
    # for away controls.  Only the opponent slice is intervened upon.
    start = 302 if team_is_home else 298
    conditioned[start:start + 4] = tactic_feature_vector(hypothesis)
    return conditioned


def _attach_opponent_counterfactuals(
    runtime,
    state,
    team_id: str,
    observation: np.ndarray,
    candidates: list[dict[str, Any]],
    belief: dict[str, Any] | None,
    *,
    horizon_s: float,
    uncertainty_penalty: float,
) -> None:
    """Evaluate each action under every latent opponent tactic."""
    if not belief or not belief.get("posterior"):
        return
    if not any(
        float(candidate.get("effective_confidence", 0.0)) > 0.0
        for candidate in candidates
    ):
        return
    team_is_home = str(team_id) == str(state.home.team_id)
    for candidate in candidates:
        candidate["state_conditioned_risk_adjusted_value"] = float(
            candidate["risk_adjusted_value"]
        )
        candidate["opponent_hypothesis_values"] = {}
    for hypothesis in OPPONENT_HYPOTHESES:
        conditioned = _condition_on_opponent_hypothesis(
            observation,
            team_is_home=team_is_home,
            hypothesis=hypothesis,
        )
        for candidate in candidates:
            action_name = str(candidate["action"])
            action = encode_high_level_action(
                action_name,
                target=np.asarray(candidate["target"], dtype=np.float32),
                horizon_s=horizon_s,
            )
            action[13] = float(candidate["physics_prior"])
            future = runtime.predict_future(
                conditioned,
                action,
                action_kind=action_name,
                horizon_s=horizon_s,
            )
            expected = (
                runtime.score_shot_action(
                    conditioned, action, attacking_home=team_is_home,
                )
                if action_name == "shot"
                else runtime.score_action(conditioned, action)
            )
            value = (
                finite_float(expected, 0.0)
                - uncertainty_penalty * float(future.state_uncertainty)
                - 0.15 * float(future.event_probabilities["turnover"])
                + float(candidate.get("multi_horizon_forecast_adjustment", 0.0))
            )
            candidate["opponent_hypothesis_values"][hypothesis] = finite_float(
                value, -1.0
            )
    reweight_counterfactual_candidates(
        {"candidates": candidates}, dict(belief["posterior"]),
    )
    entropy = float(np.clip(belief.get("normalized_entropy", 1.0), 0.0, 1.0))
    for candidate in candidates:
        sensitivity = float(np.clip(
            candidate.get("opponent_belief_value_std", 0.0) / 0.25,
            0.0,
            1.0,
        ))
        candidate["opponent_belief_entropy"] = entropy
        candidate["opponent_hypothesis_discrimination"] = (
            entropy * sensitivity
        )


def build_prematch_tactical_packet(
    runtime,
    state,
    team_id: str,
    *,
    horizon_s: float = 20.0,
    candidate_presets: tuple[str, ...] = PREMATCH_TACTICAL_CANDIDATES,
    uncertainty_penalty: float = 0.25,
    opponent_belief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank tactical policy mixtures using bounded world-model evidence."""
    if runtime is None:
        return {
            "version": DECISION_PACKET_VERSION,
            "available": False,
            "reason": "world_model_unavailable",
            "recommended_tactical_preset": "none",
            "candidates": [],
        }
    try:
        normalized = tuple(dict.fromkeys(
            resolve_tactical_preset(name) for name in candidate_presets
        ))
        observation, action_evidence = _evaluate_action_candidates(
            runtime, state, team_id, horizon_s=horizon_s,
            uncertainty_penalty=uncertainty_penalty,
            prediction_horizons_s=(0.0,),
            residual_memory=None,
        )
        opponent_belief = opponent_belief or update_opponent_belief(
            state, team_id,
        )
        _attach_opponent_counterfactuals(
            runtime,
            state,
            team_id,
            observation,
            action_evidence,
            opponent_belief,
            horizon_s=horizon_s,
            uncertainty_penalty=uncertainty_penalty,
        )
        second_order_game = attach_second_order_game(
            action_evidence,
            opponent_belief,
            getattr(state, "_wm_opponent_response_memory", None),
        )
        trajectory_values, trajectory_audit = build_predicted_state_continuations(
            runtime,
            observation,
            action_evidence,
            attacking_home=(team_id == state.home.team_id),
            horizon_s=horizon_s,
            uncertainty_penalty=uncertainty_penalty,
        )
        second_order_game = attach_second_order_game(
            action_evidence,
            opponent_belief,
            getattr(state, "_wm_opponent_response_memory", None),
            continuation_value_matrices=trajectory_values,
            trajectory_audit=trajectory_audit,
        )
        by_action = {item["action"]: item for item in action_evidence}
        distributional_frontiers = build_distributional_action_frontiers(
            action_evidence
        )
        tactical_candidates = []
        for preset_name in normalized:
            weights = _tactical_action_weights(preset_name)

            def weighted(field: str) -> float:
                return float(sum(
                    weights[action] * by_action[action][field]
                    for action in COACH_ACTIONS
                ))

            event_probabilities = {
                event: float(sum(
                    weights[action]
                    * by_action[action]["event_probabilities"].get(event, 0.0)
                    for action in COACH_ACTIONS
                ))
                for event in ("retain", "turnover", "shot", "foul", "out")
            }
            vector = TACTICAL_PRESETS[preset_name]
            fatigue_cost = float(np.clip(
                0.55 * vector.get("pressing_intensity", 0.5)
                + 0.25 * vector.get("tempo", 0.5)
                + 0.20 * vector.get("counterpress", 0.5),
                0.0, 1.0,
            ))
            structural_risk = float(np.clip(
                0.45 * vector.get("risk_budget", 0.5)
                + 0.35 * vector.get("line_height", 0.5)
                + 0.20 * (1.0 - vector.get("compactness", 0.5)),
                0.0, 1.0,
            ))
            model_value = weighted("risk_adjusted_value")
            tactical_value = (
                model_value
                + 0.10 * event_probabilities["shot"]
                + 0.06 * event_probabilities["retain"]
                - 0.08 * fatigue_cost
                - 0.10 * structural_risk
            )
            tactical_candidates.append({
                "tactical_preset": preset_name,
                "policy_action_weights": weights,
                "model_value": finite_float(model_value, -1.0),
                "risk_adjusted_value": finite_float(tactical_value, -1.0),
                "effective_confidence": weighted("effective_confidence"),
                "uncertainty": weighted("uncertainty"),
                "event_probabilities": event_probabilities,
                "fatigue_cost_proxy": fatigue_cost,
                "structural_risk_proxy": structural_risk,
            })
        eligible = [
            candidate for candidate in tactical_candidates
            if candidate["effective_confidence"] > 0.0
        ]
        best = max(
            eligible, key=lambda item: item["risk_adjusted_value"],
            default=None,
        )
        return {
            "version": DECISION_PACKET_VERSION,
            "available": best is not None,
            "reason": "ok" if best is not None else "quality_gate_closed",
            "team_id": team_id,
            "checkpoint_signature": str(
                getattr(runtime, "checkpoint_signature", "runtime_unspecified")
            ),
            "horizon_s": float(horizon_s),
            "evaluation_scope": "short_horizon_tactical_policy_proxy",
            "observation_coverage": observation_coverage(observation),
            "recommended_tactical_preset": (
                best["tactical_preset"] if best else "none"
            ),
            "recommendation_confidence": (
                best["effective_confidence"] if best else 0.0
            ),
            "candidates": tactical_candidates,
            "action_evidence": action_evidence,
            "distributional_action_frontiers": distributional_frontiers,
            "opponent_belief": opponent_belief,
            "second_order_game": second_order_game,
            "limitations": [
                "Not a full-match win-probability forecast.",
                "Opponent intent is latent and represented as a changing posterior.",
                "Opponent-conditioned values are model counterfactuals, not observed outcomes.",
                "Tactical presets are evaluated as high-level action mixtures.",
            ],
            "policy": (
                "Select only an evaluated candidate, use model uncertainty, "
                "and explain any disagreement with the recommendation."
            ),
        }
    except Exception as exc:
        return {
            "version": DECISION_PACKET_VERSION,
            "available": False,
            "reason": f"world_model_error:{type(exc).__name__}",
            "recommended_tactical_preset": "none",
            "candidates": [],
        }


def build_coach_decision_packet(
    runtime,
    state,
    team_id: str,
    *,
    horizon_s: float = 10.0,
    uncertainty_penalty: float = 0.25,
    outcome_horizons_s: tuple[float, ...] = (0.0, 60.0, 180.0),
    residual_min_samples: int = 6,
    residual_memory=None,
    environment_signature: str = "environment_unspecified",
    active_learning_config: dict[str, Any] | None = None,
    opponent_belief: dict[str, Any] | None = None,
    trajectory_branch_budget: int = 48,
) -> dict[str, Any]:
    """Compare strategic candidates without granting the LLM direct state writes."""
    from src.match_engine.world_model.outcome_calibration import (
        policy_outcome_calibration,
    )

    outcome_calibration = policy_outcome_calibration(
        getattr(state, "_wm_coach_decision_adoption", None) or [],
        min_samples=residual_min_samples,
        outcome_family="regime",
    )
    if runtime is None:
        return {
            "version": DECISION_PACKET_VERSION,
            "available": False,
            "reason": "world_model_unavailable",
            "recommended_action": "none",
            "candidates": [],
            "policy_outcome_calibration": outcome_calibration,
            "active_learning": {
                "version": 1,
                "eligible": False,
                "reason": "world_model_unavailable",
            },
        }
    memory_rejection = None
    if residual_memory is not None:
        memory_checkpoint = getattr(
            residual_memory, "checkpoint_signature", None,
        )
        memory_environment = getattr(
            residual_memory, "environment_signature", None,
        )
        runtime_checkpoint = str(getattr(
            runtime, "checkpoint_signature", "runtime_unspecified",
        ))
        if (
            memory_checkpoint is not None
            and str(memory_checkpoint) != runtime_checkpoint
        ):
            memory_rejection = "checkpoint_mismatch"
            residual_memory = None
        elif (
            memory_environment is not None
            and str(memory_environment) != str(environment_signature)
        ):
            memory_rejection = "policy_environment_mismatch"
            residual_memory = None
    try:
        observation, candidates = _evaluate_action_candidates(
            runtime, state, team_id, horizon_s=horizon_s,
            uncertainty_penalty=uncertainty_penalty,
            prediction_horizons_s=outcome_horizons_s,
            residual_memory=residual_memory,
        )
        opponent_belief = opponent_belief or update_opponent_belief(
            state, team_id,
        )
        _attach_opponent_counterfactuals(
            runtime,
            state,
            team_id,
            observation,
            candidates,
            opponent_belief,
            horizon_s=horizon_s,
            uncertainty_penalty=uncertainty_penalty,
        )
        second_order_game = attach_second_order_game(
            candidates,
            opponent_belief,
            getattr(state, "_wm_opponent_response_memory", None),
        )
        trajectory_values, trajectory_audit = build_predicted_state_continuations(
            runtime,
            observation,
            candidates,
            attacking_home=(team_id == state.home.team_id),
            horizon_s=horizon_s,
            uncertainty_penalty=uncertainty_penalty,
            max_branch_evaluations=trajectory_branch_budget,
        )
        second_order_game = attach_second_order_game(
            candidates,
            opponent_belief,
            getattr(state, "_wm_opponent_response_memory", None),
            continuation_value_matrices=trajectory_values,
            trajectory_audit=trajectory_audit,
        )
        from src.match_engine.world_model.active_learning import (
            build_active_learning_advice,
        )
        from src.match_engine.world_model.policy_outcomes import (
            capture_policy_outcome_baseline,
        )

        learning_context = capture_policy_outcome_baseline(
            state, team_id=team_id,
        )
        learning_context["team_id"] = str(team_id)
        active_learning = build_active_learning_advice(
            candidates,
            getattr(state, "_wm_coach_decision_adoption", None) or [],
            context=learning_context,
            config=active_learning_config,
        )
        eligible = [
            candidate for candidate in candidates
            if candidate["effective_confidence"] > 0.0
        ]
        best = max(
            eligible, key=lambda item: item["risk_adjusted_value"],
            default=None,
        )
        distributional_frontiers = build_distributional_action_frontiers(
            candidates
        )
        return {
            "version": DECISION_PACKET_VERSION,
            "available": best is not None,
            "reason": "ok" if best is not None else "quality_gate_closed",
            "team_id": team_id,
            "checkpoint_signature": str(
                getattr(runtime, "checkpoint_signature", "runtime_unspecified")
            ),
            "environment_signature": str(environment_signature),
            "horizon_s": float(horizon_s),
            "observation_coverage": observation_coverage(observation),
            "recommended_action": best["action"] if best else "none",
            "recommendation_confidence": (
                best["effective_confidence"] if best else 0.0
            ),
            "candidates": candidates,
            "distributional_action_frontiers": distributional_frontiers,
            "opponent_belief": opponent_belief,
            "second_order_game": second_order_game,
            "active_learning": active_learning,
            "decision_context": learning_context,
            "online_calibration": _online_calibration_diagnostics(runtime),
            "policy_outcome_calibration": outcome_calibration,
            "contextual_residual_memory": (
                residual_memory.summary()
                if residual_memory is not None else {
                    "version": 6,
                    "active_groups": 0,
                    "residual_rows": 0,
                    "reason": (
                        memory_rejection or "no_compatible_history"
                    ),
                }
            ),
            "opponent_information_query_calibration": (
                residual_memory.opponent_information_calibration_contracts(
                    horizons={
                        key
                        for candidate in candidates
                        for key in (
                            candidate.get("multi_horizon_predictions") or {}
                        )
                    }
                )
                if residual_memory is not None and hasattr(
                    residual_memory,
                    "opponent_information_calibration_contracts",
                ) else {
                    "version": 1,
                    "checkpoint_signature": str(getattr(
                        runtime, "checkpoint_signature", "runtime_unspecified",
                    )),
                    "environment_signature": str(environment_signature),
                    "horizons": {},
                    "reason": "no_compatible_history",
                }
            ),
            "llm_semantic_critic_memory": (
                getattr(state, "_wm_llm_critic_memory").summary()
                if getattr(state, "_wm_llm_critic_memory", None) is not None
                else {
                    "version": 1,
                    "active_profiles": 0,
                    "reason": "no_compatible_critic_history",
                }
            ),
            "policy": (
                "Use as uncertain evidence; retain bounded controls and never "
                "invent score, xG, or outcome facts."
            ),
        }
    except Exception as exc:
        return {
            "version": DECISION_PACKET_VERSION,
            "available": False,
            "reason": f"world_model_error:{type(exc).__name__}",
            "recommended_action": "none",
            "candidates": [],
            "active_learning": {
                "version": 1,
                "eligible": False,
                "reason": "world_model_error",
            },
        }
