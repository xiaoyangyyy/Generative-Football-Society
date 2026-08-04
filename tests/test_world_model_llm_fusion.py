"""World-model evidence flows into bounded System 2 coach decisions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import CognitiveTriggerEvent, ENTITY_TIER_COACH
from src.match_engine.cognitive.executor import (
    CognitiveExecutor,
    _policy_intervention_strength,
)
from src.match_engine.state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from src.match_engine.world_model.decision_support import (
    COACH_ACTIONS,
    PREMATCH_TACTICAL_CANDIDATES,
    build_coach_decision_packet,
    build_prematch_tactical_packet,
)
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.probabilistic import ProbabilisticFuture
from src.match_engine.world_model.opponent_information_feedback import (
    opponent_information_feedback_is_valid,
)
from src.match_engine.world_model.opponent_information_query import (
    opponent_information_question_menu_is_valid,
)
from src.match_engine.world_model.opponent_information_policy import (
    opponent_information_cognitive_policy_is_valid,
)
from src.match_engine.world_model.belief_space_meta_planner import (
    belief_space_meta_plan_is_valid,
)
from src.match_engine.world_model.llm_deliberation_encouragement import (
    task_selection_value_memory_is_valid,
)
from src.match_engine.world_model.llm_decision_brief import (
    build_llm_decision_brief,
    compact_world_model_facts_for_llm,
    deliberation_portfolio_objective,
    llm_decision_brief_diagnostics,
    llm_decision_brief_metadata_is_valid,
    llm_decision_brief_self_is_valid,
    llm_decision_brief_is_valid,
)
from src.simulation.llm_engine import SimulationLLM
from src.simulation.tactics_sync import reconcile_world_model_tactical_choice


def _state():
    home = TeamAffectiveState(
        team_id="Home",
        coach=CoachAffectiveState(
            team_id="Home",
            tactical_current={"pressing_intensity": 0.5},
        ),
    )
    away = TeamAffectiveState(
        team_id="Away", coach=CoachAffectiveState(team_id="Away"),
    )
    return MatchAffectiveState(
        home=home, away=away,
        referee=RefereeAffectiveState(strictness_base=0.55),
        crowd=CrowdState(),
    )


class _Runtime:
    def __init__(self, confidence=0.8):
        self.confidence = confidence

    def encode_state(self, state, *, attacking_home):
        return np.full(OBS_DIM, 0.4 if attacking_home else 0.6, dtype=np.float32)

    def planner_confidence(self, observation, kind="general"):
        return self.confidence

    def predict_future(self, observation, action, *, action_kind, horizon_s):
        shot = 0.45 if action_kind == "shot" else 0.05
        retain = {"hold": 0.82, "pass": 0.72, "cross": 0.48, "shot": 0.35}[action_kind]
        raw = np.array([retain, 1.0 - retain, shot, 0.02, 0.01])
        raw /= raw.sum()
        return ProbabilisticFuture(
            dict(zip(("retain", "turnover", "shot", "foul", "out"), raw)),
            (horizon_s * 0.5, 1.0),
            (-0.05, 0.1, 0.25),
            0.1 if action_kind == "shot" else 0.2,
        )

    def score_action(self, observation, action):
        return float(np.dot(action[:4], np.array([0.4, 0.0, 0.1, 0.3])))

    def score_shot_action(self, observation, action, *, attacking_home):
        return 0.72

    def online_calibration_diagnostics(self):
        return {
            "calibration_target": "same_tick_next_observation",
            "branches": {"general": {"samples": 12, "trust_factor": 0.8}},
        }


def test_decision_packet_compares_all_actions_and_recommends_risk_adjusted_best():
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    assert packet["available"]
    assert packet["recommended_action"] == "shot"
    assert {candidate["action"] for candidate in packet["candidates"]} == set(COACH_ACTIONS)
    assert all(0.0 <= candidate["uncertainty"] <= 1.0 for candidate in packet["candidates"])
    assert packet["observation_coverage"] > 0.9
    assert packet["online_calibration"]["branches"]["general"][
        "trust_factor"
    ] == 0.8
    assert packet["policy_outcome_calibration"]["samples"] == 0
    assert "active_learning" in packet
    assert "opponent_information_feedback" in packet
    assert not packet["opponent_information_feedback"]["available"]
    assert opponent_information_question_menu_is_valid(
        packet["opponent_information_question_menu"], packet,
    )
    assert packet["opponent_information_question_menu"]["available"]
    assert opponent_information_cognitive_policy_is_valid(
        packet["opponent_information_cognitive_policy"], packet,
    )
    assert all("active_learning" in candidate for candidate in packet["candidates"])
    assert all(
        "transition_epistemic_uncertainty" in candidate["active_learning"]
        for candidate in packet["candidates"]
    )
    for candidate in packet["candidates"]:
        assert set(candidate["multi_horizon_predictions"]) == {
            "transition", "60s", "180s",
        }
        assert all(
            np.isfinite(prediction["policy_utility"])
            for prediction in candidate["multi_horizon_predictions"].values()
        )
        assert abs(candidate["multi_horizon_forecast_adjustment"]) <= 0.1


def test_trained_runtime_builds_member_utility_frontiers_end_to_end():
    pytest.importorskip("torch")
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.inference import WorldModelRuntime
    from src.match_engine.world_model.model import build_model

    cfg = WorldModelConfig(
        latent_dim=16, hidden_dim=32, ensemble_size=2,
        transition_ensemble_size=2,
    )
    runtime = WorldModelRuntime(build_model(cfg), cfg, {
        "validation": {
            "planner_quality": 0.8,
            "pass_planner_quality": 0.8,
            "shot_planner_quality": 0.8,
            "weighted_obs_mse": 0.02,
        },
    })
    packet = build_coach_decision_packet(
        runtime, _state(), "Home", outcome_horizons_s=(0.0, 60.0),
    )

    brief = build_llm_decision_brief(packet)
    assert llm_decision_brief_is_valid(brief, packet)
    assert belief_space_meta_plan_is_valid(packet["belief_space_meta_plan"])
    compact_meta_plan = brief["belief_space_meta_plan"]
    assert compact_meta_plan["recommended_option_id"] in {
        option["option_id"] for option in compact_meta_plan["options"]
    }
    assert compact_meta_plan["applied_option_id"] == (
        packet["belief_space_meta_plan"]["applied_option"]["option_id"]
    )
    assert all(
        "action_values" not in (option.get("route") or {})
        for option in compact_meta_plan["options"]
    )
    assert len(json.dumps(brief)) < 0.70 * len(json.dumps(packet))
    assert [row["action"] for row in brief["candidates"]] == [
        row["action"] for row in packet["candidates"]
    ]
    for full, compact in zip(packet["candidates"], brief["candidates"]):
        assert set(compact["multi_horizon_predictions"]) == set(
            full["multi_horizon_predictions"]
        )
        for prediction in compact["multi_horizon_predictions"].values():
            distribution = prediction["distributional_policy_utility"]
            assert "member_values" not in distribution
            assert "decision_scenario_values" not in distribution
        assert "scenario_utility_paths" not in compact[
            "temporal_utility_paths"
        ]
    projected = compact_world_model_facts_for_llm({
        "score": "0-0",
        "world_model_decision_support": packet,
        "world_model_llm_decision_brief": brief,
    })
    assert projected["world_model_decision_support"] == brief
    assert "world_model_llm_decision_brief" not in projected
    assert "llm_decision_brief" not in packet
    agenda = brief["deliberation_agenda"]
    assert task_selection_value_memory_is_valid(
        packet["task_selection_value_memory"]
    )
    assert brief["task_selection_value_memory"] == packet[
        "task_selection_value_memory"
    ]
    assert brief["opponent_information_question_menu"] == packet[
        "opponent_information_question_menu"
    ]
    assert brief["opponent_information_cognitive_policy"] == packet[
        "opponent_information_cognitive_policy"
    ]
    assert len(agenda["recommended_focus"]) <= 3
    eligible_tasks = {
        task["task"] for task in agenda["tasks"] if task["eligible"]
    }
    assert set(agenda["recommended_focus"]) <= eligible_tasks
    assert agenda["version"] == 2
    assert agenda["recommended_focus"] == agenda[
        "recommended_portfolios_by_size"
    ][str(len(agenda["recommended_focus"]))]["tasks"]
    for size, portfolio in agenda[
        "recommended_portfolios_by_size"
    ].items():
        assert len(portfolio["tasks"]) == int(size)
        assert portfolio["expected_compute_credits"] <= 6
        assert set(portfolio["tasks"]) <= eligible_tasks
    for task in agenda["tasks"]:
        assert task["reasoning_domain"]
        assert 1 <= task["expected_compute_credits"] <= 2
        assert -0.08 <= task["learned_compute_value_adjustment"] <= 0.08
        assert -0.06 <= task["learned_task_selection_adjustment"] <= 0.06
        assert 0.0 <= task["portfolio_score"] <= 1.0
    same_domain = {
        "a": {"portfolio_score": 0.5, "reasoning_domain": "one"},
        "b": {"portfolio_score": 0.5, "reasoning_domain": "one"},
    }
    diverse = {
        **same_domain,
        "b": {"portfolio_score": 0.5, "reasoning_domain": "two"},
    }
    assert deliberation_portfolio_objective(["a", "b"], diverse) == (
        deliberation_portfolio_objective(["a", "b"], same_domain) + 0.04
    )
    tampered_brief = json.loads(json.dumps(brief))
    tampered_brief["deliberation_agenda"]["recommended_focus"] = [
        "rewrite_world_model"
    ]
    assert not llm_decision_brief_is_valid(tampered_brief, packet)
    tampered_portfolio = json.loads(json.dumps(brief))
    tampered_portfolio["deliberation_agenda"][
        "recommended_portfolios_by_size"
    ]["1"]["objective"] += 1.0
    assert not llm_decision_brief_is_valid(tampered_portfolio, packet)
    repaired_projection = compact_world_model_facts_for_llm({
        "world_model_decision_support": packet,
        "world_model_llm_decision_brief": tampered_brief,
    })
    assert repaired_projection["world_model_decision_support"] == brief

    assert packet["available"]
    frontiers = packet["distributional_action_frontiers"]["horizons"]
    assert frontiers["transition"]["available"]
    assert frontiers["60s"]["available"]
    assert frontiers["60s"]["pareto_actions"]
    for candidate in packet["candidates"]:
        temporal = candidate["temporal_utility_paths"]
        assert temporal["available"]
        assert temporal["member_identity_preserved_across_horizons"]
        assert not temporal["temporal_dependence_learned"]
        for prediction in candidate["multi_horizon_predictions"].values():
            distribution = prediction["distributional_policy_utility"]
            assert distribution["available"]
            assert distribution["mean_utility"] == pytest.approx(
                prediction["policy_utility"]
            )
            assert distribution[
                "spread_preserved_by_location_alignment"
            ]


def test_decision_packet_closes_recommendation_when_quality_gate_is_closed():
    packet = build_coach_decision_packet(
        _Runtime(confidence=0.0), _state(), "Home",
    )
    assert not packet["available"]
    assert packet["reason"] == "quality_gate_closed"
    assert packet["recommended_action"] == "none"


def test_decision_packet_rejects_incompatible_residual_memory():
    class _IncompatibleMemory:
        checkpoint_signature = "different-checkpoint"
        environment_signature = "env-a"

        def calibrate(self, prediction, **kwargs):
            raise AssertionError("incompatible memory must not be consulted")

    packet = build_coach_decision_packet(
        _Runtime(),
        _state(),
        "Home",
        residual_memory=_IncompatibleMemory(),
        environment_signature="env-b",
    )

    assert packet["available"]
    assert packet["contextual_residual_memory"]["reason"] == (
        "checkpoint_mismatch"
    )


def test_prematch_packet_ranks_auditable_tactical_policy_mixtures():
    packet = build_prematch_tactical_packet(_Runtime(), _state(), "Home")
    assert packet["available"]
    assert packet["evaluation_scope"] == "short_horizon_tactical_policy_proxy"
    assert len(packet["candidates"]) == len(PREMATCH_TACTICAL_CANDIDATES)
    assert packet["recommended_tactical_preset"] in PREMATCH_TACTICAL_CANDIDATES
    for candidate in packet["candidates"]:
        assert sum(candidate["policy_action_weights"].values()) == pytest.approx(1.0)
        assert 0.0 <= candidate["uncertainty"] <= 1.0
        assert 0.0 <= candidate["fatigue_cost_proxy"] <= 1.0


def test_prematch_packet_closes_quality_gate_without_model_confidence():
    packet = build_prematch_tactical_packet(
        _Runtime(confidence=0.0), _state(), "Home",
    )
    assert not packet["available"]
    assert packet["reason"] == "quality_gate_closed"
    assert packet["recommended_tactical_preset"] == "none"


def test_tactical_choice_is_constrained_to_evaluated_candidates():
    packet = build_prematch_tactical_packet(
        _Runtime(), _state(), "Home",
        candidate_presets=("balanced", "low_block"),
    )
    packet["fusion_reliability"] = {
        "evidence_tier": "matched_seed_simulator",
        "guidance": "moderate_support",
        "adjusted_recommendation_trust": 0.64,
        "matched_seed_samples": 16,
        "historical_match_records": 5,
    }
    packet["strategy_memory"] = {"opponent_specific_matches": 2}
    reconciled = reconcile_world_model_tactical_choice(
        {
            "tactical_preset": "route_one",
            "world_model_rationale": "x" * 500,
        },
        packet,
    )
    assert reconciled["tactical_preset"] == packet["recommended_tactical_preset"]
    assert reconciled["world_model_audit"]["selection_constrained"]
    assert reconciled["world_model_audit"]["selected_evidence"]
    assert reconciled["world_model_audit"]["balanced_evidence"]
    assert reconciled["world_model_audit"]["recommendation_margin"] >= 0.0
    assert reconciled["world_model_audit"]["historical_evidence_tier"] == (
        "matched_seed_simulator"
    )
    assert reconciled["world_model_audit"]["matched_seed_samples"] == 16
    assert reconciled["world_model_audit"]["opponent_specific_history"] == 2
    assert len(reconciled["world_model_rationale"]) == 300


class _PromptGateway:
    def __init__(self):
        self.client = object()
        self.config = SimpleNamespace(model="fake", timeout_s=1, max_retries=1)
        self.user_prompt = ""
        self.system_prompt = ""

    def complete(self, system_prompt, user_prompt, **kwargs):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return json.dumps({
            "formation": "4-3-3",
            "style": "balanced",
            "tactical_preset": "balanced",
            "reasoning": "Use the bounded evidence.",
            "controls": {},
            "world_model_rationale": "The quality gate is open.",
        })

    @staticmethod
    def extract_json_object(content):
        return json.loads(content)


def test_prematch_llm_prompt_receives_world_model_packet():
    gateway = _PromptGateway()
    llm = SimulationLLM(gateway=gateway)
    packet = build_prematch_tactical_packet(_Runtime(), _state(), "Home")
    response = json.loads(llm.coach_decide_tactics(
        "Home", "ready", "Away", "tired",
        world_model_decision_support=packet,
    ))
    assert "WORLD_MODEL_DECISION_SUPPORT" in gateway.user_prompt
    assert packet["recommended_tactical_preset"] in gateway.user_prompt
    assert response["world_model_rationale"] == "The quality gate is open."


def test_in_match_prompt_exposes_non_controlling_change_explanation_contract():
    gateway = _PromptGateway()
    llm = SimulationLLM(gateway=gateway)
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    packet["engine_only_scenario_sentinel"] = (
        "FULL_MEMBER_ARRAY_MUST_NOT_ENTER_LLM_PROMPT"
    )

    llm.coach_in_match_plan(
        "Home", {"world_model_decision_support": packet}, "shape_change",
    )

    assert "opponent_change_claim" in gateway.user_prompt
    assert "opponent_response_hypothesis" in gateway.user_prompt
    assert "world_model_critique" in gateway.user_prompt
    assert "world_model_event_hypothesis" in gateway.user_prompt
    assert "world_model_event_option" in gateway.user_prompt
    assert "world_model_contrastive_claim" in gateway.user_prompt
    assert "world_model_risk_constraint" in gateway.user_prompt
    assert "world_model_distributional_claim" in gateway.user_prompt
    assert "world_model_risk_preference" in gateway.user_prompt
    assert "world_model_deliberation_focus" in gateway.user_prompt
    assert "nominal consistency is not robustness" in gateway.system_prompt
    assert "trajectory_modes" in gateway.system_prompt
    assert "CRPS" in gateway.system_prompt
    assert "member-predicted states" in gateway.system_prompt
    assert "mismatched actions" in gateway.system_prompt
    assert "checks faithfulness" in gateway.system_prompt
    assert "falsifiable forecast-residual claim" in gateway.system_prompt
    assert "shadow-only" in gateway.system_prompt
    assert "two-ply belief-space policy proxy" in gateway.system_prompt
    assert "from_preset" in gateway.user_prompt
    assert "cannot" in gateway.system_prompt
    assert "change_point" in gateway.system_prompt
    assert "deliberation_agenda" in gateway.user_prompt
    assert "brief_digest" in gateway.user_prompt
    assert "source_packet_fingerprint" in gateway.user_prompt
    assert "recommended_focus" in gateway.system_prompt
    assert "unfocused" in gateway.system_prompt
    assert "FULL_MEMBER_ARRAY_MUST_NOT_ENTER_LLM_PROMPT" not in (
        gateway.user_prompt
    )


def test_contrastive_repair_prompt_freezes_action_and_all_control_fields():
    gateway = _PromptGateway()
    llm = SimulationLLM(gateway=gateway)
    llm.revise_world_model_contrastive_claim(
        team_name="Home",
        immutable_selected_action="pass",
        counterevidence={
            "claimed_directional_effect": -0.04,
            "directionally_faithful": False,
        },
        revision_contract={
            "evaluated_actions": ["pass", "hold"],
            "allowed_factors": ["score_context"],
            "revision_calls_remaining": 1,
        },
    )
    assert "selected action 'pass' is immutable" in gateway.system_prompt
    assert "cannot change" in gateway.system_prompt
    assert "revision_calls_remaining" in gateway.user_prompt
    assert "WORLD_MODEL_COUNTEREVIDENCE" in gateway.user_prompt


class _LLM:
    def __init__(self):
        self.facts = None

    def coach_in_match_plan(self, team_name, facts, kind):
        self.facts = facts
        return json.dumps({
            "reasoning": "Use the supported chance while keeping changes bounded.",
            "confidence": 0.8,
            "controls_delta": {"pressing_intensity": 0.08},
            "world_model_action": "shot",
            "world_model_rationale": "Highest risk-adjusted value with low uncertainty.",
        })


class _ExplorationRuntime(_Runtime):
    def score_action(self, observation, action):
        return 0.68


class _ExplorationLLM:
    def __init__(self):
        self.facts = None

    def coach_in_match_plan(self, team_name, facts, kind):
        self.facts = facts
        advice = facts["world_model_decision_support"]["active_learning"]
        return json.dumps({
            "reasoning": "Use the bounded learning opportunity.",
            "confidence": 0.8,
            "controls_delta": {},
            "world_model_action": advice["exploration_action"],
            "world_model_decision_mode": "explore",
            "world_model_rationale": "Low-regret information gain.",
        })


class _InvalidExplorationLLM(_ExplorationLLM):
    def coach_in_match_plan(self, team_name, facts, kind):
        self.facts = facts
        return json.dumps({
            "reasoning": "Request an unsupported experiment.",
            "confidence": 0.8,
            "controls_delta": {},
            "world_model_action": "shot",
            "world_model_decision_mode": "explore",
        })


def test_executor_injects_world_model_evidence_before_llm_and_applies_plan():
    llm = _LLM()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True), llm,
        world_model_runtime=_Runtime(),
    )
    trigger = CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    )
    state = _state()
    record = executor.process_trigger(trigger, state)
    adoption = state._wm_coach_decision_adoption[0]
    packet = llm.facts["world_model_decision_support"]
    brief = llm.facts["world_model_llm_decision_brief"]
    assert llm_decision_brief_self_is_valid(brief)
    assert adoption["llm_decision_brief_context"]["brief_digest"] == (
        brief["brief_digest"]
    )
    assert adoption["llm_decision_brief_context"][
        "full_packet_retained_for_engine_audit"
    ]
    assert llm_decision_brief_metadata_is_valid(
        adoption["llm_decision_brief_context"]
    )
    brief_diagnostics = llm_decision_brief_diagnostics([[adoption]])
    assert brief_diagnostics["brief_contexts"] == 1
    assert brief_diagnostics["all_briefs_digest_linked_and_bounded"]
    assert packet["recommended_action"] == "shot"
    assert record.plan["world_model_action"] == "shot"
    assert record.plan["world_model_adoption_id"]
    assert len(state._wm_coach_decision_adoption) == 1
    assert adoption["intervention_enabled"]
    assert 0.0 < adoption["intervention_strength"] <= 0.35
    assert record.plan["world_model_policy_intervention_enabled"]
    assert record.plan["world_model_policy_intervention_strength"] == adoption[
        "intervention_strength"
    ]
    assert record.plan["world_model_policy_experiment_arm"] in {
        "treatment", "control",
    }
    assert record.plan["world_model_policy_reliability_factor"] == 1.0
    assert record.plan["world_model_prediction_trust_factor"] == 1.0
    assert adoption["action_outcome_predictions"]["shot"]["60s"]
    assert record.applied
    assert state.home.coach.tactical_current["pressing_intensity"] > 0.5


def test_executor_audits_llm_response_branch_and_persists_planning_context():
    class ResponseLLM:
        def coach_in_match_plan(self, team_name, facts, kind):
            packet = facts["world_model_decision_support"]
            action = packet["recommended_action"]
            candidate = next(
                item for item in packet["candidates"]
                if item["action"] == action
            )
            posterior = candidate["opponent_response_prediction"][
                "response_posterior"
            ]
            response = max(posterior, key=posterior.get)
            return json.dumps({
                "reasoning": "Stress the model-supported response branch.",
                "confidence": 0.8,
                "controls_delta": {},
                "world_model_action": action,
                "opponent_response_hypothesis": {
                    "if_action": action,
                    "response_preset": response,
                    "confidence": 0.8,
                    "rationale": "Conditional scenario only.",
                },
            })

    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True, world_model_action_control_rate=0.0,
        ),
        ResponseLLM(),
        world_model_runtime=_Runtime(),
    )
    state = _state()
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)
    audit = record.plan["opponent_response_hypothesis_audit"]
    context = state._wm_coach_decision_adoption[-1][
        "opponent_response_context"
    ]

    assert audit["accepted"]
    assert audit["bounded_influence"] <= 0.15
    assert not audit["can_update_response_memory"]
    assert context["prediction"]["action"] == record.plan["world_model_action"]
    assert context["best_continuation_action"] in {
        "hold", "pass", "cross", "shot",
    }
    assert context["llm_hypothesis_audit"]["accepted"]


def test_executor_registers_falsifiable_llm_critique_in_shadow_mode():
    class CriticLLM:
        def coach_in_match_plan(self, team_name, facts, kind):
            packet = facts["world_model_decision_support"]
            action = packet["recommended_action"]
            candidate = next(
                item for item in packet["candidates"]
                if item["action"] == action
            )
            horizon = next(iter(candidate["multi_horizon_predictions"]))
            return json.dumps({
                "reasoning": "Register a falsifiable semantic residual claim.",
                "confidence": 0.8,
                "controls_delta": {},
                "world_model_action": action,
                "world_model_critique": {
                    "action": action,
                    "horizon": horizon,
                    "direction": "underestimate",
                    "confidence": 0.7,
                    "evidence_features": [
                        "zone", "epistemic_uncertainty",
                    ],
                    "rationale": "The current semantic regime may be sparse.",
                },
            })

    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True, world_model_action_control_rate=0.0,
        ),
        CriticLLM(),
        world_model_runtime=_Runtime(),
    )
    state = _state()
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)
    audit = record.plan["world_model_critique_audit"]
    context = state._wm_coach_decision_adoption[-1][
        "llm_world_model_critique_context"
    ]

    assert audit["accepted"]
    assert not audit["authority_active"]
    assert audit["reason"] == "shadow_critique_for_validation"
    assert audit["applied_ranking_correction"] == 0.0
    assert not audit["world_model_prediction_mutated"]
    assert context["critique"]["action"] == record.plan["world_model_action"]
    assert not context["can_update_world_model"]


def test_executor_registers_grounded_multiscale_event_in_shadow_mode():
    class StateScaleRuntime(_Runtime):
        def predict_policy_utility(
            self, observation, action, *, action_kind, attacking_home,
            horizon_s,
        ):
            scale = (
                "short" if horizon_s <= 20.0
                else "tactical" if horizon_s <= 120.0
                else "strategic"
            )
            state_scales = {
                "version": 2,
                "primary_scale": scale,
                "short": {"mean_progress_delta": 0.1},
                "tactical": {"final_third_probability": 0.65},
                "strategic": {"mean_goal_diff_delta": 0.0},
                "semantic_event_probabilities": {
                    "enter_final_third": 0.65,
                },
                "trajectory_modes": {
                    "version": 1,
                    "available": True,
                    "counts_trusted": True,
                    "causal_interpretation": False,
                    "ensemble_members": 8,
                    "mode_count": 2,
                    "normalized_mode_entropy": 0.5,
                    "downside_event_counts": {"lose_possession": 1},
                    "downside_event_probabilities": {
                        "lose_possession": 1.5 / 9.0,
                    },
                    "modes": [{
                        "mode_id": "mode_1",
                        "probability": 0.125,
                        "downside_events": {"lose_possession": 1.0},
                    }],
                },
            }
            return {
                "prediction_source": "test_multiscale_runtime",
                "horizon_s": horizon_s,
                "policy_utility": 0.1,
                "progress": 0.1,
                "retention_probability": 0.7,
                "xg_net_delta": 0.0,
                "goal_diff_delta": 0.0,
                "uncertainty": 0.1,
                "epistemic_uncertainty": 0.1,
                "aleatoric_uncertainty": 0.0,
                "state_scales": state_scales,
                "semantic_event_probabilities": state_scales[
                    "semantic_event_probabilities"
                ],
                "distributional_policy_utility": {
                    "version": 2,
                    "available": True,
                    "counts_trusted": True,
                    "member_identity_preserved": True,
                    "member_values": [-0.1, 0.0, 0.1, 0.2],
                    "decision_scenario_values": [-0.1, 0.0, 0.1, 0.2],
                    "distribution_scope": "epistemic_member_only",
                    "predictive_distribution_available": False,
                    "mean_utility": 0.05,
                    "utility_std": 0.1118,
                    "lower_tail_cvar_25": -0.1,
                    "upside_probability": 0.5,
                    "quantiles": {
                        "q10": -0.07, "q25": -0.025, "q50": 0.05,
                        "q75": 0.125, "q90": 0.17,
                    },
                },
            }

    class EventLLM:
        def coach_in_match_plan(self, team_name, facts, kind):
            packet = facts["world_model_decision_support"]
            action = packet["recommended_action"]
            candidate = next(
                item for item in packet["candidates"]
                if item["action"] == action
            )
            horizon = next(iter(candidate["multi_horizon_predictions"]))
            preference_horizons = list(
                candidate["multi_horizon_predictions"]
            )[:2]
            alternative = next(
                item["action"] for item in packet["candidates"]
                if item["action"] != action
            )
            return json.dumps({
                "reasoning": "Register one falsifiable event forecast.",
                "confidence": 0.8,
                "controls_delta": {},
                "world_model_action": action,
                "world_model_event_hypothesis": {
                    "action": action,
                    "horizon": horizon,
                    "event": "enter_final_third",
                    "expectation": "occur",
                    "confidence": 0.75,
                    "evidence_scales": ["short", "tactical"],
                    "rationale": "The tactical projection favors entry.",
                },
                "world_model_risk_constraint": {
                    "selected_action": action,
                    "horizon": horizon,
                    "downside_event": "lose_possession",
                    "max_violation_probability": 0.6,
                    "confidence": 0.75,
                    "rationale": "Keep the projected turnover chance bounded.",
                },
                "world_model_distributional_claim": {
                    "selected_action": action,
                    "alternative_action": alternative,
                    "horizon": horizon,
                    "criterion": "lower_tail_cvar_25",
                    "distribution_scope": "epistemic_member_only",
                    "relation": "approximately_equal",
                    "confidence": 0.75,
                    "rationale": "The two lower tails are indistinguishable.",
                },
                "world_model_risk_preference": {
                    "selected_action": action,
                    "distribution_scope": "epistemic_member_only",
                    "horizon_weights": {
                        preference_horizons[0]: 0.5,
                        preference_horizons[1]: 0.5,
                    },
                    "loss_aversion": 2.0,
                    "diminishing_sensitivity": 0.8,
                    "max_acceptable_regret": 0.05,
                    "confidence": 0.75,
                    "rationale": "Apply one stable downside preference.",
                },
            })

    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True, world_model_action_control_rate=0.0,
        ),
        EventLLM(),
        world_model_runtime=StateScaleRuntime(),
    )
    state = _state()
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)
    audit = record.plan["world_model_event_hypothesis_audit"]
    context = state._wm_coach_decision_adoption[-1][
        "llm_semantic_event_context"
    ]

    assert audit["accepted"]
    assert audit["reason"] == "shadow_semantic_event_hypothesis"
    assert not audit["policy_mutated"]
    assert context["hypothesis"]["event"] == "enter_final_third"
    assert context["llm_event_probability"] == pytest.approx(0.75)
    assert context["event_signature"].startswith("llm-event:")
    risk = record.plan["world_model_risk_certificate_audit"]
    risk_context = state._wm_coach_decision_adoption[-1][
        "llm_risk_certificate_context"
    ]
    assert risk["accepted"]
    assert risk["conservatively_certified"]
    assert not risk["certificate_can_veto_action"]
    assert risk_context["constraint"]["selected_action"] == record.plan[
        "world_model_action"
    ]
    assert risk_context["certificate_signature"].startswith(
        "llm-risk-certificate:"
    )
    distributional = record.plan["world_model_distributional_claim_audit"]
    distributional_context = state._wm_coach_decision_adoption[-1][
        "llm_distributional_claim_context"
    ]
    assert distributional["accepted"]
    assert distributional["directionally_faithful"]
    assert not distributional["can_change_selected_action"]
    assert distributional_context["claim_signature"].startswith(
        "llm-distributional-claim:"
    )
    preference = record.plan["world_model_risk_preference_audit"]
    preference_context = state._wm_coach_decision_adoption[-1][
        "llm_risk_preference_context"
    ]
    assert preference["accepted"]
    assert preference["preference_consistent"]
    assert preference["preference_robust"]
    assert preference["robustness"]["member_axis_jointly_aligned"]
    assert not preference["robustness"][
        "residual_quantile_axis_jointly_aligned"
    ]
    assert not preference["can_change_selected_action"]
    assert preference_context["preference_signature"].startswith(
        "llm-risk-preference:"
    )


def test_executor_persists_model_checked_contrastive_explanation():
    class ContrastiveRuntime(_Runtime):
        def predict_policy_utility(
            self, observation, action, *, action_kind, attacking_home,
            horizon_s,
        ):
            score_edge = float(observation[206]) - 0.5
            utility = (
                0.30 + 0.40 * score_edge
                if action_kind != "hold" else 0.20
            )
            rollout_steps = max(1, int(np.ceil(horizon_s / 30.0)))
            return {
                "prediction_source": "test_contrastive_runtime",
                "horizon_s": horizon_s,
                "rollout_steps": rollout_steps,
                "policy_utility": utility,
                "progress": 0.05,
                "retention_probability": 0.7,
                "xg_net_delta": 0.0,
                "goal_diff_delta": 0.0,
                "uncertainty": 0.1,
                "epistemic_uncertainty": 0.1,
                "aleatoric_uncertainty": 0.0,
            }

    class ContrastiveLLM:
        def __init__(self):
            self.repair_calls = 0

        def coach_in_match_plan(self, team_name, facts, kind):
            packet = facts["world_model_decision_support"]
            selected = packet["recommended_action"]
            alternative = "hold" if selected != "hold" else "pass"
            effect = (
                "supports_selected"
                if selected != "hold" else "opposes_selected"
            )
            return json.dumps({
                "reasoning": "Make the decision reason falsifiable.",
                "confidence": 0.8,
                "controls_delta": {},
                "world_model_action": selected,
                "world_model_contrastive_claim": {
                    "selected_action": selected,
                    "alternative_action": alternative,
                    "horizon": "transition",
                    "factor": "score_context",
                    "effect": effect,
                    "confidence": 0.75,
                    "rationale": "The current score context reduces the margin.",
                },
            })

        def revise_world_model_contrastive_claim(
            self, *, immutable_selected_action, **kwargs,
        ):
            self.repair_calls += 1
            selected = immutable_selected_action
            return {
                "selected_action": selected,
                "alternative_action": (
                    "hold" if selected != "hold" else "pass"
                ),
                "horizon": "transition",
                "factor": "score_context",
                "effect": (
                    "opposes_selected"
                    if selected != "hold" else "supports_selected"
                ),
                "confidence": 0.75,
                "rationale": "Correct the direction using model counterevidence.",
            }

    llm = ContrastiveLLM()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True,
            world_model_action_control_rate=0.0,
            world_model_contrastive_repair=True,
        ),
        llm,
        world_model_runtime=ContrastiveRuntime(),
    )
    state = _state()
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)
    audit = record.plan["world_model_contrastive_explanation_audit"]
    context = state._wm_coach_decision_adoption[-1][
        "llm_contrastive_explanation_context"
    ]
    repair = record.plan["world_model_contrastive_repair_audit"]
    repair_context = state._wm_coach_decision_adoption[-1][
        "llm_contrastive_repair_context"
    ]

    assert audit["accepted"]
    assert not audit["directionally_faithful"]
    assert audit["reason"] == "shadow_model_checked_contrastive_explanation"
    assert not audit["can_change_selected_action"]
    assert context["claim"]["selected_action"] == record.plan[
        "world_model_action"
    ]
    assert context["contrastive_signature"].startswith("llm-contrastive:")
    assert repair["repair_successful"]
    assert repair["selected_action"] == record.plan["world_model_action"]
    assert repair_context["effective_audit_source"] == "revision"
    assert record.plan["world_model_contrastive_claim_effective"][
        "selected_action"
    ] == record.plan["world_model_action"]
    assert llm.repair_calls == 1


def test_validated_critic_can_only_reduce_policy_bridge_authority():
    cfg = CognitiveMatchConfig(
        enabled=True, world_model_action_bias_max=0.4,
    )
    plan = {"world_model_action": "pass", "confidence": 1.0}
    baseline_packet = {
        "available": True,
        "candidates": [{
            "action": "pass", "effective_confidence": 1.0,
        }],
    }
    cautious_packet = {
        "available": True,
        "candidates": [{
            "action": "pass",
            "effective_confidence": 1.0,
            "llm_semantic_critic_bridge_factor": 0.5,
        }],
    }

    baseline = _policy_intervention_strength(plan, baseline_packet, cfg)
    cautious = _policy_intervention_strength(plan, cautious_packet, cfg)

    assert baseline == pytest.approx(0.4)
    assert cautious == pytest.approx(0.2)


def test_executor_uses_only_matching_held_out_critic_as_safety_brake():
    from src.match_engine.world_model.llm_critic import llm_critic_signature
    from src.match_engine.world_model.llm_critic_memory import (
        compile_llm_critic_memory,
    )

    signature = llm_critic_signature("rule_fallback")
    historical = []
    for _ in range(8):
        historical.append({
            "world_model_decision_adoption": {"records": [{
                "checkpoint_signature": "runtime_unspecified",
                "environment_signature": "environment_unspecified",
                "intervention_actual_action": "shot",
                "llm_world_model_critique_context": {
                    "version": 1,
                    "critic_signature": signature,
                    "critique": {"action": "shot", "horizon": "transition"},
                    "proposed_policy_utility_correction": -0.08,
                },
                "multi_horizon_regime_outcomes": {
                    "transition": {
                        "policy_utility": -0.08,
                        "world_model_prediction": {
                            "policy_utility": 0.0,
                            "raw_policy_utility": 0.0,
                        },
                    },
                },
            }]},
        })
    memory = compile_llm_critic_memory(
        historical,
        checkpoint_signature="runtime_unspecified",
        environment_signature="environment_unspecified",
        critic_signature=signature,
    )

    class CautiousCriticLLM:
        def coach_in_match_plan(self, team_name, facts, kind):
            packet = facts["world_model_decision_support"]
            return json.dumps({
                "reasoning": "Apply a previously validated semantic warning.",
                "confidence": 1.0,
                "controls_delta": {},
                "world_model_action": "shot",
                "world_model_critique": {
                    "action": "shot",
                    "horizon": "transition",
                    "direction": "overestimate",
                    "confidence": 1.0,
                    "evidence_features": ["epistemic_uncertainty"],
                    "rationale": "This regime historically overstates shot value.",
                },
            })

    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True,
            world_model_action_control_rate=0.0,
            world_model_action_bias_max=0.4,
        ),
        CautiousCriticLLM(),
        world_model_runtime=_Runtime(),
        llm_critic_memory=memory,
    )
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), _state())

    audit = record.plan["world_model_critique_audit"]
    assert audit["authority_active"]
    assert audit["correction_applied"]
    assert audit["applied_ranking_correction"] < 0.0
    assert 0.5 <= record.plan["world_model_critic_safety_factor"] < 1.0
    assert record.plan["world_model_policy_intervention_strength"] < 0.4


def test_executor_closes_llm_action_when_world_model_quality_gate_is_closed():
    llm = _LLM()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True), llm,
        world_model_runtime=_Runtime(confidence=0.0),
    )
    trigger = CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    )
    state = _state()
    record = executor.process_trigger(trigger, state)
    assert record.plan["world_model_action"] == "none"
    assert record.plan["world_model_selection_constrained"]
    assert not hasattr(state, "_wm_coach_decision_adoption")


def test_executor_scales_and_audits_an_eligible_exploration():
    llm = _ExplorationLLM()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True,
            world_model_action_control_rate=0.0,
            world_model_exploration_max_regret=0.30,
            world_model_exploration_min_information=0.0,
            world_model_exploration_strength_scale=0.20,
        ),
        llm,
        world_model_runtime=_ExplorationRuntime(),
    )
    state = _state()
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)
    advice = llm.facts["world_model_decision_support"]["active_learning"]
    adoption = state._wm_coach_decision_adoption[-1]

    assert advice["eligible"]
    assert record.plan["world_model_decision_mode"] == "explore"
    assert not record.plan["world_model_exploration_constrained"]
    assert adoption["active_learning"]["decision_mode"] == "explore"
    assert adoption["active_learning"]["intervention_scale"] == 0.20
    assert adoption["llm_selected_action"] == advice["exploration_action"]
    assert 0.0 < adoption["intervention_strength"] < 0.07


def test_executor_downgrades_exploration_for_the_wrong_action():
    llm = _InvalidExplorationLLM()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True,
            world_model_action_control_rate=0.0,
            world_model_exploration_max_regret=0.30,
            world_model_exploration_min_information=0.0,
        ),
        llm,
        world_model_runtime=_ExplorationRuntime(),
    )
    state = _state()
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)
    advice = llm.facts["world_model_decision_support"]["active_learning"]

    assert advice["eligible"]
    assert advice["exploration_action"] != "shot"
    assert record.plan["world_model_decision_mode"] == "exploit"
    assert record.plan["world_model_exploration_constrained"]
    assert state._wm_coach_decision_adoption[-1]["active_learning"][
        "decision_mode"
    ] == "exploit"


def test_executor_uses_realized_prediction_residuals_to_reduce_bridge_strength():
    llm = _LLM()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True,
            world_model_action_control_rate=0.0,
            world_model_residual_min_samples=4,
        ),
        llm,
        world_model_runtime=_Runtime(),
    )
    trigger = CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    )
    state = _state()
    state._wm_coach_decision_adoption = [
        {
            "team_id": "Home",
            "resolved": True,
            "policy_opportunity_observed": False,
            "intervention_actual_action": "shot",
            "multi_horizon_regime_outcomes": {
                horizon: {
                    "policy_utility": 0.0,
                    "world_model_prediction": {
                        "policy_utility": 1.0,
                        "uncertainty": 0.1,
                    },
                }
                for horizon in ("transition", "60s", "180s")
            },
        }
        for _ in range(4)
    ]
    record = executor.process_trigger(trigger, state)
    new_adoption = state._wm_coach_decision_adoption[-1]
    assert record.plan["world_model_prediction_trust_factor"] == 0.25
    assert record.plan["world_model_prediction_trust_audit"][
        "horizon_key"
    ] == "180s"
    assert new_adoption["outcome_prediction_trust_factor"] == 0.25
    assert new_adoption["intervention_strength"] < 0.09
    calibration = llm.facts["world_model_decision_support"][
        "policy_outcome_calibration"
    ]
    assert calibration["by_action_horizon"]["shot"]["180s"][
        "trust_factor"
    ] == 0.25


def test_cross_match_residual_memory_reaches_llm_and_policy_bridge():
    class _Memory:
        def calibrate(self, prediction, **kwargs):
            output = dict(prediction)
            output["raw_policy_utility"] = output["policy_utility"]
            output["policy_utility"] += 0.05
            output["residual_memory_trust_factor"] = 0.25
            output["residual_memory"] = {
                "scope": "action_horizon",
                "trust_factor": 0.25,
            }
            return output

        def summary(self):
            return {
                "active_groups": 3,
                "residual_rows": 24,
                "drift": {"status": "watch"},
            }

    llm = _LLM()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(
            enabled=True, world_model_action_control_rate=0.0,
        ),
        llm,
        world_model_runtime=_Runtime(),
        outcome_residual_memory=_Memory(),
    )
    state = _state()
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)
    adoption = state._wm_coach_decision_adoption[-1]
    assert record.plan["world_model_residual_memory_factor"] == 0.25
    assert record.plan["world_model_residual_memory_drift"]["status"] == (
        "watch"
    )
    assert adoption["intervention_strength"] < 0.09
    packet = llm.facts["world_model_decision_support"]
    assert packet["contextual_residual_memory"]["active_groups"] == 3
    assert opponent_information_feedback_is_valid(
        adoption["opponent_information_feedback_context"]
    )
    assert adoption["opponent_information_feedback_context"] == packet[
        "opponent_information_feedback"
    ]
    shot = next(
        candidate for candidate in packet["candidates"]
        if candidate["action"] == "shot"
    )
    assert shot["multi_horizon_predictions"]["180s"][
        "residual_memory"
    ]["scope"] == "action_horizon"
    assert shot["multi_horizon_forecast_summary"][
        "contextual_memory_active"
    ]
