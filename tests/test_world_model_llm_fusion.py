"""World-model evidence flows into bounded System 2 coach decisions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import CognitiveTriggerEvent, ENTITY_TIER_COACH
from src.match_engine.cognitive.executor import CognitiveExecutor
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


def test_decision_packet_closes_recommendation_when_quality_gate_is_closed():
    packet = build_coach_decision_packet(
        _Runtime(confidence=0.0), _state(), "Home",
    )
    assert not packet["available"]
    assert packet["reason"] == "quality_gate_closed"
    assert packet["recommended_action"] == "none"


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

    def complete(self, system_prompt, user_prompt, **kwargs):
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
    packet = llm.facts["world_model_decision_support"]
    assert packet["recommended_action"] == "shot"
    assert record.plan["world_model_action"] == "shot"
    assert record.plan["world_model_adoption_id"]
    assert len(state._wm_coach_decision_adoption) == 1
    assert record.applied
    assert state.home.coach.tactical_current["pressing_intensity"] > 0.5


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
