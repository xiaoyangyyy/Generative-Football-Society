"""World-model evidence flows into bounded System 2 coach decisions."""

import json
from types import SimpleNamespace

import numpy as np

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
    build_coach_decision_packet,
)
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.probabilistic import ProbabilisticFuture


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


def test_decision_packet_compares_all_actions_and_recommends_risk_adjusted_best():
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    assert packet["available"]
    assert packet["recommended_action"] == "shot"
    assert {candidate["action"] for candidate in packet["candidates"]} == set(COACH_ACTIONS)
    assert all(0.0 <= candidate["uncertainty"] <= 1.0 for candidate in packet["candidates"])
    assert packet["observation_coverage"] > 0.9


def test_decision_packet_closes_recommendation_when_quality_gate_is_closed():
    packet = build_coach_decision_packet(
        _Runtime(confidence=0.0), _state(), "Home",
    )
    assert not packet["available"]
    assert packet["reason"] == "quality_gate_closed"
    assert packet["recommended_action"] == "none"


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
    assert record.applied
    assert state.home.coach.tactical_current["pressing_intensity"] > 0.5
