"""Cognitive plan apply + executor fallback."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.match_engine.cognitive.apply import (
    apply_coach_plan_to_team,
    apply_referee_plan,
)
from src.match_engine.cognitive.executor import CognitiveExecutor, _rule_fallback_plan
from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import CognitiveTriggerEvent, ENTITY_TIER_COACH
from src.match_engine.state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)


def _state():
    home = TeamAffectiveState(
        team_id="Brazil",
        coach=CoachAffectiveState(team_id="Brazil", tactical_current={"pressing_intensity": 0.5}),
    )
    away = TeamAffectiveState(team_id="Argentina", coach=CoachAffectiveState(team_id="Argentina"))
    return MatchAffectiveState(
        home=home,
        away=away,
        referee=RefereeAffectiveState(strictness_base=0.55),
        crowd=CrowdState(),
    )


def test_coach_plan_clamped():
    st = _state()
    before = st.home.coach.tactical_current["pressing_intensity"]
    apply_coach_plan_to_team(
        st,
        "Brazil",
        {"controls_delta": {"pressing_intensity": 0.5}, "confidence": 0.8},
    )
    after = st.home.coach.tactical_current["pressing_intensity"]
    assert after <= 1.0
    assert after > before


def test_referee_plan():
    st = _state()
    apply_referee_plan(st, {"strictness_delta": 0.08, "bias_delta": 0.02})
    assert st.referee.strictness_base > 0.55


def test_executor_fallback():
    trig = CognitiveTriggerEvent(
        60.0, "goal", ENTITY_TIER_COACH, "coach:Brazil", team_id="Brazil", salience=1.2
    )
    plan = _rule_fallback_plan(trig)
    assert "controls_delta" in plan
    ex = CognitiveExecutor(CognitiveMatchConfig(enabled=True), llm=None, use_llm=False)
    st = _state()
    rec = ex.process_trigger(trig, st)
    assert rec.applied
