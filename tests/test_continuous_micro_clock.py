from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from src.match_engine.continuous_micro_clock import ContinuousMicroEventClock
from src.match_engine.state import BallState, MatchAffectiveState, PlayerAffectiveState, TeamAffectiveState, RefereeAffectiveState, CrowdState

class _Router:
    temporal_models = {"skillcorner": object()}
    def plan_next_mark(self, context, action):
        assert action.target_index != action.defender_index
        return SimpleNamespace(delay_seconds=1.25, kind="pass")

def _state():
    home=[PlayerAffectiveState(f"h{i}",f"h{i}","CM","H",position=np.array([.2+i*.01,.4])) for i in range(3)]
    away=[PlayerAffectiveState(f"a{i}",f"a{i}","CM","A",position=np.array([.7+i*.01,.4])) for i in range(3)]
    return MatchAffectiveState(TeamAffectiveState("H",home),TeamAffectiveState("A",away),RefereeAffectiveState(),CrowdState(),clock_seconds=10.,ball=BallState(position=home[1].position.copy(),possessor_id="h1",possession_team_id="H"))

def test_continuous_micro_clock_schedules_completed_and_turnover_receptions():
    clock=ContinuousMicroEventClock.__new__(ContinuousMicroEventClock)
    clock.router=_Router(); clock.provider="skillcorner"; clock.max_delay_s=5.; clock.next_action_at=0.; clock.pending_kind=None; clock._history=__import__('collections').deque(maxlen=5); clock.plans=0; clock.gated_ticks=0; clock.kind_counts=__import__('collections').Counter(); clock.executed_counts=__import__('collections').Counter(); clock.delay_sum_s=0.
    state=_state(); failed=SimpleNamespace(completed=False,from_id="h0",to_id="h1")
    clock.schedule_after_pass(state,failed); assert clock.plans==1
    completed=SimpleNamespace(completed=True,from_id="h0",to_id="h1")
    clock.schedule_after_pass(state,completed)
    assert clock.plans==2 and clock.next_action_at==11.25 and clock.is_gated(11.)
    assert clock.pop_due_kind(11.25)=="pass" and clock.pop_due_kind(12.) is None
    assert clock.diagnostics()["next_mark_counts"]=={"pass":2}
