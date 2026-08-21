from __future__ import annotations

import numpy as np

from src.match_engine.aerial_duel import AerialOutcome, aerial_goal_events
from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.micro_events import MicroEventType, apply_micro_event
from tests.test_affective_phase1b import _FakeAgent


def _state():
    return build_match_affective_state(
        _FakeAgent("Home"), _FakeAgent("Away"),
        rng=np.random.default_rng(17),
    )


def test_aerial_goal_uses_live_score_and_emotion_event_contract():
    state = _state()
    state.clock_seconds = 1200.0
    crosser = next(player for player in state.home.players if player.role != "GK")
    scorer = next(
        player for player in state.home.players
        if player.player_id != crosser.player_id and player.role != "GK"
    )
    outcome = AerialOutcome(
        winner_id=scorer.player_id,
        contact="header",
        xg_added=0.25,
        landed_xy=np.array([0.98, 0.5]),
        goal=True,
    )

    events = aerial_goal_events(state, crosser, outcome)

    assert [event.event_type for event in events] == [
        MicroEventType.GOAL_SCORED, MicroEventType.GOAL_CONCEDED,
    ]
    assert events[0].player_id == scorer.player_id
    assert events[0].meta == {"source": "aerial_header"}
    for event in events:
        apply_micro_event(state, event, MicroMatchConfig.fast_demo())
    assert state.home.score == 1
    assert state.away.score == 0


def test_non_goal_aerial_contact_does_not_create_score_events():
    state = _state()
    crosser = next(player for player in state.home.players if player.role != "GK")
    outcome = AerialOutcome(
        winner_id=crosser.player_id,
        contact="header",
        xg_added=0.12,
        landed_xy=np.array([0.9, 0.5]),
        goal=False,
    )

    assert aerial_goal_events(state, crosser, outcome) == []
    assert state.home.score == 0
