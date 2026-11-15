import copy

import numpy as np
import pytest

from gfs_core.contracts.events import MicroEvent, MicroEventType
from gfs_core.contracts.micro_state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    PlayerAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from gfs_core.plugins.match.actions.events import apply_micro_event
from gfs_core.plugins.society.psychology.coupling import AffectiveSpatialCoupling


def make_state(extra_player=False):
    home_players = [
        PlayerAffectiveState(
            "home-1", "Home", "CM", "home",
            z_emo=np.asarray([1.0, -0.5, 0.3, 0.2]),
            morale_logit=0.5,
            cognitive_load=0.4,
        ),
    ]
    if extra_player:
        home_players.append(PlayerAffectiveState(
            "home-2", "Home 2", "CM", "home",
            z_emo=np.asarray([-0.2, 0.8, 0.1, -0.4]),
        ))
    home = TeamAffectiveState(
        "home",
        players=home_players,
        coach=CoachAffectiveState(
            "home", z_stress=1.0, z_trust=-0.4, z_rage=0.3,
            tactical_base={"pressing_intensity": 0.5},
            tactical_current={"pressing_intensity": 0.5},
        ),
    )
    away = TeamAffectiveState(
        "away",
        players=[PlayerAffectiveState(
            "away-1", "Away", "CM", "away",
            z_emo=np.asarray([0.2, 0.4, -0.3, 0.5]),
        )],
        coach=CoachAffectiveState("away"),
    )
    return MatchAffectiveState(
        home,
        away,
        RefereeAffectiveState(z_calm=0.6),
        CrowdState(psi=0.0, home_team_id="home", is_neutral_venue=True),
    )


def quiet_step(coupling, state, dt):
    return coupling.step(
        state,
        dt,
        coordination_home=0.0,
        coordination_away=0.0,
        conflict_home=0.0,
        conflict_away=0.0,
    )


def test_linear_decay_is_equivalent_across_time_steps():
    coarse = make_state()
    fine = make_state()
    coupling = AffectiveSpatialCoupling()
    quiet_step(coupling, coarse, 30.0)
    for _ in range(30):
        quiet_step(coupling, fine, 1.0)
    assert coarse.crowd.psi == pytest.approx(fine.crowd.psi, abs=1e-12)
    for name in ("z_stress", "z_trust", "z_rage"):
        assert getattr(coarse.home.coach, name) == pytest.approx(
            getattr(fine.home.coach, name), abs=1e-12,
        )
    assert coarse.referee.z_calm == pytest.approx(fine.referee.z_calm, abs=1e-12)
    assert coarse.home.players[0].z_emo == pytest.approx(
        fine.home.players[0].z_emo, abs=1e-12,
    )
    assert coarse.home.players[0].cognitive_load == pytest.approx(
        fine.home.players[0].cognitive_load, abs=0.01,
    )
    assert coarse.home.players[0].morale_logit == pytest.approx(
        fine.home.players[0].morale_logit, abs=0.02,
    )


def test_no_event_state_is_bounded_and_returns_near_baseline():
    state = make_state()
    coupling = AffectiveSpatialCoupling()
    for _ in range(120):
        quiet_step(coupling, state, 30.0)
    assert abs(state.crowd.psi) < 1e-6
    assert abs(state.home.coach.z_stress) < 1e-6
    assert abs(state.home.coach.z_rage) < 1e-6
    assert abs(state.referee.z_calm) < 1e-6
    assert np.max(np.abs(state.home.players[0].z_emo)) < 1e-6
    assert state.home.players[0].cognitive_load == pytest.approx(0.055 / 0.06)
    for value in state.home.players[0].emotion_profile().values():
        assert np.isfinite(value)


def test_zero_peer_matrix_has_only_exponential_decay():
    state = make_state(extra_player=True)
    state.home.peer_matrix = np.zeros((2, 2))
    initial = np.stack([player.z_emo for player in state.home.players])
    quiet_step(AffectiveSpatialCoupling(), state, 30.0)
    expected = initial * np.exp(-0.08 * 30.0)
    assert np.stack([player.z_emo for player in state.home.players]) == pytest.approx(expected)


def test_uniform_peer_matrix_converges_toward_team_mean():
    state = make_state(extra_player=True)
    initial_mean = np.mean([player.z_emo for player in state.home.players], axis=0)
    quiet_step(AffectiveSpatialCoupling(), state, 30.0)
    current_mean = np.mean([player.z_emo for player in state.home.players], axis=0)
    assert current_mean == pytest.approx(initial_mean * np.exp(-0.08 * 30.0))
    assert state.home.players[0].z_emo[0] < 1.0
    assert state.home.players[1].z_emo[0] > -0.2


def test_goal_impulse_is_bounded_and_recovers_with_fine_steps():
    state = make_state()
    coupling = AffectiveSpatialCoupling(event_handler=apply_micro_event)
    goal = MicroEvent(
        0.0,
        MicroEventType.GOAL_SCORED,
        "home",
        player_id="home-1",
        intensity=1000.0,
    )
    coupling.step(state, 30.0, events=[goal])
    for _ in range(120):
        quiet_step(coupling, state, 5.5)
    assert np.isfinite(state.home.players[0].z_emo).all()
    assert np.max(np.abs(state.home.players[0].z_emo)) < 1e-6
    assert abs(state.home.coach.z_stress) < 1e-6


def test_snapshot_copy_is_independent_for_replay():
    state = make_state()
    restored = copy.deepcopy(state)
    quiet_step(AffectiveSpatialCoupling(), state, 30.0)
    assert restored.home.players[0].z_emo != pytest.approx(
        state.home.players[0].z_emo
    )
