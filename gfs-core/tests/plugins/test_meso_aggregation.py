import numpy as np
import pytest

from gfs_core.contracts.micro_state import (
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from gfs_core.plugins.society.psychology.meso import MesoAggregator


def test_meso_packet_contains_real_spatial_field_mean():
    state = MatchAffectiveState(
        TeamAffectiveState("home"),
        TeamAffectiveState("away"),
        RefereeAffectiveState(),
        CrowdState(),
    )
    state.spatial.phi_home = np.full((32, 22), 0.4)
    aggregator = MesoAggregator()
    aggregator.record_tick(
        state,
        team_id="home",
        xg_for=0.5,
        xg_against=0.2,
        controversy=0.0,
        psi=0.0,
        emotion={},
        tactical={},
        tactical_base={},
        icon_shock=0.0,
        phi=float(state.spatial.phi_home.mean()),
    )
    packet = aggregator.flush_packets("home", "away")[0]
    assert packet.phi_integral == pytest.approx(0.4)
