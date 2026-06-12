"""E1 — phi gradient at position and kinematic wiring."""

from __future__ import annotations

import numpy as np

from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.state import (
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    SpatialGridState,
    TeamAffectiveState,
)


def _state_with_phi_peak() -> MatchAffectiveState:
    nx, ny = 8, 6
    phi = np.zeros((nx, ny), dtype=float)
    phi[5, 3] = 1.0
    phi[4, 3] = 0.6
    phi[6, 3] = 0.3
    sg = SpatialGridState(
        rho_home=np.zeros((nx, ny)),
        rho_away=np.zeros((nx, ny)),
        press=np.zeros((nx, ny)),
        phi_home=phi,
        phi_away=np.zeros((nx, ny)),
    )
    return MatchAffectiveState(
        home=TeamAffectiveState(team_id="H"),
        away=TeamAffectiveState(team_id="A"),
        referee=RefereeAffectiveState(),
        crowd=CrowdState(),
        spatial=sg,
    )


def test_phi_gradient_points_toward_higher_phi():
    sie = SpatialIntelligenceEngine(MicroMatchConfig())
    state = _state_with_phi_peak()
    # Left of phi peak at grid (5,3) — gradient should point +x toward higher phi.
    p = np.array([0.64, 0.6], dtype=float)
    g = sie.phi_gradient_at(state, p, for_home=True)
    assert g[0] > 0.1


def test_phi_gradient_enabled_by_default():
    cfg = MicroMatchConfig()
    assert cfg.enable_phi_gradient_move is True
    assert cfg.phi_move_scale == 0.015
