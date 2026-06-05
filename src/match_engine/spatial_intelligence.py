"""§7 Spatial intelligence — Phi, lane quality, offside risk."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.match_engine.formation import interpolate_anchors
from src.match_engine.math_utils import sigmoid, tanh_clip
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState, TeamAffectiveState


class SpatialIntelligenceEngine:
    def __init__(self, cfg: MicroMatchConfig):
        self.cfg = cfg

    def _kernel(self, r: np.ndarray, p: np.ndarray) -> float:
        d2 = float(np.sum((r - p) ** 2))
        return float(np.exp(-d2 / (2.0 * self.cfg.sigma_player**2)))

    def compute_phi_grid(
        self,
        state: MatchAffectiveState,
        for_home: bool,
    ) -> np.ndarray:
        cfg = self.cfg
        team = state.home if for_home else state.away
        opp = state.away if for_home else state.home
        nx, ny = state.spatial.rho_home.shape
        x = np.linspace(0.02, 0.98, nx)
        y = np.linspace(0.02, 0.98, ny)
        phi = np.zeros((nx, ny), dtype=float)
        rho_own = state.spatial.rho_home if for_home else state.spatial.rho_away
        rho_opp = state.spatial.rho_away if for_home else state.spatial.rho_home

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                r = np.array([xi, yj], dtype=float)
                ro = float(rho_own[i, j])
                rop = float(rho_opp[i, j])
                opp_sum = sum(self._kernel(r, p.position) for p in opp.players if p.on_pitch)
                own_sum = sum(
                    self._kernel(r, p.position)
                    for p in team.players
                    if p.on_pitch and p.player_id != state.ball.possessor_id
                )
                z = cfg.phi_a0 + cfg.phi_a1 * (ro / (ro + rop + 1e-6)) - cfg.phi_a2 * opp_sum - cfg.phi_a3 * own_sum
                phi[i, j] = sigmoid(z)
        return phi

    def step(self, state: MatchAffectiveState) -> None:
        state.spatial.phi_home = self.compute_phi_grid(state, True)
        state.spatial.phi_away = self.compute_phi_grid(state, False)

    def phi_at(self, state: MatchAffectiveState, p: np.ndarray, for_home: bool) -> float:
        grid = state.spatial.phi_home if for_home else state.spatial.phi_away
        nx, ny = grid.shape
        ix = int(np.clip(p[0] * (nx - 1), 0, nx - 1))
        iy = int(np.clip(p[1] * (ny - 1), 0, ny - 1))
        return float(grid[ix, iy])

    def def_line_x(self, team: TeamAffectiveState, for_home: bool) -> float:
        xs = [p.position[0] for p in team.players if p.on_pitch and p.role in ("LB", "CB", "RB", "DM")]
        if not xs:
            xs = [p.position[0] for p in team.players if p.on_pitch]
        return float(np.mean(xs)) if xs else (0.25 if for_home else 0.75)

    def offside_risk(self, state: MatchAffectiveState, target: np.ndarray, attacking_home: bool) -> float:
        from src.match_engine.cognitive.assistant_dynamics import assistant_offside_modifier

        if attacking_home:
            line = self.def_line_x(state.away, False)
            z = self.cfg.gamma_offside * (target[0] - line - self.cfg.delta_offside)
        else:
            line = self.def_line_x(state.home, True)
            z = self.cfg.gamma_offside * (line - target[0] - self.cfg.delta_offside)
        z += 4.0 * assistant_offside_modifier(state)
        return float(sigmoid(z))

    def lane_quality(
        self,
        state: MatchAffectiveState,
        p_from: np.ndarray,
        p_to: np.ndarray,
        opp: TeamAffectiveState,
    ) -> float:
        cfg = self.cfg
        M = cfg.lane_samples
        total = 0.0
        for t in np.linspace(0.0, 1.0, M):
            s = (1 - t) * p_from + t * p_to
            ix = int(np.clip(s[0] * (state.spatial.press.shape[0] - 1), 0, state.spatial.press.shape[0] - 1))
            iy = int(np.clip(s[1] * (state.spatial.press.shape[1] - 1), 0, state.spatial.press.shape[1] - 1))
            press = float(state.spatial.press[ix, iy])
            opp_near = sum(self._kernel(s, o.position) for o in opp.players if o.on_pitch)
            total += cfg.lane_mu1 * press + cfg.lane_mu2 * opp_near
        return float(np.exp(-total / M))

    def numerical_advantage(self, state: MatchAffectiveState, p: np.ndarray, for_home: bool) -> float:
        own = state.home if for_home else state.away
        opp = state.away if for_home else state.home
        o_sum = sum(self._kernel(p, x.position) for x in own.players if x.on_pitch)
        d_sum = sum(self._kernel(p, x.position) for x in opp.players if x.on_pitch)
        return float(tanh_clip(o_sum - d_sum))
