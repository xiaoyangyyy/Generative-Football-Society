"""§6 Spatial field — possession density rho and press on grid."""

from __future__ import annotations

import numpy as np

from src.match_engine.math_utils import sigmoid
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.state import MatchAffectiveState, SpatialGridState


class SpatialFieldEngine:
    def __init__(self, cfg: MicroMatchConfig):
        self.cfg = cfg
        self.nx = cfg.grid_nx
        self.ny = cfg.grid_ny
        self._init_grids()

    def _init_grids(self):
        x = np.linspace(0.02, 0.98, self.nx)
        y = np.linspace(0.02, 0.98, self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

    def ensure_state(self, state: MatchAffectiveState) -> SpatialGridState:
        sg = state.spatial
        if sg.rho_home.shape != (self.nx, self.ny):
            sg.rho_home = np.zeros((self.nx, self.ny), dtype=float)
            sg.rho_away = np.zeros((self.nx, self.ny), dtype=float)
            sg.press = np.zeros((self.nx, self.ny), dtype=float)
            sg.phi_home = np.zeros((self.nx, self.ny), dtype=float)
            sg.phi_away = np.zeros((self.nx, self.ny), dtype=float)
        return sg

    def _pos_to_idx(self, p: np.ndarray) -> tuple[int, int]:
        ix = int(np.clip(p[0] * (self.nx - 1), 0, self.nx - 1))
        iy = int(np.clip(p[1] * (self.ny - 1), 0, self.ny - 1))
        return ix, iy

    def inject_ball_source(self, sg: SpatialGridState, ball_pos: np.ndarray, team_is_home: bool) -> None:
        ix, iy = self._pos_to_idx(ball_pos)
        target = sg.rho_home if team_is_home else sg.rho_away
        target[ix, iy] += 0.35

    def step(self, state: MatchAffectiveState, dt: float) -> SpatialGridState:
        cfg = self.cfg
        sg = self.ensure_state(state)
        sg.rho_home *= 1.0 - cfg.rho_decay * dt
        sg.rho_away *= 1.0 - cfg.rho_decay * dt

        # player kernels on rho
        for team, rho_t, is_home in (
            (state.home, sg.rho_home, True),
            (state.away, sg.rho_away, False),
        ):
            for p in team.players:
                if not p.on_pitch:
                    continue
                ix, iy = self._pos_to_idx(p.position)
                rho_t[ix, iy] += 0.08 * (1.0 if p.player_id == state.ball.possessor_id else 0.5)

        self.inject_ball_source(sg, state.ball.position, state.ball.possession_team_id == state.home.team_id)

        # laplacian smooth (diffusion)
        sg.rho_home = self._diffuse(sg.rho_home, cfg.D0 * dt)
        sg.rho_away = self._diffuse(sg.rho_away, cfg.D0 * dt)

        # mutual press / suppression
        sg.rho_home = np.clip(sg.rho_home - dt * cfg.lambda_press * sg.rho_home * sg.rho_away, 0, None)
        sg.rho_away = np.clip(sg.rho_away - dt * cfg.lambda_press * sg.rho_away * sg.rho_home, 0, None)

        th = state.home.coach.tactical_current or {}
        ta = state.away.coach.tactical_current or {}

        def _pm(tac: dict) -> float:
            return float(
                0.50
                + 0.40 * tac.get("pressing_intensity", 0.5)
                + 0.22 * tac.get("counterpress", 0.5)
                + 0.12 * tac.get("man_oriented_press", 0.5)
            )

        sg.press = _pm(th) * float(th.get("pressing_intensity", 0.5)) * sg.rho_away + _pm(ta) * float(
            ta.get("pressing_intensity", 0.5)
        ) * sg.rho_home

        return sg

    def _diffuse(self, grid: np.ndarray, alpha: float) -> np.ndarray:
        g = grid.copy()
        g[1:-1, 1:-1] += alpha * (
            grid[2:, 1:-1]
            + grid[:-2, 1:-1]
            + grid[1:-1, 2:]
            + grid[1:-1, :-2]
            - 4.0 * grid[1:-1, 1:-1]
        )
        return np.clip(g, 0, 2.0)

    def advantage_at(self, sg: SpatialGridState, p: np.ndarray, for_home: bool) -> float:
        ix, iy = self._pos_to_idx(p)
        rh = float(sg.rho_home[ix, iy])
        ra = float(sg.rho_away[ix, iy])
        if for_home:
            return rh / (rh + ra + 1e-6)
        return ra / (rh + ra + 1e-6)

    def press_at(self, sg: SpatialGridState, p: np.ndarray) -> float:
        ix, iy = self._pos_to_idx(p)
        return float(sg.press[ix, iy])
