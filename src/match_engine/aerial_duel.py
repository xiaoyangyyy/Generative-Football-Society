"""Aerial duels — crosses, headers, second balls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.match_engine.ball_physics import BallActionParams, TrajectoryResult, integrate_trajectory
from src.match_engine.math_utils import sigmoid
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState


@dataclass
class AerialOutcome:
    winner_id: Optional[str]
    contact: str  # header | clearance | loose
    xg_added: float
    landed_xy: np.ndarray
    goal: bool = False


def aerial_goal_events(
    state: MatchAffectiveState,
    crosser: PlayerAffectiveState,
    outcome: AerialOutcome,
):
    """Build the same live-score event contract used by normal shots."""
    if not outcome.goal or not outcome.winner_id:
        return []
    from src.match_engine.micro_events import MicroEvent, MicroEventType

    attacking_home = crosser.team_id == state.home.team_id
    opponent_id = (
        state.away.team_id if attacking_home else state.home.team_id
    )
    return [
        MicroEvent(
            t_sec=float(state.clock_seconds),
            event_type=MicroEventType.GOAL_SCORED,
            team_id=crosser.team_id,
            player_id=outcome.winner_id,
            opponent_team_id=opponent_id,
            intensity=1.0,
            meta={"source": "aerial_header"},
        ),
        MicroEvent(
            t_sec=float(state.clock_seconds) + 0.5,
            event_type=MicroEventType.GOAL_CONCEDED,
            team_id=opponent_id,
            opponent_team_id=crosser.team_id,
            intensity=0.9,
            meta={"source": "aerial_header"},
        ),
    ]


def apply_aerial_xg_to_state(
    state: MatchAffectiveState,
    xg: float,
    attacking_home: bool,
    cfg: MicroMatchConfig,
) -> None:
    """Add header/cross xG to team totals with per-match cap."""
    val = max(0.0, float(xg))
    if val <= 0.0:
        return
    cap = float(getattr(cfg, "micro_xg_match_cap", 2.2))
    if attacking_home:
        room = max(0.0, cap - state.micro_xg_home)
        state.micro_xg_home += min(val, room)
    else:
        room = max(0.0, cap - state.micro_xg_away)
        state.micro_xg_away += min(val, room)


class AerialDuelEngine:
    def __init__(self, cfg: MicroMatchConfig):
        self.cfg = cfg
        self.stats = {
            "crosses": 0,
            "headers": 0,
            "aerial_goals": 0,
            "home_headers": 0,
            "away_headers": 0,
            "header_attempts": 0,
            "home_aerial_goals": 0,
            "away_aerial_goals": 0,
        }

    def _box_target(self, attacking_home: bool, rng: np.random.Generator) -> np.ndarray:
        if attacking_home:
            x = 0.88 + 0.04 * rng.random()
        else:
            x = 0.12 - 0.04 * rng.random()
        y = 0.42 + 0.16 * rng.random()
        return np.array([x, y], dtype=float)

    def resolve_cross(
        self,
        state: MatchAffectiveState,
        crosser: PlayerAffectiveState,
        rng: np.random.Generator,
    ) -> Tuple[TrajectoryResult, AerialOutcome]:
        cfg = self.cfg
        attacking_home = crosser.team_id == state.home.team_id
        tgt = self._box_target(attacking_home, rng)
        dist = float(np.linalg.norm(tgt - crosser.position))
        elev = cfg.cross_elev_base + 0.08 * rng.random()
        v0 = cfg.cross_v0_base + 0.15 * (1.0 - min(1.0, dist))

        params = BallActionParams(
            v0=v0,
            elev=elev,
            azim=float(np.arctan2(tgt[1] - crosser.position[1], tgt[0] - crosser.position[0] + 1e-9)),
            omega=cfg.pass_curve_omega * float(crosser.abilities.curve),
            spin_axis=np.array([0.0, 0.0, 1.0]),
            knuckle_intensity=0.0,
        )
        traj = integrate_trajectory(crosser.position, params, cfg, attacking_high_x=attacking_home, rng=rng)
        self.stats["crosses"] += 1

        atk_team = state.team(crosser.team_id)
        def_team = state.away if atk_team is state.home else state.home
        outcome = self._duel_at_point(state, traj.landed, atk_team, def_team, crosser, rng, aerial=True)
        return traj, outcome

    def _duel_at_point(
        self,
        state: MatchAffectiveState,
        point: np.ndarray,
        atk_team,
        def_team,
        actor: PlayerAffectiveState,
        rng: np.random.Generator,
        aerial: bool = False,
    ) -> AerialOutcome:
        cfg = self.cfg

        def nearby(players, pt, radius):
            out = []
            for p in players:
                if p.on_pitch and float(np.linalg.norm(p.position - pt)) < radius:
                    out.append(p)
            return out

        atk_c = nearby(atk_team.players, point, cfg.aerial_radius)
        def_c = nearby(def_team.players, point, cfg.aerial_radius)
        if not atk_c and not def_c:
            return AerialOutcome(None, "loose", 0.0, point)

        def best_score(players):
            if not players:
                return -1e9, None
            scores = []
            for p in players:
                aerial_ab = float(getattr(p.abilities, "aerial", 0.5))
                head = float(getattr(p.abilities, "heading", 0.5))
                jump = cfg.jump_base * float(sigmoid(p.stamina_logit)) * (0.7 + 0.3 * aerial_ab)
                timing = rng.normal(0, 0.08)
                sc = cfg.aerial_a1 * head + cfg.aerial_a2 * jump + cfg.aerial_a3 * aerial_ab + timing
                scores.append((sc, p))
            return max(scores, key=lambda x: x[0])

        sa, pa = best_score(atk_c)
        sd, pd = best_score(def_c)
        if pa is None and pd is None:
            return AerialOutcome(None, "loose", 0.0, point)
        if pd is None or (pa is not None and sd is not None and sa >= sd):
            winner = pa
            contact = "header"
        else:
            winner = pd
            contact = "clearance"

        xg = 0.0
        goal = False
        attacking_home = atk_team.team_id == state.home.team_id
        side = "home" if attacking_home else "away"
        if contact == "header" and winner is not None:
            dist_g = abs(point[0] - (0.995 if attacking_home else 0.005))
            xg = cfg.xg_header_base * float(winner.abilities.heading) * float(np.exp(-3.5 * dist_g))
            # StatsBomb "headers" ≈ header shot attempts, not every aerial duel contact.
            attempt_min = float(getattr(cfg, "xg_header_attempt_min", 0.055))
            if xg >= attempt_min:
                self.stats["headers"] += 1
                self.stats[f"{side}_headers"] += 1
                self.stats["header_attempts"] += 1
            if rng.random() < xg:
                goal = True
                self.stats["aerial_goals"] += 1
                self.stats[f"{side}_aerial_goals"] += 1

        return AerialOutcome(
            winner.player_id if winner else None, contact, xg, point,
            goal=goal,
        )
