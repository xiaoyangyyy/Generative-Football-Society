"""Phase 3 — physics-based shots, GK model, xG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.match_engine.ball_physics import (
    BallActionParams,
    TrajectoryResult,
    azimuth_to_goal,
    integrate_trajectory,
    sample_shot_params,
)
from src.match_engine.math_utils import sigmoid, softmax
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.micro_events import MicroEvent, MicroEventType
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState, PlayerModulators


@dataclass
class ShotOutcome:
    kind: str
    xg: float
    goal: bool
    on_target: bool
    saved: bool
    traj: TrajectoryResult
    events: List[MicroEvent]


class ShotEngine:
    def __init__(self, cfg: MicroMatchConfig, sie: SpatialIntelligenceEngine):
        self.cfg = cfg
        self.sie = sie
        self.player_tracker = None
        self.stats = {
            "home_shots": 0,
            "away_shots": 0,
            "home_shots_scheduled": 0,
            "away_shots_scheduled": 0,
            "home_on_target": 0,
            "away_on_target": 0,
            "home_goals": 0,
            "away_goals": 0,
            "home_headers": 0,
            "away_headers": 0,
            "curved": 0,
            "knuckle": 0,
        }

    def _dist_to_goal(self, p: np.ndarray, attacking_home: bool) -> float:
        gx = 0.995 if attacking_home else 0.005
        return float(abs(gx - p[0]))

    def _gk_player(self, state: MatchAffectiveState, defending_team_id: str) -> PlayerAffectiveState:
        team = state.team(defending_team_id)
        for p in team.players:
            if p.role == "GK":
                return p
        return team.players[0]

    def _gk_save_prob(
        self,
        traj: TrajectoryResult,
        gk: PlayerAffectiveState,
        shot_params: BallActionParams,
        dist: float,
    ) -> float:
        cfg = self.cfg
        reflex = float(getattr(gk.abilities, "gk_reflex", 0.6))
        aerial = float(getattr(gk.abilities, "gk_aerial", 0.55))
        pos_err = float(abs(gk.position[1] - (traj.goal_y or 0.5)))
        curve_pen = cfg.gk_curve_penalty * shot_params.omega * (1.0 - reflex)
        knuckle_pen = cfg.gk_knuckle_penalty * shot_params.knuckle_intensity * (1.0 - reflex)
        z = (
            cfg.gk_b0
            + cfg.gk_b_reflex * reflex
            + cfg.gk_b_aerial * aerial
            - cfg.gk_b_dist * dist
            - cfg.gk_b_pos * pos_err
            - curve_pen
            - knuckle_pen
        )
        if traj.peak_height > 2.0:
            z += 0.15 * aerial
        return float(sigmoid(z))

    def _shot_candidates(self, carrier: PlayerAffectiveState, dist: float, rng: np.random.Generator) -> List[str]:
        kinds = ["driven", "curved", "power"]
        if dist > self.cfg.long_shot_dist:
            kinds = ["power", "curved"]
        if float(carrier.abilities.knuckle) > 0.62 and rng.random() < 0.22:
            kinds.append("knuckle")
        if dist < 0.28 and rng.random() < 0.32:
            kinds.append("header")  # close range header attempt
        return kinds

    def _utility_shot(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        kind: str,
        mod: PlayerModulators,
        attacking_home: bool,
    ) -> float:
        cfg = self.cfg
        dist = self._dist_to_goal(carrier.position, attacking_home)
        phi = self.sie.phi_at(state, carrier.position, attacking_home)
        box = float(np.exp(-dist * dist * cfg.shot_box_sharpness))
        u = cfg.shot_u_phi * phi * box + cfg.shot_u_skill * float(carrier.abilities.shot)
        u += mod.shot_utility_bias
        if kind == "curved":
            u += 0.25 * float(carrier.abilities.curve)
        if kind == "knuckle":
            u += 0.2 * float(carrier.abilities.knuckle)
        if kind == "power" and dist > 0.28:
            u += 0.2 * float(carrier.abilities.power)
        u -= cfg.shot_u_dist * dist
        return float(u)

    def resolve_shot(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        mod: PlayerModulators,
        rng: np.random.Generator,
        *,
        forced_kind: Optional[str] = None,
        scheduled_on_target: bool = False,
        from_schedule: bool = False,
    ) -> ShotOutcome:
        cfg = self.cfg
        attacking_home = carrier.team_id == state.home.team_id
        from src.match_engine.ball_path_logger import ball_log_wm_snapshot_enabled

        obs_pre = None
        if ball_log_wm_snapshot_enabled():
            from src.match_engine.world_model.observation import encode_observation

            obs_pre = encode_observation(state, attacking_home=attacking_home)
        dist = self._dist_to_goal(carrier.position, attacking_home)
        kinds = [forced_kind] if forced_kind else self._shot_candidates(carrier, dist, rng)
        utils = [self._utility_shot(state, carrier, k, mod, attacking_home) for k in kinds]
        idx = int(rng.choice(len(kinds), p=softmax(np.array(utils), tau=cfg.shot_tau * mod.tau_dec)))
        kind = kinds[idx]

        params = sample_shot_params(kind, dist, carrier.abilities, mod.shot_utility_bias, rng, cfg)
        params.azim = azimuth_to_goal(carrier.position, attacking_home)

        drag_scale = float(getattr(cfg, "shot_phys_drag_scale", 1.0))
        traj = integrate_trajectory(
            carrier.position,
            params,
            cfg,
            attacking_high_x=attacking_home,
            drag_scale=drag_scale,
            rng=rng,
        )
        if not traj.crossed_goal_line and dist < cfg.shot_max_dist:
            boosted = params
            for _ in range(8):
                boosted = BallActionParams(
                    v0=float(min(getattr(cfg, "shot_v0_max", 1.32), boosted.v0 * 1.10)),
                    elev=boosted.elev,
                    azim=boosted.azim,
                    omega=boosted.omega,
                    spin_axis=boosted.spin_axis.copy(),
                    knuckle_intensity=boosted.knuckle_intensity,
                    outside_foot=boosted.outside_foot,
                    ground_weight=boosted.ground_weight,
                )
                traj = integrate_trajectory(
                    carrier.position,
                    boosted,
                    cfg,
                    attacking_high_x=attacking_home,
                    drag_scale=drag_scale,
                    rng=rng,
                )
                params = boosted
                if traj.crossed_goal_line:
                    break

        def_team_id = state.away.team_id if attacking_home else state.home.team_id
        gk = self._gk_player(state, def_team_id)
        save_p = self._gk_save_prob(traj, gk, params, dist)
        if scheduled_on_target:
            save_p = float(np.clip(save_p * cfg.scheduled_on_target_save_mult, 0.05, 0.92))

        # xG from geometry + model
        angle_y = abs((traj.goal_y or 0.5) - 0.5)
        xg_geom = cfg.xg_geom_base * float(np.exp(-dist * cfg.xg_dist_decay)) * float(
            np.exp(-angle_y * cfg.xg_angle_penalty)
        )
        if kind == "header":
            xg_geom *= 0.85
        if kind == "knuckle":
            xg_geom *= 1.0 + cfg.xg_knuckle_boost * params.knuckle_intensity
        # Standard xG: geometric chance before GK (not discounted by save model).
        xg = float(np.clip(xg_geom, 0.01, 0.78))

        from src.match_engine.math_utils import sigmoid

        angle_y = abs((traj.goal_y or 0.5) - 0.5)
        z_ang = float(getattr(cfg, "shot_sot_z_angle", 4.4))
        z_dist = float(getattr(cfg, "shot_sot_z_dist", 2.15))
        if traj.in_goal_mouth:
            z = float(getattr(cfg, "shot_sot_z_in_goal", 0.92)) - z_ang * angle_y - z_dist * dist
        else:
            z = float(getattr(cfg, "shot_sot_z_wide", -2.4)) - 1.2 * angle_y - 0.9 * dist
        on_p = float(sigmoid(z))
        if not traj.in_goal_mouth:
            on_p *= float(getattr(cfg, "shot_sot_wide_scale", 0.10))
        if scheduled_on_target and dist < 0.22:
            on_p = max(on_p, 0.55)
        on_target = bool(rng.random() < on_p)
        goal = bool(traj.in_goal_mouth and rng.random() > save_p)
        if scheduled_on_target and on_target and not goal:
            goal = bool(rng.random() < min(0.55, xg + cfg.scheduled_on_target_goal_bonus))
        saved = bool(on_target and not goal and rng.random() < save_p)

        trk = self.player_tracker
        if trk is not None:
            trk.record_shot(
                carrier.player_id,
                xg=xg,
                goal=goal,
                on_target=on_target,
            )

        side = "home" if attacking_home else "away"
        self.stats[f"{side}_shots"] += 1
        if from_schedule:
            self.stats[f"{side}_shots_scheduled"] += 1
        if on_target:
            self.stats[f"{side}_on_target"] += 1
        if goal:
            self.stats[f"{side}_goals"] += 1
        if kind == "header":
            self.stats[f"{side}_headers"] += 1
        if kind == "curved":
            self.stats["curved"] += 1
        if kind == "knuckle":
            self.stats["knuckle"] += 1

        events: List[MicroEvent] = []
        if goal:
            events.append(
                MicroEvent(
                    t_sec=state.clock_seconds,
                    event_type=MicroEventType.GOAL_SCORED,
                    team_id=carrier.team_id,
                    player_id=carrier.player_id,
                    opponent_team_id=state.away.team_id if attacking_home else state.home.team_id,
                    intensity=1.0,
                )
            )
            opp = state.away.team_id if attacking_home else state.home.team_id
            events.append(
                MicroEvent(
                    t_sec=state.clock_seconds + 0.5,
                    event_type=MicroEventType.GOAL_CONCEDED,
                    team_id=opp,
                    intensity=0.9,
                )
            )
        elif on_target and saved:
            events.append(
                MicroEvent(
                    t_sec=state.clock_seconds,
                    event_type=MicroEventType.SAVE,
                    team_id=gk.team_id,
                    player_id=gk.player_id,
                    intensity=0.75,
                )
            )
        elif on_target:
            events.append(
                MicroEvent(
                    t_sec=state.clock_seconds,
                    event_type=MicroEventType.SHOT_ON_TARGET,
                    team_id=carrier.team_id,
                    player_id=carrier.player_id,
                    intensity=0.7,
                )
            )
        else:
            events.append(
                MicroEvent(
                    t_sec=state.clock_seconds,
                    event_type=MicroEventType.SHOT_OFF_TARGET,
                    team_id=carrier.team_id,
                    player_id=carrier.player_id,
                    intensity=0.55,
                )
            )

        cap = float(getattr(self.cfg, "micro_xg_match_cap", 2.4))
        if attacking_home:
            room = max(0.0, cap - state.micro_xg_home)
            state.micro_xg_home += min(float(xg), room)
        else:
            room = max(0.0, cap - state.micro_xg_away)
            state.micro_xg_away += min(float(xg), room)

        from src.match_engine.ball_path_logger import ball_log_wm_snapshot_enabled, record_shot

        obs_post = None
        if ball_log_wm_snapshot_enabled():
            from src.match_engine.world_model.observation import encode_observation

            obs_post = encode_observation(state, attacking_home=attacking_home)

        record_shot(
            state,
            carrier=carrier,
            gk=gk,
            kind=kind,
            xg=xg,
            goal=goal,
            on_target=on_target,
            saved=saved,
            dist=dist,
            traj_peak=float(getattr(traj, "peak_height", 0.0)),
            traj_tof=float(getattr(traj, "time_of_flight", 0.0)),
            in_goal=bool(traj.in_goal_mouth),
            obs_pre=obs_pre,
            obs_post=obs_post,
        )

        return ShotOutcome(kind, xg, goal, on_target, saved, traj, events)

    def shot_utility_max(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        mod: PlayerModulators,
    ) -> float:
        attacking_home = carrier.team_id == state.home.team_id
        kinds = self._shot_candidates(carrier, self._dist_to_goal(carrier.position, attacking_home), np.random.default_rng(0))
        if not kinds:
            return -1e9
        return max(self._utility_shot(state, carrier, k, mod, attacking_home) for k in kinds)
