"""§2b Passing — short / through / long + Phase 3b physics (curve, outside-foot, ground, intercept)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.match_engine.ball_physics import (
    azimuth_to_target,
    desired_pass_omega,
    ground_pass_weight,
    integrate_pass_trajectory,
    outside_foot_factor,
    sample_pass_delivery_params,
)
from src.match_engine.math_utils import sigmoid, softmax
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.micro_events import MicroEvent, MicroEventType
from src.match_engine.pass_calibration import (
    logit_target_blend,
    overall_logit_anchor,
    pass_mix_log_bias,
    target_logit,
)
from src.match_engine.pass_intercept import evaluate_pass_intercept
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.tactical_engine import TacticalMicroEngine
from src.match_engine.wall_pass import (
    find_wall_partner,
    wall_pass_utility,
    wall_return_target,
)
from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState, PlayerModulators
from src.match_engine.receiver_ranker import ReceiverRanker, default_receiver_ranker_path


@dataclass
class PassAction:
    pass_kind: str
    from_id: str
    to_id: str
    target: np.ndarray
    utility: float
    success_p: float
    completed: bool
    omega: float = 0.0
    landed: Optional[np.ndarray] = None
    target_miss: float = 0.0
    lateral_dev: float = 0.0
    outside_foot: float = 0.0
    ground_weight: float = 0.0
    pred_miss: float = 0.0
    intercepted: bool = False
    wall_combo: bool = False


class PassingEngine:
    def __init__(
        self,
        cfg: MicroMatchConfig,
        sie: SpatialIntelligenceEngine,
        wm_runtime=None,
        *,
        base_dir: str = ".",
    ):
        self.cfg = cfg
        self._base_dir = base_dir
        self.sie = sie
        self.wm_runtime = wm_runtime
        ranker_path = default_receiver_ranker_path(base_dir)
        try:
            self.receiver_ranker = ReceiverRanker.load(ranker_path) if cfg.enable_receiver_ranker and ranker_path.is_file() else None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            self.receiver_ranker = None
        self.player_tracker = None
        self._tac_eng = TacticalMicroEngine(cfg)
        self.stats = {
            "home_attempts": 0,
            "away_attempts": 0,
            "home_completed": 0,
            "away_completed": 0,
            "home_through": 0,
            "away_through": 0,
            "home_long": 0,
            "away_long": 0,
            "home_curved": 0,
            "away_curved": 0,
            "home_outside": 0,
            "away_outside": 0,
            "home_ground": 0,
            "away_ground": 0,
            "home_intercepts": 0,
            "away_intercepts": 0,
            "home_wall": 0,
            "away_wall": 0,
            "home_wall_ok": 0,
            "away_wall_ok": 0,
            "curve_omega_sum": 0.0,
            "curve_pass_count": 0,
        }

    def _team_attack_home(self, state: MatchAffectiveState, team_id: str) -> bool:
        return team_id == state.home.team_id

    def _get_carrier(self, state: MatchAffectiveState) -> Optional[PlayerAffectiveState]:
        if not state.ball.possessor_id:
            return None
        for team in (state.home, state.away):
            for p in team.players:
                if p.player_id == state.ball.possessor_id and p.on_pitch:
                    return p
        return None

    def _elev_hint(self, kind: str, dist: float) -> float:
        cfg = self.cfg
        if kind == "wall":
            return cfg.pass_elev_short * 0.65
        if kind == "through":
            return cfg.pass_elev_through + 0.04 * min(1.0, dist / 0.4)
        if kind == "long":
            return cfg.pass_elev_long
        return cfg.pass_elev_short + 0.02 * min(1.0, dist / cfg.short_dist_max)

    def _candidates(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        mod_map: Dict[str, PlayerModulators],
    ) -> List[Tuple[PlayerAffectiveState, str, np.ndarray]]:
        team = state.team(carrier.team_id)
        attacking_home = self._team_attack_home(state, team.team_id)
        out: List[Tuple[PlayerAffectiveState, str, np.ndarray]] = []
        dists = []
        for p in team.players:
            if not p.on_pitch or p.player_id == carrier.player_id:
                continue
            if p.role == "GK" and float(np.linalg.norm(p.position - carrier.position)) > 0.45:
                continue
            d = float(np.linalg.norm(p.position - carrier.position))
            dists.append((d, p))
        dists.sort(key=lambda x: x[0])
        near = dists[: self.cfg.pass_candidates_max]
        for d, p in near:
            if d >= self.cfg.long_dist_min:
                lead = np.array([0.02, 0.0]) if attacking_home else np.array([-0.02, 0.0])
                out.append((p, "long", np.clip(p.position + lead, 0.03, 0.97)))
            else:
                out.append((p, "short", p.position.copy()))
        risk_budget = float(team.coach.tactical_current.get("risk_budget", 0.5))
        n_through = 1 + int(risk_budget > 0.58)
        ordered = sorted(near, key=lambda x: x[1].position[0], reverse=attacking_home)
        for _d, p in ordered[:n_through]:
            fwd = self.cfg.through_forward + 0.05 * risk_budget
            tgt = p.position + (np.array([fwd, 0.0]) if attacking_home else np.array([-fwd, 0.0]))
            out.append((p, "through", np.clip(tgt, 0.03, 0.97)))
        wide = [p for p in team.players if p.on_pitch and p.role in ("LW", "RW", "LB", "RB")]
        if wide:
            far = max(
                wide,
                key=lambda q: float(np.linalg.norm(q.position - carrier.position)),
            )
            if float(np.linalg.norm(far.position - carrier.position)) >= self.cfg.long_dist_min:
                out.append((far, "long", far.position.copy()))
        outfield = [p for p in team.players if p.on_pitch and p.role != "GK" and p.player_id != carrier.player_id]
        if outfield:
            far_any = max(outfield, key=lambda q: float(np.linalg.norm(q.position - carrier.position)))
            if float(np.linalg.norm(far_any.position - carrier.position)) >= self.cfg.long_dist_min:
                lead = np.array([0.03, 0.0]) if attacking_home else np.array([-0.03, 0.0])
                out.append((far_any, "long", np.clip(far_any.position + lead, 0.03, 0.97)))
        if self.cfg.enable_wall_pass:
            wp = find_wall_partner(carrier, team.players, self.cfg)
            if wp is not None:
                out.append((wp.player, "wall", wp.player.position.copy()))
        return out

    def _team_tac(self, state: MatchAffectiveState, team_id: str) -> dict:
        return dict(state.team(team_id).coach.tactical_current or {})

    def _press_at(self, state: MatchAffectiveState, pos: np.ndarray) -> float:
        sh = state.spatial.press.shape
        ix = int(np.clip(pos[0] * (sh[0] - 1), 0, sh[0] - 1))
        iy = int(np.clip(pos[1] * (sh[1] - 1), 0, sh[1] - 1))
        return float(state.spatial.press[ix, iy])

    def _utility(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        target: np.ndarray,
        recv: PlayerAffectiveState,
        kind: str,
        mod_carrier: PlayerModulators,
        attacking_home: bool,
        *,
        omega_desired: float = 0.0,
        lane: float = 0.5,
        press: float = 0.0,
        outside_hint: float = 0.0,
        ground_hint: float = 0.0,
    ) -> float:
        cfg = self.cfg
        opp = state.away if state.home.team_id == carrier.team_id else state.home
        dist = float(np.linalg.norm(target - carrier.position))
        phi = self.sie.phi_at(state, target, attacking_home)
        off = self.sie.offside_risk(state, target, attacking_home)
        adv = self.sie.numerical_advantage(state, target, attacking_home)

        u = (
            cfg.b_phi * phi
            + cfg.b_lane * lane
            + cfg.b_off * np.log(1.0 - off + 1e-3)
            + 0.6 * adv
            - cfg.b_dist * dist
        )
        if kind == "short" and dist > cfg.short_dist_max:
            u -= 1.5 * (dist - cfg.short_dist_max)
        if kind == "through":
            u += 0.18 * float(carrier.abilities.vision)
            u += 0.12 * float(recv.abilities.pace)
            u -= 0.32
        if kind == "long":
            if dist < cfg.long_dist_min:
                u -= 1.2
            else:
                u += 0.48 + 0.28 * min(1.0, (dist - cfg.long_dist_min) / 0.15)
        if cfg.enable_pass_physics and omega_desired > cfg.pass_omega_curve_threshold:
            u += cfg.pass_u_curve * (omega_desired / cfg.pass_omega_max) * (
                cfg.pass_u_curve_lane * lane + (1.0 - lane) * (0.5 + press)
            )
        if cfg.enable_pass_physics and ground_hint > 0.55 and kind == "short":
            u += 0.18 * ground_hint * float(carrier.abilities.pass_skill)
        if cfg.enable_pass_physics and outside_hint > 0.45:
            u += 0.12 * outside_hint * float(carrier.abilities.curve)
        if kind == "wall":
            u += wall_pass_utility(carrier, recv, press, lane, cfg) + 0.25 * phi
        from src.match_engine.player_channel_passing import (
            blend_pass_risk_from_channels,
            channel_pass_utility_boost,
        )

        u += channel_pass_utility_boost(carrier, kind)
        tac = blend_pass_risk_from_channels(carrier, self._team_tac(state, carrier.team_id))
        bias = self._tac_eng.passing_bias(tac)
        if kind == "short":
            u += bias.get("short", 0.0)
        elif kind == "through":
            u += bias.get("through", 0.0)
        elif kind == "long":
            u += bias.get("long", 0.0)
        # Match real-world pass-type prevalence (StatsBomb WC2022).
        u += float(getattr(cfg, "pass_mix_bias_weight", 0.75)) * pass_mix_log_bias(
            kind, base_dir=getattr(self, "_base_dir", ".")
        )
        u *= float(carrier.abilities.pass_skill) * float(mod_carrier.vision_scale)
        if self.receiver_ranker is not None:
            defenders = [p.position for p in opp.players if p.on_pitch]
            nearest = min((float(np.linalg.norm(pos - recv.position)) for pos in defenders), default=0.5)
            direction = 1.0 if attacking_home else -1.0
            features = [
                dist,
                direction * float(recv.position[0] - carrier.position[0]),
                abs(float(recv.position[1] - carrier.position[1])),
                nearest,
                float(lane),
            ]
            # Real tracking informs receiver choice only. Physics and interception
            # remain responsible for pass completion and the official outcome.
            u += float(self.cfg.receiver_ranker_blend) * float(
                np.clip(self.receiver_ranker.score(features), -2.0, 2.0)
            )
        from src.match_engine.math_utils import finite_float

        return finite_float(u, 0.0)

    def _success_prob(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        target: np.ndarray,
        kind: str,
        lane: float,
        mod_carrier: PlayerModulators,
        *,
        land_miss: float = 0.0,
        lateral_dev: float = 0.0,
        pred_miss: float = 0.0,
        omega_desired: float = 0.0,
        omega_used: float = 0.0,
        outside_foot: float = 0.0,
        outside_desired: float = 0.0,
        press: float = 0.0,
        intercept_risk: float = 0.0,
        ground_weight: float = 0.0,
    ) -> float:
        cfg = self.cfg
        spin_exec = (omega_used - omega_desired) ** 2
        outside_exec = (outside_foot - outside_desired) ** 2
        dist_norm = float(np.linalg.norm(target - carrier.position))
        base_dir = getattr(self, "_base_dir", ".")
        pass_skill = float(carrier.abilities.pass_skill)
        # Anchor z at StatsBomb overall completion (WC 2022 open data), then vary continuously.
        z = overall_logit_anchor(base_dir=base_dir)
        z += cfg.b_lane * lane
        z += 0.95 * (pass_skill - 0.55)
        z -= 0.24 * press
        z -= 0.14 * dist_norm
        z -= 0.06 * (mod_carrier.tau_dec - cfg.tau_dec_base)
        if cfg.enable_pass_physics:
            miss_cap = float(getattr(cfg, "pass_land_miss_cap", 0.16))
            gamma = float(getattr(cfg, "pass_exec_miss_gamma", 0.55))

            def _soft_miss(raw: float, cap: float = miss_cap) -> float:
                x = min(max(0.0, raw), cap * 1.15) / max(cap, 1e-6)
                return cap * (x ** gamma)

            kind_scale = {
                "short": 0.42,
                "through": 0.68,
                "long": 1.0,
                "wall": 0.50,
            }.get(kind, 0.62)
            # Ground / short passes: StatsBomb ground ~93%; skilled players absorb small errors.
            ground_relief = 1.0 - 0.50 * max(0.0, float(ground_weight))
            skill_relief = 0.50 + 0.50 * pass_skill
            exec_scale = kind_scale * ground_relief * skill_relief

            z -= exec_scale * cfg.pass_b_land_err * _soft_miss(land_miss)
            z -= exec_scale * cfg.pass_b_lateral * _soft_miss(lateral_dev)
            z -= exec_scale * cfg.pass_b_pred_miss * _soft_miss(pred_miss, miss_cap + 0.05)
            z -= exec_scale * cfg.pass_b_spin_exec * spin_exec
            z -= exec_scale * cfg.pass_outside_exec_penalty * outside_exec
            z += 0.10 * float(carrier.abilities.curve) * min(1.0, omega_desired / cfg.pass_omega_max)
            z -= 0.24 * intercept_risk
        if kind == "wall":
            z -= cfg.wall_second_z_penalty
        z_tgt = target_logit(kind, dist_norm, base_dir=base_dir)
        w = logit_target_blend(base_dir=base_dir)
        z = (1.0 - w) * z + w * z_tgt
        return float(sigmoid(z))

    def _execute_wall_return(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        wall_player: PlayerAffectiveState,
        attacking_home: bool,
        opp_team,
        mod_carrier: PlayerModulators,
        press: float,
        rng: np.random.Generator,
    ) -> Tuple[bool, np.ndarray, List[MicroEvent]]:
        """Second ODE leg: wall → sprinting carrier."""
        cfg = self.cfg
        events: List[MicroEvent] = []
        team = state.team(carrier.team_id)
        risk = float(team.coach.tactical_current.get("risk_budget", 0.5))
        ret_tgt = wall_return_target(carrier, attacking_home, risk, cfg)
        lane = self.sie.lane_quality(state, wall_player.position, ret_tgt, opp_team)
        traj, omega_d, params = self._deliver_pass(state, wall_player, ret_tgt, "short", lane, rng)
        land_xy = traj.landed
        opp_pool = [p for p in opp_team.players if p.on_pitch and p.role != "GK"]
        ic = evaluate_pass_intercept(
            land_xy, traj.time_of_flight, carrier, opp_pool, press, lane, cfg, rng, base_dir=getattr(self, "_base_dir", ".")
        )
        p2 = self._success_prob(
            state,
            wall_player,
            ret_tgt,
            "short",
            lane,
            mod_carrier,
            land_miss=traj.target_miss,
            lateral_dev=traj.lateral_dev,
            pred_miss=ic.pred_miss,
            omega_desired=omega_d,
            omega_used=traj.omega_used,
            outside_foot=traj.outside_foot,
            outside_desired=0.0,
            press=press,
            intercept_risk=ic.intercept_risk,
            ground_weight=traj.ground_weight,
        )
        ok = bool(rng.random() < p2) and not ic.intercepted
        if ok:
            state.ball.position = land_xy.copy()
            state.ball.possessor_id = carrier.player_id
            state.ball.possession_team_id = carrier.team_id
            state.ball.omega = traj.omega_used
            state.ball.spin_axis = params.spin_axis.copy()
            carrier.position = 0.65 * carrier.position + 0.35 * land_xy
            if attacking_home:
                state.micro_xg_home += cfg.wall_combo_xg_chip
            else:
                state.micro_xg_away += cfg.wall_combo_xg_chip
            events.append(
                MicroEvent(
                    t_sec=state.clock_seconds + 0.3,
                    event_type=MicroEventType.KEY_PASS,
                    team_id=carrier.team_id,
                    player_id=wall_player.player_id,
                    intensity=0.8,
                )
            )
            events.append(
                MicroEvent(
                    t_sec=state.clock_seconds + 0.5,
                    event_type=MicroEventType.ASSIST,
                    team_id=carrier.team_id,
                    player_id=wall_player.player_id,
                    intensity=0.7,
                )
            )
        return ok, land_xy, events

    def _deliver_pass(
        self,
        state: MatchAffectiveState,
        carrier: PlayerAffectiveState,
        target: np.ndarray,
        kind: str,
        lane: float,
        rng: np.random.Generator,
    ):
        cfg = self.cfg
        dist = float(np.linalg.norm(target - carrier.position))
        press = self._press_at(state, carrier.position)
        curve = float(carrier.abilities.curve)
        omega_d = desired_pass_omega(carrier.position, target, curve, lane, press, cfg)
        elev = self._elev_hint(kind, dist)
        az = azimuth_to_target(carrier.position, target)
        params = sample_pass_delivery_params(
            dist,
            elev,
            omega_d,
            curve,
            rng,
            cfg,
            start_xy=carrier.position,
            target_xy=target,
            azim=az,
        )
        traj = integrate_pass_trajectory(
            carrier.position,
            target,
            params,
            cfg,
            curve_skill=curve,
            rng=rng,
        )
        return traj, omega_d, params

    def step(
        self,
        state: MatchAffectiveState,
        mod_home: List[PlayerModulators],
        mod_away: List[PlayerModulators],
        rng: np.random.Generator,
    ) -> Tuple[Optional[PassAction], List[MicroEvent]]:
        cfg = self.cfg
        carrier = self._get_carrier(state)
        events: List[MicroEvent] = []
        if carrier is None:
            self._assign_random_possession(state, rng)
            return None, events

        attacking_home = self._team_attack_home(state, carrier.team_id)
        mod_list = mod_home if attacking_home else mod_away
        mod_map = {m.player_id: m for m in mod_list}
        mod_carrier = mod_map.get(carrier.player_id, mod_list[0] if mod_list else PlayerModulators(carrier.player_id))

        cands = self._candidates(state, carrier, mod_map)
        if not cands:
            return None, events

        from src.match_engine.ball_path_logger import ball_log_wm_snapshot_enabled

        obs_pre = obs_post = None
        if ball_log_wm_snapshot_enabled():
            from src.match_engine.world_model.observation import encode_observation

            obs_pre = encode_observation(state, attacking_home=attacking_home)

        opp_team = state.away if attacking_home else state.home
        utilities, meta = [], []
        for recv, kind, tgt in cands:
            lane = self.sie.lane_quality(state, carrier.position, tgt, opp_team)
            press = self._press_at(state, carrier.position)
            omega_d = (
                desired_pass_omega(carrier.position, tgt, float(carrier.abilities.curve), lane, press, cfg)
                if cfg.enable_pass_physics and cfg.enable_phase3
                else 0.0
            )
            elev_h = self._elev_hint(kind, float(np.linalg.norm(tgt - carrier.position)))
            g_h = ground_pass_weight(elev_h, cfg) if cfg.enable_pass_physics else 0.0
            out_h = (
                outside_foot_factor(carrier.position, tgt, float(carrier.abilities.curve), omega_d, cfg)
                if cfg.enable_pass_physics
                else 0.0
            )
            utilities.append(
                self._utility(
                    state,
                    carrier,
                    tgt,
                    recv,
                    kind,
                    mod_carrier,
                    attacking_home,
                    omega_desired=omega_d,
                    lane=lane,
                    press=press,
                    outside_hint=out_h,
                    ground_hint=g_h,
                )
            )
            success_prior = self._success_prob(
                state,
                carrier,
                tgt,
                kind,
                lane,
                mod_carrier,
                omega_desired=omega_d,
                press=press,
            )
            meta.append((recv, kind, tgt, lane, press, omega_d, success_prior))

        if self.wm_runtime is not None:
            from src.match_engine.world_model.planner import pass_imagination_bonuses

            bonuses = pass_imagination_bonuses(
                self.wm_runtime, state, carrier, meta, attacking_home
            )
            utilities = [u + b for u, b in zip(utilities, bonuses)]

        tau = cfg.pass_tau_base * max(0.15, float(mod_carrier.tau_dec))
        util_arr = np.nan_to_num(np.array(utilities, dtype=float), nan=0.0)
        probs = softmax(util_arr, tau=max(0.15, tau))
        idx = int(rng.choice(len(probs), p=probs))
        recv, kind, tgt, lane, press, omega_d, success_prior = meta[idx]

        if getattr(state, "_wm_recorder", None) is not None and getattr(state, "_wm_obs_pre", None) is not None:
            from src.match_engine.world_model.action_codec import encode_pass_candidate

            state._wm_last_action = encode_pass_candidate(  # noqa: SLF001
                state,
                carrier,
                recv,
                kind,
                tgt,
                success_p=success_prior,
                horizon_s=float(self.cfg.dt_default),
            )

        use_physics = cfg.enable_pass_physics and cfg.enable_phase3
        land_miss = lateral_dev = pred_miss = 0.0
        omega_used = outside_f = ground_w = 0.0
        land_xy = tgt.copy()
        tof = 0.0
        spin_axis = np.array([0.0, 0.0, 1.0])
        outside_desired = 0.0

        if use_physics:
            traj, omega_d, params = self._deliver_pass(state, carrier, tgt, kind, lane, rng)
            land_xy = traj.landed
            land_miss = traj.target_miss
            lateral_dev = traj.lateral_dev
            omega_used = traj.omega_used
            outside_f = traj.outside_foot
            ground_w = traj.ground_weight
            tof = traj.time_of_flight
            spin_axis = params.spin_axis.copy()
            outside_desired = outside_foot_factor(
                carrier.position, tgt, float(carrier.abilities.curve), omega_d, cfg
            )

        intercept_risk = 0.0
        intercepted = False
        interceptor_id = None
        if use_physics:
            opp_pool = [p for p in opp_team.players if p.on_pitch and p.role != "GK"]
            ic = evaluate_pass_intercept(
                land_xy, tof, recv, opp_pool, press, lane, cfg, rng, base_dir=getattr(self, "_base_dir", ".")
            )
            pred_miss = ic.pred_miss
            intercept_risk = ic.intercept_risk
            intercepted = ic.intercepted
            interceptor_id = ic.interceptor_id

        p_ok = self._success_prob(
            state,
            carrier,
            tgt,
            kind,
            lane,
            mod_carrier,
            land_miss=land_miss,
            lateral_dev=lateral_dev,
            pred_miss=pred_miss,
            omega_desired=omega_d,
            omega_used=omega_used,
            outside_foot=outside_f,
            outside_desired=outside_desired,
            press=press,
            intercept_risk=intercept_risk,
            ground_weight=ground_w if use_physics else 0.0,
        )
        completed = bool(rng.random() < p_ok) and not intercepted

        trk = self.player_tracker
        if trk is not None:
            trk.record_pass(
                carrier.player_id,
                recv.player_id if completed else None,
                kind,
                completed,
            )

        side = "home" if attacking_home else "away"
        if attacking_home:
            self.stats["home_attempts"] += 1
            if completed:
                self.stats["home_completed"] += 1
            if kind == "through":
                self.stats["home_through"] += 1
            if kind == "long":
                self.stats["home_long"] += 1
        else:
            self.stats["away_attempts"] += 1
            if completed:
                self.stats["away_completed"] += 1
            if kind == "through":
                self.stats["away_through"] += 1
            if kind == "long":
                self.stats["away_long"] += 1

        if use_physics and omega_used >= cfg.pass_omega_curve_threshold:
            self.stats[f"{side}_curved"] += 1
            self.stats["curve_omega_sum"] += omega_used
            self.stats["curve_pass_count"] += 1
        if use_physics and outside_f > 0.42:
            self.stats[f"{side}_outside"] += 1
        if use_physics and ground_w > 0.55:
            self.stats[f"{side}_ground"] += 1
        if intercepted:
            self.stats[f"{side}_intercepts"] += 1
        if kind == "wall":
            self.stats[f"{side}_wall"] += 1

        wall_combo_ok = False
        if completed and kind == "wall" and cfg.enable_wall_pass:
            state.ball.position = land_xy.copy()
            state.ball.possessor_id = recv.player_id
            state.ball.possession_team_id = carrier.team_id
            recv.position = 0.75 * recv.position + 0.25 * land_xy
            wall_ok, _, wall_ev = self._execute_wall_return(
                state, carrier, recv, attacking_home, opp_team, mod_carrier, press, rng
            )
            events.extend(wall_ev)
            wall_combo_ok = wall_ok
            if wall_ok:
                self.stats[f"{side}_wall_ok"] += 1
            completed = wall_ok
            state.ball.height = 0.0
        elif completed:
            state.ball.position = land_xy.copy()
            state.ball.possessor_id = recv.player_id
            state.ball.possession_team_id = carrier.team_id
            state.ball.height = 0.0
            if use_physics:
                state.ball.omega = omega_used
                state.ball.spin_axis = spin_axis
            recv.position = 0.7 * recv.position + 0.3 * land_xy
            if kind in ("through", "long") or omega_used >= cfg.pass_omega_curve_threshold:
                events.append(
                    MicroEvent(
                        t_sec=state.clock_seconds,
                        event_type=MicroEventType.KEY_PASS,
                        team_id=carrier.team_id,
                        player_id=carrier.player_id,
                        intensity=0.75 if outside_f > 0.45 else 0.65,
                    )
                )

        else:
            opp_players = [p for p in opp_team.players if p.on_pitch and p.role != "GK"]
            if intercepted and interceptor_id:
                for p in opp_players:
                    if p.player_id == interceptor_id:
                        state.ball.possessor_id = p.player_id
                        state.ball.possession_team_id = opp_team.team_id
                        state.ball.position = land_xy.copy()
                        state.ball.omega = 0.0
                        events.append(
                            MicroEvent(
                                t_sec=state.clock_seconds,
                                event_type=MicroEventType.TACKLE_WON,
                                team_id=opp_team.team_id,
                                player_id=p.player_id,
                                intensity=0.72,
                            )
                        )
                        events.append(
                            MicroEvent(
                                t_sec=state.clock_seconds,
                                event_type=MicroEventType.DISPOSSESSED,
                                team_id=carrier.team_id,
                                player_id=carrier.player_id,
                                intensity=0.65,
                            )
                        )
                        break
            elif opp_players:
                taker = opp_players[int(rng.integers(0, len(opp_players)))]
                state.ball.possessor_id = taker.player_id
                state.ball.possession_team_id = opp_team.team_id
                state.ball.position = land_xy.copy() if use_physics else taker.position.copy()
                state.ball.omega = 0.0
                events.append(
                    MicroEvent(
                        t_sec=state.clock_seconds,
                        event_type=MicroEventType.DISPOSSESSED,
                        team_id=carrier.team_id,
                        player_id=carrier.player_id,
                        intensity=0.6,
                    )
                )

        if getattr(state, "_wm_last_action", None) is not None:
            from src.match_engine.world_model.schema import PASS_OUTCOME_INDEX

            state._wm_last_action[PASS_OUTCOME_INDEX] = 1.0 if completed else 0.0

        interceptor_player = None
        if intercepted and interceptor_id:
            for p in opp_team.players:
                if p.player_id == interceptor_id:
                    interceptor_player = p
                    break

        if ball_log_wm_snapshot_enabled():
            from src.match_engine.world_model.observation import encode_observation

            obs_post = encode_observation(state, attacking_home=attacking_home)

        from src.match_engine.ball_path_logger import record_pass

        record_pass(
            state,
            carrier=carrier,
            receiver=recv,
            kind=kind,
            target=tgt,
            landed=land_xy,
            completed=completed,
            success_p=p_ok,
            press=press,
            lane=lane,
            intercepted=intercepted,
            interceptor=interceptor_player,
            omega=omega_used,
            outside_foot=outside_f,
            ground_weight=ground_w,
            time_of_flight=tof,
            target_miss=land_miss,
            lateral_dev=lateral_dev,
            wall_combo=wall_combo_ok,
            obs_pre=obs_pre,
            obs_post=obs_post,
        )

        action = PassAction(
            pass_kind=kind,
            from_id=carrier.player_id,
            to_id=recv.player_id,
            target=tgt,
            utility=float(utilities[idx]),
            success_p=p_ok,
            completed=completed,
            omega=omega_used,
            landed=land_xy.copy(),
            target_miss=land_miss,
            lateral_dev=lateral_dev,
            outside_foot=outside_f,
            ground_weight=ground_w,
            pred_miss=pred_miss,
            intercepted=intercepted,
            wall_combo=wall_combo_ok,
        )
        state.pass_log.append(
            {
                "t": state.clock_seconds,
                "kind": kind,
                "from": carrier.player_id,
                "to": recv.player_id,
                "ok": completed,
                "p": p_ok,
                "omega": omega_used,
                "outside": outside_f,
                "ground": ground_w,
                "pred_miss": pred_miss,
                "intercept": intercepted,
            }
        )
        return action, events

    def _assign_random_possession(self, state: MatchAffectiveState, rng: np.random.Generator) -> None:
        if rng.random() < state.home.possession_share:
            team = state.home
        else:
            team = state.away
        pool = [p for p in team.players if p.on_pitch and p.role != "GK"]
        if not pool:
            pool = team.players
        p = pool[int(rng.integers(0, len(pool)))]
        state.ball.possessor_id = p.player_id
        state.ball.possession_team_id = team.team_id
        state.ball.position = p.position.copy()
