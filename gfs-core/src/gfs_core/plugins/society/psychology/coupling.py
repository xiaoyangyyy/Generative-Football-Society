"""Continuous affective coupling for players, coaches, referees and crowds."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from gfs_core.contracts.affective import AffectiveConfig
from gfs_core.contracts.events import MicroEvent
from gfs_core.contracts.micro_math import (
    clip01,
    finite_float,
    sigmoid,
    softmax,
    tanh_clip,
)
from gfs_core.contracts.micro_state import (
    EMOTION_KEYS,
    MatchAffectiveState,
    PlayerModulators,
    TeamAffectiveState,
)

# re-export for squad / state convenience
__all__ = ["AffectiveSpatialCoupling", "softmax", "sigmoid", "tanh_clip"]


class AffectiveSpatialCoupling:
    def __init__(self, config: Optional[AffectiveConfig] = None, event_handler=None):
        self.cfg = config or AffectiveConfig()
        self.event_handler = event_handler

    def step(
        self,
        state: MatchAffectiveState,
        dt: float,
        events: Optional[List[MicroEvent]] = None,
        score_diff_home: float = 0.0,
        xg_swing_home: float = 0.0,
        coordination_home: float = 0.6,
        coordination_away: float = 0.6,
        conflict_home: float = 0.12,
        conflict_away: float = 0.12,
        press_home: float = 0.0,
        press_away: float = 0.0,
    ) -> Tuple[Dict[str, List[PlayerModulators]], Dict[str, float]]:
        """
        One integration step.
        Returns per-team player modulators and match-level scalars (strictness, lambda_foul, psi).
        """
        cfg = self.cfg
        if getattr(cfg, "disable_affective_dynamics", False):
            mod_home = [PlayerModulators(player_id=p.player_id) for p in state.home.players]
            mod_away = [PlayerModulators(player_id=p.player_id) for p in state.away.players]
            strict = float(state.referee.strictness_effective(0.0, cfg))
            lam_foul = float(cfg.lambda_foul_base)
            return {
                "home": mod_home,
                "away": mod_away,
            }, {
                "psi": 0.0,
                "strictness_effective": strict,
                "lambda_foul": lam_foul,
                "ref_calm": state.referee.calm(),
            }
        dt = float(max(1e-6, dt))
        events = events or []
        g_psi = 0.0

        for ev in events:
            if self.event_handler is None:
                raise RuntimeError("an event handler is required for affective events")
            g_psi += self.event_handler(state, ev, cfg)

        remaining = dt
        maximum_step = max(1e-6, float(getattr(cfg, "affective_max_dt", 1.0)))
        while remaining > 1e-9:
            step_dt = min(remaining, maximum_step)
            state.clock_seconds += step_dt
            self._integrate_crowd(state, g_psi, step_dt)
            self._integrate_coach(
                state.home,
                step_dt,
                score_diff_home,
                xg_swing_home,
                coordination_home,
                conflict_home,
                state.crowd,
                is_home=True,
            )
            self._integrate_coach(
                state.away,
                step_dt,
                -score_diff_home,
                -xg_swing_home,
                coordination_away,
                conflict_away,
                state.crowd,
                is_home=False,
            )
            self._integrate_referee(state, step_dt)
            self._integrate_players(state.home, step_dt, state.crowd, press_home)
            self._integrate_players(state.away, step_dt, state.crowd, press_away)
            remaining -= step_dt

        mod_home = [self.compute_modulators(p, state) for p in state.home.players]
        mod_away = [self.compute_modulators(p, state) for p in state.away.players]

        strict = state.referee.strictness_effective(state.crowd.psi, cfg)
        lam_foul = cfg.lambda_foul_base * float(
            np.exp(
                cfg.delta_strictness_foul * (strict - 0.5)
                + cfg.delta_press_foul * 0.5 * (press_home + press_away)
            )
        )

        scalars = {
            "psi": state.crowd.psi,
            "strictness_effective": strict,
            "lambda_foul": lam_foul,
            "ref_calm": state.referee.calm(),
        }
        return {"home": mod_home, "away": mod_away}, scalars

    def _integrate_crowd(self, state: MatchAffectiveState, g_pulse: float, dt: float) -> None:
        cfg = self.cfg
        c = state.crowd
        home_adv = 0.0 if c.is_neutral_venue else cfg.home_advantage_base
        score_diff = state.home.score - state.away.score
        # home crowd energised when home leads; away surge when away leads
        H = home_adv * tanh_clip(score_diff * 0.35)
        if c.home_team_id and c.home_team_id == state.away.team_id:
            H = -H
        c.psi = _integrate_linear(
            c.psi,
            cfg.psi_goal_pulse * g_pulse + 0.15 * H,
            cfg.psi_decay,
            dt,
        )
        c.psi = float(np.clip(c.psi, -2.5, 2.5))
        c.event_intensity_integral += abs(g_pulse) * dt

    def _integrate_coach(
        self,
        team: TeamAffectiveState,
        dt: float,
        score_diff: float,
        xg_swing: float,
        coordination: float,
        conflict_heat: float,
        crowd,
        is_home: bool,
    ) -> None:
        cfg = self.cfg
        coach = team.coach
        stress_drive = cfg.coach_a_score * abs(score_diff) + cfg.coach_a_xg_swing * max(0.0, -xg_swing)
        trust_drive = cfg.coach_a_coordination * coordination - cfg.coach_a_conflict * conflict_heat
        crowd_term = 0.0
        if is_home and not crowd.is_neutral_venue:
            crowd_term = cfg.coach_a_crowd_stress * tanh_clip(-crowd.psi) * float(max(0, -score_diff))

        coach.z_stress = _integrate_linear(
            coach.z_stress, stress_drive + crowd_term, cfg.coach_stress_decay, dt,
        )
        coach.z_trust = _integrate_linear(
            coach.z_trust, trust_drive, cfg.coach_trust_decay, dt,
        )
        coach.z_rage = _integrate_linear(
            coach.z_rage, cfg.coach_a_conflict * conflict_heat, cfg.coach_rage_decay, dt,
        )
        coach.z_stress = float(np.clip(coach.z_stress, -6.0, 6.0))
        coach.z_trust = float(np.clip(coach.z_trust, -6.0, 6.0))
        coach.z_rage = float(np.clip(coach.z_rage, -6.0, 6.0))

        stress = coach.stress()
        trust = coach.trust()
        rage = coach.rage()

        cur = dict(coach.tactical_current)
        base = coach.tactical_base
        delta_press = cfg.coach_k_press_stress * stress + cfg.coach_k_risk_trust * trust * 0.5
        delta_risk = cfg.coach_k_risk_stress * stress + cfg.coach_k_risk_trust * trust
        delta_sub = cfg.coach_k_sub_stress * sigmoid(stress - trust)

        cur["pressing_intensity"] = clip01(
            cur.get("pressing_intensity", 0.5) + dt * delta_press
        )
        cur["risk_budget"] = clip01(cur.get("risk_budget", 0.5) + dt * delta_risk)
        cur["rotation_aggressiveness"] = clip01(
            cur.get("rotation_aggressiveness", 0.5) + dt * delta_sub
        )
        # line drifts slightly with rage (emotional high line or drop-back calibrated by sign)
        cur["line_height"] = clip01(
            cur.get("line_height", 0.5) + dt * 0.02 * tanh_clip(rage - 0.5)
        )

        clip = cfg.coach_tactical_step_clip
        for k in cur:
            b = base.get(k, 0.5)
            cur[k] = float(np.clip(cur[k], b - clip * 8, b + clip * 8))
        coach.tactical_current = cur
        coach.volatility_accum += dt * abs(delta_press) + dt * rage * 0.02

    def _integrate_referee(self, state: MatchAffectiveState, dt: float) -> None:
        cfg = self.cfg
        ref = state.referee
        controversy = ref.controversy_integral
        ref.z_calm = _integrate_linear(
            ref.z_calm,
            cfg.ref_a_controversy * controversy + cfg.ref_a_crowd * abs(state.crowd.psi),
            cfg.ref_calm_decay,
            dt,
        )
        ref.z_defensive = _integrate_linear(
            ref.z_defensive, 0.05 * ref.card_load, 0.1, dt,
        )

    def _integrate_players(
        self,
        team: TeamAffectiveState,
        dt: float,
        crowd,
        press_local: float,
    ) -> None:
        cfg = self.cfg
        n = team.n_players
        if n == 0:
            return
        Z = np.stack([p.z_emo for p in team.players], axis=0)
        peer = team.peer_matrix
        if peer is None or peer.shape != (n, n):
            peer = np.ones((n, n), dtype=float) * 0.02
            np.fill_diagonal(peer, 0.0)

        mean_z = np.mean(Z, axis=0, keepdims=True)
        weights = np.sum(peer, axis=1, keepdims=True)
        weighted_mean = (peer @ Z) / np.maximum(weights, 1e-12)
        weighted_mean = np.where(weights > 1e-12, weighted_mean, mean_z)
        coupling = cfg.peer_coupling_scale * (weighted_mean - mean_z)

        crowd_vec = np.zeros(4, dtype=float)
        sign = 1.0 if team.team_id == crowd.home_team_id else -1.0
        if not crowd.is_neutral_venue:
            crowd_vec[0] = cfg.k_crowd_z_emo * tanh_clip(sign * crowd.psi) * 0.3
            crowd_vec[2] = -cfg.k_crowd_z_emo * tanh_clip(-sign * crowd.psi) * 0.25

        for i, p in enumerate(team.players):
            if not p.on_pitch:
                continue
            p.z_emo = np.clip(_integrate_linear(
                p.z_emo,
                coupling[i] + crowd_vec * p.fan_affinity,
                cfg.z_emo_decay,
                dt,
            ), -12.0, 12.0)

            e = softmax(p.z_emo, tau=cfg.emotion_tau)
            pride, anger, fear, det = float(e[0]), float(e[1]), float(e[2]), float(e[3])

            p.morale_logit = _integrate_linear(
                p.morale_logit,
                cfg.k_crowd_morale * tanh_clip(sign * crowd.psi) * p.fan_affinity
                + 0.04 * (pride - fear)
                + 0.02 * det,
                cfg.morale_decay,
                dt,
            )

            p.cognitive_load = float(clip01(_integrate_linear(
                p.cognitive_load,
                0.08 * press_local + 0.12 * anger + 0.10 * fear,
                0.06,
                dt,
            )))

    def compute_modulators(self, player, state: MatchAffectiveState) -> PlayerModulators:
        cfg = self.cfg
        e = softmax(player.z_emo, tau=cfg.emotion_tau)
        pride, anger, fear, det = float(e[0]), float(e[1]), float(e[2]), float(e[3])
        q = float(player.cognitive_load)
        s_stamina = float(sigmoid(player.stamina_logit))
        s_spatial = float(sigmoid(player.spatial_cognition_logit))

        tau_dec = cfg.tau_dec_base * (1.0 + cfg.d_q * q + cfg.d_fear * fear + cfg.d_anger * anger)
        tau_dec = max(0.08, finite_float(tau_dec, cfg.tau_dec_base))
        vision = cfg.vision_base * (1.0 - cfg.v_fear * fear - cfg.v_anger * anger)
        vision = float(np.clip(vision, 0.35, 1.15))

        move_alpha = (
            cfg.move_alpha_base
            * s_stamina
            * s_spatial
            * (1.0 + cfg.b_determination * det - cfg.b_fear_move * fear)
        )
        move_alpha = float(np.clip(move_alpha, 0.15, 1.6))

        shot_bias = cfg.rho_shot_anger * anger - cfg.rho_shot_fear * fear + 0.08 * pride
        foul_impulse = cfg.rho_foul_anger * anger + cfg.rho_foul_fear * fear * 0.5
        foul_impulse += 0.1 * state.referee.card_load * anger

        return PlayerModulators(
            player_id=player.player_id,
            tau_dec=float(tau_dec),
            vision_scale=vision,
            move_alpha=move_alpha,
            shot_utility_bias=float(shot_bias),
            foul_impulse=float(foul_impulse),
            lane_sampling_scale=vision,
        )

    def team_emotion_mean(self, team: TeamAffectiveState) -> Dict[str, float]:
        if not team.players:
            return {k: 0.0 for k in EMOTION_KEYS}
        acc = np.zeros(4, dtype=float)
        w = 0.0
        for p in team.players:
            if not p.on_pitch:
                continue
            acc += softmax(p.z_emo, tau=self.cfg.emotion_tau)
            w += 1.0
        if w < 1:
            return {k: 0.0 for k in EMOTION_KEYS}
        acc /= w
        return {k: float(v) for k, v in zip(EMOTION_KEYS, acc)}


def _integrate_linear(value, drive: float, decay: float, dt: float):
    if decay <= 0:
        return value + drive * dt
    factor = math.exp(-decay * dt)
    return value * factor + drive * (-math.expm1(-decay * dt)) / decay
