"""Continuous parameters for Phase 1b affective coupling (no hard thresholds)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AffectiveConfig:
    """Tunable kernel — all gates are smooth (sigmoid/tanh/exp), not if-threshold rules."""

    # --- integration ---
    dt_default: float = 5.5  # seconds per tick (~982 ticks / 90 min)
    match_seconds: float = 90.0 * 60.0
    emotion_tau: float = 0.85  # softmax temperature on z_emo
    peer_coupling_scale: float = 0.15
    z_emo_decay: float = 0.08
    morale_decay: float = 0.05

    # --- player emotion impulses (applied to z_emo logits) ---
    impulse_goal_pride: float = 1.35
    impulse_goal_fear_opp: float = -0.95
    impulse_assist_pride: float = 0.55
    impulse_key_pass_pride: float = 0.28
    impulse_tackle_won_pride: float = 0.18
    impulse_dispossessed_anger: float = 0.75
    impulse_dispossessed_fear: float = 0.45
    impulse_shot_miss_determination: float = 0.35
    impulse_shot_miss_fear: float = 0.22
    impulse_concede_fear: float = 0.85
    impulse_concede_anger: float = 0.55
    impulse_foul_committed_anger: float = 0.40
    impulse_foul_suffered_anger: float = 0.65
    impulse_yellow_anger: float = 0.90
    impulse_yellow_fear: float = 0.35
    impulse_red_anger: float = 1.20
    impulse_red_fear: float = 0.80
    impulse_save_pride: float = 0.70  # GK
    impulse_var_controversy_anger: float = 0.55
    impulse_var_norm: float = 0.45

    # --- crowd ↔ player morale / emotion ---
    k_crowd_morale: float = 0.018
    k_crowd_z_emo: float = 0.12
    k_crowd_coach_stress: float = 0.022
    k_crowd_ref_calm_down: float = 0.015
    psi_decay: float = 0.12
    psi_goal_pulse: float = 0.55
    psi_red_pulse: float = 0.42
    psi_shot_pulse: float = 0.08
    psi_save_pulse: float = 0.15
    psi_var_pulse: float = 0.38
    psi_neutral: float = 0.0
    home_advantage_base: float = 0.12

    # --- spatial modulators (§3.2) ---
    tau_dec_base: float = 0.45
    d_q: float = 0.35
    d_fear: float = 0.55
    d_anger: float = 0.40
    vision_base: float = 1.0
    v_fear: float = 0.25
    v_anger: float = 0.18
    move_alpha_base: float = 1.0
    b_determination: float = 0.22
    b_fear_move: float = 0.28
    rho_shot_anger: float = 0.35
    rho_shot_fear: float = 0.30
    rho_foul_anger: float = 0.25
    rho_foul_fear: float = 0.12

    # --- coach continuous state ---
    coach_stress_decay: float = 0.10
    coach_trust_decay: float = 0.08
    coach_rage_decay: float = 0.14
    coach_a_score: float = 0.14
    coach_a_xg_swing: float = 0.20
    coach_a_coordination: float = 0.18
    coach_a_conflict: float = 0.22
    coach_a_crowd_stress: float = 0.16
    coach_k_press_stress: float = 0.06
    coach_k_risk_stress: float = -0.04
    coach_k_risk_trust: float = 0.05
    coach_k_sub_stress: float = 0.08
    coach_tactical_step_clip: float = 0.025

    # --- referee ---
    ref_calm_decay: float = 0.11
    ref_a_controversy: float = 0.25
    ref_a_crowd: float = 0.12
    kappa_rpsi: float = 0.08
    kappa_rc_calm: float = 0.10
    kappa_grievance_strict: float = 0.15
    lambda_foul_base: float = 0.06
    delta_strictness_foul: float = 1.40
    delta_press_foul: float = 0.35

    # --- meso ---
    meso_window_seconds: float = 15.0 * 60.0

    @classmethod
    def fast_demo(cls) -> "AffectiveConfig":
        """Coarser time step for quick smoke runs."""
        c = cls()
        c.dt_default = 30.0
        return c
