"""Combined config for Phase 1b + 2a + 2b."""

from __future__ import annotations

from dataclasses import dataclass

from src.match_engine.config import AffectiveConfig


@dataclass
class MicroMatchConfig(AffectiveConfig):
    # Research modules remain gated until matched-seed calibration passes.
    enable_hierarchical_policy: bool = False
    enable_continuous_event_clock: bool = False
    continuous_clock_provider: str = "skillcorner"
    continuous_clock_artifact: str = "data/frame_world/continuous_intervals_v93.json"
    continuous_clock_max_delay_s: float = 5.0
    # --- grid ---
    grid_nx: int = 32
    grid_ny: int = 22

    # --- spatial field §6 ---
    D0: float = 0.08
    lambda_press: float = 0.35
    alpha_width: float = 0.25
    rho_decay: float = 0.04

    # --- spatial intelligence §7 ---
    phi_a0: float = 0.0
    phi_a1: float = 1.2
    phi_a2: float = 1.8
    phi_a3: float = 0.55
    sigma_player: float = 0.06
    lane_mu1: float = 2.2
    lane_mu2: float = 1.4
    lane_samples: int = 8
    gamma_offside: float = 8.0
    delta_offside: float = 0.02
    enable_phi_gradient_move: bool = True  # E1 — phi drives kinematics (gate passed @ scale=0.015)
    w_phi_move: float = 0.50
    w_lane_move: float = 0.22
    w_supp_move: float = 0.18
    w_form_move: float = 0.28
    phi_move_scale: float = 0.015  # gradient step magnitude per tick (E1 tuned)
    kappa_phase: float = 0.14
    orient_rate: float = 2.5
    max_speed: float = 0.12  # per tick normalized coords

    # --- passing §2b ---
    pass_candidates_max: int = 6
    enable_receiver_ranker: bool = True
    receiver_ranker_blend: float = 0.16
    pass_tau_base: float = 0.42
    b_lane: float = 1.6
    b_phi: float = 1.4
    b_off: float = 2.0
    b_dist: float = 1.8
    short_dist_max: float = 0.22
    through_forward: float = 0.10
    long_dist_min: float = 0.26
    v_short: float = 0.14
    v_through: float = 0.28
    v_long: float = 0.22
    turnover_noise: float = 0.08
    xg_chi0: float = 0.045
    possession_home_prior: float = 0.52
    # Map eff_status gap → home possession prior (0.5 = even); higher = favorites hold ball more
    possession_strength_beta: float = 0.38

    # --- Phase 3 ball physics ---
    phys_gravity: float = 9.2
    phys_drag: float = 0.42
    phys_magnus: float = 0.038
    phys_knuckle: float = 0.22
    phys_dt: float = 0.012
    phys_max_steps: int = 400
    pass_curve_omega: float = 8.0

    # --- Phase 3b curved pass physics ---
    enable_pass_physics: bool = True
    pass_v0_base: float = 0.16
    pass_v0_dist: float = 0.42
    pass_v0_min: float = 0.10
    pass_v0_max: float = 0.62
    pass_elev_min: float = 0.01
    pass_elev_max: float = 0.28
    pass_elev_short: float = 0.03
    pass_elev_through: float = 0.07
    pass_elev_long: float = 0.16
    pass_elev_jitter: float = 0.015
    pass_omega_base: float = 14.0
    pass_omega_max: float = 28.0
    pass_omega_jitter: float = 1.2
    pass_omega_curve_threshold: float = 6.0
    pass_land_tol: float = 0.050
    pass_max_steps: int = 280
    pass_max_tof: float = 2.2
    # Execution-error logits (tuned vs StatsBomb WC2022; sublinear via passing_engine)
    pass_b_land_err: float = 0.30
    pass_b_lateral: float = 0.22
    pass_b_spin_exec: float = 0.003
    pass_u_curve: float = 0.42
    pass_u_curve_lane: float = 0.35
    # outside-foot (trivela): spin axis tilt
    pass_outside_tilt_max: float = 0.72  # rad, max axis tilt from vertical
    pass_outside_a0: float = -0.35
    pass_outside_a_curve: float = 2.1
    pass_outside_a_wide: float = 1.8
    pass_outside_a_omega: float = 1.4
    pass_outside_exec_penalty: float = 0.006
    pass_mix_bias_weight: float = 0.82
    # ground / carpet pass (θ_elev ≈ 0)
    pass_ground_elev_sigma: float = 0.045
    pass_ground_drag_boost: float = 0.46
    pass_ground_roll_mu: float = 0.86
    pass_ground_contact_z: float = 0.040
    # near-ground "stable roll" segment to reduce late random drift on carpet passes
    pass_roll_stabilize_speed: float = 0.070
    pass_roll_stabilize_gain: float = 0.55
    pass_ground_v0_boost: float = 0.08
    # moving intercept vs p_pred(t+Δt)
    pass_pred_vel_scale: float = 1.0
    pass_pred_dt_scale: float = 1.0
    pass_intercept_radius: float = 0.022
    pass_intercept_recv_margin: float = 0.062
    pass_b_pred_miss: float = 0.16
    pass_intercept_b0: float = -0.15
    pass_intercept_b_dist: float = 8.5
    pass_intercept_b_pace: float = 1.2
    pass_intercept_b_press: float = 0.55
    # defender pursuit toward predicted land point
    pass_pursuit_weight_base: float = 0.28
    pass_pursuit_press_gain: float = 0.18
    pass_pursuit_pace_scale: float = 0.14
    pass_pursuit_lane_gain: float = 0.22
    pass_recv_pursuit_weight: float = 0.58
    pass_pursuit_substeps: int = 4
    # wall pass / give-and-go (two-touch combo)
    enable_wall_pass: bool = True
    wall_radius: float = 0.13
    wall_return_forward: float = 0.11
    wall_u_base: float = 0.28
    wall_u_press: float = 0.55
    wall_second_z_penalty: float = 0.14
    wall_combo_xg_chip: float = 0.012

    # --- discipline: tick fouls driven by lambda_foul + live foul_impulse ---
    discipline_tick_fouls: bool = True
    discipline_foul_target_base: float = 14.0
    discipline_foul_target_ref_slope: float = 11.0
    discipline_yellow_on_foul_base: float = 0.020
    discipline_yellow_on_foul_ref: float = 0.038
    discipline_red_on_foul_base: float = 0.00045
    cross_tick_target: float = 20.0
    discipline_red_drama_gain: float = 0.0012

    # --- Phase 3 shots / action selection ---
    action_pass_base: float = 1.62
    action_shot_base: float = 0.42
    action_shot_dist_bonus: float = 2.95
    action_shot_cooldown_sec: float = 12.0
    action_shot_volume_decay: float = 0.065
    action_shot_skew_decay: float = 0.11
    shot_finish_xg_scale: float = 1.12
    shot_finish_save_scale: float = 0.68
    action_cross_base: float = 0.65
    shot_tau: float = 0.38
    action_tau: float = 0.40
    shot_max_dist: float = 0.50
    long_shot_dist: float = 0.30
    shot_v0_base: float = 0.48
    shot_v0_dist: float = 0.42
    shot_v0_max: float = 1.32
    shot_reach_dist_scale: float = 2.25
    shot_reach_bias: float = 0.58
    shot_phys_drag_scale: float = 0.88
    shot_elev_base: float = 0.12
    shot_omega_curve: float = 22.0
    shot_omega_knuckle: float = 2.5
    shot_omega_low: float = 1.0
    knuckle_intensity_scale: float = 0.85
    shot_u_phi: float = 1.5
    shot_u_skill: float = 1.2
    shot_u_dist: float = 2.4
    shot_box_sharpness: float = 18.0
    xg_geom_base: float = 0.54
    xg_dist_decay: float = 4.2
    xg_angle_penalty: float = 4.0
    xg_knuckle_boost: float = 0.18
    xg_header_base: float = 0.09
    xg_header_attempt_min: float = 0.038
    gk_b0: float = -0.08
    gk_b_reflex: float = 3.34
    gk_b_aerial: float = 1.0
    gk_b_dist: float = 1.85
    gk_b_pos: float = 2.5
    gk_curve_penalty: float = 0.04
    gk_knuckle_penalty: float = 0.12
    # Shot on-target logit (continuous; tuned vs StatsBomb SOT ~35%)
    shot_sot_z_in_goal: float = 0.92
    shot_sot_z_wide: float = -2.4
    shot_sot_z_angle: float = 4.4
    shot_sot_z_dist: float = 2.15
    shot_sot_wide_scale: float = 0.10

    # --- crosses / aerial ---
    cross_v0_base: float = 0.38
    cross_elev_base: float = 0.42
    aerial_radius: float = 0.14
    aerial_a1: float = 2.0
    aerial_a2: float = 1.6
    aerial_a3: float = 1.2
    jump_base: float = 0.75

    # --- micro → Poisson ---
    lambda0: float = 0.95
    gamma_xg: float = 0.88
    eta_status_blend: float = 0.62

    # --- feature flags ---
    disable_affective_dynamics: bool = False
    enable_passing: bool = True
    enable_spatial: bool = True
    enable_phase3: bool = True
    # True: no pre-seeded goals on calendar; official score = goals from shot physics (state.score).
    use_micro_goals: bool = False
    # Scheduled SHOT_ON_TARGET: place carrier in box + boost conversion
    scheduled_shot_box_x_home: float = 0.86
    scheduled_shot_box_x_away: float = 0.14
    scheduled_shot_box_y_spread: float = 0.28
    scheduled_on_target_save_mult: float = 0.78
    scheduled_on_target_goal_bonus: float = 0.12
    # Optional calendar shot density (MATCH_SCHEDULED_SHOTS=calibrated); not first-principles physics.
    # xg_mult / cap only used when calendar is enabled — heuristic ~8–14 shots/team from macro xG prior.
    xg_supplement_threshold: float = 99.0
    xg_supplement_cap: int = 1
    xg_supplement_shrink: float = 0.75
    micro_xg_match_cap: float = 2.2
    pass_land_miss_cap: float = 0.16
    pass_exec_miss_gamma: float = 0.55
    schedule_shots_xg_mult: float = 1.35
    schedule_shots_poisson: float = 0.6
    max_scheduled_shots_per_team: int = 16

    @classmethod
    def fast_demo(cls) -> "MicroMatchConfig":
        c = cls()
        c.dt_default = 30.0
        c.grid_nx = 16
        c.grid_ny = 11
        c.lane_samples = 5
        return c
