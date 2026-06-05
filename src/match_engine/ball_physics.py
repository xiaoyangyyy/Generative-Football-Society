"""
Ball trajectory ODE — drag, Magnus (banana/curve), knuckle flutter.
Coordinates: x,y normalized pitch [0,1], z height in [0, ~2.5] metres scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.match_engine.math_utils import sigmoid
from src.match_engine.micro_config import MicroMatchConfig


@dataclass
class BallActionParams:
    v0: float
    elev: float  # radians
    azim: float  # radians in pitch plane
    omega: float  # rad/s spin magnitude
    spin_axis: np.ndarray  # unit 3d
    knuckle_intensity: float = 0.0
    outside_foot: float = 0.0  # 0=instep … 1=full outside-foot wrap
    ground_weight: float = 0.0  # continuous carpet-pass weight from θ_elev


@dataclass
class TrajectoryResult:
    landed: np.ndarray  # x,y
    peak_height: float
    time_of_flight: float
    crossed_goal_line: bool
    goal_y: Optional[float]  # y at goal line if crossed
    in_goal_mouth: bool
    curve_amount: float


@dataclass
class PassTrajectoryResult:
    """Ground / floated pass — lands near target with lateral bend from Magnus."""

    landed: np.ndarray
    peak_height: float
    time_of_flight: float
    target_miss: float  # ||landed - target||
    lateral_dev: float  # perpendicular miss vs intended chord
    curve_amount: float
    omega_used: float
    outside_foot: float = 0.0
    ground_weight: float = 0.0


def _goal_line_x(attacking_high_x: bool) -> float:
    return 0.995 if attacking_high_x else 0.005


def _goal_mouth_y() -> Tuple[float, float]:
    return 0.38, 0.62


def integrate_trajectory(
    start_xy: np.ndarray,
    params: BallActionParams,
    cfg: MicroMatchConfig,
    *,
    attacking_high_x: bool = True,
    pitch_length_m: float = 105.0,
    drag_scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> TrajectoryResult:
    """
    Semi-implicit Euler integration in 3D.
    Horizontal speeds scaled to normalized coords per second.
    """
    rng = rng or np.random.default_rng()
    g = cfg.phys_gravity
    c_d = cfg.phys_drag * max(0.35, float(drag_scale))
    c_m = cfg.phys_magnus
    c_k = cfg.phys_knuckle

    v0 = float(max(0.01, params.v0))
    elev = float(params.elev)
    azim = float(params.azim)
    # horizontal plane velocity (normalized / sec)
    vx = v0 * np.cos(elev) * np.cos(azim)
    vy = v0 * np.cos(elev) * np.sin(azim)
    vz = v0 * np.sin(elev)

    pos = np.array([float(start_xy[0]), float(start_xy[1]), 0.05], dtype=float)
    vel = np.array([vx, vy, vz], dtype=float)
    omega_vec = float(params.omega) * np.asarray(params.spin_axis, dtype=float)
    if np.linalg.norm(omega_vec) < 1e-9:
        omega_vec = np.zeros(3)

    eta = 0.0
    dt = cfg.phys_dt
    t = 0.0
    peak_z = pos[2]
    goal_x = _goal_line_x(attacking_high_x)
    crossed = False
    goal_y_at_cross: Optional[float] = None
    in_goal = False
    y_lo, y_hi = _goal_mouth_y()

    max_steps = cfg.phys_max_steps
    for _ in range(max_steps):
        speed = float(np.linalg.norm(vel)) + 1e-9
        drag = -c_d * speed * vel
        magnus = c_m * np.cross(omega_vec, vel) if params.omega > 0.1 else np.zeros(3)
        knuckle = -c_k * eta * vel if params.knuckle_intensity > 0.05 else np.zeros(3)
        acc = np.array([drag[0], drag[1], -g + drag[2] + magnus[2] + knuckle[2]])
        acc[0] += magnus[0] + knuckle[0]
        acc[1] += magnus[1] + knuckle[1]

        vel = vel + dt * acc
        pos = pos + dt * vel
        peak_z = max(peak_z, float(pos[2]))
        t += dt

        # OU knuckle noise
        if params.knuckle_intensity > 0.05:
            eta += dt * (-0.6 * eta + 0.4 * rng.normal())

        # ground bounce damping (soft)
        if pos[2] < 0.0:
            pos[2] = 0.0
            vel[2] = abs(vel[2]) * 0.25
            vel[0] *= 0.88
            vel[1] *= 0.88

        # goal line crossing
        if not crossed:
            if attacking_high_x and pos[0] >= goal_x and vel[0] > 0:
                crossed = True
                goal_y_at_cross = float(np.clip(pos[1], 0, 1))
                in_goal = y_lo <= goal_y_at_cross <= y_hi and pos[2] < 2.6
            elif (not attacking_high_x) and pos[0] <= goal_x and vel[0] < 0:
                crossed = True
                goal_y_at_cross = float(np.clip(pos[1], 0, 1))
                in_goal = y_lo <= goal_y_at_cross <= y_hi and pos[2] < 2.6

        if pos[2] < -0.05 and t > 0.15:
            break
        if t > 4.0:
            break
        if crossed and t > 0.05:
            break

    curve_amt = float(params.omega / (v0 + 1e-6))
    return TrajectoryResult(
        landed=pos[:2].copy(),
        peak_height=peak_z,
        time_of_flight=t,
        crossed_goal_line=crossed,
        goal_y=goal_y_at_cross,
        in_goal_mouth=in_goal,
        curve_amount=curve_amt,
    )


def sample_shot_params(
    shot_kind: str,
    dist_to_goal: float,
    abilities,
    mod_shot_bias: float,
    rng: np.random.Generator,
    cfg: MicroMatchConfig,
) -> BallActionParams:
    """shot_kind: driven | curved | knuckle | power | header"""
    tech = float(getattr(abilities, "tech", 0.55))
    shot = float(getattr(abilities, "shot", 0.55))
    curve = float(getattr(abilities, "curve", 0.55))
    power = float(getattr(abilities, "power", 0.55))
    knuckle = float(getattr(abilities, "knuckle", 0.5))

    base_v = cfg.shot_v0_base + cfg.shot_v0_dist * (1.0 - min(1.0, dist_to_goal))
    elev = cfg.shot_elev_base + rng.normal(0, 0.04)

    if shot_kind == "curved":
        omega = cfg.shot_omega_curve * (0.5 + curve) * (0.8 + 0.4 * rng.random())
        spin = np.array([0.0, 0.0, 1.0 if rng.random() > 0.5 else -1.0])
        elev += 0.06
        v0 = base_v * (0.85 + 0.15 * tech)
        kn_i = 0.0
    elif shot_kind == "knuckle":
        omega = cfg.shot_omega_knuckle * rng.random()
        spin = np.array([1.0, 0.0, 0.0])
        v0 = base_v * (0.9 + 0.2 * power)
        kn_i = knuckle * cfg.knuckle_intensity_scale
        elev += 0.12
    elif shot_kind == "power":
        omega = cfg.shot_omega_low
        spin = np.array([0.0, 1.0, 0.0])
        v0 = base_v * (1.0 + 0.25 * power)
        kn_i = 0.0
        elev += 0.03
    elif shot_kind == "header":
        omega = 0.0
        spin = np.array([0.0, 0.0, 1.0])
        v0 = base_v * 0.55
        elev = 0.25 + 0.1 * rng.random()
        kn_i = 0.0
    else:  # driven
        omega = cfg.shot_omega_low
        spin = np.array([0.0, 0.0, 1.0])
        v0 = base_v * (0.9 + 0.15 * shot)
        kn_i = 0.0

    v0 *= 1.0 + 0.08 * mod_shot_bias
    reach_v0 = cfg.shot_reach_bias + cfg.shot_reach_dist_scale * float(dist_to_goal)
    v0 = max(v0, reach_v0)
    v0_max = float(getattr(cfg, "shot_v0_max", 1.22))
    # azimuth set by caller toward goal
    return BallActionParams(
        v0=float(np.clip(v0, 0.08, v0_max)),
        elev=float(np.clip(elev, 0.02, 0.55)),
        azim=0.0,
        omega=float(omega),
        spin_axis=spin / (np.linalg.norm(spin) + 1e-9),
        knuckle_intensity=float(kn_i),
    )


def azimuth_to_goal(from_xy: np.ndarray, attacking_high_x: bool) -> float:
    gx = _goal_line_x(attacking_high_x)
    gy = 0.5
    d = np.array([gx - from_xy[0], gy - from_xy[1]])
    return float(np.arctan2(d[1], d[0] + 1e-9))


def azimuth_to_target(from_xy: np.ndarray, target_xy: np.ndarray) -> float:
    d = target_xy - from_xy
    return float(np.arctan2(d[1], d[0] + 1e-9))


def _lateral_deviation(start_xy: np.ndarray, target_xy: np.ndarray, land_xy: np.ndarray) -> float:
    chord = target_xy - start_xy
    ln = float(np.linalg.norm(chord)) + 1e-9
    unit = chord / ln
    along = float(np.dot(land_xy - start_xy, unit))
    proj = start_xy + along * unit
    return float(np.linalg.norm(land_xy - proj))


def desired_pass_omega(
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    curve_skill: float,
    lane: float,
    press: float,
    cfg: MicroMatchConfig,
) -> float:
    """
    Continuous spin demand — wide angle + pressure → more bend (banana pass).
    No discrete pass-type label.
    """
    chord = target_xy - start_xy
    dist = float(np.linalg.norm(chord))
    if dist < 1e-4:
        return 0.0
    # angle between pass chord and central channel (y=0.5)
    mid_y = 0.5 * (start_xy[1] + target_xy[1])
    wide = float(abs(mid_y - 0.5))
    angle_need = wide / (dist + 0.08)
    press_boost = max(0.0, press - 0.35)
    lane_need = max(0.0, 0.55 - lane)
    raw = cfg.pass_omega_base * (0.35 + 0.65 * curve_skill)
    raw *= float(sigmoid(2.2 * angle_need + 1.4 * lane_need + 1.1 * press_boost))
    return float(np.clip(raw, 0.0, cfg.pass_omega_max))


def ground_pass_weight(elev: float, cfg: MicroMatchConfig) -> float:
    """Continuous carpet-pass weight: 1 when θ_elev≈0, →0 for floated balls."""
    sigma = max(1e-4, cfg.pass_ground_elev_sigma)
    return float(np.exp(-float(elev) / sigma))


def outside_foot_factor(
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    curve_skill: float,
    omega_target: float,
    cfg: MicroMatchConfig,
) -> float:
    """
    Continuous outside-foot (trivela) intensity — wide geometry + curve demand.
    No discrete foot label.
    """
    chord = target_xy - start_xy
    dist = float(np.linalg.norm(chord))
    if dist < 1e-4:
        return 0.0
    mid_y = 0.5 * (start_xy[1] + target_xy[1])
    wide = float(abs(mid_y - 0.5)) / (dist + 0.06)
    omega_need = float(omega_target / max(1e-6, cfg.pass_omega_max))
    z = (
        cfg.pass_outside_a0
        + cfg.pass_outside_a_curve * curve_skill
        + cfg.pass_outside_a_wide * wide
        + cfg.pass_outside_a_omega * omega_need
    )
    return float(np.clip(sigmoid(z), 0.0, 1.0))


def build_pass_spin_axis(
    azim: float,
    spin_sign: float,
    outside_foot: float,
    cfg: MicroMatchConfig,
) -> np.ndarray:
    """
    Spin axis: mostly vertical (instep) → tilted toward horizontal perpendicular
    to pass direction when outside_foot → 1 (trivela).
    """
    z_axis = np.array([0.0, 0.0, float(spin_sign)], dtype=float)
    perp = np.array([-np.sin(azim), np.cos(azim), 0.0], dtype=float)
    tilt = float(outside_foot) * cfg.pass_outside_tilt_max
    axis = np.cos(tilt) * z_axis + np.sin(tilt) * perp
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return z_axis
    return axis / n


def sample_pass_delivery_params(
    dist: float,
    elev_hint: float,
    omega_target: float,
    curve_skill: float,
    rng: np.random.Generator,
    cfg: MicroMatchConfig,
    *,
    start_xy: Optional[np.ndarray] = None,
    target_xy: Optional[np.ndarray] = None,
    azim: float = 0.0,
) -> BallActionParams:
    """Sample (v0, elev, ω, spin axis) — short/through/long via elev_hint & dist only."""
    elev = float(np.clip(elev_hint + rng.normal(0, cfg.pass_elev_jitter), cfg.pass_elev_min, cfg.pass_elev_max))
    g_w = ground_pass_weight(elev, cfg)
    v0 = cfg.pass_v0_base + cfg.pass_v0_dist * min(1.0, dist / 0.55)
    v0 *= 1.0 + cfg.pass_ground_v0_boost * g_w
    omega = float(
        np.clip(
            omega_target * (0.85 + 0.3 * curve_skill) + rng.normal(0, cfg.pass_omega_jitter),
            0.0,
            cfg.pass_omega_max,
        )
    )
    outside = 0.0
    if start_xy is not None and target_xy is not None:
        outside = outside_foot_factor(start_xy, target_xy, curve_skill, omega_target, cfg)
    spin_sign = 1.0 if rng.random() > 0.5 else -1.0
    spin = build_pass_spin_axis(azim, spin_sign, outside, cfg)
    return BallActionParams(
        v0=float(np.clip(v0, cfg.pass_v0_min, cfg.pass_v0_max)),
        elev=elev,
        azim=azim,
        omega=omega,
        spin_axis=spin,
        knuckle_intensity=0.0,
        outside_foot=outside,
        ground_weight=g_w,
    )


def integrate_pass_trajectory(
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    params: BallActionParams,
    cfg: MicroMatchConfig,
    *,
    curve_skill: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> PassTrajectoryResult:
    """
    Same drag + Magnus ODE as shots; terminate near target or max time.
    curve_skill scales effective Magnus (θ^curve).
    """
    rng = rng or np.random.default_rng()
    g = cfg.phys_gravity
    g_w = float(params.ground_weight if params.ground_weight > 0 else ground_pass_weight(params.elev, cfg))
    c_d_base = cfg.phys_drag
    c_m = cfg.phys_magnus * (0.55 + 0.9 * float(curve_skill))
    # tilted spin axis → slightly stronger horizontal Magnus component
    c_m *= 1.0 + 0.25 * float(params.outside_foot)
    dt = cfg.phys_dt
    tol = cfg.pass_land_tol

    v0 = float(max(0.01, params.v0))
    elev = float(params.elev)
    azim = float(params.azim)
    vx = v0 * np.cos(elev) * np.cos(azim)
    vy = v0 * np.cos(elev) * np.sin(azim)
    vz = v0 * np.sin(elev)

    pos = np.array([float(start_xy[0]), float(start_xy[1]), 0.04], dtype=float)
    vel = np.array([vx, vy, vz], dtype=float)
    omega_vec = float(params.omega) * np.asarray(params.spin_axis, dtype=float)
    if np.linalg.norm(omega_vec) < 1e-9:
        omega_vec = np.zeros(3)

    t = 0.0
    peak_z = pos[2]
    max_steps = min(cfg.phys_max_steps, cfg.pass_max_steps)
    max_tof = cfg.pass_max_tof

    for _ in range(max_steps):
        speed = float(np.linalg.norm(vel)) + 1e-9
        c_d = c_d_base * (1.0 + cfg.pass_ground_drag_boost * g_w)
        drag = -c_d * speed * vel
        magnus = c_m * np.cross(omega_vec, vel) if params.omega > 0.08 else np.zeros(3)
        acc = np.array([drag[0], drag[1], -g + drag[2] + magnus[2]], dtype=float)
        acc[0] += magnus[0]
        acc[1] += magnus[1]
        # carpet pass: extra grass rolling friction when ball hugs ground
        if pos[2] < cfg.pass_ground_contact_z and g_w > 0.15:
            roll = cfg.pass_ground_roll_mu * g_w
            acc[0] -= roll * vel[0]
            acc[1] -= roll * vel[1]
            # Stable rolling: when very low and slow, blend velocity toward target direction.
            vxy = vel[:2]
            vxy_n = float(np.linalg.norm(vxy))
            if vxy_n < cfg.pass_roll_stabilize_speed:
                to_tgt = target_xy - pos[:2]
                to_tgt_n = float(np.linalg.norm(to_tgt))
                if to_tgt_n > 1e-6:
                    align = to_tgt / to_tgt_n
                    gain = cfg.pass_roll_stabilize_gain * g_w
                    vel[:2] = (1.0 - gain) * vel[:2] + gain * (vxy_n * align)

        vel = vel + dt * acc
        pos = pos + dt * vel
        peak_z = max(peak_z, float(pos[2]))
        t += dt

        if pos[2] < 0.0:
            pos[2] = 0.0
            damp_v = 0.12 if g_w > 0.5 else 0.2
            damp_h = 0.82 if g_w > 0.5 else 0.9
            vel[2] = abs(vel[2]) * damp_v
            vel[0] *= damp_h
            vel[1] *= damp_h

        miss = float(np.linalg.norm(pos[:2] - target_xy))
        if miss < tol:
            break
        if t > max_tof:
            break
        if pos[2] < 0.02 and speed < 0.04 and t > 0.08:
            break

    land = np.clip(pos[:2], 0.02, 0.98)
    miss = float(np.linalg.norm(land - target_xy))
    lat = _lateral_deviation(start_xy, target_xy, land)
    curve_amt = float(params.omega / (v0 + 1e-6))
    return PassTrajectoryResult(
        landed=land,
        peak_height=peak_z,
        time_of_flight=t,
        target_miss=miss,
        lateral_dev=lat,
        curve_amount=curve_amt,
        omega_used=float(params.omega),
        outside_foot=float(params.outside_foot),
        ground_weight=g_w,
    )
