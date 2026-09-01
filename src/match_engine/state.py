"""State containers for Phase 1b affective match simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

EMOTION_KEYS = ("pride", "anger", "fear", "determination")
TACTICAL_KEYS = ("pressing_intensity", "risk_budget", "line_height", "rotation_aggressiveness")


@dataclass
class PlayerModulators:
    """Spatial / decision coupling outputs for one player (§3.2)."""

    player_id: str
    tau_dec: float = 0.45
    vision_scale: float = 1.0
    move_alpha: float = 1.0
    shot_utility_bias: float = 0.0
    foul_impulse: float = 0.0
    lane_sampling_scale: float = 1.0  # scales effective vision for pass lines


@dataclass
class PlayerAbilities:
    tech: float = 0.55
    pass_skill: float = 0.55
    vision: float = 0.55
    spatial: float = 0.55
    pace: float = 0.55
    press: float = 0.50
    curve: float = 0.50
    shot: float = 0.55
    power: float = 0.55
    knuckle: float = 0.45
    aerial: float = 0.50
    heading: float = 0.50
    gk_reflex: float = 0.58
    gk_aerial: float = 0.55


@dataclass
class PlayerAffectiveState:
    player_id: str
    name: str
    role: str
    team_id: str
    is_icon: bool = False
    fan_affinity: float = 0.5
    mental: float = 0.5
    z_emo: np.ndarray = field(default_factory=lambda: np.zeros(4))
    morale_logit: float = 0.0
    cognitive_load: float = 0.0
    stamina_logit: float = 0.0
    spatial_cognition_logit: float = 0.0
    on_pitch: bool = True
    norm_violation_accum: float = 0.0
    # Phase 2a kinematics
    position: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5], dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    orientation: float = 0.0
    abilities: PlayerAbilities = field(default_factory=PlayerAbilities)
    cb_wide: float = 0.0
    channel_affinities: Dict[str, float] = field(default_factory=dict)
    primary_channel: str = ""
    availability: float = 1.0
    squad_role: str = "starter"

    def emotion_profile(self) -> Dict[str, float]:
        from src.match_engine.math_utils import softmax

        p = softmax(self.z_emo, tau=1.0)
        return {k: float(v) for k, v in zip(EMOTION_KEYS, p)}


@dataclass
class CoachAffectiveState:
    team_id: str
    coach_name: str = ""
    z_stress: float = 0.0
    z_trust: float = 0.0
    z_rage: float = 0.0
    tactical_base: Dict[str, float] = field(default_factory=dict)
    tactical_current: Dict[str, float] = field(default_factory=dict)
    volatility_accum: float = 0.0

    def stress(self) -> float:
        from src.match_engine.math_utils import sigmoid

        return float(sigmoid(np.clip(self.z_stress, -12.0, 12.0)))

    def trust(self) -> float:
        from src.match_engine.math_utils import sigmoid

        return float(sigmoid(np.clip(self.z_trust, -12.0, 12.0)))

    def rage(self) -> float:
        from src.match_engine.math_utils import sigmoid

        return float(sigmoid(np.clip(self.z_rage, -12.0, 12.0)))


@dataclass
class AssistantRefereeState:
    side: str = "left"  # left | right
    offside_strictness: float = 0.5
    trust_with_center: float = 0.6
    flag_delay_sec: float = 0.0


@dataclass
class RefereeAffectiveState:
    profile_name: str = "balanced"
    strictness_base: float = 0.55
    bias_home: float = 0.0
    z_calm: float = 0.0
    z_defensive: float = 0.0
    card_load: float = 0.0
    controversy_integral: float = 0.0
    assistants: List["AssistantRefereeState"] = field(default_factory=list)

    def calm(self) -> float:
        from src.match_engine.math_utils import sigmoid

        return float(sigmoid(np.clip(self.z_calm, -12.0, 12.0)))

    def strictness_effective(self, psi: float, cfg) -> float:
        from src.match_engine.affective_coupling import sigmoid, tanh_clip

        s = (
            self.strictness_base
            + cfg.kappa_rpsi * tanh_clip(psi - cfg.psi_neutral)
            - cfg.kappa_rc_calm * self.calm()
        )
        return float(sigmoid(s * 4.0 - 2.0))


@dataclass
class CrowdState:
    psi: float = 0.0
    home_team_id: str = ""
    is_neutral_venue: bool = False
    event_intensity_integral: float = 0.0


@dataclass
class BallState:
    position: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5], dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    height: float = 0.0
    omega: float = 0.0
    spin_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    possessor_id: Optional[str] = None
    possession_team_id: str = ""


@dataclass
class SpatialGridState:
    rho_home: np.ndarray = field(default_factory=lambda: np.zeros((32, 22)))
    rho_away: np.ndarray = field(default_factory=lambda: np.zeros((32, 22)))
    press: np.ndarray = field(default_factory=lambda: np.zeros((32, 22)))
    phi_home: np.ndarray = field(default_factory=lambda: np.zeros((32, 22)))
    phi_away: np.ndarray = field(default_factory=lambda: np.zeros((32, 22)))


@dataclass
class TeamAffectiveState:
    team_id: str
    players: List[PlayerAffectiveState] = field(default_factory=list)
    coach: CoachAffectiveState = field(default_factory=lambda: CoachAffectiveState(team_id=""))
    peer_matrix: Optional[np.ndarray] = None  # (n,n) coupling
    score: int = 0
    xg: float = 0.0
    icon_player_id: Optional[str] = None
    attacks_high_x: bool = True
    phase: float = 0.5
    possession_share: float = 0.5
    formation_key: str = "433"

    @property
    def n_players(self) -> int:
        return len(self.players)


@dataclass
class MatchAffectiveState:
    home: TeamAffectiveState
    away: TeamAffectiveState
    referee: RefereeAffectiveState
    crowd: CrowdState
    clock_seconds: float = 0.0
    stage_pressure: float = 0.3
    pending_events: List[Any] = field(default_factory=list)
    ball: BallState = field(default_factory=BallState)
    spatial: SpatialGridState = field(default_factory=SpatialGridState)
    pass_log: List[Dict[str, Any]] = field(default_factory=list)
    ball_path_log: List[Dict[str, Any]] = field(default_factory=list)
    micro_xg_home: float = 0.0
    micro_xg_away: float = 0.0
    last_shot_clock_home: float = -999.0
    last_shot_clock_away: float = -999.0

    def team(self, team_id: str) -> TeamAffectiveState:
        if team_id == self.home.team_id:
            return self.home
        if team_id == self.away.team_id:
            return self.away
        raise KeyError(team_id)


@dataclass
class MicroMatchSummary:
    """Extends affective summary with spatial / passing stats."""

    home_team: str
    away_team: str
    ticks: int
    final_psi: float
    home_coach_stress: float
    away_coach_stress: float
    ref_strictness_mean: float
    home_emotion_mean: Dict[str, float]
    away_emotion_mean: Dict[str, float]
    icon_shock_home: float
    icon_shock_away: float
    tactical_drift_home: float
    tactical_drift_away: float
    controversy_integral: float
    possession_home: float = 0.5
    passes_home: int = 0
    passes_away: int = 0
    pass_completion_home: float = 0.0
    pass_completion_away: float = 0.0
    through_balls_home: int = 0
    through_balls_away: int = 0
    long_passes_home: int = 0
    long_passes_away: int = 0
    curved_passes_home: int = 0
    curved_passes_away: int = 0
    pass_curve_omega_mean: float = 0.0
    outside_foot_passes_home: int = 0
    outside_foot_passes_away: int = 0
    ground_passes_home: int = 0
    ground_passes_away: int = 0
    pass_intercepts_home: int = 0
    pass_intercepts_away: int = 0
    wall_passes_home: int = 0
    wall_passes_away: int = 0
    wall_combos_home: int = 0
    wall_combos_away: int = 0
    phi_integral_home: float = 0.0
    phi_integral_away: float = 0.0
    micro_xg_home: float = 0.0
    micro_xg_away: float = 0.0
    goals_micro_home: int = 0
    goals_micro_away: int = 0
    goals_physics_home: int = 0
    goals_physics_away: int = 0
    xg_supplement_applied: bool = False
    xg_supplement_meta: Dict[str, Any] = field(default_factory=dict)
    shots_home: int = 0
    shots_away: int = 0
    shots_scheduled_home: int = 0
    shots_scheduled_away: int = 0
    shots_on_target_home: int = 0
    shots_on_target_away: int = 0
    fouls_committed_home: int = 0
    fouls_committed_away: int = 0
    yellow_cards_home: int = 0
    yellow_cards_away: int = 0
    red_cards_home: int = 0
    red_cards_away: int = 0
    tackles_home: int = 0
    tackles_away: int = 0
    curved_shots: int = 0
    knuckle_shots: int = 0
    headers_attempted: int = 0
    crosses_attempted: int = 0
    lambda_home: float = 0.0
    lambda_away: float = 0.0
    player_stats: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    substitutions: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_triggers: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_plans: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_tier_usage: Dict[str, int] = field(default_factory=dict)
    meso_packets: List[Dict[str, Any]] = field(default_factory=list)
    timeline_snippet: List[str] = field(default_factory=list)
    ball_log_path: str = ""
    continuous_clock: Dict[str, Any] = field(default_factory=dict)
    subtick_reception_queue: Dict[str, Any] = field(default_factory=dict)
    world_model_online_calibration: Dict[str, Any] = field(default_factory=dict)
    world_model_decision_adoption: Dict[str, Any] = field(default_factory=dict)
    world_model_action_adoption: Dict[str, Any] = field(default_factory=dict)
    world_model_branch_anchor: Dict[str, Any] = field(default_factory=dict)
    world_model_runtime: Dict[str, Any] = field(default_factory=dict)
    continuity_state: Dict[str, Any] = field(default_factory=dict)
    manager_effects: Dict[str, Any] = field(default_factory=dict)
    in_match_management: Dict[str, Any] = field(default_factory=dict)
    tactical_execution: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AffectiveMatchSummary:
    home_team: str
    away_team: str
    ticks: int
    final_psi: float
    home_coach_stress: float
    away_coach_stress: float
    ref_strictness_mean: float
    home_emotion_mean: Dict[str, float]
    away_emotion_mean: Dict[str, float]
    icon_shock_home: float
    icon_shock_away: float
    tactical_drift_home: float
    tactical_drift_away: float
    controversy_integral: float
    meso_packets: List[Dict[str, Any]] = field(default_factory=list)
    timeline_snippet: List[str] = field(default_factory=list)
