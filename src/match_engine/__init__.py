"""
GFS match_engine — microscopic simulation layers.

Phase 1b: affective coupling | 2a/2b: spatial + passing | 3: ball physics + shots.
"""

from src.match_engine.adapter import simulate_match_score_micro
from src.match_engine.affective_coupling import AffectiveSpatialCoupling
from src.match_engine.config import AffectiveConfig
from src.match_engine.goal_generator import lambdas_from_micro, simulate_match_score_from_micro
from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.match_affective_runner import run_match_affective_simulation
from src.match_engine.match_micro_runner import run_match_micro_simulation
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.shot_engine import ShotEngine
from src.match_engine.state import (
    AffectiveMatchSummary,
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    MicroMatchSummary,
    PlayerAffectiveState,
    PlayerModulators,
    RefereeAffectiveState,
    TeamAffectiveState,
)

__all__ = [
    "AffectiveConfig",
    "AffectiveSpatialCoupling",
    "AffectiveMatchSummary",
    "CoachAffectiveState",
    "CrowdState",
    "MatchAffectiveState",
    "MicroMatchSummary",
    "PlayerAffectiveState",
    "PlayerModulators",
    "RefereeAffectiveState",
    "TeamAffectiveState",
    "build_match_affective_state",
    "run_match_affective_simulation",
    "run_match_micro_simulation",
    "simulate_match_score_micro",
    "lambdas_from_micro",
    "simulate_match_score_from_micro",
    "MicroMatchConfig",
    "ShotEngine",
]
