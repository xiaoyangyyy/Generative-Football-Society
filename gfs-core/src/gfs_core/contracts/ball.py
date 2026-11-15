from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallActionParams:
    v0: float
    elev: float
    azim: float
    omega: float
    spin_axis: np.ndarray
    knuckle_intensity: float = 0.0
    outside_foot: float = 0.0
    ground_weight: float = 0.0


@dataclass
class TrajectoryResult:
    landed: np.ndarray
    peak_height: float
    time_of_flight: float
    crossed_goal_line: bool
    goal_y: float | None
    in_goal_mouth: bool
    curve_amount: float


@dataclass
class PassTrajectoryResult:
    landed: np.ndarray
    peak_height: float
    time_of_flight: float
    target_miss: float
    lateral_dev: float
    curve_amount: float
    omega_used: float
    outside_foot: float = 0.0
    ground_weight: float = 0.0
