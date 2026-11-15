from gfs_core.contracts.micro_config import MicroMatchConfig

from .ball import (
    azimuth_to_goal,
    azimuth_to_target,
    desired_pass_omega,
    ground_pass_weight,
    integrate_pass_trajectory,
    integrate_trajectory,
    outside_foot_factor,
    sample_pass_delivery_params,
    sample_shot_params,
)
from .kinematics import KinematicPositionLayer


class PhysicsService:
    def __init__(self, spatial) -> None:
        self._spatial = spatial

    def create_kinematics(
        self,
        config: MicroMatchConfig,
        *,
        spatial_intelligence=None,
    ) -> KinematicPositionLayer:
        return KinematicPositionLayer(
            config,
            self._spatial,
            spatial_intelligence,
        )

    integrate_trajectory = staticmethod(integrate_trajectory)
    integrate_pass_trajectory = staticmethod(integrate_pass_trajectory)
    sample_shot_params = staticmethod(sample_shot_params)
    sample_pass_delivery_params = staticmethod(sample_pass_delivery_params)
    azimuth_to_goal = staticmethod(azimuth_to_goal)
    azimuth_to_target = staticmethod(azimuth_to_target)
    desired_pass_omega = staticmethod(desired_pass_omega)
    ground_pass_weight = staticmethod(ground_pass_weight)
    outside_foot_factor = staticmethod(outside_foot_factor)
