from gfs_core.contracts.micro_config import MicroMatchConfig

from .field import SpatialFieldEngine
from .formation import interpolate_anchors, mirror_for_away, roles_for_formation
from .intelligence import SpatialIntelligenceEngine
from .keys import normalize_formation_key


class SpatialService:
    @staticmethod
    def create_field(config: MicroMatchConfig) -> SpatialFieldEngine:
        return SpatialFieldEngine(config)

    @staticmethod
    def create_intelligence(config: MicroMatchConfig) -> SpatialIntelligenceEngine:
        return SpatialIntelligenceEngine(config)

    @staticmethod
    def interpolate_anchors(*args, **kwargs):
        return interpolate_anchors(*args, **kwargs)

    @staticmethod
    def mirror_for_away(position):
        return mirror_for_away(position)

    @staticmethod
    def roles_for_formation(formation: str):
        return roles_for_formation(formation)

    @staticmethod
    def normalize_formation_key(formation: str):
        return normalize_formation_key(formation)
