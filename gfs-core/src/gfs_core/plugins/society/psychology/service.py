from gfs_core.contracts.affective import AffectiveConfig

from .coupling import AffectiveSpatialCoupling
from .discipline import emit_tick_cross, emit_tick_discipline, emit_tick_tackle
from .meso import MesoAggregator
from .schedule import build_event_schedule
from .tactical import step_in_match_pressure


class PsychologyService:
    def __init__(self, actions) -> None:
        self._actions = actions

    def create_coupling(self, config: AffectiveConfig | None = None):
        return AffectiveSpatialCoupling(
            config,
            self._actions.apply_micro_event,
        )

    @staticmethod
    def build_event_schedule(state, **kwargs):
        return build_event_schedule(state, **kwargs)

    emit_tick_cross = staticmethod(emit_tick_cross)
    emit_tick_discipline = staticmethod(emit_tick_discipline)
    emit_tick_tackle = staticmethod(emit_tick_tackle)
    step_in_match_pressure = staticmethod(step_in_match_pressure)
    create_meso = staticmethod(MesoAggregator)
