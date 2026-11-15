from .action_codec import encode_high_level_action
from .observation import encode_observation
from .schema import strip_outcome_leakage


class WorldModelTraceCodec:
    version = "world-model-v9-obs307-action18"

    def observation(self, state):
        return encode_observation(state)

    def action(self, kind: str, target, horizon_s: float):
        if kind in {"start", "kickoff"}:
            kind = "hold"
        return encode_high_level_action(kind, target, horizon_s=horizon_s)

    def strip_action(self, action):
        return strip_outcome_leakage(action)
