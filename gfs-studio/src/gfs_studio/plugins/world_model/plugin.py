from gfs_core.contracts import PluginContext, PluginMetadata

from .model import load_checkpoint, save_checkpoint
from .runtime import WorldModelRuntime
from .trace_codec import WorldModelTraceCodec


class CheckpointCodec:
    load = staticmethod(load_checkpoint)
    save = staticmethod(save_checkpoint)


class LatentWorldModelPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.world-model.latent",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.world_model",
        provides=("latent-models", "checkpoint-codec", "trace-codec"),
        requires=(),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("latent-models", WorldModelRuntime)
        context.provide("checkpoint-codec", CheckpointCodec())
        context.provide("trace-codec", WorldModelTraceCodec())

    def deactivate(self) -> None:
        pass
