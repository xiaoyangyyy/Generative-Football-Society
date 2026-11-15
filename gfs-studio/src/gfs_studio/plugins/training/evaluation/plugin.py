from gfs_core.contracts import PluginContext, PluginMetadata

from .validator import CandidateEvaluator


class EvaluationPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.training.evaluation",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.training",
        provides=("model-evaluation",),
        requires=("gfs.studio.world-model.latent",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("model-evaluation", CandidateEvaluator)

    def deactivate(self) -> None:
        pass
