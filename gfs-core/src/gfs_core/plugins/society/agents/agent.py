from gfs_core.contracts import PluginContext, PluginMetadata

from .squad_factory import SquadFactory


class SquadService:
    def __init__(self, rosters, spatial) -> None:
        self._factory = SquadFactory(rosters, spatial)

    def build(self, profile, rng, **kwargs):
        return self._factory.build(profile, rng, **kwargs)


class AgentsPlugin:
    metadata = PluginMetadata(
        id="gfs.core.society.agents",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.society",
        provides=("agents",),
        requires=("gfs.core.data.rosters", "gfs.core.match.spatial"),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide(
            "agents",
            SquadService(
                context.require("rosters"),
                context.require("spatial"),
            ),
        )

    def deactivate(self) -> None:
        pass
