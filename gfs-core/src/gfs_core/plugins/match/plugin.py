from gfs_core.contracts import PluginContext, PluginMetadata

from .runner import MatchService


class MatchPlugin:
    metadata = PluginMetadata(
        id="gfs.core.match",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.match",
        provides=("match-runner",),
        requires=(
            "gfs.core.match.physics",
            "gfs.core.match.actions",
            "gfs.core.match.spatial",
            "gfs.core.match.scoring",
            "gfs.core.society.agents",
            "gfs.core.society.psychology",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide(
            "match-runner",
            MatchService(
                agents=context.require("agents"),
                spatial=context.require("spatial"),
                physics=context.require("physics"),
                actions=context.require("actions"),
                psychology=context.require("psychology"),
                scoring=context.require("scoring"),
            ),
        )

    def deactivate(self) -> None:
        pass
