from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.contracts.micro_config import MicroMatchConfig

from .service import ActionService


class ActionsPlugin:
    metadata = PluginMetadata(
        id="gfs.core.match.actions",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.match",
        provides=("actions",),
        requires=("gfs.core.match.physics", "gfs.core.match.spatial"),
    )

    def __init__(
        self,
        config: MicroMatchConfig | None = None,
        *,
        ball_log=None,
        base_dir=".",
    ) -> None:
        self._config = config or MicroMatchConfig()
        self._ball_log = ball_log
        self._base_dir = base_dir

    def activate(self, context: PluginContext) -> None:
        context.provide(
            "actions",
            ActionService(
                self._config,
                context.require("physics"),
                self._ball_log,
                self._base_dir,
                context.require("spatial"),
            ),
        )

    def deactivate(self) -> None:
        pass
