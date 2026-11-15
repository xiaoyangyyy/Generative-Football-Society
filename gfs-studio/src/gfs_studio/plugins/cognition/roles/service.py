from gfs_core.contracts import PluginContext, PluginMetadata

from .coach import CoachRole
from .media import MediaRole
from .player import PlayerRole


class RoleService:
    def __init__(self, cognition) -> None:
        self.coach = CoachRole(cognition)
        self.player = PlayerRole()
        self.media = MediaRole()


class RolesPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.cognition.roles",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.cognition",
        provides=("coach-role", "player-role", "media-role"),
        requires=("gfs.studio.cognition.gateway",),
    )

    def __init__(self, cognition, service=None) -> None:
        self.service = service or RoleService(cognition)

    def activate(self, context: PluginContext) -> None:
        context.provide("coach-role", self.service.coach)
        context.provide("player-role", self.service.player)
        context.provide("media-role", self.service.media)

    def deactivate(self) -> None:
        pass
