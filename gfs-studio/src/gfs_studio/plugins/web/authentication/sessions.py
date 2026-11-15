from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_studio.contracts.web import HttpError


class LoopbackAuthentication:
    def authorize(self, host: str, headers: dict) -> dict:
        name = host.rsplit(":", 1)[0].lower()
        if name not in {"127.0.0.1", "localhost"}:
            raise HttpError(403, "loopback host is required")
        if any("forwarded" in key.lower() for key in headers):
            raise HttpError(403, "forwarded requests require an external trusted proxy")
        return {
            "principal": "local_development_operator",
            "transport": "loopback",
            "production_authenticated": False,
        }

    def status(self) -> dict:
        return {
            "mode": "loopback_development_boundary",
            "remote_access": False,
            "production_authentication": False,
        }


class AuthenticationPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.web.authentication",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.web",
        provides=("authentication",),
        requires=("gfs.studio.web.gateway",),
    )

    def activate(self, context: PluginContext) -> None:
        gateway = context.require("http-gateway")
        authentication = LoopbackAuthentication()
        gateway.authentication = authentication
        context.provide("authentication", authentication)

    def deactivate(self) -> None:
        pass
