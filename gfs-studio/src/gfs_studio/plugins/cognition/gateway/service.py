from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_studio.contracts.cognition import CognitionRequest

from .provider import GatewayConfig, LLMGateway, RuleBasedProvider


class CognitionService:
    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway

    def reflect(self, *, role: str, topic: str, match_id: str, seed: int = 42):
        request = CognitionRequest(
            role=role,
            system_prompt=(
                "You are a bounded football review assistant. "
                "Do not claim real-world causal authority."
            ),
            user_prompt=topic,
            match_id=match_id,
            seed=seed,
        )
        return self.gateway.generate(request).content

    def status(self) -> dict:
        return self.gateway.status()


class CognitionPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.cognition.gateway",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.cognition",
        provides=("provider-gateway",),
    )

    def __init__(self, gateway: LLMGateway | None = None) -> None:
        self._gateway = gateway or LLMGateway(
            RuleBasedProvider(), GatewayConfig(),
        )
        self.service = CognitionService(self._gateway)

    def activate(self, context: PluginContext) -> None:
        context.provide("provider-gateway", self.service)

    def deactivate(self) -> None:
        pass
