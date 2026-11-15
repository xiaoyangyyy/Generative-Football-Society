from __future__ import annotations

from gfs_core.contracts import PluginMetadata


class StubPlugin:
    def __init__(
        self,
        plugin_id: str,
        *,
        requires: tuple[str, ...] = (),
        provides: tuple[str, ...] = (),
        api: int = 1,
        contract_version: int = 1,
        events: list[str] | None = None,
        fail_activate: bool = False,
        fail_deactivate: bool = False,
    ) -> None:
        self.metadata = PluginMetadata(
            id=plugin_id,
            version="1.0.0",
            api=api,
            contract="tests.contract",
            provides=provides,
            requires=requires,
            contract_version=contract_version,
        )
        self.events = events if events is not None else []
        self.fail_activate = fail_activate
        self.fail_deactivate = fail_deactivate
        self.context = None

    def activate(self, context) -> None:
        if self.fail_activate:
            raise RuntimeError(f"activate: {self.metadata.id}")
        self.context = context
        self.events.append(f"activate:{self.metadata.id}")
        for capability in self.metadata.provides:
            context.provide(capability, self)

    def deactivate(self) -> None:
        self.events.append(f"deactivate:{self.metadata.id}")
        if self.fail_deactivate:
            raise RuntimeError(f"deactivate: {self.metadata.id}")
        self.context = None
