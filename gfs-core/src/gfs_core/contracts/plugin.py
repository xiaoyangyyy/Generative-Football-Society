from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

PLUGIN_API_VERSION = 1
PLUGIN_CONTRACT_VERSION = 1
_PLUGIN_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    id: str
    version: str
    api: int
    contract: str
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    contract_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "provides", tuple(self.provides))
        object.__setattr__(self, "requires", tuple(self.requires))
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version < 1
        ):
            raise ValueError("plugin contract version must be a positive integer")
        for field in (self.id, self.version, self.contract):
            if not isinstance(field, str) or not field.strip():
                raise ValueError("plugin id, version, and contract must be non-empty strings")
        if not _PLUGIN_ID.fullmatch(self.id):
            raise ValueError(f"invalid plugin id: {self.id}")
        if isinstance(self.api, bool) or not isinstance(self.api, int) or self.api < 1:
            raise ValueError("plugin api must be a positive integer")
        for field, values in (("provides", self.provides), ("requires", self.requires)):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"plugin {field} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"plugin {field} must not contain duplicates")
        if self.id in self.requires:
            raise ValueError(f"plugin cannot depend on itself: {self.id}")
        for plugin_id in self.requires:
            if not _PLUGIN_ID.fullmatch(plugin_id):
                raise ValueError(f"invalid required plugin id: {plugin_id}")


class PluginContext(Protocol):
    def provide(self, capability: str, service: Any) -> None: ...

    def require(self, capability: str) -> Any: ...

    def publish(self, topic: str, payload: Any = None) -> None: ...

    def subscribe(
        self, topic: str, callback: Callable[[Any], None],
    ) -> Callable[[], None]: ...


@runtime_checkable
class Plugin(Protocol):
    metadata: PluginMetadata

    def activate(self, context: PluginContext) -> None: ...

    def deactivate(self) -> None: ...
