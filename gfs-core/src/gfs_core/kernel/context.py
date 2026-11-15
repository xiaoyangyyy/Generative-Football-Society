from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gfs_core.kernel.errors import CapabilityError
from gfs_core.kernel.events import EventBus


class PluginContext:
    def __init__(self, events: EventBus | None = None) -> None:
        self._events = events or EventBus()
        self._services: dict[str, Any] = {}

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._services))

    def provide(self, capability: str, service: Any) -> None:
        if not isinstance(capability, str) or not capability.strip():
            raise ValueError("capability must be a non-empty string")
        if service is None:
            raise ValueError("capability service must not be None")
        if capability in self._services:
            raise CapabilityError(f"capability already provided: {capability}")
        self._services[capability] = service
        self.publish("capability.provided", {
            "capability": capability,
            "service": service,
        })

    def require(self, capability: str) -> Any:
        try:
            return self._services[capability]
        except KeyError as error:
            raise CapabilityError(f"capability not available: {capability}") from error

    def find(self, capability: str, default: Any = None) -> Any:
        return self._services.get(capability, default)

    def publish(self, topic: str, payload: Any = None) -> None:
        self._events.publish(topic, payload)

    def subscribe(
        self, topic: str, callback: Callable[[Any], None],
    ) -> Callable[[], None]:
        return self._events.subscribe(topic, callback)

    def clear(self) -> None:
        self._services.clear()
        self._events.clear()
