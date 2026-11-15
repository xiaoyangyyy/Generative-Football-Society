from __future__ import annotations

from collections.abc import Iterable

from gfs_core.kernel.context import PluginContext
from gfs_core.kernel.errors import (
    PluginActivationError,
    PluginError,
    PluginStateError,
)
from gfs_core.kernel.registry import PluginRegistry
from gfs_core.kernel.resolver import resolve_metadata


class PluginKernel:
    def __init__(
        self, registry: PluginRegistry, context: PluginContext | None = None,
    ) -> None:
        self._registry = registry
        self._context = context or PluginContext()
        self._active: list[str] = []
        self._running = False

    @property
    def context(self) -> PluginContext:
        return self._context

    @property
    def running(self) -> bool:
        return self._running

    @property
    def active_ids(self) -> tuple[str, ...]:
        return tuple(self._active)

    def start(self, selected: Iterable[str] | None = None) -> PluginKernel:
        if self._running or self._active:
            raise PluginStateError("plugin kernel is already running")
        plan = resolve_metadata(self._registry.metadata, selected)
        activated: list[str] = []
        for metadata in plan:
            plugin = self._registry.get(metadata.id)
            try:
                plugin.activate(self._context)
            except Exception as error:
                self._deactivate_all(activated)
                self._context.clear()
                raise PluginActivationError(metadata.id) from error
            activated.append(metadata.id)
        self._active = activated
        self._running = True
        return self

    def stop(self) -> None:
        errors = self._deactivate_all(self._active)
        self._active = []
        self._running = False
        self._context.clear()
        if errors:
            raise PluginError("one or more plugins failed to deactivate") from errors[0]

    def _deactivate_all(self, plugin_ids: Iterable[str]) -> list[Exception]:
        errors: list[Exception] = []
        for plugin_id in reversed(tuple(plugin_ids)):
            try:
                self._registry.get(plugin_id).deactivate()
            except Exception as error:
                errors.append(error)
        return errors
