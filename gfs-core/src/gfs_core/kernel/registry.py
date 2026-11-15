from __future__ import annotations

from collections.abc import Iterator

from gfs_core.contracts.plugin import (
    PLUGIN_API_VERSION,
    PLUGIN_CONTRACT_VERSION,
    Plugin,
    PluginMetadata,
)
from gfs_core.kernel.errors import DuplicatePluginError


class PluginRegistry:
    def __init__(self, plugins: Iterator[Plugin] | list[Plugin] | tuple[Plugin, ...] = ()) -> None:
        self._plugins: dict[str, Plugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: Plugin) -> PluginMetadata:
        if not isinstance(plugin, Plugin):
            raise TypeError("registered object does not implement Plugin")
        metadata = plugin.metadata
        if metadata.api != PLUGIN_API_VERSION:
            raise ValueError(
                f"unsupported plugin api for {metadata.id}: {metadata.api}"
            )
        if metadata.contract_version != PLUGIN_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported contract version for {metadata.id}: {metadata.contract_version}"
            )
        if metadata.id in self._plugins:
            raise DuplicatePluginError(f"plugin already registered: {metadata.id}")
        self._plugins[metadata.id] = plugin
        return metadata

    def get(self, plugin_id: str) -> Plugin:
        return self._plugins[plugin_id]

    @property
    def metadata(self) -> tuple[PluginMetadata, ...]:
        return tuple(self._plugins[key].metadata for key in sorted(self._plugins))

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self) -> Iterator[Plugin]:
        for plugin_id in sorted(self._plugins):
            yield self._plugins[plugin_id]
