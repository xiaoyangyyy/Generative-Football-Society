from __future__ import annotations

import heapq
from collections.abc import Iterable

from gfs_core.contracts.plugin import PluginMetadata
from gfs_core.kernel.errors import (
    DependencyCycleError,
    DuplicatePluginError,
    MissingDependencyError,
)


def resolve_metadata(
    metadata: Iterable[PluginMetadata], selected: Iterable[str] | None = None,
) -> tuple[PluginMetadata, ...]:
    available: dict[str, PluginMetadata] = {}
    for item in metadata:
        if item.id in available:
            raise DuplicatePluginError(f"plugin already available: {item.id}")
        available[item.id] = item

    if selected is None:
        required = set(available)
    else:
        required = set()
        pending = list(selected)
        while pending:
            plugin_id = pending.pop()
            if plugin_id in required:
                continue
            try:
                item = available[plugin_id]
            except KeyError as error:
                raise MissingDependencyError(
                    f"selected plugin is not registered: {plugin_id}"
                ) from error
            required.add(plugin_id)
            pending.extend(item.requires)

    for plugin_id in required:
        for dependency in available[plugin_id].requires:
            if dependency not in available:
                raise MissingDependencyError(
                    f"{plugin_id} requires unregistered plugin: {dependency}"
                )

    dependencies = {
        plugin_id: set(available[plugin_id].requires)
        for plugin_id in required
    }
    dependents: dict[str, list[str]] = {plugin_id: [] for plugin_id in required}
    for plugin_id, required_ids in dependencies.items():
        for dependency in required_ids:
            dependents[dependency].append(plugin_id)

    ready = [plugin_id for plugin_id, required_ids in dependencies.items() if not required_ids]
    heapq.heapify(ready)
    resolved: list[PluginMetadata] = []
    while ready:
        plugin_id = heapq.heappop(ready)
        resolved.append(available[plugin_id])
        for dependent in sorted(dependents[plugin_id]):
            dependencies[dependent].remove(plugin_id)
            if not dependencies[dependent]:
                heapq.heappush(ready, dependent)

    if len(resolved) != len(required):
        cyclic = sorted(plugin_id for plugin_id, value in dependencies.items() if value)
        raise DependencyCycleError(
            "plugin dependency cycle: " + ", ".join(cyclic)
        )
    return tuple(resolved)
