from __future__ import annotations

import tomllib
from collections.abc import Iterable
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from gfs_core.contracts.plugin import Plugin, PluginMetadata
from gfs_core.kernel.registry import PluginRegistry


def load_manifest(path: str | Path) -> PluginMetadata:
    manifest_path = Path(path)
    with manifest_path.open("rb") as handle:
        document = tomllib.load(handle)
    data = document["plugin"]
    return PluginMetadata(
        id=data["id"],
        version=data["version"],
        api=data["api"],
        contract=data["contract"],
        provides=tuple(data.get("provides", ())),
        requires=tuple(data.get("requires", ())),
        contract_version=data.get("contract-version", 1),
    )


def load_manifests(root: str | Path) -> tuple[PluginMetadata, ...]:
    root_path = Path(root)
    paths = {
        *root_path.rglob("plugin.toml"),
        *root_path.rglob("*.plugin.toml"),
    }
    return tuple(load_manifest(path) for path in sorted(paths))


def load_entry_points(
    group: str, entries: Iterable[Any] | None = None,
) -> tuple[Plugin, ...]:
    discovered = (
        importlib_metadata.entry_points(group=group)
        if entries is None else entries
    )
    plugins: list[Plugin] = []
    for entry in discovered:
        factory = entry.load()
        if not callable(factory):
            raise TypeError(f"plugin entry point is not callable: {entry.name}")
        plugin = factory()
        if not isinstance(plugin, Plugin):
            raise TypeError(f"entry point did not create a Plugin: {entry.name}")
        plugins.append(plugin)
    return tuple(plugins)


def register_entry_points(
    registry: PluginRegistry,
    group: str,
    entries: Iterable[Any] | None = None,
) -> tuple[PluginMetadata, ...]:
    return tuple(
        registry.register(plugin)
        for plugin in load_entry_points(group, entries)
    )
