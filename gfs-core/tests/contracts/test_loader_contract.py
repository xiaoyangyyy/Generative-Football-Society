from pathlib import Path

import pytest

from gfs_core.kernel import (
    PluginKernel,
    PluginRegistry,
    load_entry_points,
    load_manifest,
    load_manifests,
    register_entry_points,
    resolve_metadata,
)
from tests.helpers import StubPlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_manifest_loading_supports_simple_and_composite_plugins(tmp_path):
    composite = tmp_path / "composite" / "plugin.toml"
    composite.parent.mkdir()
    composite.write_text(
        '[plugin]\nid = "example.composite"\nversion = "1.0.0"\n'
        'api = 1\ncontract = "example.contract"\n'
        'provides = ["composite"]\nrequires = ["example.simple"]\n',
        encoding="utf-8",
    )
    simple = tmp_path / "simple.plugin.toml"
    simple.write_text(
        '[plugin]\nid = "example.simple"\nversion = "1.0.0"\n'
        'api = 1\ncontract = "example.contract"\nprovides = ["simple"]\n',
        encoding="utf-8",
    )
    assert load_manifest(simple).provides == ("simple",)
    assert [item.id for item in load_manifests(tmp_path)] == [
        "example.composite", "example.simple",
    ]


def test_repository_manifests_have_closed_acyclic_dependencies():
    root = PROJECT_ROOT / "src" / "gfs_core" / "plugins"
    metadata = load_manifests(root)
    assert len(metadata) == 30
    resolved = resolve_metadata(metadata)
    assert {item.id for item in resolved} == {item.id for item in metadata}
    for item in metadata:
        contract = PROJECT_ROOT / "src" / Path(*item.contract.split(".")).with_suffix(".py")
        assert contract.is_file(), item.contract


class FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


def test_entry_points_create_register_and_start_plugins():
    entry = FakeEntryPoint(
        "alpha", lambda: StubPlugin("example.alpha", provides=("alpha",)),
    )
    plugins = load_entry_points("gfs_core.plugins", [entry])
    assert len(plugins) == 1
    registry = PluginRegistry()
    metadata = register_entry_points(registry, "gfs_core.plugins", [entry])
    assert metadata[0].id == "example.alpha"
    kernel = PluginKernel(registry).start()
    assert isinstance(kernel.context.require("alpha"), StubPlugin)
    kernel.stop()


def test_entry_points_reject_non_factory_and_invalid_plugin():
    with pytest.raises(TypeError, match="not callable"):
        load_entry_points("gfs_core.plugins", [FakeEntryPoint("bad", object())])
    with pytest.raises(TypeError, match="did not create a Plugin"):
        load_entry_points("gfs_core.plugins", [FakeEntryPoint("bad", lambda: object())])
