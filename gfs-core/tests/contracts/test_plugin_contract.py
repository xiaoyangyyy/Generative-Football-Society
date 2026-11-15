import pytest

from gfs_core.contracts import Plugin, PluginMetadata
from gfs_core.kernel import DuplicatePluginError, PluginRegistry
from tests.helpers import StubPlugin


def test_metadata_normalizes_collections_and_implements_plugin_contract():
    plugin = StubPlugin("example.alpha", provides=["alpha"], requires=[])
    assert plugin.metadata.provides == ("alpha",)
    assert plugin.metadata.requires == ()
    assert isinstance(plugin, Plugin)


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "Invalid ID"},
        {"api": 0},
        {"provides": ("alpha", "alpha")},
        {"requires": ("example.alpha",)},
    ],
)
def test_metadata_rejects_invalid_contracts(changes):
    values = {
        "id": "example.alpha",
        "version": "1.0.0",
        "api": 1,
        "contract": "example.contract",
    }
    values.update(changes)
    with pytest.raises(ValueError):
        PluginMetadata(**values)


def test_registry_rejects_duplicate_ids_and_unsupported_api():
    registry = PluginRegistry([StubPlugin("example.alpha")])
    with pytest.raises(DuplicatePluginError):
        registry.register(StubPlugin("example.alpha"))
    with pytest.raises(ValueError, match="unsupported plugin api"):
        registry.register(StubPlugin("example.beta", api=2))


def test_registry_rejects_objects_missing_lifecycle_methods():
    class IncompletePlugin:
        metadata = StubPlugin("example.incomplete").metadata

    with pytest.raises(TypeError, match="does not implement Plugin"):
        PluginRegistry().register(IncompletePlugin())
