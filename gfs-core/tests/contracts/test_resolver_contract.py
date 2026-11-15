import pytest

from gfs_core.kernel import (
    DependencyCycleError,
    MissingDependencyError,
    resolve_metadata,
)
from tests.helpers import StubPlugin


def metadata(plugin_id, requires=()):
    return StubPlugin(plugin_id, requires=requires).metadata


def test_resolver_selects_transitive_dependencies_in_order():
    alpha = metadata("example.alpha")
    beta = metadata("example.beta", ("example.alpha",))
    gamma = metadata("example.gamma", ("example.beta",))
    unused = metadata("example.unused")
    resolved = resolve_metadata(
        [gamma, unused, beta, alpha], selected=["example.gamma"],
    )
    assert [item.id for item in resolved] == [
        "example.alpha", "example.beta", "example.gamma",
    ]


def test_resolver_is_deterministic_for_independent_plugins():
    alpha = metadata("example.alpha")
    beta = metadata("example.beta")
    gamma = metadata("example.gamma", ("example.alpha", "example.beta"))
    resolved = resolve_metadata([gamma, beta, alpha])
    assert [item.id for item in resolved] == [
        "example.alpha", "example.beta", "example.gamma",
    ]


def test_resolver_rejects_missing_selection_and_dependency():
    with pytest.raises(MissingDependencyError, match="selected plugin"):
        resolve_metadata([], selected=["example.missing"])
    broken = metadata("example.broken", ("example.missing",))
    with pytest.raises(MissingDependencyError, match="unregistered plugin"):
        resolve_metadata([broken])


def test_resolver_rejects_cycles():
    alpha = metadata("example.alpha", ("example.beta",))
    beta = metadata("example.beta", ("example.alpha",))
    with pytest.raises(DependencyCycleError, match="dependency cycle"):
        resolve_metadata([alpha, beta])
