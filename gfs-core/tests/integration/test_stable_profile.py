import tomllib
from pathlib import Path

from gfs_core.bootstrap.builtin_plugins import build_stable_registry
from gfs_core.kernel import PluginKernel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_stable_profile_starts_all_declared_plugin_bundles():
    profile = tomllib.loads((PROJECT_ROOT / "profiles" / "stable.toml").read_text())
    kernel = PluginKernel(build_stable_registry(PROJECT_ROOT))
    kernel.start(profile["profile"]["plugins"])
    assert kernel.context.require("data-bundle")["results"] is not None
    assert kernel.context.require("status-bundle")["snapshot"] is not None
    assert set(kernel.context.require("society-bundle")) == {"agents", "psychology", "memory"}
    assert set(kernel.context.require("persistence-bundle")) == {"workspace", "transactions", "recovery", "micro_checkpoints"}
    assert set(kernel.context.require("presentation-bundle")) == {"cli", "json"}
    assert kernel.context.require("match-runner") is not None
    assert kernel.context.require("tournament-runner") is not None
    kernel.stop()
