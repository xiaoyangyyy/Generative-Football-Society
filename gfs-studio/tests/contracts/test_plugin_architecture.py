import ast
import importlib
import tomllib
from pathlib import Path

from gfs_core.kernel import PluginKernel
from gfs_studio.bootstrap.builtin_plugins import build_studio_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = PROJECT_ROOT.parent / "gfs-core"


def test_complete_studio_profile_starts_every_declared_plugin():
    with (PROJECT_ROOT / "profiles" / "studio.toml").open("rb") as handle:
        selected = tomllib.load(handle)["profile"]["plugins"]
    registry = build_studio_registry(PROJECT_ROOT)
    kernel = PluginKernel(registry)
    try:
        kernel.start(item.id for item in registry.metadata)
        assert set(selected) <= set(kernel.active_ids)
        assert {item.id for item in registry.metadata} == set(kernel.active_ids)
        assert kernel.context.require("web-bundle")["gateway"] is not None
        assert kernel.context.require("training-bundle")["checkpoints"] is not None
        assert kernel.context.require("studio-workflows").policy_assessor is not None
        assert kernel.context.require("studio-workflows").trace_codec is not None
        assert kernel.context.require("participants-bundle")["sessions"].status()["registered_participants"] == 0
    finally:
        kernel.stop()


def test_plugin_sources_do_not_import_other_plugin_implementations():
    for root, package in (
        (CORE_ROOT / "src" / "gfs_core" / "plugins", "gfs_core.plugins"),
        (PROJECT_ROOT / "src" / "gfs_studio" / "plugins", "gfs_studio.plugins"),
    ):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module and node.module.startswith(package):
                    raise AssertionError(f"cross-plugin implementation import: {path}:{node.lineno}")
                if node.module and node.module in {"gfs_core.bootstrap", "gfs_studio.bootstrap"}:
                    raise AssertionError(f"plugin imports composition root: {path}:{node.lineno}")
                if node.level <= 1:
                    continue
                relative = path.relative_to(root)
                if package == "gfs_studio.plugins":
                    raise AssertionError(f"parent plugin implementation import: {path}:{node.lineno}")
                domain = relative.parts[0]
                ancestors = relative.parent.parts[:1 - node.level]
                assert ancestors[0] == domain, f"cross-domain import: {path}:{node.lineno}"


def test_every_manifest_has_an_importable_implementation():
    for root in (
        CORE_ROOT / "src" / "gfs_core" / "plugins",
        PROJECT_ROOT / "src" / "gfs_studio" / "plugins",
    ):
        for path in root.rglob("*.toml"):
            with path.open("rb") as handle:
                document = tomllib.load(handle)
                implementation = document["plugin"]["implementation"]
            assert document["plugin"].get("contract-version", 1) == 1, path
            assert implementation != "pending", path
            module_name, separator, attribute = implementation.partition(":")
            assert separator, path
            assert callable(getattr(importlib.import_module(module_name), attribute)), path
