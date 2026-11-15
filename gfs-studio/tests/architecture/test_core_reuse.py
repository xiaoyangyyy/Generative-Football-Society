import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "gfs_studio"


def test_studio_has_no_parallel_plugin_kernel():
    assert not (SOURCE_ROOT / "kernel").exists()


def test_studio_plugins_do_not_import_plugin_implementations():
    for path in (SOURCE_ROOT / "plugins").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            assert all(not module.startswith("gfs_core.plugins") for module in modules), path
            assert all(not module.startswith("gfs_studio.plugins") for module in modules), path
