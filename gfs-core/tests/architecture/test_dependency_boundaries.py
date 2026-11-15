import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "gfs_core"


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_core_never_depends_on_studio():
    for path in SOURCE_ROOT.rglob("*.py"):
        assert all(
            not module.startswith("gfs_studio")
            for module in imported_modules(path)
        ), path


def test_plugins_do_not_import_other_plugin_implementations():
    plugins_root = SOURCE_ROOT / "plugins"
    for path in plugins_root.rglob("*.py"):
        assert all(
            not module.startswith("gfs_core.plugins")
            for module in imported_modules(path)
        ), path


def test_contracts_do_not_depend_on_runtime_or_product_layers():
    for path in (SOURCE_ROOT / "contracts").rglob("*.py"):
        for module in imported_modules(path):
            assert not module.startswith("gfs_core.kernel"), path
            assert not module.startswith("gfs_core.plugins"), path
            assert not module.startswith("gfs_core.bootstrap"), path
