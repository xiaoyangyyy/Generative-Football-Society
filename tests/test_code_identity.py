from pathlib import Path

import pytest

from src.infrastructure.code_identity import (
    EXPLICIT_FILES_V1,
    TRANSITIVE_LOCAL_IMPORTS_V1,
    code_identity_manifest,
)


def _write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _dependency_tree(root: Path) -> None:
    _write(root, "src/__init__.py", "ROOT = True\n")
    _write(root, "src/pkg/__init__.py", "from . import package_hook\n")
    _write(root, "src/pkg/package_hook.py", "HOOK = 1\n")
    _write(
        root,
        "src/pkg/controller.py",
        "from src.pkg import helper\n"
        "from .leaf import VALUE\n"
        "import importlib\n"
        "plugin = importlib.import_module('src.pkg.dynamic')\n",
    )
    _write(root, "src/pkg/helper.py", "from . import leaf\n")
    _write(root, "src/pkg/leaf.py", "from .helper import helper_value\nVALUE = 1\n")
    _write(root, "src/pkg/dynamic.py", "DYNAMIC = True\n")


def test_explicit_identity_preserves_legacy_file_set(tmp_path):
    _dependency_tree(tmp_path)

    identity = code_identity_manifest(
        tmp_path,
        ["src/pkg/controller.py"],
        mode=EXPLICIT_FILES_V1,
    )

    assert list(identity) == ["src/pkg/controller.py"]


def test_transitive_identity_closes_local_imports_packages_cycles_and_dynamic_imports(
    tmp_path,
):
    _dependency_tree(tmp_path)

    identity = code_identity_manifest(
        tmp_path,
        ["src/pkg/controller.py"],
        mode=TRANSITIVE_LOCAL_IMPORTS_V1,
    )

    assert list(identity) == sorted(identity)
    assert set(identity) == {
        "src/__init__.py",
        "src/pkg/__init__.py",
        "src/pkg/controller.py",
        "src/pkg/dynamic.py",
        "src/pkg/helper.py",
        "src/pkg/leaf.py",
        "src/pkg/package_hook.py",
    }


def test_transitive_leaf_change_invalidates_identity(tmp_path):
    _dependency_tree(tmp_path)
    before = code_identity_manifest(
        tmp_path,
        ["src/pkg/controller.py"],
        mode=TRANSITIVE_LOCAL_IMPORTS_V1,
    )

    _write(tmp_path, "src/pkg/leaf.py", "VALUE = 2\n")
    after = code_identity_manifest(
        tmp_path,
        ["src/pkg/controller.py"],
        mode=TRANSITIVE_LOCAL_IMPORTS_V1,
    )

    assert before != after
    assert before["src/pkg/controller.py"] == after["src/pkg/controller.py"]
    assert before["src/pkg/leaf.py"] != after["src/pkg/leaf.py"]


@pytest.mark.parametrize(
    "relative",
    ["../outside.py", str(Path.cwd().resolve() / "absolute.py")],
)
def test_identity_rejects_nonportable_or_escaping_paths(tmp_path, relative):
    with pytest.raises(ValueError, match="project-relative|escapes root"):
        code_identity_manifest(tmp_path, [relative])


def test_identity_rejects_missing_invalid_and_empty_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        code_identity_manifest(tmp_path, ["src/missing.py"])
    with pytest.raises(ValueError, match="unsupported"):
        code_identity_manifest(tmp_path, ["src/missing.py"], mode="future")
    with pytest.raises(ValueError, match="at least one"):
        code_identity_manifest(tmp_path, [])


def test_transitive_identity_fails_closed_on_unparseable_declared_code(tmp_path):
    _write(tmp_path, "src/broken.py", "def broken(:\n")

    with pytest.raises(ValueError, match="cannot parse"):
        code_identity_manifest(
            tmp_path,
            ["src/broken.py"],
            mode=TRANSITIVE_LOCAL_IMPORTS_V1,
        )
