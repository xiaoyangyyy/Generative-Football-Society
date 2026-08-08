import os
import tomllib
from pathlib import Path

from src import app


ROOT = Path(__file__).resolve().parents[1]


def test_wheel_preserves_the_src_package_named_by_console_entrypoint():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["scripts"]["gfs"] == "src.cli:main"
    discovery = payload["tool"]["setuptools"]["packages"]["find"]
    assert discovery["where"] == ["."]
    assert "src" in discovery["include"] and "src.*" in discovery["include"]


def test_project_root_honors_explicit_external_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("GFS_PROJECT_ROOT", str(tmp_path))
    assert app.project_root() == tmp_path.resolve()


def test_project_root_discovers_workspace_from_child_directory(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "data/releases").mkdir(parents=True)
    (workspace / "data/releases/current.json").write_text("{}", encoding="utf-8")
    child = workspace / "nested/tool"
    child.mkdir(parents=True)
    monkeypatch.delenv("GFS_PROJECT_ROOT", raising=False)
    previous = Path.cwd()
    try:
        os.chdir(child)
        assert app.project_root() == workspace.resolve()
    finally:
        os.chdir(previous)
