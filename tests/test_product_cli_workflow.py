from argparse import Namespace

import pytest

from src.cli import cmd_studio_run
from src.product.workspace import ProductWorkspace, StudioConfig


def test_single_run_entry_reuses_matching_workspace(tmp_path, monkeypatch):
    ProductWorkspace.create(tmp_path, StudioConfig(name="Demo", mode="stable", seed=9))
    workspace = ProductWorkspace.load(tmp_path)
    monkeypatch.setattr(workspace, "readiness", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(ProductWorkspace, "load", lambda root: workspace)
    monkeypatch.setattr(workspace, "run_match", lambda *args, **kwargs: {
        "result": {"score": {"home": 1, "away": 0}},
        "report_path": "report.json", "dashboard_path": "report.html",
    })
    args = Namespace(
        base_dir=str(tmp_path), name="Demo", mode="stable", seed=9,
        replace=False, home="Brazil", away="Argentina", fast=True,
    )
    assert cmd_studio_run(args) == 0


def test_single_run_entry_refuses_implicit_workspace_reconfiguration(tmp_path):
    ProductWorkspace.create(tmp_path, StudioConfig(name="Demo", mode="stable", seed=9))
    args = Namespace(
        base_dir=str(tmp_path), name="Other", mode=None, seed=None,
        replace=False, home="Brazil", away="Argentina", fast=True,
    )
    with pytest.raises(ValueError, match="--replace"):
        cmd_studio_run(args)
