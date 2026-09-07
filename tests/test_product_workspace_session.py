import json

import pytest

import src.product.workspace as workspace_module
from src.product.workspace import ProductWorkspace, StudioConfig
from src.product.workspace_session import load_workspace_session
from scripts.verify_product_recovery import CODE_IDENTITY_FILES


def _load_direct(workspace: ProductWorkspace):
    return load_workspace_session(
        root=workspace.root,
        session_path=workspace.session_path,
        season_history_view=workspace._season_history_view,
        sporting_brief_for_transition=workspace._sporting_brief_for_transition,
    )


def test_workspace_session_repository_loads_a_minimal_valid_session(tmp_path):
    assert "src/product/workspace_session.py" in CODE_IDENTITY_FILES
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Session boundary", seed=23),
    )

    session = _load_direct(workspace)

    assert session["schema_version"] == 1
    assert session["name"] == "Session boundary"
    assert session["seed"] == 23


def test_workspace_session_repository_fails_closed_on_registry_tampering(
    tmp_path,
):
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Tamper boundary"),
    )
    session = json.loads(workspace.session_path.read_text(encoding="utf-8"))
    session["finance_registry"]["schema_version"] = 99
    workspace.session_path.write_text(
        json.dumps(session), encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid finance registry"):
        _load_direct(workspace)


def test_product_workspace_session_facade_forwards_exact_dependencies(
    monkeypatch, tmp_path,
):
    workspace = ProductWorkspace(
        tmp_path, StudioConfig(name="Facade boundary"),
    )
    observed = {}

    def fake_loader(**kwargs):
        observed.update(kwargs)
        return {"loaded": True}

    monkeypatch.setattr(
        workspace_module, "load_workspace_session", fake_loader,
    )

    assert workspace._session() == {"loaded": True}
    assert observed["root"] == tmp_path.resolve()
    assert observed["session_path"] == workspace.session_path
    assert observed["season_history_view"].__self__ is workspace
    assert observed["sporting_brief_for_transition"].__self__ is workspace
