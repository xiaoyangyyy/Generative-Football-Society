import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "verify_deployment",
    PROJECT_ROOT.parent / "scripts" / "verify_deployment.py",
)
verify_deployment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_deployment)
verify = verify_deployment.verify


def test_static_deployment_contract_passes_without_claiming_runtime():
    result = verify(PROJECT_ROOT.parent)
    assert result["static_verified"] is True
    assert result["runtime_verified"] is False
    assert result["external_calls_made"] is False
    assert result["errors"] == []
