from scripts.verify_deployment import ROOT, verify


def test_plugin_deployment_runs_only_studio_with_embedded_core():
    report = verify(ROOT)
    assert report["static_verified"], report["errors"]
    assert report["runtime_verified"] is False
