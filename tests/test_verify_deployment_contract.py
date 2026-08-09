from scripts.verify_deployment_contract import verify_deployment_contract


def test_deployment_contract_is_secure_and_honest_about_unbuilt_state():
    report = verify_deployment_contract()
    assert report["passed"]
    assert all(report["checks"].values())
    assert report["image_built"] is False
    assert report["deployment_started"] is False
    assert report["external_calls_made"] is False
    if not report["docker_available"]:
        assert report["status"] == "passed_static_contract_unbuilt"
        assert report["docker_compose_validation"] == "docker_unavailable"
    assert any("not been built" in item for item in report["limitations"])
