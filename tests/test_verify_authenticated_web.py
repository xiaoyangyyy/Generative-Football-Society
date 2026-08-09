from scripts.verify_authenticated_web import verify_authenticated_web


def test_authenticated_web_verifier_is_honest_and_zero_execution():
    report = verify_authenticated_web()
    assert report["passed"]
    assert report["status"] == "passed_authenticated_socket_boundary"
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert all(report["checks"].values())
    assert any("no container was built" in item for item in report["limitations"])
