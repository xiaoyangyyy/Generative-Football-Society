from scripts.verify_product_web import verify_web_beta


def test_web_beta_verifier_is_honest_and_zero_call():
    report = verify_web_beta()
    assert report["passed"]
    assert report["status"] == "passed_local_beta"
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert all(report["checks"].values())
    assert any("not a production deployment" in item for item in report["limitations"])
    assert any("no 100-match soak" in item for item in report["limitations"])
