from scripts.verify_product_recovery import verify_product_recovery


def test_recovery_verifier_is_complete_honest_and_zero_execution():
    report = verify_product_recovery()
    assert report["passed"]
    assert report["status"] == "passed_local_recovery"
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert all(report["checks"].values())
    assert any("not an operational deployment drill" in item for item in report["limitations"])
