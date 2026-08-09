from scripts.verify_product_tasks import verify_product_tasks


def test_task_verifier_is_honest_persistent_and_zero_real_execution():
    report = verify_product_tasks()
    assert report["passed"]
    assert report["status"] == "passed_isolated_task_lifecycle"
    assert report["external_calls_made"] is False
    assert report["real_matches_executed"] == 0
    assert report["training_executed"] is False
    assert all(report["checks"].values())
    assert any("not a real match" in item for item in report["limitations"])
