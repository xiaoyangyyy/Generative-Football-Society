from scripts.verify_process_recovery import verify_process_recovery


def test_real_process_kill_recovery_is_honest_and_zero_execution():
    report = verify_process_recovery()
    assert report["passed"]
    assert report["status"] == "passed_local_process_kill_recovery"
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert report["formal_experiment_executed"] is False
    assert report["recovery_point"] == "zero_loss_for_files_committed_before_kill"
    assert 0 < report["recovery_time_seconds"] <= 15
    assert all(report["checks"].values())
    assert any("not a production RTO" in item for item in report["limitations"])
    assert any("synthetic" in item for item in report["limitations"])
