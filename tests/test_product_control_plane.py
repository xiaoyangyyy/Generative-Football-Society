import json
import socket

import pytest

from src.product.control_plane import ProductControlPlane


def test_control_plane_composes_training_and_decision_state(tmp_path):
    run = tmp_path / "data/training/runs/shot-head"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({
        "schema_version": 1, "state": "running", "last_completed_epoch": 2,
    }), encoding="utf-8")
    decision = tmp_path / "data/evaluation/formal_ablation"
    decision.mkdir(parents=True)
    (decision / "M1_calibrated_decision.json").write_text(json.dumps({
        "decision": "research_only_default_off",
    }), encoding="utf-8")
    snapshot = ProductControlPlane(tmp_path).snapshot()
    assert snapshot["training_summary"]["running"] == 1
    assert snapshot["decisions"]["world_model_formal_m1"]["available"]


def test_control_plane_requests_only_cooperative_stop_for_running_job(tmp_path):
    run = tmp_path / "data/training/runs/model-a"
    run.mkdir(parents=True)
    status = run / "status.json"
    status.write_text(json.dumps({"state": "running"}), encoding="utf-8")
    stop = ProductControlPlane(tmp_path).request_stop("model-a")
    assert stop.read_text(encoding="utf-8").startswith("requested_by_gfs_studio")
    snapshot = ProductControlPlane(tmp_path).snapshot()
    assert snapshot["training_jobs"][0]["stop_requested"]
    assert snapshot["training_summary"]["stopping"] == 1
    status.write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        ProductControlPlane(tmp_path).request_stop("model-a")
    with pytest.raises(ValueError):
        ProductControlPlane(tmp_path).request_stop("../escape")


def test_control_plane_marks_dead_local_process_as_stale(tmp_path):
    run = tmp_path / "data/training/runs/dead-job"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({
        "state": "running", "pid": 2147483647, "hostname": socket.gethostname(),
    }), encoding="utf-8")
    snapshot = ProductControlPlane(tmp_path).snapshot()
    assert snapshot["training_jobs"][0]["observed_state"] == "stale"
    assert snapshot["training_summary"]["stale"] == 1
    assert snapshot["training_summary"]["running"] == 0
    with pytest.raises(RuntimeError, match="stale"):
        ProductControlPlane(tmp_path).request_stop("dead-job")
