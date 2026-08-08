import json

import pytest

from src.training.job import TrainingJob


def test_training_job_persists_logs_progress_and_terminal_state(tmp_path):
    torch = pytest.importorskip("torch")
    job = TrainingJob(tmp_path / "run", {"epochs": 3, "model": "unit"})
    job.start(total_epochs=3, resume=False)
    job.record_epoch(1, {"loss": 0.5})
    job.save_progress({
        "config_fingerprint": job.config_fingerprint,
        "epoch": 1,
        "tensor": torch.tensor([2.0]),
    })
    restored = job.load_progress()
    assert restored["epoch"] == 1
    assert restored["tensor"].item() == 2.0
    assert json.loads(job.log_path.read_text(encoding="utf-8"))["loss"] == 0.5
    first_attempt = json.loads(job.log_path.read_text(encoding="utf-8"))["attempt_id"]
    assert first_attempt == job.attempt_id
    job.complete(artifact="candidate.pt", epoch=3)
    assert json.loads(job.status_path.read_text(encoding="utf-8"))["state"] == "completed"


def test_training_job_rejects_accidental_overwrite_and_config_drift(tmp_path):
    run_dir = tmp_path / "run"
    first = TrainingJob(run_dir, {"epochs": 3})
    first.start(total_epochs=3, resume=False)
    with pytest.raises(FileExistsError):
        TrainingJob(run_dir, {"epochs": 3}).start(total_epochs=3, resume=False)
    with pytest.raises(ValueError):
        TrainingJob(run_dir, {"epochs": 4}).start(total_epochs=4, resume=True)


def test_training_job_supports_cooperative_stop(tmp_path):
    torch = pytest.importorskip("torch")
    job = TrainingJob(tmp_path / "run", {"epochs": 3})
    job.start(total_epochs=3, resume=False)
    job.stop_path.write_text("stop\n", encoding="utf-8")
    assert job.stop_requested()
    job.save_progress({
        "config_fingerprint": job.config_fingerprint,
        "epoch": 1,
        "tensor": torch.tensor([1.0]),
    })
    job.mark_stopped(epoch=1)
    assert json.loads(job.status_path.read_text(encoding="utf-8"))["state"] == "stopped"


def test_resume_clears_consumed_stop_and_rejects_duplicate_writer(tmp_path):
    torch = pytest.importorskip("torch")
    run = tmp_path / "run"
    first = TrainingJob(run, {"epochs": 3})
    first.start(total_epochs=3, resume=False)
    first.save_progress({
        "config_fingerprint": first.config_fingerprint,
        "epoch": 1,
        "tensor": torch.tensor([1.0]),
    })
    duplicate = TrainingJob(run, {"epochs": 3})
    with pytest.raises(RuntimeError, match="already active"):
        duplicate.start(total_epochs=3, resume=True)
    first.stop_path.write_text("stop\n", encoding="utf-8")
    first.mark_stopped(epoch=1)

    resumed = TrainingJob(run, {"epochs": 3})
    resumed.start(total_epochs=3, resume=True)
    assert not resumed.stop_path.exists()
    assert resumed.attempt_id != first.attempt_id
    assert len(json.loads(resumed.status_path.read_text(encoding="utf-8"))["attempts"]) == 2
    resumed.fail(RuntimeError("test shutdown"))


def test_stop_requires_checkpoint_for_exact_resume(tmp_path):
    job = TrainingJob(tmp_path / "run", {"epochs": 3})
    job.start(total_epochs=3, resume=False)
    with pytest.raises(RuntimeError, match="checkpointing"):
        job.mark_stopped(epoch=1)
    job.fail(RuntimeError("expected test cleanup"))


def test_terminal_training_job_rejects_later_mutation(tmp_path):
    job = TrainingJob(tmp_path / "run", {"epochs": 1})
    job.start(total_epochs=1, resume=False)
    job.complete(artifact="candidate.pt", epoch=1)
    with pytest.raises(RuntimeError, match="not in an active"):
        job.record_epoch(2, {"loss": 0.0})
    with pytest.raises(RuntimeError, match="not in an active"):
        job.fail(RuntimeError("late failure"))
