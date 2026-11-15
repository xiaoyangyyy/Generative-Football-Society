from pathlib import Path

import pytest

from gfs_core.bootstrap.application import CoreApplication
from gfs_core.kernel.workspace import WorkspaceRepository
from gfs_core.plugins.society.memory.service import SocietyMemory

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_micro_resume_converges_to_uninterrupted_summary(tmp_path):
    expected = CoreApplication(
        PROJECT_ROOT, workspace_root=tmp_path / "expected",
    ).run_micro_match(fast=True, seed=42)
    app = CoreApplication(PROJECT_ROOT, workspace_root=tmp_path)
    interrupted = app.run_micro_match(
        fast=True,
        seed=42,
        checkpoint="match",
        checkpoint_interval=30,
        stop_at_tick=90,
    )
    assert interrupted["next_tick"] == 90
    assert interrupted["status"] == "interrupted"
    resumed = CoreApplication(
        PROJECT_ROOT, workspace_root=tmp_path,
    ).run_micro_match(fast=True, seed=42, resume="match")
    assert resumed == expected
    completed = WorkspaceRepository(tmp_path).read("micro-checkpoints/match.json")
    assert completed["status"] == "completed"
    assert completed["next_tick"] == 180


def test_micro_resume_rejects_request_or_state_divergence(tmp_path):
    app = CoreApplication(PROJECT_ROOT, workspace_root=tmp_path)
    document = app.run_micro_match(
        fast=True, seed=42, checkpoint="match", stop_at_tick=60,
    )
    with pytest.raises(ValueError, match="request identity mismatch"):
        app.run_micro_match(fast=True, seed=43, resume="match")
    document["state_sha256"] = "0" * 64
    WorkspaceRepository(tmp_path).write("micro-checkpoints/match.json", document)
    with pytest.raises(ValueError, match="replay diverged at tick 60"):
        CoreApplication(PROJECT_ROOT, workspace_root=tmp_path).run_micro_match(
            fast=True, seed=42, resume="match",
        )


def test_replay_with_historical_memory_requires_explicit_time_cursor(tmp_path):
    SocietyMemory(tmp_path).append("Brazil", "previous match", importance=0.8)
    app = CoreApplication(PROJECT_ROOT, workspace_root=tmp_path)
    app.run_micro_match(
        fast=True, seed=42, checkpoint="match", stop_at_tick=1,
    )
    with pytest.raises(ValueError, match="memory_as_of"):
        CoreApplication(PROJECT_ROOT, workspace_root=tmp_path).run_micro_match(
            fast=True, seed=42, resume="match",
        )
