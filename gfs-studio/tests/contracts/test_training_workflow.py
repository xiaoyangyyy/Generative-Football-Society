import json
from pathlib import Path

import numpy as np
import pytest

from gfs_core.kernel import PluginKernel
from gfs_studio.bootstrap.application import StudioApplication
from gfs_studio.bootstrap.builtin_plugins import build_training_registry
from gfs_studio.plugins.training.datasets.manifest import TraceDataset
from gfs_studio.plugins.world_model.model import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_traces(path: Path) -> None:
    rng = np.random.default_rng(42)
    with path.open("w", encoding="utf-8") as handle:
        for group in range(4):
            for _ in range(4):
                observation = rng.random(307, dtype=np.float32)
                action = rng.random(18, dtype=np.float32)
                target = np.clip(
                    observation + rng.normal(0, 0.01, 307), 0, 1,
                ).astype(np.float32)
                handle.write(json.dumps({
                    "observation": observation.tolist(),
                    "action": action.tolist(),
                    "next_observation": target.tolist(),
                    "group_id": f"match-{group}",
                }) + "\n")


def test_grouped_dataset_split_does_not_leak_validation_group(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_traces(path)
    dataset = TraceDataset.from_jsonl(path)
    train, validation, split = dataset.split()
    assert split.train_groups == 3
    assert split.validation_groups == 1
    assert split.train_rows == 12
    assert split.validation_rows == 4
    assert set(train.groups).isdisjoint(set(validation.groups))


def test_real_training_checkpoint_and_optimizer_resume(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_traces(path)
    app = StudioApplication(tmp_path)
    first = app.train(path, "smoke", epochs=1)
    assert first["status"] == "completed"
    assert first["epochs"] == 1
    assert first["resumed"] is False
    assert first["qualification"]["scope"] == "transition_prediction_only_not_policy_or_deployment"

    second = app.train(path, "smoke", epochs=2, resume=True)
    assert second["status"] == "completed"
    assert second["epochs"] == 2
    assert second["resumed"] is True
    _, _, metadata = load_checkpoint(second["checkpoint"])
    assert metadata["validation"]["model_mse"] == second["qualification"]["model_mse"]
    assert len(metadata["history"]) == 2
    status = json.loads((tmp_path / "data" / "training" / "runs" / "smoke" / "status.json").read_text())
    assert status["state"] == "completed"
    assert (tmp_path / "data" / "training" / "runs" / "smoke" / "optimizer.pt").is_file()


def test_training_bundle_registers_real_checkpoint(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_traces(path)
    StudioApplication(tmp_path).train(path, "smoke")
    kernel = PluginKernel(build_training_registry(
        tmp_path, core_root=PROJECT_ROOT.parent / "gfs-core",
    ))
    try:
        kernel.start(["gfs.studio.training"])
        bundle = kernel.context.require("training-bundle")
        assert set(bundle) == {"datasets", "trace_export", "trainers", "evaluation", "checkpoints"}
        record = bundle["checkpoints"].register("smoke")
        assert record["immutable"] is True
        assert len(record["checkpoint_sha256"]) == 64
        assert record["epoch"] == 1
    finally:
        kernel.stop()


def test_dataset_rejects_invalid_shapes(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps({"observation": [0.0], "action": [0.0], "next_observation": [0.0]}) + "\n")
    with pytest.raises(ValueError):
        TraceDataset.from_jsonl(path)
