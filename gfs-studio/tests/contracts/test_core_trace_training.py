import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gfs_core.kernel import PluginKernel
from gfs_core.kernel.identity import content_identity
from gfs_studio.bootstrap.application import StudioApplication
from gfs_studio.bootstrap.builtin_plugins import build_training_registry
from gfs_studio.plugins.training.datasets.manifest import TraceDataset
from gfs_studio.plugins.world_model.model import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def export_two_matches(tmp_path):
    app = StudioApplication(PROJECT_ROOT)
    exported = [
        app.export_traces(
            tmp_path / f"match-{seed}.jsonl", seed=seed, stop_at_tick=4,
        )
        for seed in (42, 43)
    ]
    rows = []
    for manifest in exported:
        assert manifest["rows"] == 4
        assert manifest["source"] == "gfs.core.match"
        rows.extend(Path(manifest["trace"]).read_text().splitlines())
    combined = tmp_path / "combined.jsonl"
    combined.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source": "gfs.core.match",
        "trace_contract": "gfs.core.trace/v1",
        "codec_version": exported[0]["codec_version"],
        "config_identity": content_identity([item["config_identity"] for item in exported]),
        "trace_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
        "rows": len(rows),
        "groups": sorted({item["groups"][0] for item in exported}),
        "observation_dim": 307,
        "action_dim": 18,
        "split_seed": 42,
        "status": "interrupted",
    }
    combined.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return combined, manifest


def write_manifest(path, manifest):
    manifest["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def assert_same_weights(left, right):
    left_model, _, _ = load_checkpoint(left)
    right_model, _, _ = load_checkpoint(right)
    for name, value in left_model.state_dict().items():
        assert torch.equal(value, right_model.state_dict()[name]), name


def test_real_core_traces_have_identity_and_leakage_free_actions(tmp_path):
    path, manifest = export_two_matches(tmp_path)
    dataset = TraceDataset.from_jsonl(path)
    assert dataset.identity == manifest["trace_sha256"]
    assert dataset.source == "gfs.core.match"
    assert len(dataset.observations) == 8
    assert np.all(dataset.actions[:, 14:17] == 0.0)
    train, validation, split = dataset.split()
    assert split.train_groups == 1
    assert split.validation_groups == 1
    assert set(train.groups).isdisjoint(set(validation.groups))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="content identity mismatch"):
        TraceDataset.from_jsonl(path)


def test_resume_matches_fresh_training_and_rejects_changed_data(tmp_path):
    path, manifest = export_two_matches(tmp_path)
    resumed_app = StudioApplication(tmp_path / "resumed")
    first = resumed_app.train(path, "run", epochs=1)
    assert first["dataset_identity"] == manifest["trace_sha256"]
    second = resumed_app.train(path, "run", epochs=2, resume=True)
    fresh = StudioApplication(tmp_path / "fresh").train(path, "run", epochs=2)
    assert second["resumed"] is True
    assert_same_weights(second["checkpoint"], fresh["checkpoint"])
    kernel = PluginKernel(build_training_registry(
        resumed_app.project_root, core_root=PROJECT_ROOT.parent / "gfs-core",
    ))
    kernel.start(["gfs.studio.training.checkpoints"])
    registry = kernel.context.require("checkpoint-registry")
    registered = registry.register("run")
    assert registry.register("run") == registered
    assert Path(registered["checkpoint"]) != Path(second["checkpoint"])
    third = resumed_app.train(path, "run", epochs=3, resume=True)
    assert registry.get(registered["checkpoint_sha256"])["immutable"] is True
    assert hashlib.sha256(Path(third["checkpoint"]).read_bytes()).hexdigest() != registered["checkpoint_sha256"]
    assert registry.register("run")["checkpoint_sha256"] != registered["checkpoint_sha256"]
    kernel.stop()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["observation"][0] = 1.0 - rows[0]["observation"][0]
    path.write_text("\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n")
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        resumed_app.train(path, "run", epochs=3, resume=True)
