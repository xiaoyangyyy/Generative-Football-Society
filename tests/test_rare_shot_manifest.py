import json

from src.data_engine.dataset_registry import (
    augment_training_manifest,
    build_trace_manifest,
    stable_partition,
    verify_trace_manifest,
)
from src.match_engine.world_model.action_codec import ACTION_DIM
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.schema import SHOT_GOAL_INDEX


def _group_for(split: str, token: str) -> str:
    for index in range(10000):
        group = f"{token}_{index}"
        if stable_partition(group, 42) == split:
            return group
    raise AssertionError(f"unable to create {split} group")


def _trace(path, marker: float) -> None:
    obs = [0.0] * OBS_DIM
    nxt = [0.0] * OBS_DIM
    action = [0.0] * ACTION_DIM
    obs[0] = marker
    nxt[0] = marker + 0.01
    action[1] = 1.0
    action[SHOT_GOAL_INDEX] = float(marker > 0.5)
    path.write_text(json.dumps({
        "obs": obs, "action": action, "next_obs": nxt, "meta": {},
    }) + "\n", encoding="utf-8")


def test_rare_shot_augmentation_preserves_the_original_sealed_groups(tmp_path):
    markers = {"train": 0.1, "dev": 0.2, "sealed_test": 0.3}
    for split, marker in markers.items():
        _trace(tmp_path / f"{_group_for(split, 'base')}.jsonl", marker)
    base = build_trace_manifest(tmp_path, seed=42)
    original_sealed = [
        entry for entry in base["files"] if entry["split"] == "sealed_test"
    ]

    added_train = _group_for("train", "rare_shot_v10")
    excluded_sealed = _group_for("sealed_test", "rare_shot_v10")
    _trace(tmp_path / f"{added_train}.jsonl", 0.7)
    _trace(tmp_path / f"{excluded_sealed}.jsonl", 0.8)

    augmented = augment_training_manifest(
        base, tmp_path, include_token="rare_shot_v10",
    )
    assert [
        entry for entry in augmented["files"] if entry["split"] == "sealed_test"
    ] == original_sealed
    assert f"{added_train}.jsonl" in augmented["augmentation"]["added_files"]
    assert f"{excluded_sealed}.jsonl" in augmented["augmentation"]["excluded_sealed_candidates"]
    verify_trace_manifest(augmented)
