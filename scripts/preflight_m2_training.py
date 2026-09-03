#!/usr/bin/env python3
"""Validate M2 training feasibility without optimizing or writing a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_world_model import (
    _match_group,
    _multi_step_curriculum_weight,
    _sequential_transition_pairs,
)
from src.data_engine.dataset_registry import (
    files_for_split,
    load_manifest,
    verify_trace_manifest,
)
from src.match_engine.world_model.action_codec import decode_action_kinds
from src.match_engine.world_model.config import WorldModelConfig
from src.match_engine.world_model.m2_contract import (
    FROZEN_TRAINING_CONFIGURATION,
)
from src.match_engine.world_model.policy_utility import (
    transition_policy_utility_numpy,
)
from src.match_engine.world_model.recorder import load_trace_batches


DEFAULT_MANIFEST = (
    ROOT / "data/world_model/dataset_manifest_formal_v8_candidate.json"
)
DEFAULT_TRACES = ROOT / "data/world_model/traces"


def _split_arrays(trace_dir: Path, manifest: dict, split: str) -> dict[str, Any]:
    observations, actions, futures, raw_groups = load_trace_batches(
        str(trace_dir),
        return_groups=True,
        allowed_files=files_for_split(manifest, {split}),
    )
    observations = np.clip(
        np.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    futures = np.clip(
        np.nan_to_num(futures, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0)
    groups = np.asarray([_match_group(value) for value in raw_groups], dtype=str)
    valid = (
        (np.std(observations, axis=1) > 0.02)
        & (np.std(futures, axis=1) > 0.02)
    )
    observations = observations[valid]
    actions = actions[valid]
    futures = futures[valid]
    groups = groups[valid]
    kinds = decode_action_kinds(actions)
    indices = np.arange(len(observations), dtype=int)
    pair_left = _sequential_transition_pairs(
        observations, futures, groups, indices,
    )
    action_counts = {
        action: int(np.sum(kinds == action))
        for action in ("pass", "shot", "hold", "cross", "intercept", "tackle")
    }
    action_groups = {
        action: int(len(np.unique(groups[kinds == action])))
        for action in action_counts
    }
    utility = transition_policy_utility_numpy(
        observations, futures,
    )["policy_utility"]
    pass_utility = np.asarray(utility)[kinds == "pass"]
    return {
        "rows": int(len(observations)),
        "groups": int(len(np.unique(groups))),
        "action_counts": action_counts,
        "action_groups": action_groups,
        "two_step_pairs": int(len(pair_left)),
        "two_step_groups": (
            int(len(np.unique(groups[pair_left]))) if len(pair_left) else 0
        ),
        "pass_utility_persistence_mse": (
            float(np.mean(pass_utility ** 2)) if len(pass_utility) else 0.0
        ),
        "pass_utility_standard_deviation": (
            float(np.std(pass_utility)) if len(pass_utility) else 0.0
        ),
    }


def readiness_checks(
    *,
    manifest: dict,
    train: dict[str, Any],
    dev: dict[str, Any],
    epochs: int,
    warmup_fraction: float,
    policy_utility_loss_weight: float,
    multi_step_loss_weight: float,
    transition_ensemble_size: int,
) -> dict[str, bool]:
    entries = list(manifest.get("files") or [])
    split_groups = {
        split: {str(row.get("group")) for row in entries if row.get("split") == split}
        for split in ("train", "dev", "sealed_test")
    }
    disjoint = all(
        not (split_groups[left] & split_groups[right])
        for left, right in (
            ("train", "dev"),
            ("train", "sealed_test"),
            ("dev", "sealed_test"),
        )
    )
    final_policy_weight = _multi_step_curriculum_weight(
        policy_utility_loss_weight,
        epochs - 1,
        epochs,
        warmup_fraction=warmup_fraction,
    ) if epochs > 0 else 0.0
    final_two_step_weight = _multi_step_curriculum_weight(
        multi_step_loss_weight,
        epochs - 1,
        epochs,
        warmup_fraction=warmup_fraction,
    ) if epochs > 0 else 0.0
    sealed_summary = (manifest.get("summary") or {}).get("sealed_test") or {}
    return {
        "manifest_schema_and_policy_valid": bool(
            manifest.get("schema_version") == 1
            and manifest.get("sealed_test_policy")
            == "evaluation-only; never train or tune"
        ),
        "train_dev_sealed_groups_complete_and_disjoint": bool(
            disjoint and all(split_groups.values())
        ),
        "training_rows_sufficient": train["rows"] >= 128,
        "development_rows_sufficient": dev["rows"] >= 32,
        "development_pass_support_sufficient": bool(
            dev["action_counts"]["pass"] >= 96
            and dev["action_groups"]["pass"] >= 6
        ),
        "sealed_pass_support_declared_sufficient": bool(
            int(sealed_summary.get("passes", 0)) >= 96
            and int(sealed_summary.get("groups", 0)) >= 6
        ),
        "development_pass_utility_target_non_degenerate": bool(
            dev["pass_utility_persistence_mse"] > 1e-12
            and dev["pass_utility_standard_deviation"] > 1e-8
        ),
        "training_two_step_support_sufficient": bool(
            train["two_step_pairs"] >= 32
            and train["two_step_groups"] >= 4
        ),
        "development_two_step_support_sufficient": bool(
            dev["two_step_pairs"] >= 32
            and dev["two_step_groups"] >= 4
        ),
        "policy_utility_curriculum_will_activate": final_policy_weight > 0.0,
        "two_step_curriculum_will_activate": final_two_step_weight > 0.0,
        "transition_ensemble_supports_uncertainty": transition_ensemble_size >= 2,
    }


def preflight(
    *,
    root: Path = ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
    trace_dir: Path = DEFAULT_TRACES,
    epochs: int = 45,
    warmup_fraction: float = 0.20,
    policy_utility_loss_weight: float = 0.25,
    multi_step_loss_weight: float = 0.25,
    transition_ensemble_size: int = 0,
    batch_size: int = 128,
    learning_rate: float = 0.0003,
    transition: str = "gru",
    semantic_event_loss_weight: float = 0.20,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    trace_dir = trace_dir.resolve()
    manifest_path.relative_to(root)
    trace_dir.relative_to(root)
    manifest = load_manifest(manifest_path)
    verified_trace_root = verify_trace_manifest(manifest, base_dir=root)
    if verified_trace_root != trace_dir:
        raise ValueError("manifest source_root does not match the requested trace directory")
    train = _split_arrays(trace_dir, manifest, "train")
    dev = _split_arrays(trace_dir, manifest, "dev")
    effective_ensemble = int(transition_ensemble_size)
    if effective_ensemble <= 0:
        effective_ensemble = int(WorldModelConfig.from_env().transition_ensemble_size)
    checks = readiness_checks(
        manifest=manifest,
        train=train,
        dev=dev,
        epochs=epochs,
        warmup_fraction=warmup_fraction,
        policy_utility_loss_weight=policy_utility_loss_weight,
        multi_step_loss_weight=multi_step_loss_weight,
        transition_ensemble_size=effective_ensemble,
    )
    configuration = {
        "trainer": "world_model_v9_m2",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "transition": str(transition),
        "transition_ensemble_size": effective_ensemble,
        "multi_step_loss_weight": float(multi_step_loss_weight),
        "multi_step_warmup_fraction": float(warmup_fraction),
        "semantic_event_loss_weight": float(semantic_event_loss_weight),
        "policy_utility_loss_weight": float(policy_utility_loss_weight),
        "dataset_manifest": manifest_path.relative_to(root).as_posix(),
        "ball_log_enabled": False,
        "sealed_test_used": False,
    }
    checks["configuration_matches_frozen_m2_protocol"] = (
        configuration == FROZEN_TRAINING_CONFIGURATION
    )
    ready = all(checks.values())
    return {
        "schema_version": 1,
        "status": "ready_for_m2_training" if ready else "blocked_before_training",
        "ready": ready,
        "checks": checks,
        "manifest": manifest_path.relative_to(root).as_posix(),
        "trace_dir": trace_dir.relative_to(root).as_posix(),
        "manifest_summary": manifest.get("summary") or {},
        "training_split": train,
        "development_split": dev,
        "sealed_test": {
            "used_for_training_or_tuning": False,
            "rows_loaded_for_model_selection": 0,
            "support_read_from_frozen_manifest_only": True,
        },
        "configuration": configuration,
        "training_executed": False,
        "checkpoint_written": False,
        "next_action": (
            "run the frozen M2 training command"
            if ready else "resolve every failed preflight check before training"
        ),
        "frozen_training_command": (
            "python scripts/train_world_model.py --trace-dir "
            "data/world_model/traces --dataset-manifest "
            "data/world_model/dataset_manifest_formal_v8_candidate.json "
            "--epochs 45 --batch-size 128 --lr 0.0003 --transition gru "
            "--transition-ensemble-size 3 --multi-step-loss-weight 0.25 "
            "--multi-step-warmup-fraction 0.2 "
            "--semantic-event-loss-weight 0.2 "
            "--policy-utility-loss-weight 0.25 --out "
            "data/world_model/m2_outcome_aligned_candidate.pt --run-dir "
            "data/training/runs/m2_outcome_aligned_candidate"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--multi-step-warmup-fraction", type=float, default=0.20)
    parser.add_argument("--policy-utility-loss-weight", type=float, default=0.25)
    parser.add_argument("--multi-step-loss-weight", type=float, default=0.25)
    parser.add_argument("--transition-ensemble-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--transition", choices=("gru", "transformer"), default="gru")
    parser.add_argument("--semantic-event-loss-weight", type=float, default=0.20)
    args = parser.parse_args()
    report = preflight(
        manifest_path=args.manifest,
        trace_dir=args.trace_dir,
        epochs=args.epochs,
        warmup_fraction=args.multi_step_warmup_fraction,
        policy_utility_loss_weight=args.policy_utility_loss_weight,
        multi_step_loss_weight=args.multi_step_loss_weight,
        transition_ensemble_size=args.transition_ensemble_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        transition=args.transition,
        semantic_event_loss_weight=args.semantic_event_loss_weight,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
