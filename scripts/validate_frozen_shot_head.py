#!/usr/bin/env python3
"""Evaluate a frozen shot-head candidate once on sealed groups and promote safely."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_frozen_shot_head import _shot_features
from src.data_engine.dataset_registry import (
    file_sha256,
    load_manifest,
    verify_trace_manifest,
    write_json_atomic,
)
from src.match_engine.world_model.shot_head import FrozenShotHead


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trace-dir", default=str(ROOT / "data/world_model/traces"))
    parser.add_argument(
        "--report", default=str(ROOT / "data/evaluation/frozen_shot_head_decision.json"),
    )
    parser.add_argument(
        "--promoted", default=str(ROOT / "data/world_model/frozen_shot_head_v1.json"),
    )
    args = parser.parse_args()
    from src.match_engine.world_model.inference import WorldModelRuntime

    manifest = load_manifest(args.manifest)
    verify_trace_manifest(manifest, base_dir=ROOT)
    candidate_payload = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    head = FrozenShotHead.from_dict(candidate_payload)
    runtime = WorldModelRuntime.load(args.checkpoint)
    if head.base_checkpoint_sha256 != runtime.checkpoint_signature:
        raise ValueError("candidate and validation checkpoint identities differ")
    training = head.evidence.get("training") or {}
    if training.get("manifest_sha256") != file_sha256(Path(args.manifest)):
        raise ValueError("candidate and validation manifest identities differ")

    features, truth, prior = _shot_features(
        runtime, args.trace_dir, manifest, "sealed_test",
    )
    probabilities = np.asarray([
        head.predict(row[: head.latent_dim], row[head.latent_dim :])
        for row in features
    ])
    sealed = {
        "samples": int(len(truth)),
        "goals": int(truth.sum()),
        "model_brier": float(np.mean((probabilities - truth) ** 2)),
        "physics_xg_prior_brier": float(np.mean((prior - truth) ** 2)),
        "sealed_only": True,
        "groups": int(manifest["summary"]["sealed_test"]["groups"]),
    }
    gates = {
        "samples": sealed["samples"] >= 40,
        "goals": sealed["goals"] >= 5,
        "proper_score_gain": sealed["model_brier"] < sealed["physics_xg_prior_brier"],
        "backbone_frozen": training.get("backbone_frozen") is True,
        "sealed_not_used_for_training": training.get("sealed_test_used") is False,
    }
    accepted = all(gates.values())
    promoted = {
        **candidate_payload,
        "evidence": {**dict(head.evidence), "sealed_test": sealed},
        "accepted": accepted,
    }
    report = {
        "schema_version": 1,
        "decision": "promote_optional_shot_head" if accepted else "retain_physics_xg_fallback",
        "accepted": accepted,
        "gates": gates,
        "sealed_test": sealed,
        "candidate": str(Path(args.candidate)),
        "candidate_sha256": file_sha256(Path(args.candidate)),
        "checkpoint_signature": runtime.checkpoint_signature,
        "promotion_artifact": str(Path(args.promoted)) if accepted else None,
        "promotion_artifact_sha256": None,
    }
    if accepted:
        write_json_atomic(args.promoted, promoted)
        report["promotion_artifact_sha256"] = file_sha256(Path(args.promoted))
    write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
