#!/usr/bin/env python3
"""Freeze targeted rare-shot traces without changing the existing sealed set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from src.data_engine.dataset_registry import (
    augment_training_manifest,
    load_manifest,
    verify_trace_manifest,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-manifest",
        default=str(ROOT / "data/world_model/dataset_manifest_formal_v8_candidate.json"),
    )
    parser.add_argument("--trace-dir", default=str(ROOT / "data/world_model/traces"))
    parser.add_argument("--include-token", default="rare_shot_v10")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data/world_model/dataset_manifest_rare_shot_v10.json"),
    )
    args = parser.parse_args()
    base = load_manifest(args.base_manifest)
    verify_trace_manifest(base, base_dir=ROOT)
    manifest = augment_training_manifest(
        base, args.trace_dir, include_token=args.include_token,
    )
    verify_trace_manifest(manifest, base_dir=ROOT)
    write_json_atomic(args.out, manifest)
    print(manifest["summary"])
    print(manifest["augmentation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
