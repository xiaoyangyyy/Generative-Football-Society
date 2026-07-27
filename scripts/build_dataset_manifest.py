#!/usr/bin/env python3
"""Freeze world-model traces into group-safe train/dev/sealed-test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import build_trace_manifest, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default=str(ROOT / "data" / "world_model" / "traces"))
    parser.add_argument("--out", default=str(ROOT / "data" / "world_model" / "dataset_manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_trace_manifest(args.trace_dir, seed=args.seed)
    write_json_atomic(args.out, manifest)
    print(manifest["summary"])
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
