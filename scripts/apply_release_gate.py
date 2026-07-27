#!/usr/bin/env python3
"""Bind sealed-test evidence to a checkpoint and disable failed capabilities."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload.setdefault("meta", {})
    validation = meta.setdefault("validation", {})
    gates = report.get("gates", {})
    if not gates.get("pass_balanced_accuracy", False):
        validation["pass_planner_quality"] = 0.0
    if not gates.get("shot_evidence", False):
        validation["shot_planner_quality"] = 0.0
    if not gates.get("one_step_mse", False) or not gates.get("ten_step_mse", False):
        validation["transition_quality"] = 0.0
    validation["planner_quality"] = min(
        float(validation.get("pass_planner_quality", 0.0)),
        float(validation.get("shot_planner_quality", 0.0)),
    )
    meta["sealed_release_gate"] = report
    fd, temporary = tempfile.mkstemp(dir=checkpoint.parent, prefix=f".{checkpoint.name}-", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    print(json.dumps({"checkpoint": str(checkpoint), "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
