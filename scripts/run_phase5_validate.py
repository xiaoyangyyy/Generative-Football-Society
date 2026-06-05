#!/usr/bin/env python3
"""Phase 5 validate — M0 gate prerequisite + M1/C1 layer gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 5 narrative isolation validate.")
    p.add_argument("--quick", action="store_true", help="Short layer gate runs")
    p.add_argument("--skip-m0-gate", action="store_true", help="Assume M0 gate already passed")
    args = p.parse_args()

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    if not args.skip_m0_gate:
        gate_path = ROOT / "reports" / "calibration_gate.json"
        if not gate_path.is_file() or not json.loads(gate_path.read_text()).get("all_pass"):
            print("[Phase5] Running M0 calibration gate...", file=sys.stderr)
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_calibration_gate.py"), "--pipeline", "M0"],
                cwd=str(ROOT),
                env=env,
            )
            if r.returncode != 0:
                return r.returncode

    print("[Phase5] M1/C1 layer gates...", file=sys.stderr)
    layer_args = [sys.executable, str(ROOT / "scripts" / "run_layer_gate.py")]
    if args.quick:
        layer_args.append("--quick")
    r = subprocess.run(layer_args, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        return r.returncode

    print("\nPHASE 5 COMPLETE — L0 isolated; narrative layer gates passed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
