#!/usr/bin/env python3
"""Replay historical anomaly fixtures with current stack; report anomaly rate."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.benchmark_core import run_micro_benchmark_rows
from src.match_engine.calibration.narrative_gate import evaluate_narrative_rows
from src.match_engine.tactical_catalog import TACTICAL_PRESETS


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12, help="Max fixtures to replay (default 12)")
    args = ap.parse_args()

    path = ROOT / "outputs" / "narrative_debug.jsonl"
    if not path.is_file():
        print(f"No log at {path}", file=sys.stderr)
        return 1
    flagged = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "match_result" and rec.get("anomalies"):
            flagged.append((rec["home"], rec["away"]))
    # dedupe preserve order
    seen = set()
    fixtures = []
    for h, a in flagged:
        key = (h, a)
        if key not in seen:
            seen.add(key)
            fixtures.append(key)
    fixtures = fixtures[: max(1, args.limit)]
    print(f"replay_fixtures={len(fixtures)} (from {len(flagged)} flagged records)")

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    tac = dict(TACTICAL_PRESETS["low_block"])
    spec = PIPELINE_PRESETS["M1"]
    with ablation_context(spec) as cfg:
        rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=fixtures,
            samples=1,
            match_seconds=5400.0,
            spec=spec,
            cfg=cfg,
            seed_start=42,
            tactical_override_home=tac,
            tactical_override_away=tac,
        )
    ev = evaluate_narrative_rows(rows)
    print(json.dumps(ev, indent=2))
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
