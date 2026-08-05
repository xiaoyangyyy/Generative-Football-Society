#!/usr/bin/env python3
"""Merge recoverable fixture shards into one frozen external benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_sim_vs_open_data import _bootstrap_mean_intervals
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES
from src.match_engine.calibration.contract import (
    evaluate_rows,
    load_contract,
    load_statsbomb_baselines,
)


def merge_reports(payloads: list[dict], *, seed: int = 42) -> dict:
    rows = [row for payload in payloads for row in payload.get("raw_rows", [])]
    fixtures = sorted({str(row.get("fixture", "")) for row in rows})
    expected = sorted(f"{home}_vs_{away}" for home, away in DEFAULT_FIXTURES)
    if fixtures != expected:
        raise ValueError(f"fixture coverage mismatch: {fixtures} != {expected}")
    counts = {fixture: sum(row.get("fixture") == fixture for row in rows)
              for fixture in fixtures}
    if set(counts.values()) != {3}:
        raise ValueError(f"expected exactly three seeds per fixture: {counts}")
    evaluation = evaluate_rows(
        rows, contract=load_contract(), baselines=load_statsbomb_baselines(),
    )
    return {
        "schema_version": 1,
        "protocol": "six_fixtures_three_seeds_full_90min",
        "pipeline": "M0",
        "fixtures": fixtures,
        "fixture_samples": counts,
        "samples_total": len(rows),
        "uncertainty": {
            "method": "deterministic_match_bootstrap",
            "draws": 5000,
            "seed": seed,
            "metrics": _bootstrap_mean_intervals(rows, seed=seed, draws=5000),
        },
        "report": evaluation["hard_metrics"],
        "soft_constraints": evaluation["soft_constraints"],
        "failed_metrics": evaluation["failed_hard"],
        "failed_soft": evaluation["failed_soft"],
        "all_pass": evaluation["all_pass"],
        "all_hard_pass": evaluation["all_hard_pass"],
        "all_soft_pass": evaluation["all_soft_pass"],
        "raw_rows": rows,
    }


def _load_payload(path: Path) -> dict:
    """Load a JSON shard or recover JSON following deterministic runner logs."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON payload in {path}")
    return json.loads(text[start:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+")
    parser.add_argument(
        "--out", default="data/evaluation/m0_formal_baseline_v1.json",
    )
    args = parser.parse_args()
    payloads = [_load_payload(Path(path)) for path in args.shards]
    report = merge_reports(payloads)
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "samples_total": report["samples_total"],
        "all_pass": report["all_pass"],
        "failed_metrics": report["failed_metrics"],
        "failed_soft": report["failed_soft"],
    }, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
