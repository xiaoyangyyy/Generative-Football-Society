#!/usr/bin/env python3
"""Freeze the formal decision for the sealed-accepted calibrated M1 candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_formal_ablation_results import METRICS, paired_intervals


def _keyed(rows: list[dict]) -> dict[tuple[str, int], dict]:
    result, counts = {}, {}
    for row in rows:
        fixture = str(row["fixture"])
        index = int(row.get("sample_index", counts.get(fixture, 0)))
        counts[fixture] = counts.get(fixture, 0) + 1
        key = (fixture, index)
        if key in result:
            raise ValueError(f"duplicate formal sample: {key}")
        result[key] = row
    return result


def build_decision(baseline: dict, candidate: dict) -> dict:
    if candidate.get("variant") != "M1" or candidate.get("samples_total") != 18:
        raise ValueError("candidate is not a complete formal M1 report")
    provenance = candidate.get("candidate") or {}
    digest = str(provenance.get("checkpoint_sha256") or "")
    if (
        not provenance.get("explicit_checkpoint")
        or not provenance.get("checkpoint")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("formal M1 candidate lacks checkpoint identity provenance")
    base, cand = _keyed(baseline["raw_rows"]), _keyed(candidate["raw_rows"])
    if set(base) != set(cand):
        raise ValueError("candidate samples do not match frozen M0")
    deltas = {
        metric: [float(cand[key][metric]) - float(base[key][metric]) for key in sorted(base)]
        for metric in METRICS
    }
    changed = {
        metric: {
            "changed_samples": sum(abs(value) > 1e-7 for value in values),
            "max_absolute_delta": max(map(abs, values)),
            "mean_delta": sum(values) / len(values),
        }
        for metric, values in deltas.items()
    }
    resolved = paired_intervals(baseline["raw_rows"], candidate["raw_rows"], draws=2000)
    interval = resolved["delta_loss"]
    behavior_changed = any(row["changed_samples"] for row in changed.values())
    promotion_supported = bool(
        candidate.get("all_pass") and interval["ci95_high"] < 0 and behavior_changed
    )
    return {
        "schema_version": 1,
        "review": "sealed_calibrated_world_model_formal_m1",
        "protocol": candidate["protocol"],
        "baseline": "data/evaluation/m0_formal_baseline_v1.json",
        "candidate_report": "data/evaluation/formal_ablation/M1_calibrated.json",
        "candidate": provenance,
        "samples_total": candidate["samples_total"],
        "all_pass": candidate["all_pass"],
        "loss": candidate["loss"],
        "baseline_loss": candidate["baseline_loss"],
        "observed_delta_loss": candidate["delta_loss"],
        "paired_uncertainty": resolved,
        "behavior_changed_above_tolerance": behavior_changed,
        "metric_differences": changed,
        "runtime": candidate.get("runtime") or {},
        "promotion_supported": promotion_supported,
        "decision": (
            "promotion_candidate_pending_final_release_review"
            if promotion_supported else "research_only_default_off"
        ),
        "decision_reason": (
            "formal external loss improves with a paired 95% interval below zero and behavior changes"
            if promotion_supported else
            "sealed prediction skill did not produce a measurable formal behavior or external-loss improvement"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/evaluation/m0_formal_baseline_v1.json")
    parser.add_argument("--candidate", default="data/evaluation/formal_ablation/M1_calibrated.json")
    parser.add_argument("--out", default="data/evaluation/formal_ablation/M1_calibrated_decision.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    baseline = json.loads((ROOT / args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads((ROOT / args.candidate).read_text(encoding="utf-8"))
    decision = build_decision(baseline, candidate)
    output = ROOT / args.out
    if args.check:
        if json.loads(output.read_text(encoding="utf-8")) != decision:
            raise SystemExit(f"stale calibrated M1 decision: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision["decision"],
        "promotion_supported": decision["promotion_supported"],
        "behavior_changed": decision["behavior_changed_above_tolerance"],
        "elapsed_seconds": decision["runtime"].get("elapsed_seconds"),
        "output": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
