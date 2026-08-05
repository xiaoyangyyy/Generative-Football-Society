#!/usr/bin/env python3
"""Audit the frozen football evidence boundary without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_release_manifest import verify_artifacts


def _read(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sum_rows(manifest: dict[str, Any], key: str) -> int:
    return sum(int(row.get(key, 0)) for row in manifest.get("matches", []))


def build_report() -> dict[str, Any]:
    freeze = _read("data/evaluation/research_freeze_v1.json")
    policy = _read("data/evaluation/module_policy_v1.json")
    baseline = _read("data/calibration/statsbomb_match_baselines.json")
    current = _read("data/releases/current.json")
    active_manifest = _read(current["manifest"])
    release_failures, release_normalized = verify_artifacts(active_manifest)
    shot = _read("reports/acceptance/shot_v7.json")
    passing = _read("reports/acceptance/joint_pass_lodo_v7.json")
    skillcorner = _read("data/external/skillcorner/derived/manifest.json")
    statsbomb = _read("data/external/statsbomb360/derived/manifest.json")
    sportec = _read("data/external/sportec/derived/manifest.json")
    temporal = _read("data/frame_world/v94_temporal_providers/manifest.json")
    ablation = _read("data/evaluation/ablation_diagnostic_v1.json")

    metric_names = set(baseline.get("metrics", {}))
    frozen_metrics = set(freeze["frozen_external_metrics"])
    external_gate = bool(
        baseline.get("n_team_matches", 0) >= 128
        and frozen_metrics <= metric_names
        and len(frozen_metrics) >= 15
    )

    completion = {
        provider: fold.get("completion")
        for provider, fold in passing.get("folds", {}).items()
        if fold.get("completion") is not None
    }
    sufficient_pass_providers = sorted(
        provider for provider, row in completion.items()
        if row.get("evidence_sufficient")
        and int(row.get("positives", 0)) >= 100
        and int(row.get("negatives", 0)) >= 100
    )
    pass_gate = len(sufficient_pass_providers) >= 2
    shot_gate = bool(
        shot.get("candidate_ready")
        and shot.get("test", {}).get("shots", 0) >= 1000
        and shot.get("test", {}).get("goals", 0) >= 100
        and shot.get("test", {}).get("brier", 1.0)
        < shot.get("test", {}).get("baseline_brier", 0.0)
    )

    temporal_by_provider: dict[str, dict[str, int]] = {}
    for row in temporal.get("matches", []):
        bucket = temporal_by_provider.setdefault(
            str(row["provider"]), {"matches": 0, "events": 0, "observed": 0},
        )
        bucket["matches"] += 1
        bucket["events"] += int(row.get("events", 0))
        bucket["observed"] += int(row.get("observed", 0))
    qualified_temporal_providers = sorted(
        provider for provider, row in temporal_by_provider.items()
        if row["matches"] >= 5 and row["observed"] >= 500
    )
    continuous_gate = len(qualified_temporal_providers) >= 2

    report = {
        "schema_version": 1,
        "freeze_id": freeze["freeze_id"],
        "project_identity": freeze["primary_identity"],
        "release_boundary": {
            "active": current["active"],
            "expected": freeze["stable_simulator_release"],
            "artifact_count": len(active_manifest.get("artifacts", [])),
            "normalized_text_artifact_count": len(release_normalized),
            "artifact_failures": release_failures,
            "valid": bool(
                current["active"] == freeze["stable_simulator_release"]
                and not release_failures
            ),
        },
        "external_benchmark": {
            "source": baseline.get("source"),
            "team_matches": int(baseline.get("n_team_matches", 0)),
            "metric_count": len(metric_names),
            "frozen_metric_count": len(frozen_metrics),
            "gate_pass": external_gate,
        },
        "data_coverage": {
            "skillcorner": {
                "matches": len(skillcorner.get("matches", [])),
                "passes": _sum_rows(skillcorner, "passes"),
                "shots": _sum_rows(skillcorner, "shots"),
                "sampled_tracking_frames": _sum_rows(skillcorner, "sampled_frames"),
            },
            "statsbomb360": {
                "matches": len(statsbomb.get("matches", [])),
                "passes": _sum_rows(statsbomb, "passes"),
                "shots": _sum_rows(statsbomb, "shots"),
            },
            "sportec": {
                "matches": len(sportec.get("matches", [])),
                "passes": _sum_rows(sportec, "passes"),
                "shots": _sum_rows(sportec, "shots"),
                "tracking_frames": _sum_rows(sportec, "tracking_frames"),
            },
            "pass_completion_lodo": completion,
            "sufficient_pass_providers": sufficient_pass_providers,
            "shot_holdout": shot.get("test", {}),
            "continuous_state": temporal_by_provider,
            "qualified_continuous_state_providers": qualified_temporal_providers,
        },
        "gates": {
            "external_benchmark_frozen": external_gate,
            "failed_pass_labels_cross_provider": pass_gate,
            "shot_and_goal_holdout": shot_gate,
            "continuous_state_cross_provider": continuous_gate,
            "stable_release_pointer": bool(
                current["active"] == freeze["stable_simulator_release"]
                and not release_failures
            ),
        },
        "module_policy": policy["modules"],
        "ablation": ablation,
    }
    report["research_data_ready"] = all(report["gates"].values())
    report["production_promotion_ready"] = False
    report["production_blockers"] = [
        "full matched-seed external-calibration ablation is not frozen",
        "M0 pure micro baseline does not yet pass the external observable contract",
        "advanced world-model and LLM layers have no demonstrated external simulation lift",
        "diagnostic ablation is not a full multi-fixture promotion experiment",
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="data/evaluation/research_evidence_v1.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_report()
    output = ROOT / args.out
    if args.check:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit(f"stale evidence report: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "report": str(output.relative_to(ROOT)),
        "research_data_ready": report["research_data_ready"],
        "production_promotion_ready": report["production_promotion_ready"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
