#!/usr/bin/env python3
"""Prove the staged research plan from current repository evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _run(command: list[str]) -> dict:
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors="replace")
    return {
        "command": " ".join(command),
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "tail": (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [],
    }


def calibration_is_noop(plan: dict) -> bool:
    return plan.get("action") == "no_op_freeze_parameters"


def build_report(*, run_tests: bool) -> dict:
    m0 = _load("data/evaluation/m0_formal_baseline_v1.json")
    calibration = _load("data/evaluation/m0_calibration_plan_v1.json")
    ablation = _load("data/evaluation/formal_ablation_summary_v1.json")
    phase5 = _load("data/evaluation/phase5_research_layer_validation_v1.json")
    policy = _load("data/evaluation/module_policy_v1.json")
    commands = {
        "v7_artifacts": _run([sys.executable, "scripts/verify_v7_release.py", "--artifacts-only"]),
        "release_boundary": _run([sys.executable, "scripts/audit_release_boundary.py"]),
        "research_evidence": _run([sys.executable, "scripts/audit_research_evidence.py", "--check"]),
    }
    if run_tests:
        commands["pytest"] = _run([sys.executable, "-m", "pytest", "-q"])
    stages = {
        "1_formal_m0": {
            "complete": m0.get("samples_total") == 18 and bool(m0.get("all_pass")),
            "evidence": "data/evaluation/m0_formal_baseline_v1.json",
        },
        "2_calibration_non_regression": {
            "complete": calibration_is_noop(calibration),
            "evidence": "data/evaluation/m0_calibration_plan_v1.json",
        },
        "3_calibration": {
            "complete": calibration_is_noop(calibration),
            "outcome": "no parameter change required because formal M0 passed",
        },
        "4_formal_ablation": {
            "complete": bool(ablation.get("complete")) and len(ablation.get("variants", {})) == 7,
            "evidence": "data/evaluation/formal_ablation_summary_v1.json",
            "policy": policy.get("policy_id"),
        },
        "5_research_layers": {
            "complete": bool(phase5.get("phase_complete")),
            "candidate_accepted": phase5["world_model_candidate"]["accepted"],
            "contracts_passed": phase5["mechanism_and_active_probe_contracts"]["passed"],
            "live_llm_evaluable": phase5["live_llm_evidence"]["evaluable"],
            "evidence": "data/evaluation/phase5_research_layer_validation_v1.json",
        },
        "6_release_candidate_validation": {
            "complete": all(item["passed"] for item in commands.values()),
            "commands": commands,
        },
    }
    return {
        "schema_version": 1,
        "objective": "complete_the_frozen_research_plan_in_stages",
        "all_stages_complete": all(stage["complete"] for stage in stages.values()),
        "production_promotion_ready": False,
        "promotion_decision": "keep_deployed_v7_and_research_layers_default_off",
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/evaluation/staged_completion_v1.json")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()
    report = build_report(run_tests=not args.skip_pytest)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_stages_complete": report["all_stages_complete"], "output": args.out}, indent=2))
    return 0 if report["all_stages_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
