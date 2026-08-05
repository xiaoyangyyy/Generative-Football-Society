#!/usr/bin/env python3
"""Reproduce the phase-5 world-model and cognitive research-layer audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEST_FILES = (
    "tests/test_world_model_active_probe.py",
    "tests/test_world_model_active_probe_memory.py",
    "tests/test_world_model_active_probe_portfolio.py",
    "tests/test_world_model_active_probe_sequential_policy.py",
    "tests/test_world_model_predictive_mechanism.py",
    "tests/test_world_model_predictive_mechanism_chain.py",
    "tests/test_world_model_mechanism_stress_test.py",
    "tests/test_world_model_llm_fusion.py",
    "tests/test_world_model_online_calibration.py",
    "tests/test_fusion_audit.py",
    "tests/test_fusion_reliability.py",
)


def _last_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("validator emitted no JSON object")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="data/world_model/latent_wm_formal_v8_candidate.pt"
    )
    parser.add_argument(
        "--manifest", default="data/world_model/dataset_manifest_formal_v8_candidate.json"
    )
    parser.add_argument(
        "--out", default="data/evaluation/phase5_research_layer_validation_v1.json"
    )
    args = parser.parse_args()

    validate_cmd = [
        sys.executable, "scripts/validate_world_model.py",
        "--trace-dir", "data/world_model/traces",
        "--checkpoint", args.checkpoint,
        "--require-two-step-planning", "--require-semantic-event-heads",
    ]
    validation = subprocess.run(
        validate_cmd, cwd=ROOT, capture_output=True, text=True, errors="replace"
    )
    validation_payload = _last_json(validation.stdout)

    test_cmd = [sys.executable, "-m", "pytest", *TEST_FILES, "-q"]
    tests = subprocess.run(
        test_cmd, cwd=ROOT, capture_output=True, text=True, errors="replace"
    )
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    cognitive_root = ROOT / "data/persistence/cognitive_log"
    cognitive_logs = sorted(cognitive_root.glob("*.jsonl")) if cognitive_root.exists() else []
    fusion_path = ROOT / "data/persistence/world_model_fusion.jsonl"
    fusion_records = 0
    if fusion_path.exists():
        fusion_records = sum(1 for line in fusion_path.read_text(encoding="utf-8").splitlines() if line.strip())

    semantic = validation_payload.get("semantic_event_head_gates", {})
    active_semantic = {
        depth: sorted(name for name, gate in gates.items() if gate.get("active"))
        for depth, gates in semantic.items()
    }
    report = {
        "schema_version": 1,
        "phase": "world_model_mechanism_chain_and_llm_active_probe_revalidation",
        "dataset": {
            "manifest": args.manifest,
            "summary": manifest.get("summary", {}),
            "group_safe_split": True,
        },
        "world_model_candidate": {
            "checkpoint": args.checkpoint,
            "strict_command": " ".join(validate_cmd[1:]),
            "strict_exit_code": validation.returncode,
            "accepted": bool(validation_payload.get("ok")),
            "checkpoint_version": validation_payload.get("checkpoint_version"),
            "transition_quality": validation_payload.get("transition_quality"),
            "transition_ensemble_trained": validation_payload.get("transition_ensemble_trained"),
            "transition_ensemble_size": validation_payload.get("transition_ensemble_size"),
            "two_step_planning_gate": validation_payload.get("two_step_planning_gate"),
            "semantic_event_heads_ready": validation_payload.get("semantic_event_heads_ready"),
            "active_semantic_events": active_semantic,
            "pass_planner_active": validation_payload.get("pass_planner_active"),
            "shot_planner_active": validation_payload.get("shot_planner_active"),
            "decision": "reject_keep_stable_checkpoint" if validation.returncode else "eligible_for_separate_promotion_review",
        },
        "mechanism_and_active_probe_contracts": {
            "command": " ".join(test_cmd[1:]),
            "exit_code": tests.returncode,
            "passed": tests.returncode == 0,
            "summary": tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else "",
            "test_files": list(TEST_FILES),
        },
        "live_llm_evidence": {
            "cognitive_log_files": len(cognitive_logs),
            "fusion_records": fusion_records,
            "evaluable": bool(cognitive_logs and fusion_records),
            "status": "available" if cognitive_logs and fusion_records else "not_evaluable_no_real_provider_logs",
            "claim_boundary": "contract correctness only; no live-provider benefit claim without prospective logs",
        },
        "phase_complete": tests.returncode == 0,
        "production_promotion_ready": bool(
            validation.returncode == 0 and tests.returncode == 0 and cognitive_logs and fusion_records
        ),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "phase_complete": report["phase_complete"],
        "candidate_accepted": report["world_model_candidate"]["accepted"],
        "contracts_passed": report["mechanism_and_active_probe_contracts"]["passed"],
        "live_llm_evaluable": report["live_llm_evidence"]["evaluable"],
        "output": args.out,
    }, indent=2))
    return 0 if report["phase_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
