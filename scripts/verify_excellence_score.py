#!/usr/bin/env python3
"""Verify evidence-derived product and academic maturity without running compute."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import file_sha256  # noqa: E402
from src.product.control_plane import ProductControlPlane  # noqa: E402
from src.product.completion_plan import STEPS  # noqa: E402
from src.product.excellence import validate_scoring_contract  # noqa: E402

CONTRACT_PATH = ROOT / "data/evaluation/excellence_scoring_contract_v1.json"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def verify_excellence_score(root: Path = ROOT) -> dict:
    contract_path = root / "data/evaluation/excellence_scoring_contract_v1.json"
    roadmap_path = root / "data/evaluation/excellence_roadmap_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    roadmap = json.loads(roadmap_path.read_text(encoding="utf-8"))
    contract_checks = validate_scoring_contract(contract)
    release = ProductControlPlane(root).release_readiness()
    excellence = release.get("excellence") or {}
    tracks = excellence.get("tracks") or {}
    gates = {row["id"]: row["passed"] for row in release.get("gates") or []}
    referenced = {
        gate
        for rows in contract.get("tracks", {}).values()
        for row in rows
        for gate in row.get("completion_requires") or []
    }
    scores = {name: row.get("score") for name, row in tracks.items()}
    all_full = excellence.get("all_tracks_full_maturity") is True
    completion_plan = release.get("completion_plan") or {}
    plan_steps = completion_plan.get("steps") or []
    checks = {
        "scoring_contract_is_valid": all(contract_checks.values()),
        "all_required_gate_ids_exist": referenced <= set(gates),
        "control_plane_uses_current_contract": (
            excellence.get("contract_id") == contract.get("contract_id")
            and excellence.get("scoring_contract")
            == "data/evaluation/excellence_scoring_contract_v1.json"
        ),
        "roadmap_points_are_declared_non_authoritative": (
            roadmap.get("dynamic_scoring_contract")
            == "data/evaluation/excellence_scoring_contract_v1.json"
            and roadmap.get("dynamic_score_verification")
            == "data/evaluation/excellence_score_verification_v1.json"
        ),
        "scores_equal_category_point_sums": (
            set(tracks) == {"product", "academic"}
            and all(
                row.get("score")
                == sum(category["points"] for category in row.get("categories") or [])
                for row in tracks.values()
            )
        ),
        "scores_are_bounded_and_have_100_maximum": all(
            0 <= int(row.get("score", -1)) <= row.get("maximum") == 100
            for row in tracks.values()
        ),
        "full_maturity_equivalence_is_exact": (
            all_full
            == all(
                row.get("score") == row.get("maximum") == 100
                for row in tracks.values()
            )
        ),
        "release_ready_cannot_precede_full_maturity": (
            release.get("release_ready") is not True or all_full
        ),
        "current_scores_are_not_read_from_roadmap_points": (
            excellence.get("contract_checks") == contract_checks
        ),
        "completion_plan_is_dependency_aware_and_zero_execution": (
            completion_plan.get("zero_execution_plan") is True
            and {row.get("gate_id") for row in plan_steps}
            == {step.gate_id for step in STEPS}
            and completion_plan.get("open_step_count")
            == sum(row.get("passed") is not True for row in plan_steps)
            and all(
                isinstance(row.get("commands"), list)
                and row.get("commands")
                and isinstance(row.get("required_inputs"), list)
                and row.get("required_inputs")
                and isinstance(row.get("blocking_gate_ids"), list)
                for row in plan_steps
            )
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_evidence_derived_excellence_score",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_dynamic_scoring" if passed else "failed",
        "passed": passed,
        "scores": scores,
        "all_tracks_full_maturity": all_full,
        "release_ready": release.get("release_ready") is True,
        "passed_gate_count": release.get("passed_gate_count"),
        "open_gate_count": release.get("open_gate_count"),
        "tracks": tracks,
        "checks": checks,
        "artifact_sha256": {
            "data/evaluation/excellence_scoring_contract_v1.json": file_sha256(
                contract_path
            ),
            "data/evaluation/excellence_roadmap_v1.json": file_sha256(roadmap_path),
            "src/product/excellence.py": file_sha256(
                root / "src/product/excellence.py"
            ),
            "src/product/control_plane.py": file_sha256(
                root / "src/product/control_plane.py"
            ),
            "src/product/completion_plan.py": file_sha256(
                root / "src/product/completion_plan.py"
            ),
            "src/cli.py": file_sha256(root / "src/cli.py"),
            "src/product/web.py": file_sha256(root / "src/product/web.py"),
            "scripts/verify_excellence_score.py": file_sha256(Path(__file__)),
        },
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_excellence_score()
    if args.out is not None:
        _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
