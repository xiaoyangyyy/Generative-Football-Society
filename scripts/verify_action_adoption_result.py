#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import action_adoption_study as study  # noqa: E402
from src.infrastructure import file_sha256  # noqa: E402


def _artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def verify(root: Path = ROOT) -> dict:
    protocol_path = root / "data/evaluation/action_adoption_protocol_v1.json"
    protocol = study._read_json(protocol_path)
    progress = study._read_json(root / protocol["outputs"]["progress"])
    decision = study._read_json(root / protocol["outputs"]["decision"])
    identity = study.execution_identity(protocol_path, protocol, root)
    replay = study.analyze(protocol, progress, expected_identity=identity)
    recorded = dict(decision)
    recorded.pop("exports", None)
    if recorded != replay:
        raise ValueError("recorded action-adoption decision does not replay")
    exports = decision.get("exports") or dict()
    csv_record = exports.get("rows_csv") or dict()
    summary_record = exports.get("summary_markdown") or dict()
    csv_path = _artifact(root, str(csv_record.get("path") or ""))
    summary_path = _artifact(root, str(summary_record.get("path") or ""))
    for path, record in ((csv_path, csv_record), (summary_path, summary_record)):
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise ValueError("action-adoption export identity mismatch")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    arm_counts = dict(
        (arm, sum(row.get("arm") == arm for row in rows)) for arm in study.ARM_IDS
    )
    if len(rows) != 24 or any(arm_counts[arm] != 12 for arm in study.ARM_IDS):
        raise ValueError("action-adoption CSV violates fixed paired budget")
    summary = summary_path.read_text(encoding="utf-8")
    if decision["status"] not in summary or "does not authorize product" not in summary:
        raise ValueError("action-adoption summary omits claim boundary")
    return dict(
        schema_version=1,
        verification="gfs_action_adoption_result",
        passed=True,
        status=decision["status"],
        runs_executed=len(rows),
        arm_counts=arm_counts,
        execution_identity_verified=True,
        decision_replayed=True,
        exports_verified=True,
        training_executed=decision["training_executed"],
        provider_calls_made=decision["provider_calls_made"],
        promotion_authorized=decision["promotion_authorized"],
    )


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
