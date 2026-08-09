#!/usr/bin/env python3
"""Verify Studio backup and restore behavior in an isolated temporary workspace."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.recovery import ProductRecovery  # noqa: E402
from src.product.tasks import ProductTaskQueue  # noqa: E402


def _fixture(root: Path) -> tuple[Path, ...]:
    report = root / "outputs/studio/recovery/matches/0001.json"
    dashboard = report.with_suffix(".html")
    ball_log = root / "outputs/ball_log/0001.jsonl"
    cognitive = root / "data/persistence/cognitive_log/studio_0001.json"
    artifacts = {
        "ball_log": ball_log.relative_to(root).as_posix(),
        "cognitive_log": cognitive.relative_to(root).as_posix(),
    }
    contents = {
        report: json.dumps({"artifacts": artifacts}),
        dashboard: "<html><body>recovery verification</body></html>",
        ball_log: '{"event":"kickoff"}\n',
        cognitive: '{"successful_calls":1}\n',
    }
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    session = root / "data/persistence/product_session.json"
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text(json.dumps({
        "schema_version": 1,
        "product": "Generative Football Society Studio",
        "name": "Recovery Verification",
        "slug": "recovery",
        "mode": "cognitive",
        "seed": 42,
        "matches": [{
            "match_id": "0001",
            "report": report.relative_to(root).as_posix(),
            "dashboard": dashboard.relative_to(root).as_posix(),
        }],
    }), encoding="utf-8")
    return session, report, dashboard, ball_log, cognitive


def verify_product_recovery() -> dict:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="gfs-recovery-") as temporary:
        root = Path(temporary)
        paths = _fixture(root)
        original = {path: path.read_bytes() for path in paths}
        bundle = root / "backups/studio.zip"
        recovery = ProductRecovery(root)
        created = recovery.create_backup(bundle)
        checks["backup_created"] = bundle.is_file() and created.get("valid") is True
        checks["all_referenced_files_included"] = created.get("file_count") == len(paths)
        checks["final_bundle_path_reported"] = created.get("bundle") == str(bundle.resolve())

        verified = ProductRecovery.verify_backup(bundle)
        checks["integrity_verification_passed"] = (
            verified.get("valid") is True
            and verified.get("studio", {}).get("name") == "Recovery Verification"
        )

        paths[0].write_text(paths[0].read_text(encoding="utf-8").replace(
            "Recovery Verification", "Current Session",
        ), encoding="utf-8")
        try:
            recovery.restore_backup(bundle)
        except FileExistsError:
            checks["implicit_replacement_rejected"] = True
        else:
            checks["implicit_replacement_rejected"] = False

        queue = ProductTaskQueue(root)
        old_task, _ = queue.submit_match("France", "Spain", fast=True)
        claimed = queue.claim_next("completed-before-restore")
        queue.complete(claimed["task_id"], "completed-before-restore", {
            "match_id": "old-session-match",
        })
        restored = recovery.restore_backup(bundle, replace=True)
        checks["explicit_restore_roundtrip"] = (
            {path: path.read_bytes() for path in paths} == original
        )
        checks["cross_session_task_history_reset"] = (
            restored.get("task_history_reset") is True
            and queue.list_tasks() == []
            and old_task["task_id"] not in json.dumps(queue.list_tasks())
        )

        tampered = root / "backups/tampered.zip"
        tampered.write_bytes(bundle.read_bytes())
        with zipfile.ZipFile(tampered, "a") as archive:
            archive.writestr("outputs/studio/untracked.html", "untracked")
        try:
            ProductRecovery.verify_backup(tampered)
        except ValueError:
            checks["untracked_member_rejected"] = True
        else:
            checks["untracked_member_rejected"] = False

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_product_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_local_recovery" if passed else "failed",
        "passed": passed,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "checks": checks,
        "limitations": [
            "local temporary-workspace verification is not an operational deployment drill",
            "backup integrity hashes detect corruption but do not authenticate an untrusted bundle",
            "recovery time and recovery point objectives are not yet measured on production storage",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_product_recovery()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
