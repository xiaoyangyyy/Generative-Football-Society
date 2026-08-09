#!/usr/bin/env python3
"""Verify persistent product tasks without executing the football engine."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.tasks import (  # noqa: E402
    BackgroundMatchWorker, ProductTaskQueue, TaskConflict,
)


def verify_product_tasks() -> dict:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="gfs-tasks-") as temporary:
        root = Path(temporary)
        report = root / "outputs/studio/verify/matches/0001.json"
        dashboard = report.with_suffix(".html")
        report.parent.mkdir(parents=True)
        report.write_text("{}\n", encoding="utf-8")
        dashboard.write_text("<html></html>\n", encoding="utf-8")

        class VerificationWorkspace:
            def run_match(self, home, away, *, fast):
                return {
                    "match_id": "verification-0001",
                    "fixture": {"home": home, "away": away, "fast": fast},
                    "result": {"score": {"home": 1, "away": 0}},
                    "integrity": {"accepted": True, "state": "accepted", "blockers": []},
                    "report_path": str(report), "dashboard_path": str(dashboard),
                }

        queue = ProductTaskQueue(root)
        submissions = []

        def submit():
            submissions.append(queue.submit_match(
                "Brazil", "Argentina", fast=True,
                idempotency_key="verification-request",
            ))

        threads = [threading.Thread(target=submit) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        task_ids = {item[0]["task_id"] for item in submissions}
        checks["concurrent_idempotency"] = (
            len(submissions) == 6 and len(task_ids) == 1
            and sum(item[1] for item in submissions) == 1
        )
        try:
            queue.submit_match(
                "France", "Spain", fast=True,
                idempotency_key="verification-request",
            )
        except TaskConflict:
            checks["idempotency_payload_conflict_rejected"] = True
        else:
            checks["idempotency_payload_conflict_rejected"] = False
        task_id = next(iter(task_ids))
        worker = BackgroundMatchWorker(
            queue, workspace_loader=lambda _root: VerificationWorkspace(),
        )
        checks["worker_claimed_task"] = worker.run_once()
        completed = queue.get_task(task_id)
        checks["terminal_result_persisted"] = (
            completed.get("state") == "completed"
            and completed.get("result", {}).get("dashboard")
            == "outputs/studio/verify/matches/0001.html"
        )
        checks["public_state_redacts_internal_identity"] = (
            "idempotency_hash" not in completed and "worker_id" not in completed
            and str(root) not in json.dumps(completed)
        )

        interrupted, _ = queue.submit_match("France", "Spain", fast=True)
        queue.claim_next("simulated-crashed-worker")
        checks["startup_recovery_visible"] = (
            queue.recover_running() == 1
            and queue.get_task(interrupted["task_id"])["state"] == "interrupted"
        )
        checks["queue_persisted_on_disk"] = queue.path.is_file()

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_product_background_tasks",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_isolated_task_lifecycle" if passed else "failed",
        "passed": passed,
        "external_calls_made": False,
        "real_matches_executed": 0,
        "training_executed": False,
        "checks": checks,
        "limitations": [
            "the worker lifecycle uses an injected deterministic workspace, not a real match",
            "process-kill recovery is modeled by persisted running-state reconciliation",
            "multi-host task execution is out of scope for the loopback Beta",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_product_tasks()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
