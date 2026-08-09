"""Exercise persistent product operations without matches, training, or network calls."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.tasks import ProductTaskQueue  # noqa: E402


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def verify(*, task_count: int = 100) -> dict:
    if isinstance(task_count, bool) or not isinstance(task_count, int) or not 10 <= task_count <= 500:
        raise ValueError("task_count must be between 10 and 500")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gfs-operations-soak-") as temporary:
        root = Path(temporary)
        queue = ProductTaskQueue(root)
        observed: list[tuple[dict, bool]] = []
        failures: list[str] = []
        observed_lock = threading.Lock()

        def submit(index: int) -> None:
            try:
                result = queue.submit_match(
                    f"Soak Home {index}", f"Soak Away {index}", fast=True,
                    idempotency_key=f"soak-key-{index}",
                )
                with observed_lock:
                    observed.append(result)
            except Exception as exc:
                with observed_lock:
                    failures.append(type(exc).__name__)

        threads = [
            threading.Thread(target=submit, args=(index,))
            for index in range(task_count) for _duplicate in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        threads_stopped = all(not thread.is_alive() for thread in threads)

        completed_target = task_count - 5
        failed_target = 3
        for index in range(task_count):
            task = queue.claim_next("operations-verifier")
            if task is None:
                failures.append("MissingClaim")
                break
            if index < completed_target:
                queue.complete(
                    task["task_id"], "operations-verifier",
                    {"verification_index": index},
                )
            elif index < completed_target + failed_target:
                queue.fail(task["task_id"], "operations-verifier", "SyntheticFailure")
        restarted = ProductTaskQueue(root)
        recovered = restarted.recover_running(reason="verification_restart")
        tasks = restarted.list_tasks(limit=500)
        ids = [task["task_id"] for task in tasks]
        states = Counter(task["state"] for task in tasks)
        created_count = sum(created for _task, created in observed)
        replay_count = len(observed) - created_count
        telemetry = restarted.telemetry.snapshot()
        raw_telemetry = restarted.telemetry.path.read_text(encoding="utf-8")
        forbidden_values_absent = all(
            value not in raw_telemetry
            for value in ("Soak Home", "Soak Away", "soak-key-")
        )
        checks = {
            "all_submission_threads_stopped": threads_stopped,
            "no_submission_exceptions": not failures,
            "exactly_one_task_per_idempotency_key": (
                len(tasks) == task_count and len(set(ids)) == task_count
            ),
            "all_duplicates_resolved_to_same_task": (
                len(observed) == task_count * 2
                and created_count == task_count and replay_count == task_count
            ),
            "terminal_state_distribution_exact": states == Counter({
                "completed": completed_target,
                "failed": failed_target,
                "interrupted": 2,
            }),
            "restart_recovered_exactly_two_running_tasks": recovered == 2,
            "no_queued_or_running_tasks_remain": not ({"queued", "running"} & set(states)),
            "telemetry_has_no_corrupt_records": (
                telemetry["retention"]["corrupt_records"] == 0
            ),
            "telemetry_has_no_write_failures": (
                telemetry["retention"]["write_failures_since_start"] == 0
            ),
            "telemetry_task_counts_exact": (
                telemetry["tasks"]["new_submissions"] == task_count
                and telemetry["tasks"]["idempotent_replays"] == task_count
                and telemetry["tasks"]["task_claimed"] == task_count
                and telemetry["tasks"]["task_completed"] == completed_target
                and telemetry["tasks"]["task_failed"] == failed_target
                and telemetry["tasks"]["recovered_tasks"] == 2
            ),
            "fixture_names_and_idempotency_keys_absent_from_telemetry": (
                forbidden_values_absent
            ),
        }
        passed = all(checks.values())
        return {
            "schema_version": 1,
            "verification": "gfs_product_operations_control_plane",
            "status": (
                "passed_control_plane_soak_no_match_execution"
                if passed else "failed_control_plane_soak"
            ),
            "passed": passed,
            "task_count": task_count,
            "submission_attempts": task_count * 2,
            "state_counts": dict(sorted(states.items())),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "checks": checks,
            "external_calls_made": False,
            "matches_executed": 0,
            "training_executed": False,
            "formal_experiment_executed": False,
            "claim_boundary": {
                "closes_100_match_soak_gate": False,
                "proves": [
                    "concurrent idempotent queue submission",
                    "persistent terminal task transitions",
                    "restart interruption recovery",
                    "privacy-bounded telemetry consistency",
                ],
                "does_not_prove": [
                    "100 match-engine executions",
                    "long-duration deployment stability",
                    "container, TLS, or external network behavior",
                    "scientific model quality or paper claims",
                ],
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "data/evaluation/product_operations_verification_v1.json",
    )
    args = parser.parse_args()
    report = verify(task_count=args.tasks)
    _write_report(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
