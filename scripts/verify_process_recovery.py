#!/usr/bin/env python3
"""Kill and restart a real Studio Web process without executing a match."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import FileLease  # noqa: E402
from src.product.tasks import ProductTaskQueue  # noqa: E402
from src.product.web import create_product_web_server  # noqa: E402
from src.product.workspace import _atomic_json  # noqa: E402


PROCESS_READY_TIMEOUT_SECONDS = 45.0
LOCAL_RECOVERY_OBJECTIVE_SECONDS = 15.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _request(opener, url: str, *, method: str = "GET", payload=None, csrf: str = ""):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if csrf:
        headers["X-GFS-CSRF"] = csrf
    try:
        response = opener.open(
            Request(url, data=body, headers=headers, method=method), timeout=10,
        )
    except HTTPError as exc:
        response = exc
    raw = response.read()
    parsed = (
        json.loads(raw)
        if response.headers.get_content_type() == "application/json"
        else None
    )
    return response.status, dict(response.headers), parsed


def _child(
    root: Path,
    ready_file: Path,
    *,
    seed_running_task: bool,
    observe_task_id: str,
) -> int:
    server = create_product_web_server(root, port=0)
    worker_stopped_for_drill = False
    seeded_task: dict | None = None
    if seed_running_task:
        worker_stopped_for_drill = server.product_worker.stop(timeout=5)
        if not worker_stopped_for_drill:
            server.server_close()
            return 3
        queue = ProductTaskQueue(root)
        submitted, _ = queue.submit_match(
            "Process Drill Home", "Process Drill Away", fast=True,
            idempotency_key="process-recovery-drill",
        )
        claimed = queue.claim_next("process-recovery-drill-owner")
        if claimed is None or claimed["task_id"] != submitted["task_id"]:
            server.server_close()
            return 4
        seeded_task = queue.get_task(submitted["task_id"])

    observed_task = None
    if observe_task_id:
        observed_task = ProductTaskQueue(root).get_task(observe_task_id)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _atomic_json(ready_file, {
        "schema_version": 1,
        "pid": os.getpid(),
        "port": server.server_port,
        "worker_alive": server.product_worker.is_alive,
        "worker_stopped_for_drill": worker_stopped_for_drill,
        "seeded_task": seeded_task,
        "observed_task": observed_task,
    })
    try:
        command = sys.stdin.readline().strip()
        return 0 if command == "stop" else 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "DEEPSEEK_API_KEY",
        "API_KEY",
        "OPENAI_API_KEY",
        "GFS_WEB_ACCESS_TOKEN",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(ROOT)
    return environment


def _start_child(
    root: Path,
    ready_file: Path,
    *,
    seed_running_task: bool = False,
    observe_task_id: str = "",
) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-root",
        str(root),
        "--ready-file",
        str(ready_file),
    ]
    if seed_running_task:
        command.append("--seed-running-task")
    if observe_task_id:
        command.extend(("--observe-task-id", observe_task_id))
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=_child_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=creationflags,
    )


def _wait_ready(
    process: subprocess.Popen,
    ready_file: Path,
    timeout: float = PROCESS_READY_TIMEOUT_SECONDS,
) -> dict:
    started = time.monotonic()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_file.is_file():
            return json.loads(ready_file.read_text(encoding="utf-8"))
        if process.poll() is not None:
            raise RuntimeError(
                f"drill child exited before readiness with code {process.returncode}"
            )
        time.sleep(0.05)
    elapsed = time.monotonic() - started
    state = "running" if process.poll() is None else f"exited:{process.returncode}"
    raise TimeoutError(
        "drill child did not become ready "
        f"within {timeout:.1f}s (elapsed={elapsed:.3f}s, pid={process.pid}, "
        f"state={state}, ready_file={ready_file})"
    )


def _stop_child(process: subprocess.Popen, timeout: float = 15) -> bool:
    if process.poll() is not None:
        return process.returncode == 0
    assert process.stdin is not None
    process.stdin.write("stop\n")
    process.stdin.flush()
    try:
        return process.wait(timeout=timeout) == 0
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        return False


def verify_process_recovery() -> dict:
    checks: dict[str, bool] = {}
    recovery_seconds = 0.0
    first: subprocess.Popen | None = None
    second: subprocess.Popen | None = None
    with tempfile.TemporaryDirectory(prefix="gfs-process-recovery-") as temporary:
        root = Path(temporary)
        first_ready_path = root / "drill-first-ready.json"
        second_ready_path = root / "drill-second-ready.json"
        first = _start_child(root, first_ready_path, seed_running_task=True)
        try:
            first_ready = _wait_ready(first, first_ready_path)
            first_base = f"http://127.0.0.1:{first_ready['port']}"
            opener = build_opener()
            status, headers, _ = _request(opener, first_base + "/")
            csrf = headers.get("X-GFS-CSRF-Token", "")
            checks["first_process_served_real_http"] = status == 200 and bool(csrf)
            seeded_task = first_ready.get("seeded_task") or {}
            task_id = str(seeded_task.get("task_id") or "")
            checks["synthetic_task_claimed_without_match_worker"] = (
                first_ready.get("worker_stopped_for_drill") is True
                and first_ready.get("worker_alive") is False
                and seeded_task.get("state") == "running"
                and bool(task_id)
            )

            status, _, studio = _request(
                opener,
                first_base + "/api/v1/studio",
                method="POST",
                payload={"name": "Process Recovery Drill", "mode": "stable", "seed": 42},
                csrf=csrf,
            )
            checks["studio_committed_before_kill"] = (
                status == 201 and studio.get("configured") is True
            )
            status, _, backup = _request(
                opener,
                first_base + "/api/v1/backups",
                method="POST",
                payload={},
                csrf=csrf,
            )
            backup_id = str((backup or {}).get("backup_id") or "")
            checks["managed_backup_committed_before_kill"] = (
                status == 201 and backup.get("valid") is True and bool(backup_id)
            )

            session_path = root / "data/persistence/product_session.json"
            backup_path = root / "backups/studio" / f"{backup_id}.zip"
            before_hashes = {
                "session": _sha256(session_path),
                "backup": _sha256(backup_path),
            }
            checks["server_lease_excludes_second_process"] = FileLease.is_held(
                root / "data/persistence/product_web.lock"
            )
            try:
                duplicate = create_product_web_server(root, port=0)
            except RuntimeError:
                duplicate_rejected = True
            else:
                duplicate_rejected = False
                duplicate.server_close()
            checks["duplicate_server_rejected_cross_process"] = duplicate_rejected

            recovery_started = time.monotonic()
            first.kill()
            first.wait(timeout=10)
            checks["owned_child_was_force_killed"] = first.returncode not in (None, 0)
            checks["os_lease_released_after_process_death"] = not FileLease.is_held(
                root / "data/persistence/product_web.lock"
            )
            after_kill_task = ProductTaskQueue(root).get_task(task_id)
            checks["running_task_survived_process_death"] = (
                after_kill_task.get("state") == "running"
            )
            checks["committed_session_and_backup_survived_kill"] = (
                _sha256(session_path) == before_hashes["session"]
                and _sha256(backup_path) == before_hashes["backup"]
            )

            second = _start_child(
                root,
                second_ready_path,
                observe_task_id=task_id,
            )
            second_ready = _wait_ready(second, second_ready_path)
            second_base = f"http://127.0.0.1:{second_ready['port']}"
            status, _, health = _request(build_opener(), second_base + "/healthz")
            recovery_seconds = time.monotonic() - recovery_started
            checks["service_recovered_on_second_process"] = (
                status == 200
                and health.get("status") == "ok"
                and health.get("background_worker_alive") is True
                and second_ready.get("pid") != first_ready.get("pid")
            )
            checks["local_recovery_time_measured"] = recovery_seconds > 0
            observed = second_ready.get("observed_task") or {}
            checks["orphaned_running_task_reconciled"] = (
                observed.get("task_id") == task_id
                and observed.get("state") == "interrupted"
                and observed.get("reason") == "web_worker_restarted"
            )

            second_opener = build_opener()
            status, second_headers, _ = _request(second_opener, second_base + "/")
            second_csrf = second_headers.get("X-GFS-CSRF-Token", "")
            status, _, restored_studio = _request(
                second_opener, second_base + "/api/v1/studio"
            )
            checks["studio_readable_after_restart"] = (
                status == 200
                and restored_studio.get("configured") is True
                and restored_studio.get("studio", {}).get("name")
                == "Process Recovery Drill"
            )
            status, _, verified = _request(
                second_opener,
                f"{second_base}/api/v1/backups/{backup_id}/verify",
                method="POST",
                payload={},
                csrf=second_csrf,
            )
            checks["managed_backup_verified_after_restart"] = (
                status == 200 and verified.get("valid") is True
            )
            checks["zero_loss_for_committed_drill_files"] = (
                _sha256(session_path) == before_hashes["session"]
                and _sha256(backup_path) == before_hashes["backup"]
                and ProductTaskQueue(root).get_task(task_id).get("state")
                == "interrupted"
            )
            session_payload = json.loads(session_path.read_text(encoding="utf-8"))
            match_output = root / "outputs/studio"
            checks["no_match_transaction_or_artifact_created"] = (
                session_payload.get("matches") == []
                and session_payload.get("runs") == []
                and (
                    not match_output.exists()
                    or not any(path.is_file() for path in match_output.rglob("*"))
                )
            )
            checks["no_partial_temporary_files_remain"] = not any(
                path.is_file() and path.suffix == ".tmp"
                for path in root.rglob("*")
            )
            checks["second_process_stopped_cleanly"] = _stop_child(second)
            second = None
            checks["lease_released_after_clean_shutdown"] = not FileLease.is_held(
                root / "data/persistence/product_web.lock"
            )
        finally:
            for process in (second, first):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_cross_process_recovery_drill",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_local_process_kill_recovery" if passed else "failed",
        "passed": passed,
        "recovery_time_seconds": round(recovery_seconds, 3),
        "recovery_objective_seconds": LOCAL_RECOVERY_OBJECTIVE_SECONDS,
        "recovery_objective_met": (
            0 < recovery_seconds <= LOCAL_RECOVERY_OBJECTIVE_SECONDS
        ),
        "readiness_timeout_seconds": PROCESS_READY_TIMEOUT_SECONDS,
        "production_rto_authorized": False,
        "recovery_point": "zero_loss_for_files_committed_before_kill" if passed else "unproven",
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "checks": checks,
        "limitations": [
            "the drill uses a temporary local filesystem, not a deployment volume or container orchestrator",
            "the forced kill occurs between committed requests, not during backup, restore, or filesystem replacement",
            "the running task is synthetic and claimed only after the match worker is stopped",
            "the measured local recovery time is evidence for this run, not a production RTO or SLA",
            "the 15-second local objective is reported separately and is not a functional recovery gate",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed-running-task", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--observe-task-id", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_root is not None:
        if args.ready_file is None:
            parser.error("--ready-file is required in child mode")
        return _child(
            args.child_root,
            args.ready_file,
            seed_running_task=args.seed_running_task,
            observe_task_id=args.observe_task_id,
        )
    report = verify_process_recovery()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
