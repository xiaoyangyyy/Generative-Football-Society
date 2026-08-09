#!/usr/bin/env python3
"""Verify the integrated Web recovery center over a local socket."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener
from wsgiref.simple_server import make_server

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.web import ProductWebApp  # noqa: E402


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
    parsed = json.loads(raw) if response.headers.get_content_type() == "application/json" else None
    return response.status, dict(response.headers), raw, parsed


def verify_web_recovery() -> dict:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="gfs-web-recovery-") as temporary:
        root = Path(temporary)
        app = ProductWebApp(root)
        server = make_server("127.0.0.1", 0, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        opener = build_opener()
        try:
            status, headers, document, _ = _request(opener, base + "/")
            csrf = headers.get("X-GFS-CSRF-Token", "")
            checks["accessible_recovery_ui_present"] = (
                status == 200 and bool(csrf)
                and b'id="backup-list"' in document
                and b'id="restore-form"' in document
                and b'type="file"' not in document
            )
            status, _, _, studio = _request(
                opener, base + "/api/v1/studio", method="POST",
                payload={"name": "Recovery Socket", "mode": "stable", "seed": 42},
                csrf=csrf,
            )
            checks["studio_created"] = status == 201 and studio.get("configured") is True
            status, _, _, empty = _request(opener, base + "/api/v1/recovery")
            checks["empty_catalog_available"] = status == 200 and empty.get("backups") == []

            status, _, _, denied = _request(
                opener, base + "/api/v1/backups", method="POST", payload={},
            )
            checks["backup_requires_csrf"] = (
                status == 403 and denied.get("error", {}).get("code") == "csrf_rejected"
            )
            status, _, _, created = _request(
                opener, base + "/api/v1/backups", method="POST", payload={}, csrf=csrf,
            )
            backup_id = str(created.get("backup_id") or "")
            checks["backup_created_and_verified"] = (
                status == 201 and created.get("valid") is True and bool(backup_id)
            )
            checks["server_path_not_exposed"] = str(root) not in json.dumps(created)
            status, _, _, catalog = _request(opener, base + "/api/v1/recovery")
            checks["catalog_contains_managed_id"] = (
                status == 200 and catalog.get("count") == 1
                and catalog["backups"][0]["backup_id"] == backup_id
                and str(root) not in json.dumps(catalog)
            )
            status, _, _, verified = _request(
                opener, f"{base}/api/v1/backups/{backup_id}/verify",
                method="POST", payload={}, csrf=csrf,
            )
            checks["explicit_verify_passed"] = status == 200 and verified.get("valid") is True

            session = root / "data/persistence/product_session.json"
            session.write_text(session.read_text(encoding="utf-8").replace(
                "Recovery Socket", "Changed Session",
            ), encoding="utf-8")
            status, _, _, rejected = _request(
                opener, f"{base}/api/v1/backups/{backup_id}/restore", method="POST",
                payload={"confirmation": "错误确认", "replace": True}, csrf=csrf,
            )
            checks["wrong_confirmation_is_non_mutating"] = (
                status == 422
                and rejected.get("error", {}).get("code") == "restore_confirmation_required"
                and "Changed Session" in session.read_text(encoding="utf-8")
            )

            task, _ = app.task_queue.submit_match("Home", "Away", fast=True)
            status, _, _, blocked = _request(
                opener, f"{base}/api/v1/backups/{backup_id}/restore", method="POST",
                payload={"confirmation": backup_id, "replace": True}, csrf=csrf,
            )
            checks["active_task_blocks_restore"] = (
                status == 409 and blocked.get("error", {}).get("code") == "restore_blocked"
            )
            claimed = app.task_queue.claim_next("recovery-verifier")
            app.task_queue.complete(claimed["task_id"], "recovery-verifier", {"ok": True})
            status, _, _, restored = _request(
                opener, f"{base}/api/v1/backups/{backup_id}/restore", method="POST",
                payload={"confirmation": backup_id, "replace": True}, csrf=csrf,
            )
            checks["transactional_restore_completed"] = (
                status == 200 and restored.get("restored") is True
                and "Recovery Socket" in session.read_text(encoding="utf-8")
            )
            checks["cross_session_task_history_reset"] = (
                restored.get("task_history_reset") is True
                and app.task_queue.list_tasks() == []
                and task["task_id"] not in json.dumps(app.task_queue.list_tasks())
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        checks["server_shutdown_cleanly"] = not thread.is_alive()

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_integrated_web_recovery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_integrated_local_web_recovery" if passed else "failed",
        "passed": passed,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "checks": checks,
        "limitations": [
            "the verifier uses a temporary local socket and filesystem",
            "backup and restore run synchronously in a threaded Web request",
            "no deployment-volume RPO/RTO drill or process-kill fault was performed",
            "managed backups remain on deployment storage and are not downloaded by the Web API",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_web_recovery()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
