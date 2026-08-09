#!/usr/bin/env python3
"""Verify the local Web Beta over a real HTTP socket without running a match."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.web import create_product_web_server  # noqa: E402


def _request(url: str, *, payload=None, csrf: str | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if csrf:
        headers["X-GFS-CSRF"] = csrf
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        response = exc
    body = response.read()
    return response.status, dict(response.headers), body


def verify_web_beta() -> dict:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="gfs-web-beta-") as temporary:
        server = create_product_web_server(temporary, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            status, headers, root = _request(base + "/")
            text = root.decode("utf-8")
            csrf = headers.get("X-GFS-CSRF-Token", "")
            checks["root_http_200"] = status == 200
            checks["content_security_policy"] = bool(headers.get("Content-Security-Policy"))
            checks["accessibility_landmarks"] = all(marker in text for marker in (
                'class="skip"', 'id="main"', 'aria-live="polite"',
                "prefers-reduced-motion",
            ))

            status, _, body = _request(base + "/healthz")
            health = json.loads(body)
            checks["liveness_http_200"] = status == 200 and health.get("status") == "ok"
            checks["zero_external_calls"] = health.get("external_calls_made") is False

            request = Request(base + "/api/v1/studio", headers={"Host": "attacker.example"})
            try:
                response = urlopen(request, timeout=5)
            except HTTPError as exc:
                response = exc
            checks["dns_rebinding_host_rejected"] = response.status == 400
            response.read()

            status, _, _ = _request(base + "/api/v1/studio", payload={
                "name": "Verification Studio", "mode": "stable", "seed": 42,
            })
            checks["csrf_rejects_missing_token"] = status == 403

            status, _, body = _request(
                base + "/api/v1/studio",
                payload={"name": "Verification Studio", "mode": "stable", "seed": 42},
                csrf=csrf,
            )
            created = json.loads(body)
            checks["studio_created_over_http"] = (
                status == 201 and created.get("studio", {}).get("name") == "Verification Studio"
            )

            status, _, body = _request(base + "/api/v1/studio")
            observed = json.loads(body)
            checks["persisted_status_over_http"] = (
                status == 200 and observed.get("configured") is True
            )

            status, _, body = _request(base + "/readyz")
            readiness = json.loads(body)
            checks["readiness_gate_is_honest"] = (
                status == 503 and readiness.get("ready") is False
                and bool(readiness.get("blockers"))
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        checks["server_shutdown_cleanly"] = not thread.is_alive()

    try:
        create_product_web_server(Path.cwd(), host="0.0.0.0")
    except ValueError:
        checks["remote_binding_rejected"] = True
    else:
        checks["remote_binding_rejected"] = False

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_product_web_beta",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_local_beta" if passed else "failed",
        "passed": passed,
        "external_calls_made": False,
        "matches_executed": 0,
        "checks": checks,
        "limitations": [
            "local_loopback verification is not a production deployment",
            "no 100-match soak has been executed",
            "accessibility landmarks are automated checks, not an external WCAG audit",
            "no user-value study has been conducted",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_web_beta()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
