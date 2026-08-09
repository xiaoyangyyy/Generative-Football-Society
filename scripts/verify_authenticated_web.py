#!/usr/bin/env python3
"""Verify authenticated Web policy over a real socket without external calls."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.web import create_product_web_server  # noqa: E402


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(
    opener, url: str, *, host: str, proto: str = "https",
    payload=None, csrf: str = "", cookie: str = "",
    forwarded_for: str = "198.51.100.20",
):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Host": host, "X-Forwarded-Proto": proto}
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for
    if body is not None:
        headers["Content-Type"] = "application/json"
    if csrf:
        headers["X-GFS-CSRF"] = csrf
    if cookie:
        headers["Cookie"] = cookie
    request = Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        response = opener.open(request, timeout=5)
    except HTTPError as exc:
        response = exc
    return response.status, dict(response.headers), response.read()


def verify_authenticated_web() -> dict:
    checks: dict[str, bool] = {}
    access_token = "verification-only-product-token-" + "x" * 32
    with tempfile.TemporaryDirectory(prefix="gfs-auth-web-") as temporary:
        server = create_product_web_server(
            temporary, host="127.0.0.1", port=0,
            allow_remote=True, access_token=access_token,
            allowed_hosts=("studio.example",),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        opener = build_opener(_NoRedirect())
        try:
            status, _, body = _request(
                opener, base + "/healthz", host="127.0.0.1", proto="http",
            )
            health = json.loads(body)
            checks["public_liveness_is_minimal"] = (
                status == 200 and health.get("status") == "ok"
                and health.get("external_calls_made") is False
            )

            status, _, _ = _request(
                opener, base + "/login", host="studio.example", proto="http",
            )
            checks["plaintext_proxy_rejected"] = status == 426
            status, _, _ = _request(
                opener, base + "/login", host="attacker.example",
            )
            checks["host_allowlist_enforced"] = status == 400

            status, headers, login_page = _request(
                opener, base + "/login", host="studio.example",
            )
            csrf = headers.get("X-GFS-CSRF-Token", "")
            checks["login_page_hardened"] = (
                status == 200 and bool(csrf)
                and bool(headers.get("Content-Security-Policy"))
                and b"API Key" in login_page
            )
            status, headers, _ = _request(
                opener, base + "/", host="studio.example",
            )
            checks["unauthenticated_root_redirects"] = (
                status == 302 and headers.get("Location") == "/login"
            )
            status, _, body = _request(
                opener, base + "/api/v1/login", host="studio.example",
                payload={"access_token": "wrong-token"}, csrf=csrf,
            )
            checks["invalid_token_rejected"] = (
                status == 401 and access_token.encode("utf-8") not in body
            )
            status, headers, body = _request(
                opener, base + "/api/v1/login", host="studio.example",
                payload={"access_token": access_token}, csrf=csrf,
            )
            cookie_header = headers.get("Set-Cookie", "")
            cookie = cookie_header.split(";", 1)[0]
            checks["secure_session_issued"] = (
                status == 200
                and all(value in cookie_header for value in (
                    "__Host-", "HttpOnly", "Secure", "SameSite=Strict", "Max-Age=",
                ))
                and access_token.encode("utf-8") not in body
            )
            status, second_headers, _ = _request(
                opener, base + "/api/v1/login", host="studio.example",
                payload={"access_token": access_token}, csrf=csrf,
            )
            second_cookie = second_headers.get("Set-Cookie", "").split(";", 1)[0]
            checks["independent_sessions_issued"] = (
                status == 200 and bool(second_cookie) and second_cookie != cookie
            )
            status, _, body = _request(
                opener, base + "/api/v1/studio", host="studio.example", cookie=cookie,
            )
            payload = json.loads(body)
            checks["authenticated_api_available"] = (
                status == 200
                and payload.get("access", {}).get("mode") == "remote_authenticated"
                and access_token not in json.dumps(payload)
            )
            attempts = []
            for _ in range(6):
                attempts.append(_request(
                    opener, base + "/api/v1/login", host="studio.example",
                    payload={"access_token": "wrong-token"}, csrf=csrf,
                ))
            limited_status, limited_headers, limited_body = attempts[-1]
            checks["login_rate_limit_enforced"] = (
                all(item[0] == 401 for item in attempts[:5])
                and limited_status == 429
                and int(limited_headers.get("Retry-After", "0")) >= 1
                and access_token.encode("utf-8") not in limited_body
            )
            logout_status, logout_headers, _ = _request(
                opener, base + "/api/v1/logout", host="studio.example",
                payload={}, csrf=csrf, cookie=cookie,
            )
            old_status, _, _ = _request(
                opener, base + "/api/v1/studio", host="studio.example", cookie=cookie,
            )
            other_status, _, _ = _request(
                opener, base + "/api/v1/studio", host="studio.example",
                cookie=second_cookie,
            )
            checks["single_session_logout_enforced"] = (
                logout_status == 200
                and "Max-Age=0" in logout_headers.get("Set-Cookie", "")
                and old_status == 401 and other_status == 200
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        checks["server_shutdown_cleanly"] = not thread.is_alive()

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_authenticated_web",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_authenticated_socket_boundary" if passed else "failed",
        "passed": passed,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "checks": checks,
        "limitations": [
            "the socket verifier uses loopback while exercising remote policy headers",
            "TLS is asserted at the trusted-proxy boundary and is not terminated by this WSGI server",
            "Docker/Compose is unavailable in the current environment, so no container was built",
            "no public DNS, certificate issuance, firewall, or penetration test was performed",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_authenticated_web()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
