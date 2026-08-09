"""Dependency-free local Web Beta over the canonical product workspace."""

from __future__ import annotations

import html
import json
import logging
import re
import secrets
import threading
import time
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Iterable
from urllib.parse import unquote
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from src.infrastructure import FileLease, LeaseUnavailable
from src.product.control_plane import ProductControlPlane
from src.product.recovery import ProductRecovery
from src.product.tasks import BackgroundMatchWorker, ProductTaskQueue, TaskConflict
from src.product.telemetry import ProductTelemetry, route_template
from src.product.web_security import WebAccessPolicy, is_loopback_host
from src.product.workspace import ProductWorkspace, StudioConfig
from src.simulation.llm_gateway import provider_preflight


LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 64 * 1024
JSON_HEADERS = [("Content-Type", "application/json; charset=utf-8")]
STATUS_TEXT = {
    200: "OK", 201: "Created", 202: "Accepted", 302: "Found",
    400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
    404: "Not Found", 405: "Method Not Allowed", 409: "Conflict",
    413: "Payload Too Large", 415: "Unsupported Media Type",
    422: "Unprocessable Entity", 500: "Internal Server Error",
    426: "Upgrade Required", 429: "Too Many Requests",
    503: "Service Unavailable",
}


class WebRequestError(Exception):
    def __init__(
        self, status: int, code: str, message: str,
        *, headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _is_loopback_host(host: str) -> bool:
    return is_loopback_host(host)


class ProductWebApp:
    """Small WSGI adapter; all domain decisions stay in ProductWorkspace."""

    def __init__(
        self, root: str | Path, *, task_queue: ProductTaskQueue | None = None,
        task_worker: BackgroundMatchWorker | None = None,
        access_policy: WebAccessPolicy | None = None,
        telemetry: ProductTelemetry | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.csrf_token = secrets.token_urlsafe(32)
        self._mutation_lock = threading.Lock()
        self.telemetry = telemetry or (
            task_queue.telemetry if task_queue is not None else ProductTelemetry(self.root)
        )
        self.task_queue = task_queue or ProductTaskQueue(
            self.root, telemetry=self.telemetry,
        )
        self.task_queue.telemetry = self.telemetry
        self.task_worker = task_worker
        self.access_policy = access_policy or WebAccessPolicy()

    def __call__(
        self, environ: dict[str, Any],
        start_response: Callable[[str, list[tuple[str, str]]], Any],
    ) -> Iterable[bytes]:
        request_id = secrets.token_hex(8)
        started = time.perf_counter()
        error_code: str | None = None
        try:
            status, headers, body = self._dispatch(environ)
        except WebRequestError as exc:
            status = exc.status
            error_code = exc.code
            headers = list(JSON_HEADERS) + list(exc.headers)
            body = _json_bytes({
                "error": {"code": exc.code, "message": exc.message},
                "request_id": request_id,
            })
        except Exception:
            LOGGER.exception("Unhandled Web Beta request error request_id=%s", request_id)
            status = 500
            error_code = "internal_error"
            headers = list(JSON_HEADERS)
            body = _json_bytes({
                "error": {
                    "code": "internal_error",
                    "message": "The request failed. Use the request ID in local logs.",
                },
                "request_id": request_id,
            })
        common = [
            ("Cache-Control", "no-store"),
            ("Referrer-Policy", "no-referrer"),
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("X-Request-ID", request_id),
            ("Content-Length", str(len(body))),
        ]
        start_response(f"{status} {STATUS_TEXT[status]}", headers + common)
        observed_method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        if observed_method not in {
            "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD",
        }:
            observed_method = "OTHER"
        self.telemetry.try_record(
            "web_request", request_id=request_id, method=observed_method,
            route=route_template(str(environ.get("PATH_INFO", "/"))),
            status=status, duration_ms=(time.perf_counter() - started) * 1000,
            error_code=error_code,
        )
        return [body]

    def _dispatch(
        self, environ: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        if method == "GET" and path == "/healthz":
            return self._json_response(200, {
                "schema_version": 1,
                "status": "ok",
                "service": "gfs-product-web",
                "external_calls_made": False,
                "background_worker_alive": (
                    self.task_worker.is_alive if self.task_worker is not None else None
                ),
            })
        if not self.access_policy.host_allowed(environ):
            raise WebRequestError(400, "invalid_host", "Host is not allowed")
        if not self.access_policy.secure_transport(environ):
            raise WebRequestError(
                426, "https_required", "Remote Web access requires the trusted HTTPS proxy",
            )
        if method == "GET" and path == "/login":
            if not self.access_policy.remote:
                return self._redirect_response("/")
            return self._login_html_response()
        if method == "POST" and path == "/api/v1/login":
            if not self.access_policy.remote:
                raise WebRequestError(404, "not_found", "Resource not found")
            self._require_csrf(environ)
            return self._login(environ, self._read_json(environ))
        if method == "POST" and path == "/api/v1/logout":
            if not self.access_policy.remote:
                raise WebRequestError(404, "not_found", "Resource not found")
            self._require_csrf(environ)
            revoked = self.access_policy.revoke_session(environ)
            return self._json_response(200, {
                "schema_version": 1, "authenticated": False, "revoked": revoked,
            }, headers=[
                ("Set-Cookie", self.access_policy.clear_session_cookie_header()),
            ])
        if not self.access_policy.authenticated(environ):
            if method == "GET" and path == "/":
                return self._redirect_response("/login")
            raise WebRequestError(401, "authentication_required", "Authentication required")
        if method == "GET" and path == "/":
            return self._html_response()
        if method == "GET" and path == "/readyz":
            payload = self._studio_status()
            ready = bool(payload.get("configured") and (
                payload.get("studio", {}).get("readiness", {}).get("ready")
            ))
            worker_ready = self.task_worker is None or self.task_worker.is_alive
            operations = payload["operations"]
            telemetry_ready = (
                operations["retention"]["corrupt_records"] == 0
                and operations["retention"]["write_failures_since_start"] == 0
            )
            ready = ready and worker_ready and telemetry_ready
            blockers = list(
                payload.get("studio", {}).get("readiness", {}).get("blockers", []),
            )
            if not worker_ready:
                blockers.append("background_worker_not_alive")
            if not telemetry_ready:
                blockers.append("product_telemetry_degraded")
            return self._json_response(200 if ready else 503, {
                "schema_version": 1, "ready": ready,
                "configured": payload["configured"],
                "blockers": blockers, "telemetry_ready": telemetry_ready,
            })
        if method == "GET" and path == "/api/v1/studio":
            return self._json_response(200, self._studio_status(), csrf=True)
        if method == "GET" and path == "/api/v1/operations":
            return self._json_response(200, self.telemetry.snapshot())
        if method == "GET" and path == "/api/v1/recovery":
            return self._recovery_status()
        if method == "POST" and path == "/api/v1/studio":
            self._require_csrf(environ)
            return self._create_studio(self._read_json(environ))
        if method == "POST" and path == "/api/v1/matches":
            self._require_csrf(environ)
            return self._queue_match(environ, self._read_json(environ))
        if method == "POST" and path == "/api/v1/backups":
            self._require_csrf(environ)
            return self._create_managed_backup()
        backup_action = re.fullmatch(
            r"/api/v1/backups/([^/]+)/(verify|restore)", path,
        )
        if method == "POST" and backup_action:
            self._require_csrf(environ)
            backup_id, action = backup_action.groups()
            if action == "verify":
                return self._verify_managed_backup(backup_id)
            return self._restore_managed_backup(
                backup_id, self._read_json(environ),
            )
        if method == "GET" and path == "/api/v1/tasks":
            return self._json_response(200, {
                "schema_version": 1,
                "tasks": [self._task_for_web(task) for task in self.task_queue.list_tasks()],
            })
        if method == "GET" and path.startswith("/api/v1/tasks/"):
            task_id = path.removeprefix("/api/v1/tasks/")
            if "/" in task_id:
                raise WebRequestError(404, "task_not_found", "Task not found")
            try:
                task = self.task_queue.get_task(task_id)
            except (FileNotFoundError, ValueError) as exc:
                raise WebRequestError(404, "task_not_found", "Task not found") from exc
            return self._json_response(200, {"task": self._task_for_web(task)})
        if method == "GET" and path.startswith("/artifacts/"):
            return self._artifact_response(path.removeprefix("/artifacts/"))
        if path in {
            "/api/v1/studio", "/api/v1/matches", "/api/v1/tasks",
            "/api/v1/recovery", "/api/v1/backups",
        } or backup_action:
            raise WebRequestError(405, "method_not_allowed", "Method not allowed")
        raise WebRequestError(404, "not_found", "Resource not found")

    def _json_response(
        self, status: int, payload: Any, *, csrf: bool = False,
        headers: list[tuple[str, str]] | None = None,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        response_headers = list(JSON_HEADERS) + list(headers or ())
        if csrf:
            response_headers.append(("X-GFS-CSRF-Token", self.csrf_token))
        return status, response_headers, _json_bytes(payload)

    def _html_response(self) -> tuple[int, list[tuple[str, str]], bytes]:
        nonce = secrets.token_urlsafe(18)
        document = _INDEX_HTML.replace("__NONCE__", html.escape(nonce, quote=True))
        document = document.replace(
            "__CSRF__", html.escape(self.csrf_token, quote=True),
        )
        document = document.replace(
            "__REMOTE_LOGOUT_HIDDEN__", "" if self.access_policy.remote else " hidden",
        )
        headers = [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Security-Policy", (
                "default-src 'none'; connect-src 'self'; img-src 'self' data:; "
                f"script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
            )),
            ("X-GFS-CSRF-Token", self.csrf_token),
        ]
        return 200, headers, document.encode("utf-8")

    def _login_html_response(self) -> tuple[int, list[tuple[str, str]], bytes]:
        nonce = secrets.token_urlsafe(18)
        document = _LOGIN_HTML.replace("__NONCE__", html.escape(nonce, quote=True))
        document = document.replace(
            "__CSRF__", html.escape(self.csrf_token, quote=True),
        )
        return 200, [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Security-Policy", (
                "default-src 'none'; connect-src 'self'; "
                f"script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; "
                "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
            )),
            ("X-GFS-CSRF-Token", self.csrf_token),
        ], document.encode("utf-8")

    @staticmethod
    def _redirect_response(location: str):
        return 302, [("Location", location), ("Content-Type", "text/plain; charset=utf-8")], b"Redirecting"

    def _login(
        self, environ: dict[str, Any], payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        candidate = payload.get("access_token")
        observed = candidate if isinstance(candidate, str) else ""
        decision, retry_after = self.access_policy.authenticate_login(environ, observed)
        if decision == "rate_limited":
            raise WebRequestError(
                429, "login_rate_limited", "Too many login attempts; try again later",
                headers=(("Retry-After", str(retry_after)),),
            )
        if decision != "accepted":
            raise WebRequestError(401, "invalid_credentials", "Invalid credentials")
        headers = list(JSON_HEADERS)
        headers.append(("Set-Cookie", self.access_policy.session_cookie_header()))
        return 200, headers, _json_bytes({
            "schema_version": 1, "authenticated": True,
        })

    def _read_json(self, environ: dict[str, Any]) -> dict[str, Any]:
        media_type = str(environ.get("CONTENT_TYPE", "")).split(";", 1)[0].strip()
        if media_type != "application/json":
            raise WebRequestError(
                415, "json_required", "Content-Type must be application/json",
            )
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except (TypeError, ValueError) as exc:
            raise WebRequestError(400, "invalid_length", "Invalid Content-Length") from exc
        if length <= 0:
            raise WebRequestError(400, "empty_body", "A JSON object is required")
        if length > MAX_REQUEST_BYTES:
            raise WebRequestError(413, "body_too_large", "Request body exceeds 64 KiB")
        raw = environ["wsgi.input"].read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebRequestError(400, "invalid_json", "Malformed UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise WebRequestError(422, "object_required", "JSON body must be an object")
        return payload

    def _require_csrf(self, environ: dict[str, Any]) -> None:
        supplied = str(environ.get("HTTP_X_GFS_CSRF", ""))
        if not secrets.compare_digest(supplied, self.csrf_token):
            raise WebRequestError(403, "csrf_rejected", "Missing or invalid CSRF token")

    @staticmethod
    def _text_field(payload: dict[str, Any], key: str, *, maximum: int) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise WebRequestError(422, f"invalid_{key}", f"{key} must not be empty")
        value = value.strip()
        if len(value) > maximum or any(ord(char) < 32 for char in value):
            raise WebRequestError(422, f"invalid_{key}", f"{key} is not valid")
        return value

    def _studio_status(self) -> dict[str, Any]:
        session_path = self.root / "data/persistence/product_session.json"
        provider = provider_preflight()
        if not session_path.is_file():
            return {
                "schema_version": 1, "configured": False,
                "studio": None,
                "control_plane": ProductControlPlane(self.root).snapshot(),
                "provider": provider,
                "access": self.access_policy.public_summary(),
                "operations": self.telemetry.snapshot(),
                "tasks": [
                    self._task_for_web(task)
                    for task in self.task_queue.list_tasks(limit=20)
                ],
            }
        try:
            status = ProductWorkspace.load(self.root).status()
        except (OSError, ValueError) as exc:
            raise WebRequestError(
                409, "invalid_session", "The persisted Studio session is invalid",
            ) from exc
        return {
            "schema_version": 1, "configured": True,
            "studio": status, "provider": provider,
            "access": self.access_policy.public_summary(),
            "operations": self.telemetry.snapshot(),
            "tasks": [
                self._task_for_web(task)
                for task in self.task_queue.list_tasks(limit=20)
            ],
        }

    def _create_studio(
        self, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        name = self._text_field(payload, "name", maximum=80)
        mode = payload.get("mode", "stable")
        if mode not in {"stable", "research", "cognitive"}:
            raise WebRequestError(422, "invalid_mode", "Unsupported Studio mode")
        seed = payload.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
            raise WebRequestError(422, "invalid_seed", "seed must be a 32-bit non-negative integer")
        if not self._mutation_lock.acquire(blocking=False):
            raise WebRequestError(409, "operation_in_progress", "Another mutation is running")
        try:
            try:
                workspace = ProductWorkspace.create(
                    self.root, StudioConfig(name=name, mode=str(mode), seed=seed),
                )
            except FileExistsError as exc:
                raise WebRequestError(
                    409, "studio_exists", "A Studio already exists; use the CLI for explicit replacement",
                ) from exc
            return self._json_response(201, {
                "schema_version": 1, "configured": True,
                "studio": workspace.status(),
            })
        finally:
            self._mutation_lock.release()

    def _recovery_status(self) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            catalog = ProductRecovery(self.root).list_managed_backups()
        except LeaseUnavailable as exc:
            raise WebRequestError(
                409, "operation_in_progress", "Another recovery operation is running",
            ) from exc
        return self._json_response(200, catalog)

    def _create_managed_backup(self) -> tuple[int, list[tuple[str, str]], bytes]:
        if not self._mutation_lock.acquire(blocking=False):
            raise WebRequestError(409, "operation_in_progress", "Another mutation is running")
        try:
            try:
                result = ProductRecovery(self.root).create_managed_backup()
            except FileNotFoundError as exc:
                raise WebRequestError(
                    409, "studio_missing", "Create a Studio before making a backup",
                ) from exc
            except LeaseUnavailable as exc:
                raise WebRequestError(
                    409, "operation_in_progress", "Another product operation is running",
                ) from exc
            except RuntimeError as exc:
                raise WebRequestError(
                    409, "backup_capacity_reached",
                    "Managed backup capacity is full; archive older backups using deployment storage",
                ) from exc
            except ValueError as exc:
                raise WebRequestError(
                    409, "backup_source_invalid", "Studio state is not backup-ready",
                ) from exc
            return self._json_response(201, result)
        finally:
            self._mutation_lock.release()

    def _verify_managed_backup(
        self, backup_id: str,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            result = ProductRecovery(self.root).verify_managed_backup(backup_id)
        except FileNotFoundError as exc:
            raise WebRequestError(404, "backup_not_found", "Backup not found") from exc
        except ValueError as exc:
            raise WebRequestError(
                409, "backup_invalid", "Backup failed integrity or semantic verification",
            ) from exc
        except LeaseUnavailable as exc:
            raise WebRequestError(
                409, "operation_in_progress", "Another recovery operation is running",
            ) from exc
        return self._json_response(200, result)

    def _restore_managed_backup(
        self, backup_id: str, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        confirmation = payload.get("confirmation")
        confirmed = isinstance(confirmation, str) and secrets.compare_digest(
            confirmation.encode("utf-8"), backup_id.encode("ascii"),
        )
        if not confirmed:
            raise WebRequestError(
                422, "restore_confirmation_required",
                "Type the exact backup ID to confirm restore",
            )
        if payload.get("replace") is not True:
            raise WebRequestError(
                422, "explicit_replace_required", "Restore requires replace=true",
            )
        if not self._mutation_lock.acquire(blocking=False):
            raise WebRequestError(409, "operation_in_progress", "Another mutation is running")
        try:
            try:
                result = ProductRecovery(self.root).restore_managed_backup(
                    backup_id, replace=True,
                )
            except FileNotFoundError as exc:
                raise WebRequestError(404, "backup_not_found", "Backup not found") from exc
            except LeaseUnavailable as exc:
                raise WebRequestError(
                    409, "operation_in_progress", "Another product operation is running",
                ) from exc
            except RuntimeError as exc:
                raise WebRequestError(
                    409, "restore_blocked",
                    "Restore is blocked while match tasks are queued or running",
                ) from exc
            except ValueError as exc:
                raise WebRequestError(
                    409, "backup_invalid", "Backup failed integrity or semantic verification",
                ) from exc
            return self._json_response(200, result)
        finally:
            self._mutation_lock.release()

    def _queue_match(
        self, environ: dict[str, Any], payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        home = self._text_field(payload, "home", maximum=80)
        away = self._text_field(payload, "away", maximum=80)
        if home.casefold() == away.casefold():
            raise WebRequestError(422, "same_team", "Home and away teams must differ")
        fast = payload.get("fast", False)
        if not isinstance(fast, bool):
            raise WebRequestError(422, "invalid_fast", "fast must be boolean")
        if not self._mutation_lock.acquire(blocking=False):
            raise WebRequestError(
                409, "operation_in_progress", "Another mutation is running",
            )
        try:
            try:
                workspace = ProductWorkspace.load(self.root)
            except FileNotFoundError as exc:
                raise WebRequestError(
                    404, "studio_missing", "Create a Studio first",
                ) from exc
            readiness = workspace.readiness()
            if not readiness["ready"]:
                raise WebRequestError(
                    409, "match_blocked", "Match blocked by readiness gates",
                )
            try:
                task, created = self.task_queue.submit_match(
                    home, away, fast=fast,
                    idempotency_key=str(environ.get("HTTP_IDEMPOTENCY_KEY", "")),
                )
            except TaskConflict as exc:
                raise WebRequestError(409, "idempotency_conflict", str(exc)) from exc
            except ValueError as exc:
                raise WebRequestError(422, "invalid_idempotency_key", str(exc)) from exc
            except RuntimeError as exc:
                raise WebRequestError(409, "task_queue_full", "Task queue is full") from exc
            return self._json_response(202 if created else 200, {
                "schema_version": 1,
                "created": created,
                "task": self._task_for_web(task),
            })
        finally:
            self._mutation_lock.release()

    @staticmethod
    def _task_for_web(task: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(task, ensure_ascii=False, default=str))
        result = payload.get("result") or {}
        dashboard = str(result.get("dashboard") or "")
        if (
            dashboard.startswith("outputs/studio/")
            and chr(92) not in dashboard
            and ".." not in Path(dashboard).parts
        ):
            result["dashboard_url"] = "/artifacts/" + dashboard
        return payload

    def _artifact_response(
        self, raw_relative: str,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        relative = unquote(raw_relative)
        if chr(92) in relative or not relative:
            raise WebRequestError(404, "artifact_not_found", "Artifact not found")
        allowed_root = (self.root / "outputs/studio").resolve()
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(allowed_root)
        except ValueError as exc:
            raise WebRequestError(404, "artifact_not_found", "Artifact not found") from exc
        if candidate.suffix.lower() != ".html" or not candidate.is_file():
            raise WebRequestError(404, "artifact_not_found", "Artifact not found")
        return 200, [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Security-Policy", (
                "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            )),
        ], candidate.read_bytes()


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True

    def server_close(self) -> None:
        worker = getattr(self, "product_worker", None)
        stopped = worker.stop(timeout=30.0) if worker is not None else True
        try:
            super().server_close()
        finally:
            lease = getattr(self, "product_server_lease", None)
            if lease is not None and stopped:
                lease.release()
        if not stopped:
            raise RuntimeError("background match worker did not stop; server lease remains held")


def create_product_web_server(
    root: str | Path, *, host: str = "127.0.0.1", port: int = 8765,
    allow_remote: bool = False, access_token: str = "",
    allowed_hosts: tuple[str, ...] = (),
):
    """Create a local server or an authenticated trusted-proxy boundary."""
    if not _is_loopback_host(host) and not allow_remote:
        raise ValueError("non-loopback Web binding requires --allow-remote")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be 0 (automatic) or between 1 and 65535")
    resolved_root = Path(root).resolve()
    access_policy = WebAccessPolicy(
        remote=allow_remote, access_token=access_token,
        allowed_hosts=allowed_hosts,
    )
    server_lease = FileLease(
        resolved_root / "data/persistence/product_web.lock", timeout=0.0,
    ).acquire()
    queue = ProductTaskQueue(resolved_root)
    try:
        queue.recover_running()
        worker = BackgroundMatchWorker(queue)
        server = make_server(
            host, port, ProductWebApp(
                resolved_root, task_queue=queue, task_worker=worker,
                access_policy=access_policy,
            ),
            server_class=ThreadingWSGIServer,
            handler_class=WSGIRequestHandler,
        )
        server.product_server_lease = server_lease
        server.product_worker = worker
        worker.start()
        return server
    except BaseException:
        server_lease.release()
        raise


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="gfs-csrf" content="__CSRF__">
  <title>GFS Studio · Football Society Lab</title>
  <style>
    :root { color-scheme: dark; --ink:#f7f6f0; --muted:#aab2ad; --panel:#151b19;
      --line:#31413a; --accent:#7ee2a8; --warn:#ffd166; --danger:#ff7b72; }
    * { box-sizing:border-box } body { margin:0; color:var(--ink); background:#090d0c;
      font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif }
    .skip { position:absolute; left:-999px; top:0 } .skip:focus { left:1rem; top:1rem;
      padding:.7rem 1rem; background:var(--ink); color:#000; z-index:3 }
    header,main { width:min(1120px,calc(100% - 2rem)); margin:auto }
    header { padding:3.5rem 0 1.5rem } .eyebrow { color:var(--accent); letter-spacing:.14em;
      text-transform:uppercase; font-size:.76rem; font-weight:750 }
    h1 { max-width:820px; margin:.35rem 0; font-size:clamp(2.2rem,7vw,5.5rem); line-height:.94 }
    .lede { max-width:720px; color:var(--muted); font-size:1.08rem }
    .grid { display:grid; grid-template-columns:repeat(12,1fr); gap:1rem; padding:1rem 0 4rem }
    section { grid-column:span 12; border:1px solid var(--line); border-radius:18px;
      background:linear-gradient(145deg,#18201d,#101513); padding:clamp(1rem,3vw,1.6rem) }
    @media(min-width:800px){ .overview{grid-column:span 7}.action{grid-column:span 5}.wide{grid-column:span 12} }
    h2 { margin:.1rem 0 1rem; font-size:1.15rem } .cards { display:grid;
      grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:.7rem }
    .card { padding:1rem; border-radius:12px; background:#0c1210; border:1px solid #26342e }
    .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em }
    .value { display:block; margin-top:.25rem; font-size:1.25rem; font-weight:720; overflow-wrap:anywhere }
    form { display:grid; gap:.8rem } label { display:grid; gap:.35rem; color:var(--muted); font-size:.9rem }
    input,select,button { min-block-size:44px; font:inherit; border-radius:9px; border:1px solid #455b52; padding:.72rem .8rem }
    input,select { color:var(--ink); background:#0b100f } button { color:#07100c; background:var(--accent);
      border-color:transparent; font-weight:750; cursor:pointer } button:disabled { opacity:.48; cursor:wait }
    input:focus-visible,select:focus-visible,button:focus-visible,a:focus-visible,
    [tabindex="-1"]:focus-visible { outline:3px solid var(--warn); outline-offset:3px }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:.7rem } .check { display:flex; align-items:center; gap:.6rem }
    .check input { width:1.1rem; height:1.1rem } .status { min-height:1.6rem; color:var(--muted) }
    .error { color:var(--danger) } .ok { color:var(--accent) } a { color:var(--accent) }
    .toolbar { display:flex; flex-wrap:wrap; gap:.7rem; align-items:center }
    .backup-list { display:grid; gap:.7rem; margin:1rem 0 }
    .danger-button { background:var(--danger) }
    code { overflow-wrap:anywhere }
    pre { max-height:340px; overflow:auto; padding:1rem; border-radius:10px; background:#070a09;
      color:#cbd5d0; white-space:pre-wrap; overflow-wrap:anywhere }
    [hidden] { display:none!important }
    @media(max-width:520px){ .row { grid-template-columns:1fr } header { padding-top:2rem } }
    @media(prefers-reduced-motion:no-preference){ section { animation:rise .45s ease both }
      @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}} }
    @media(forced-colors:active){ section,.card,input,select,button { border:1px solid CanvasText }
      .skip:focus { border:2px solid CanvasText } }
  </style>
</head>
<body>
<a class="skip" href="#main">跳到主要内容</a>
<header><div class="eyebrow">Generative Football Society</div><h1>一个入口，观察完整足球社会。</h1>
<p class="lede">配置工作区、核验模型与证据、运行比赛，并在同一条可审计工作流中查看结果。</p></header>
<main id="main" class="grid" tabindex="-1">
  <section class="overview" aria-labelledby="overview-title"><h2 id="overview-title">系统状态</h2>
    <div id="cards" class="cards" aria-live="polite" aria-atomic="true"></div><p id="workflow" class="status"></p></section>
  <section class="action" aria-labelledby="action-title"><h2 id="action-title">下一步操作</h2>
    <form id="setup-form"><label>工作区名称<input name="name" maxlength="80" required value="My GFS Studio"></label>
      <div class="row"><label>模式<select name="mode"><option value="stable">稳定</option><option value="research">研究</option><option value="cognitive">认知</option></select></label>
      <label>随机种子<input name="seed" type="number" min="0" max="2147483647" value="42" required></label></div>
      <button type="submit">创建工作区</button></form>
    <form id="match-form" hidden><div class="row"><label>主队<input name="home" maxlength="80" value="Brazil" required></label>
      <label>客队<input name="away" maxlength="80" value="Argentina" required></label></div>
      <label class="check"><input name="fast" type="checkbox" checked>快速模式</label><button type="submit">运行比赛</button></form>
    <button id="logout-button" type="button"__REMOTE_LOGOUT_HIDDEN__>安全退出</button>
    <p id="message" class="status" role="status" aria-live="polite" aria-atomic="true"></p><p id="report-link" hidden></p></section>
  <section class="wide" aria-labelledby="recovery-title"><h2 id="recovery-title">恢复中心</h2>
    <p>备份保存在受控部署存储中。恢复前必须校验，并输入完整备份 ID 确认替换当前工作区。</p>
    <div class="toolbar"><button id="create-backup" type="button">创建完整备份</button><span id="recovery-message" class="status" role="status" aria-live="polite" aria-atomic="true"></span></div>
    <div id="backup-list" class="backup-list" role="list" aria-label="托管备份" aria-busy="true"></div>
    <form id="restore-form" hidden aria-labelledby="restore-title"><h3 id="restore-title">确认恢复</h3><p>准备恢复：<code id="selected-backup"></code></p>
      <label>输入完整备份 ID<input name="confirmation" maxlength="64" autocomplete="off" required></label>
      <label class="check"><input name="replace" type="checkbox" required>我确认使用该备份替换当前工作区并重置任务历史</label>
      <div class="toolbar"><button class="danger-button" type="submit">确认恢复</button><button id="cancel-restore" type="button">取消</button></div>
    </form></section>
  <section class="wide" aria-labelledby="release-title"><h2 id="release-title">产品与论文发布门禁</h2>
    <p id="release-summary" class="status" role="status" aria-live="polite" aria-atomic="true">正在读取发布状态……</p>
    <div id="release-gates" class="cards" role="list" aria-label="发布门禁列表"></div></section>
  <section class="wide" aria-labelledby="evidence-title"><h2 id="evidence-title">证据与运行详情</h2><pre id="details" tabindex="0" aria-label="当前工作区和任务的 JSON 证据">正在读取……</pre></section>
</main>
<noscript><p role="alert" aria-live="assertive" aria-atomic="true">GFS Studio 需要 JavaScript 才能执行受控工作流。</p></noscript>
<script nonce="__NONCE__">
const csrf=document.querySelector('meta[name="gfs-csrf"]').content;
const cards=document.querySelector('#cards'),setup=document.querySelector('#setup-form'),match=document.querySelector('#match-form');let activePoll='',restoreTrigger=null;
const releaseSummary=document.querySelector('#release-summary'),releaseGates=document.querySelector('#release-gates');
const message=document.querySelector('#message'),details=document.querySelector('#details'),workflow=document.querySelector('#workflow'),report=document.querySelector('#report-link'),logoutButton=document.querySelector('#logout-button');
const createBackupButton=document.querySelector('#create-backup'),backupList=document.querySelector('#backup-list'),recoveryMessage=document.querySelector('#recovery-message'),restoreForm=document.querySelector('#restore-form'),selectedBackup=document.querySelector('#selected-backup'),cancelRestore=document.querySelector('#cancel-restore');
const esc=v=>String(v??'—');
function announce(node,text,kind='status',focus=false){node.className='status '+(kind==='error'?'error':kind==='success'?'ok':'');node.setAttribute('role',kind==='error'?'alert':'status');node.setAttribute('aria-live',kind==='error'?'assertive':'polite');node.setAttribute('aria-atomic','true');node.textContent=text;if(focus){node.setAttribute('tabindex','-1');node.focus()}}
function setBusy(node,busy){node.setAttribute('aria-busy',busy?'true':'false')}
function card(label,value){const el=document.createElement('div');el.className='card';const a=document.createElement('span');a.className='label';a.textContent=label;const b=document.createElement('strong');b.className='value';b.textContent=esc(value);el.append(a,b);return el}
function renderRelease(data){releaseGates.replaceChildren();const cp=data.control_plane||data.studio?.control_plane||{},release=cp.release,ex=cp.excellence||{};if(!release){releaseSummary.textContent='发布状态尚不可用。';return}const product=release.scores?.product??ex.tracks?.product?.score??'—',academic=release.scores?.academic??ex.tracks?.academic?.score??'—',plan=release.completion_plan||{},actions=new Map((plan.steps||[]).map(step=>[step.gate_id,step])),kit=plan.evidence_kit?.ready?'就绪':'待检查';releaseSummary.textContent=`产品 ${product}/100 · 学术 ${academic}/100 · 代码契约${release.code_ready?'通过':'未通过'} · 最终发布${release.release_ready?'就绪':'未就绪'} · 证据包${kit} · ${release.open_gate_count} 个开放门禁 · 下一步 ${plan.recommended_gate_id||'无'}`;for(const gate of release.gates||[]){const action=actions.get(gate.id),state=gate.passed?'通过':action?.state||'待完成',detail=action&&!gate.passed?`${state} · 责任角色 ${action.operator}`:state,item=card(gate.label,detail);item.setAttribute('role','listitem');item.dataset.gateId=gate.id;if(action)item.dataset.actionState=action.state;releaseGates.append(item)}}
function showReport(url){if(!url)return;report.replaceChildren();const a=document.createElement('a');a.href=url;a.target='_blank';a.rel='noopener';a.textContent='打开比赛仪表板（新窗口）';report.append(a);report.hidden=false}
function render(data){createBackupButton.disabled=!data.configured;cards.replaceChildren();const s=data.studio,tasks=data.tasks||[],latest=tasks[0];if(!data.configured){cards.append(card('工作区','未配置'),card('API 调用','0'),card('任务',tasks.length));setup.hidden=false;match.hidden=true;workflow.textContent='创建工作区后，系统会先执行证据就绪检查。'}else{const r=s.readiness||{},w=s.workflow||{};cards.append(card('模式',s.mode),card('就绪',r.ready?'是':'否'),card('已完成比赛',s.matches_played),card('最近任务',latest?.state||'无'));setup.hidden=true;match.hidden=false;workflow.textContent=r.ready?'证据门禁通过，可以提交后台比赛任务。':'阻塞项：'+(r.blockers||[]).join(', ')}if(latest?.state==='completed')showReport(latest.result?.dashboard_url);if(['queued','running'].includes(latest?.state)&&activePoll!==latest.task_id){activePoll=latest.task_id;void pollTask(latest.task_id)}details.textContent=JSON.stringify(data,null,2)}
function setRecoveryMessage(text,isError=false,focus=false){announce(recoveryMessage,text,isError?'error':'success',focus)}
function closeRestore(returnFocus=true){restoreForm.reset();restoreForm.hidden=true;delete restoreForm.dataset.backupId;if(returnFocus&&restoreTrigger?.isConnected)restoreTrigger.focus();restoreTrigger=null}
function prepareRestore(id,trigger){restoreTrigger=trigger;selectedBackup.textContent=id;restoreForm.dataset.backupId=id;restoreForm.reset();restoreForm.hidden=false;restoreForm.querySelector('input[name="confirmation"]').focus()}
function renderBackups(data){backupList.replaceChildren();if(!data.backups.length){const empty=document.createElement('p');empty.className='status';empty.textContent='尚无托管备份。';backupList.append(empty);return}for(const backup of data.backups){const item=document.createElement('div');item.className='card';item.setAttribute('role','listitem');const title=document.createElement('strong');title.textContent=backup.backup_id;const meta=document.createElement('p');meta.className='status';meta.textContent=new Date(backup.modified_at).toLocaleString()+' · '+backup.size_bytes+' bytes';const actions=document.createElement('div');actions.className='toolbar';const verify=document.createElement('button');verify.type='button';verify.textContent='校验';verify.setAttribute('aria-label','校验备份 '+backup.backup_id);verify.addEventListener('click',async()=>{verify.disabled=true;setBusy(item,true);try{await api('/api/v1/backups/'+encodeURIComponent(backup.backup_id)+'/verify',{method:'POST'});setRecoveryMessage('备份完整性与语义校验通过。')}catch(e){setRecoveryMessage(e.message,true,true)}finally{verify.disabled=false;setBusy(item,false)}});const restore=document.createElement('button');restore.type='button';restore.className='danger-button';restore.textContent='准备恢复';restore.setAttribute('aria-label','准备恢复备份 '+backup.backup_id);restore.addEventListener('click',()=>prepareRestore(backup.backup_id,restore));actions.append(verify,restore);item.append(title,meta,actions);backupList.append(item)}}
async function refreshRecovery(){setBusy(backupList,true);try{renderBackups(await api('/api/v1/recovery'))}catch(e){setRecoveryMessage(e.message,true,true)}finally{setBusy(backupList,false)}}
async function api(path,options={}){const response=await fetch(path,{...options,headers:{'Content-Type':'application/json','X-GFS-CSRF':csrf,...options.headers}});const data=await response.json();if(!response.ok)throw new Error(data.error?.message||'请求失败');return data}
async function refresh(){try{const data=await api('/api/v1/studio');render(data);renderRelease(data)}catch(e){announce(message,e.message,'error',true)}}
async function pollTask(id){try{while(true){const data=await api('/api/v1/tasks/'+encodeURIComponent(id)),task=data.task;details.textContent=JSON.stringify(data,null,2);announce(message,task.state==='queued'?'比赛任务已排队……':'比赛正在后台运行……');if(task.state==='completed'){announce(message,'比赛完成。','success',true);showReport(task.result?.dashboard_url);await refresh();return}if(['failed','interrupted'].includes(task.state)){announce(message,task.error?.message||'任务中断，可检查状态后重新提交。','error',true);await refresh();return}await new Promise(resolve=>setTimeout(resolve,750))}}catch(e){announce(message,e.message,'error',true)}finally{activePoll=''}}
async function submit(form,path,payload,headers={}){const button=form.querySelector('button');button.disabled=true;setBusy(form,true);announce(message,'正在提交……');report.hidden=true;try{const data=await api(path,{method:'POST',body:JSON.stringify(payload),headers});if(data.task){announce(message,data.created?'任务已持久化排队。':'已返回同一幂等任务。');activePoll=data.task.task_id;await pollTask(data.task.task_id)}else{announce(message,'操作成功。','success',true);await refresh()}}catch(e){announce(message,e.message,'error',true)}finally{button.disabled=false;setBusy(form,false)}}
setup.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(setup);submit(setup,'/api/v1/studio',{name:f.get('name'),mode:f.get('mode'),seed:Number(f.get('seed'))})});
match.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(match),key=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random();submit(match,'/api/v1/matches',{home:f.get('home'),away:f.get('away'),fast:f.get('fast')==='on'},{'Idempotency-Key':key})});
createBackupButton.addEventListener('click',async()=>{createBackupButton.disabled=true;createBackupButton.setAttribute('aria-busy','true');setRecoveryMessage('正在创建并校验备份……');try{await api('/api/v1/backups',{method:'POST'});setRecoveryMessage('备份已创建并通过校验。');await refreshRecovery()}catch(e){setRecoveryMessage(e.message,true,true)}finally{createBackupButton.disabled=false;createBackupButton.setAttribute('aria-busy','false')}});
restoreForm.addEventListener('submit',async e=>{e.preventDefault();const id=restoreForm.dataset.backupId,f=new FormData(restoreForm),button=restoreForm.querySelector('button[type="submit"]');button.disabled=true;setBusy(restoreForm,true);setRecoveryMessage('正在执行事务恢复……');try{await api('/api/v1/backups/'+encodeURIComponent(id)+'/restore',{method:'POST',body:JSON.stringify({confirmation:f.get('confirmation'),replace:f.get('replace')==='on'})});closeRestore(false);setRecoveryMessage('恢复完成，任务历史已安全重置。',false,true);await Promise.all([refresh(),refreshRecovery()])}catch(error){setRecoveryMessage(error.message,true,true)}finally{button.disabled=false;setBusy(restoreForm,false)}});
cancelRestore.addEventListener('click',()=>closeRestore(true));
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!restoreForm.hidden){event.preventDefault();closeRestore(true)}});
logoutButton.addEventListener('click',async()=>{logoutButton.disabled=true;logoutButton.setAttribute('aria-busy','true');try{await api('/api/v1/logout',{method:'POST',body:'{}'});location.replace('/login')}catch(e){announce(message,e.message,'error',true);logoutButton.disabled=false;logoutButton.setAttribute('aria-busy','false')}});
void Promise.all([refresh(),refreshRecovery()]);
</script>
</body></html>"""


_LOGIN_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="gfs-csrf" content="__CSRF__"><title>GFS Studio 登录</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#090d0c;color:#f7f6f0;font:16px/1.5 system-ui,sans-serif}.panel{width:min(440px,calc(100% - 2rem));padding:2rem;border:1px solid #31413a;border-radius:18px;background:#151b19}h1{margin:.2rem 0}.muted{color:#aab2ad}form{display:grid;gap:.8rem;margin-top:1.4rem}label{display:grid;gap:.35rem}input,button{min-block-size:44px;font:inherit;padding:.75rem;border-radius:9px}input{color:inherit;background:#0b100f;border:1px solid #455b52}button{font-weight:750;background:#7ee2a8;border:0;color:#07100c}input:focus-visible,button:focus-visible,[tabindex="-1"]:focus-visible{outline:3px solid #ffd166;outline-offset:3px}.error{color:#ff7b72;min-height:1.5rem}@media(forced-colors:active){input,button,.panel{border:1px solid CanvasText}}
</style></head><body><main class="panel"><p class="muted">Authenticated deployment</p><h1>进入 GFS Studio</h1><p class="muted">请输入独立的产品访问令牌。它不是模型提供商 API Key。</p><form id="login" aria-busy="false"><label>产品访问令牌<input name="token" type="password" autocomplete="current-password" required></label><button type="submit">安全登录</button></form><p id="message" class="error" role="alert" aria-live="assertive" aria-atomic="true" tabindex="-1"></p></main><script nonce="__NONCE__">
const form=document.querySelector('#login'),message=document.querySelector('#message'),csrf=document.querySelector('meta[name="gfs-csrf"]').content;form.addEventListener('submit',async e=>{e.preventDefault();const button=form.querySelector('button'),token=new FormData(form).get('token');button.disabled=true;form.setAttribute('aria-busy','true');message.textContent='';try{const response=await fetch('/api/v1/login',{method:'POST',headers:{'Content-Type':'application/json','X-GFS-CSRF':csrf},body:JSON.stringify({access_token:token})});const data=await response.json();if(!response.ok)throw new Error(data.error?.message||'登录失败');location.replace('/')}catch(error){message.textContent=error.message;message.focus()}finally{button.disabled=false;form.setAttribute('aria-busy','false');form.reset()}});
</script></body></html>"""
