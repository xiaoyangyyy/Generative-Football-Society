"""Dependency-free local Web Beta over the canonical product workspace."""

from __future__ import annotations

import html
import json
import logging
import math
import re
import secrets
import threading
import time
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import unquote
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from src.infrastructure import FileLease, LeaseUnavailable
from src.product.control_plane import ProductControlPlane
from src.product.match_plan import (
    MatchPlan, PairedMatchPlan, WorldModelForkPlan, playable_tactic_catalog,
)
from src.product.season import ManagerDecision, SeasonPlan
from src.product.player_promises import PlayerPromisePlan
from src.product.tactical_study import TacticalStudyPlan
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


def _bounded_public_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    number = float(value)
    return max(0, min(100_000, int(number))) if math.isfinite(number) else 0


def _match_capabilities() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiences": [
            {
                "id": "observational", "label": "原生观赛",
                "score_path": "macro_replay",
                "claim_boundary": "descriptive",
            },
            {
                "id": "tactical_lab", "label": "战术实验室",
                "score_path": "physics_official",
                "claim_boundary": "paired_seed_required_for_attribution",
                "modes": ["research", "cognitive"],
            },
        ],
        "tactics": playable_tactic_catalog(),
        "tactical_study": {
            "modes": ["research"],
            "min_pairs": 4,
            "max_pairs": 32,
            "primary_metric": "focus_xg_difference",
            "interim_analysis": "withheld_until_fixed_budget_complete",
            "claim_boundary": "fixed_fixture_and_seed_set_only",
        },
        "paired_match": {
            "modes": ["research", "cognitive"],
            "seed": "explicit_shared_seed",
            "intervention": "exactly_one_tactical_side",
            "score_path": "physics_official",
            "cognitive_claim_boundary": "descriptive_only",
        },
        "world_model_fork": {
            "modes": ["research"],
            "seed": "explicit_shared_seed",
            "intervention": "predict_only_to_action_policy",
            "branch_time_seconds": {"min": 0, "max": 5400, "default": 2700},
            "branch_execution": "identity_bound_deterministic_replay",
            "resume_capability": "deterministic_replay_only",
            "fixed_controls": [
                "fixture", "tactics", "fast_configuration", "checkpoint",
            ],
            "score_path": "physics_official",
            "evidence_chain": [
                "isolated_policy_assignment",
                "identical_pre_intervention_branch_anchor",
                "realized_action_change",
                "direct_runtime_identity",
                "descriptive_downstream_windows",
            ],
            "downstream_causal_attribution": False,
            "claim_boundary": "single_fixture_seed_simulator_contrast_only",
        },
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
        sporting_plan = re.fullmatch(
            r"/api/v1/sporting-plans/([^/]+)", path,
        )
        if method == "GET" and sporting_plan:
            return self._sporting_plan(unquote(sporting_plan.group(1)))
        recruitment_market = re.fullmatch(
            r"/api/v1/recruitment-markets/([^/]+)", path,
        )
        if method == "GET" and recruitment_market:
            return self._recruitment_market(
                unquote(recruitment_market.group(1)),
            )
        lifecycle_preview = re.fullmatch(
            r"/api/v1/lifecycle-previews/([^/]+)", path,
        )
        if method == "GET" and lifecycle_preview:
            return self._lifecycle_preview(
                unquote(lifecycle_preview.group(1)),
            )
        free_agent_market = re.fullmatch(
            r"/api/v1/free-agent-markets/([^/]+)", path,
        )
        if method == "GET" and free_agent_market:
            return self._free_agent_market(
                unquote(free_agent_market.group(1)),
            )
        if method == "GET" and path == "/api/v1/operations":
            return self._json_response(200, self.telemetry.snapshot())
        if method == "GET" and path == "/api/v1/excellence/evidence-kit.zip":
            return self._evidence_kit_response()
        if method == "GET" and path == "/api/v1/recovery":
            return self._recovery_status()
        if method == "POST" and path == "/api/v1/studio":
            self._require_csrf(environ)
            return self._create_studio(self._read_json(environ))
        if method == "POST" and path == "/api/v1/seasons":
            self._require_csrf(environ)
            return self._create_season(self._read_json(environ))
        if method == "POST" and path == "/api/v1/scouting-reports":
            self._require_csrf(environ)
            return self._scout_free_agent(self._read_json(environ))
        if method == "POST" and path == "/api/v1/seasons/next-matchday":
            self._require_csrf(environ)
            return self._queue_season_matchday(environ)
        if method == "POST" and path == "/api/v1/seasons/decision":
            self._require_csrf(environ)
            return self._set_season_decision(self._read_json(environ))
        if method == "POST" and path == "/api/v1/seasons/decision-preview":
            self._require_csrf(environ)
            return self._preview_season_decision(self._read_json(environ))
        if method == "POST" and path == "/api/v1/seasons/decision-advice":
            self._require_csrf(environ)
            return self._request_season_decision_advice(self._read_json(environ))
        if method == "POST" and path == "/api/v1/seasons/player-promises":
            self._require_csrf(environ)
            return self._set_player_promises(self._read_json(environ))
        if method == "POST" and path == "/api/v1/matches":
            self._require_csrf(environ)
            return self._queue_match(environ, self._read_json(environ))
        if method == "POST" and path == "/api/v1/paired-matches":
            self._require_csrf(environ)
            return self._queue_paired_match(environ, self._read_json(environ))
        if method == "POST" and path == "/api/v1/world-model-forks":
            self._require_csrf(environ)
            return self._queue_world_model_fork(
                environ, self._read_json(environ),
            )
        if method == "POST" and path == "/api/v1/tactical-studies":
            self._require_csrf(environ)
            return self._queue_tactical_study(environ, self._read_json(environ))
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
        task_requeue = re.fullmatch(r"/api/v1/tasks/([a-zA-Z0-9]+)/requeue", path)
        if method == "POST" and task_requeue:
            self._require_csrf(environ)
            return self._requeue_task(
                task_requeue.group(1), self._read_json(environ),
            )
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
            "/api/v1/studio", "/api/v1/matches", "/api/v1/paired-matches",
            "/api/v1/world-model-forks",
            "/api/v1/tactical-studies",
            "/api/v1/seasons", "/api/v1/seasons/next-matchday",
            "/api/v1/seasons/decision",
            "/api/v1/tasks",
            "/api/v1/recovery", "/api/v1/backups",
            "/api/v1/excellence/evidence-kit.zip",
        } or backup_action or task_requeue:
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

    def _evidence_kit_response(
        self,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        import hashlib

        from scripts.build_excellence_evidence_kit import build_archive, verify_kit

        report = verify_kit(self.root)
        if report.get("passed") is not True:
            raise WebRequestError(
                503, "evidence_kit_unavailable",
                "The template-only evidence kit failed its current audit",
            )
        archive, _ = build_archive(self.root)
        digest = hashlib.sha256(archive).hexdigest()
        if digest != report.get("archive_sha256"):
            raise WebRequestError(
                503, "evidence_kit_identity_mismatch",
                "The evidence kit changed after verification",
            )
        return 200, [
            ("Content-Type", "application/zip"),
            (
                "Content-Disposition",
                'attachment; filename="gfs-excellence-evidence-kit-v1.zip"',
            ),
            ("X-GFS-Artifact-SHA256", digest),
            ("X-GFS-Template-Only", "true"),
        ], archive

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
        raw_tasks = self.task_queue.list_tasks(limit=51)
        task_history_truncated = len(raw_tasks) > 50
        web_tasks = [
            self._task_for_web(task)
            for task in raw_tasks[:50]
        ]
        if not session_path.is_file():
            return {
                "schema_version": 1, "configured": False,
                "studio": None,
                "match_capabilities": _match_capabilities(),
                "control_plane": ProductControlPlane(self.root).snapshot(),
                "provider": provider,
                "access": self.access_policy.public_summary(),
                "operations": self.telemetry.snapshot(),
                "tasks": web_tasks[:20],
                "evidence_library": {
                    "schema_version": 1, "limit": 50,
                    "matches": [], "pairs": [], "forks": [], "studies": [],
                    "truncated": False,
                },
            }
        try:
            status = ProductWorkspace.load(self.root).status()
        except (OSError, ValueError) as exc:
            raise WebRequestError(
                409, "invalid_session", "The persisted Studio session is invalid",
            ) from exc
        last_match = status.get("last_match") or {}
        if isinstance(last_match, dict):
            dashboard_url = self._safe_artifact_url(last_match.get("dashboard"))
            comparison_url = self._safe_artifact_url(
                last_match.get("comparison_dashboard")
            )
            if dashboard_url:
                last_match["dashboard_url"] = dashboard_url
            if comparison_url:
                last_match["comparison_url"] = comparison_url
        season = status.get("season")
        if isinstance(season, dict):
            for fixture in season.get("fixtures") or []:
                if not isinstance(fixture, dict):
                    continue
                dashboard_url = self._safe_artifact_url(fixture.get("dashboard"))
                if dashboard_url:
                    fixture["dashboard_url"] = dashboard_url
            profile = season.get("manager_profile") or {}
            journal = profile.get("journal") if isinstance(profile, dict) else None
            if isinstance(journal, list):
                safe_by_fixture = {
                    fixture.get("fixture_id"): fixture.get("dashboard_url")
                    for fixture in season.get("fixtures") or []
                    if isinstance(fixture, dict) and fixture.get("dashboard_url")
                }
                for entry in journal:
                    if isinstance(entry, dict):
                        entry["dashboard_url"] = safe_by_fixture.get(
                            entry.get("fixture_id")
                        )
        for archived in status.get("season_history") or []:
            if not isinstance(archived, dict):
                continue
            profile = archived.get("manager_profile") or {}
            if not isinstance(profile, dict):
                continue
            journal = profile.get("journal")
            if not isinstance(journal, list):
                continue
            for entry in journal:
                if not isinstance(entry, dict):
                    continue
                dashboard_url = self._safe_artifact_url(entry.get("dashboard"))
                if dashboard_url:
                    entry["dashboard_url"] = dashboard_url
        library_matches = []
        for match in status.get("match_history") or []:
            if not isinstance(match, Mapping):
                continue
            item = {
                key: match.get(key) for key in (
                    "match_id", "home", "away", "seed", "fast", "score",
                    "integrity", "experience", "home_tactic", "away_tactic",
                )
            }
            item["dashboard_url"] = self._safe_artifact_url(
                match.get("dashboard")
            )
            item["comparison_url"] = self._safe_artifact_url(
                match.get("comparison_dashboard")
            )
            library_matches.append(item)
        library_studies = []
        library_pairs = []
        library_forks = []
        for task in web_tasks:
            if task.get("kind") == "world_model_fork":
                request = task.get("request") or {}
                try:
                    plan = WorldModelForkPlan.from_payload(
                        request.get("plan") or {},
                    )
                except ValueError:
                    continue
                result = task.get("result") or {}
                raw_propagation = result.get("propagation") or {}
                propagation = (
                    {
                        "available": bool(raw_propagation.get("available")),
                        "status": str(
                            raw_propagation.get("status") or "unknown"
                        )[:80],
                        "changed_decisions": _bounded_public_count(
                            raw_propagation.get("changed_decisions"),
                        ),
                        "directly_observed_changes": _bounded_public_count(
                            raw_propagation.get("directly_observed_changes"),
                        ),
                        "locally_attributable_changes": _bounded_public_count(
                            raw_propagation.get("locally_attributable_changes"),
                        ),
                        "replay_windows_available": bool(
                            raw_propagation.get("replay_windows_available") is True
                        ),
                        "downstream_causal_attribution_authorized": False,
                        **({
                            "branch_at_sec": raw_propagation.get("branch_at_sec"),
                            "branch_anchor_verified": bool(
                                raw_propagation.get("branch_anchor_verified")
                            ),
                            "branch_state_identity": str(
                                raw_propagation.get("branch_state_identity") or ""
                            )[:64],
                        } if raw_propagation.get("branch_at_sec") is not None else {}),
                    }
                    if isinstance(raw_propagation, Mapping) else
                    {"available": False, "status": "unavailable"}
                )
                library_forks.append({
                    "task_id": task.get("task_id"),
                    "home": request.get("home"), "away": request.get("away"),
                    "seed": plan.seed,
                    "branch_at_sec": float(plan.branch_at_sec),
                    "home_tactic": plan.home_tactic,
                    "away_tactic": plan.away_tactic,
                    "state": task.get("state"),
                    "baseline_match_id": result.get("baseline_match_id"),
                    "treatment_match_id": result.get("treatment_match_id"),
                    "baseline_url": result.get("baseline_url"),
                    "treatment_url": result.get("treatment_url"),
                    "comparison_url": result.get("comparison_url"),
                    "propagation": propagation,
                })
                continue
            if task.get("kind") == "paired_match":
                request = task.get("request") or {}
                raw_plan = request.get("plan") or {}
                try:
                    plan = PairedMatchPlan.from_payload(raw_plan)
                except ValueError:
                    continue
                result = task.get("result") or {}
                library_pairs.append({
                    "task_id": task.get("task_id"),
                    "home": request.get("home"), "away": request.get("away"),
                    "seed": plan.seed, "focus_side": plan.focus_side,
                    "baseline_tactic": getattr(
                        plan, f"baseline_{plan.focus_side}_tactic"
                    ),
                    "treatment_tactic": getattr(
                        plan, f"treatment_{plan.focus_side}_tactic"
                    ),
                    "state": task.get("state"),
                    "baseline_match_id": result.get("baseline_match_id"),
                    "treatment_match_id": result.get("treatment_match_id"),
                    "baseline_url": result.get("baseline_url"),
                    "treatment_url": result.get("treatment_url"),
                    "comparison_url": result.get("comparison_url"),
                })
                continue
            if task.get("kind") != "tactical_study":
                continue
            raw_plan = ((task.get("request") or {}).get("plan") or {})
            try:
                plan = TacticalStudyPlan.from_payload(raw_plan)
            except ValueError:
                continue
            progress = task.get("study_progress") or {}
            library_studies.append({
                "task_id": task.get("task_id"),
                "study_id": plan.study_id,
                "home": plan.home, "away": plan.away,
                "focus_side": plan.focus_side,
                "baseline_tactic": getattr(
                    plan, f"baseline_{plan.focus_side}_tactic"
                ),
                "treatment_tactic": getattr(
                    plan, f"treatment_{plan.focus_side}_tactic"
                ),
                "state": task.get("state"),
                "pairs_completed": progress.get("pairs_completed", 0),
                "fixed_pair_budget": len(plan.seeds),
                "analysis_withheld": (
                    task.get("state") != "completed"
                    or not (task.get("result") or {}).get("study_url")
                ),
                "study_url": (task.get("result") or {}).get("study_url"),
            })
        return {
            "schema_version": 1, "configured": True,
            "studio": status, "provider": provider,
            "match_capabilities": _match_capabilities(),
            "access": self.access_policy.public_summary(),
            "operations": self.telemetry.snapshot(),
            "tasks": web_tasks[:20],
            "evidence_library": {
                "schema_version": 1, "limit": 50,
                "matches": library_matches,
                "pairs": library_pairs,
                "forks": library_forks,
                "studies": library_studies,
                "truncated": bool(status.get("match_history_truncated"))
                or task_history_truncated,
            },
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

    def _create_season(
        self, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        start_next = payload.get("start_next", False)
        if not isinstance(start_next, bool):
            raise WebRequestError(422, "invalid_season", "start_next must be boolean")
        try:
            plan = SeasonPlan.from_payload(payload.get("plan") or payload)
        except ValueError as exc:
            raise WebRequestError(422, "invalid_season", str(exc)) from exc
        try:
            workspace = ProductWorkspace.load(self.root)
            season = (
                workspace.create_season(plan, replace=True)
                if start_next else workspace.create_season(plan)
            )
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except FileExistsError as exc:
            raise WebRequestError(409, "season_exists", "A Studio season already exists") from exc
        except ValueError as exc:
            raise WebRequestError(409, "season_transition_invalid", str(exc)) from exc
        return self._json_response(201, {
            "schema_version": 1, "season": season,
        })

    def _recruitment_market(
        self, team: str,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        if not team.strip() or len(team) > 96:
            raise WebRequestError(422, "invalid_team", "Invalid recruitment team")
        try:
            market = ProductWorkspace.load(self.root).recruitment_market(team)
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except ValueError as exc:
            raise WebRequestError(409, "recruitment_unavailable", str(exc)) from exc
        return self._json_response(200, {
            "schema_version": 1, "market": market,
        })

    def _sporting_plan(
        self, team: str,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        if not team.strip() or len(team) > 96:
            raise WebRequestError(422, "invalid_sporting_team", "Invalid sporting team")
        try:
            plan = ProductWorkspace.load(self.root).sporting_plan(team)
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except ValueError as exc:
            raise WebRequestError(409, "sporting_plan_unavailable", str(exc)) from exc
        return self._json_response(200, {
            "schema_version": 1, "plan": plan,
        })

    def _lifecycle_preview(
        self, team: str,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        if not team.strip() or len(team) > 96:
            raise WebRequestError(422, "invalid_team", "Invalid lifecycle team")
        try:
            preview = ProductWorkspace.load(self.root).lifecycle_preview(team)
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except ValueError as exc:
            raise WebRequestError(409, "lifecycle_unavailable", str(exc)) from exc
        return self._json_response(200, {
            "schema_version": 1, "preview": preview,
        })

    def _free_agent_market(
        self, team: str,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        if not team.strip() or len(team) > 96:
            raise WebRequestError(422, "invalid_team", "Invalid free-agent team")
        try:
            market = ProductWorkspace.load(self.root).free_agent_market(team)
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except ValueError as exc:
            raise WebRequestError(409, "free_agent_market_unavailable", str(exc)) from exc
        return self._json_response(200, {
            "schema_version": 1, "market": market,
        })

    def _scout_free_agent(
        self, payload: Mapping[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        team = str(payload.get("team") or "").strip()
        player_id = str(payload.get("player_id") or "").strip()
        if not team or len(team) > 96 or not player_id or len(player_id) > 128:
            raise WebRequestError(422, "invalid_scouting_request", "Invalid scouting request")
        try:
            market = ProductWorkspace.load(self.root).scout_free_agent(
                team, player_id,
            )
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except ValueError as exc:
            raise WebRequestError(409, "scouting_unavailable", str(exc)) from exc
        return self._json_response(200, {
            "schema_version": 1, "market": market,
        })

    def _queue_season_matchday(
        self, environ: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            workspace = ProductWorkspace.load(self.root)
            season = workspace.season_status()
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        if season is None:
            raise WebRequestError(409, "season_missing", "Create a season first")
        matchday = season.get("next_matchday")
        if matchday is None:
            raise WebRequestError(409, "season_complete", "The season is already complete")
        key = str(environ.get("HTTP_IDEMPOTENCY_KEY", "")).strip()
        if not key:
            key = (
                f"{season['season_id']}:matchday:{matchday}:"
                f"revision:{int(season.get('revision', 0))}"
            )
        try:
            task, created = self.task_queue.submit_season_matchday(
                season_id=str(season["season_id"]), matchday=int(matchday),
                season_revision=int(season.get("revision", 0)),
                idempotency_key=key,
            )
        except TaskConflict as exc:
            raise WebRequestError(409, "idempotency_conflict", str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise WebRequestError(409, "season_matchday_unavailable", str(exc)) from exc
        return self._json_response(202 if created else 200, {
            "schema_version": 1, "created": created,
            "task": self._task_for_web(task),
        })

    def _set_season_decision(
        self, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            decision = ManagerDecision.from_payload(payload.get("decision") or {})
            fixture_id = self._text_field(payload, "fixture_id", maximum=32)
            expected_revision = payload.get("expected_revision")
            if expected_revision is not None and (
                isinstance(expected_revision, bool)
                or not isinstance(expected_revision, int)
                or expected_revision < 0
            ):
                raise ValueError("manager decision preview revision is invalid")
            workspace = ProductWorkspace.load(self.root)
            options: dict[str, Any] = {"fixture_id": fixture_id}
            if expected_revision is not None:
                options["expected_revision"] = expected_revision
            advice_adoption = payload.get("advice_adoption")
            if advice_adoption is not None:
                if not isinstance(advice_adoption, dict):
                    raise ValueError("manager advice adoption payload is invalid")
                options["advice_adoption"] = advice_adoption
            season = workspace.set_manager_decision(decision, **options)
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except ValueError as exc:
            raise WebRequestError(422, "invalid_manager_decision", str(exc)) from exc
        return self._json_response(200, {
            "schema_version": 1, "season": season,
        })

    def _preview_season_decision(
        self, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            decision = ManagerDecision.from_payload(payload.get("decision") or {})
            fixture_id = self._text_field(payload, "fixture_id", maximum=32)
            preview = ProductWorkspace.load(self.root).preview_manager_decision(
                decision, fixture_id=fixture_id,
            )
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except ValueError as exc:
            raise WebRequestError(422, "invalid_manager_decision", str(exc)) from exc
        return self._json_response(200, {
            "schema_version": 1, "preview": preview,
        })

    def _request_season_decision_advice(
        self, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            if set(payload) != {"fixture_id"}:
                raise ValueError("manager advice request has unsupported fields")
            fixture_id = self._text_field(payload, "fixture_id", maximum=32)
            advice = ProductWorkspace.load(self.root).request_manager_decision_advice(
                fixture_id=fixture_id,
            )
        except FileNotFoundError as exc:
            raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
        except ValueError as exc:
            raise WebRequestError(422, "manager_advice_unavailable", str(exc)) from exc
        return self._json_response(200, {
            "schema_version": 1, "advice": advice,
        })

    def _set_player_promises(
        self, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            promise_plan = PlayerPromisePlan.from_payload(payload)
            season = ProductWorkspace.load(self.root).set_player_role_promises(
                promise_plan,
            )
        except FileNotFoundError as exc:
            raise WebRequestError(
                404, "studio_missing", "Create a Studio first",
            ) from exc
        except ValueError as exc:
            raise WebRequestError(
                422, "invalid_player_promises", str(exc),
            ) from exc
        return self._json_response(200, {
            "schema_version": 1, "season": season,
        })

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
        try:
            plan = MatchPlan.from_payload(payload.get("plan"))
        except ValueError as exc:
            raise WebRequestError(422, "invalid_match_plan", str(exc)) from exc
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
                plan.validate_for_mode(workspace.config.mode)
                seed_override = (
                    workspace.replay_seed(home, away)
                    if plan.reuse_last_seed else None
                )
            except ValueError as exc:
                raise WebRequestError(422, "invalid_match_plan", str(exc)) from exc
            try:
                task, created = self.task_queue.submit_match(
                    home, away, fast=fast, plan=plan,
                    seed_override=seed_override,
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

    def _requeue_task(
        self, task_id: str, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        reason = self._text_field(payload, "reason", maximum=100)
        if not self._mutation_lock.acquire(blocking=False):
            raise WebRequestError(
                409, "operation_in_progress", "Another mutation is running",
            )
        try:
            try:
                task = self.task_queue.requeue_interrupted(
                    task_id, reason=reason,
                )
            except FileNotFoundError as exc:
                raise WebRequestError(
                    404, "task_not_found", "Task not found",
                ) from exc
            except ValueError as exc:
                raise WebRequestError(
                    422, "invalid_requeue", str(exc),
                ) from exc
            except RuntimeError as exc:
                raise WebRequestError(
                    409, "task_not_interrupted", str(exc),
                ) from exc
            return self._json_response(200, {
                "schema_version": 1, "task": self._task_for_web(task),
            })
        finally:
            self._mutation_lock.release()

    def _queue_tactical_study(
        self, environ: dict[str, Any], payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            plan = TacticalStudyPlan.from_payload(payload.get("plan") or payload)
        except ValueError as exc:
            raise WebRequestError(422, "invalid_tactical_study", str(exc)) from exc
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
            if workspace.config.mode != "research":
                raise WebRequestError(
                    422, "study_requires_research_mode",
                    "Inferential tactical studies require research mode",
                )
            if not workspace.readiness()["ready"]:
                raise WebRequestError(
                    409, "study_blocked", "Study blocked by readiness gates",
                )
            try:
                task, created = self.task_queue.submit_tactical_study(
                    plan,
                    idempotency_key=str(
                        environ.get("HTTP_IDEMPOTENCY_KEY", "")
                    ),
                )
            except TaskConflict as exc:
                raise WebRequestError(
                    409, "idempotency_conflict", str(exc),
                ) from exc
            except ValueError as exc:
                raise WebRequestError(
                    422, "invalid_idempotency_key", str(exc),
                ) from exc
            except RuntimeError as exc:
                raise WebRequestError(
                    409, "task_queue_full", "Task queue is full",
                ) from exc
            return self._json_response(202 if created else 200, {
                "schema_version": 1, "created": created,
                "task": self._task_for_web(task),
            })
        finally:
            self._mutation_lock.release()

    def _queue_paired_match(
        self, environ: dict[str, Any], payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        home = self._text_field(payload, "home", maximum=80)
        away = self._text_field(payload, "away", maximum=80)
        if home.casefold() == away.casefold():
            raise WebRequestError(422, "same_team", "Teams must differ")
        fast = payload.get("fast", False)
        if not isinstance(fast, bool):
            raise WebRequestError(422, "invalid_fast", "fast must be boolean")
        try:
            plan = PairedMatchPlan.from_payload(payload.get("plan"))
        except ValueError as exc:
            raise WebRequestError(422, "invalid_paired_match", str(exc)) from exc
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
            try:
                plan.validate_for_mode(workspace.config.mode)
            except ValueError as exc:
                raise WebRequestError(
                    422, "invalid_paired_match", str(exc),
                ) from exc
            if not workspace.readiness()["ready"]:
                raise WebRequestError(
                    409, "paired_match_blocked",
                    "Paired match blocked by readiness gates",
                )
            try:
                task, created = self.task_queue.submit_paired_match(
                    home, away, fast=fast, plan=plan,
                    idempotency_key=str(
                        environ.get("HTTP_IDEMPOTENCY_KEY", "")
                    ),
                )
            except TaskConflict as exc:
                raise WebRequestError(
                    409, "idempotency_conflict", str(exc),
                ) from exc
            except ValueError as exc:
                raise WebRequestError(
                    422, "invalid_idempotency_key", str(exc),
                ) from exc
            except RuntimeError as exc:
                raise WebRequestError(
                    409, "task_queue_full", "Task queue is full",
                ) from exc
            return self._json_response(202 if created else 200, {
                "schema_version": 1, "created": created,
                "task": self._task_for_web(task),
            })
        finally:
            self._mutation_lock.release()

    def _queue_world_model_fork(
        self, environ: dict[str, Any], payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        home = self._text_field(payload, "home", maximum=80)
        away = self._text_field(payload, "away", maximum=80)
        if home.casefold() == away.casefold():
            raise WebRequestError(422, "same_team", "Teams must differ")
        fast = payload.get("fast", False)
        if not isinstance(fast, bool):
            raise WebRequestError(422, "invalid_fast", "fast must be boolean")
        try:
            plan = WorldModelForkPlan.from_payload(payload.get("plan"))
        except ValueError as exc:
            raise WebRequestError(
                422, "invalid_world_model_fork", str(exc),
            ) from exc
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
            try:
                plan.validate_for_mode(workspace.config.mode)
            except ValueError as exc:
                raise WebRequestError(
                    422, "invalid_world_model_fork", str(exc),
                ) from exc
            if not workspace.readiness()["ready"]:
                raise WebRequestError(
                    409, "world_model_fork_blocked",
                    "World-model fork blocked by readiness gates",
                )
            try:
                task, created = self.task_queue.submit_world_model_fork(
                    home, away, fast=fast, plan=plan,
                    idempotency_key=str(
                        environ.get("HTTP_IDEMPOTENCY_KEY", "")
                    ),
                )
            except TaskConflict as exc:
                raise WebRequestError(
                    409, "idempotency_conflict", str(exc),
                ) from exc
            except ValueError as exc:
                raise WebRequestError(
                    422, "invalid_idempotency_key", str(exc),
                ) from exc
            except RuntimeError as exc:
                raise WebRequestError(
                    409, "task_queue_full", "Task queue is full",
                ) from exc
            return self._json_response(202 if created else 200, {
                "schema_version": 1, "created": created,
                "task": self._task_for_web(task),
            })
        finally:
            self._mutation_lock.release()

    @staticmethod
    def _safe_artifact_url(value: Any) -> str | None:
        relative = str(value or "")
        if (
            relative.startswith("outputs/studio/")
            and chr(92) not in relative
            and ".." not in Path(relative).parts
            and Path(relative).suffix.lower() == ".html"
        ):
            return "/artifacts/" + relative
        return None

    def _task_for_web(self, task: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(task, ensure_ascii=False, default=str))
        result = payload.get("result") or {}
        for field, url_field in (
            ("dashboard", "dashboard_url"),
            ("baseline_dashboard", "baseline_url"),
            ("treatment_dashboard", "treatment_url"),
            ("comparison_dashboard", "comparison_url"),
            ("study_dashboard", "study_url"),
        ):
            dashboard_url = ProductWebApp._safe_artifact_url(result.get(field))
            if dashboard_url:
                result[url_field] = dashboard_url
        if payload.get("kind") == "tactical_study":
            plan_payload = ((payload.get("request") or {}).get("plan") or {})
            try:
                validated_plan = TacticalStudyPlan.from_payload(plan_payload)
                study_id = validated_plan.study_id
                if payload.get("state") == "queued":
                    payload["study_progress"] = {
                        "state": "queued",
                        "pairs_completed": 0,
                        "fixed_pair_budget": len(validated_plan.seeds),
                        "analysis_withheld": True,
                    }
                    return payload
                workspace = ProductWorkspace.load(self.root)
                progress_path = (
                    workspace.output_root / "studies" / study_id / "progress.json"
                )
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                completed = int(progress.get("pairs_completed", 0))
                budget = int(progress.get("fixed_pair_budget", 0))
                if (
                    progress.get("analysis") is not None
                    or progress.get("interim_effects_disclosed") is not False
                    or not 0 <= completed <= budget
                    or budget != len(validated_plan.seeds)
                ):
                    raise ValueError("invalid public study progress")
                payload["study_progress"] = {
                    "state": str(progress.get("state") or "unknown"),
                    "pairs_completed": completed,
                    "fixed_pair_budget": budget,
                    "analysis_withheld": True,
                }
            except (
                OSError, ValueError, TypeError, AttributeError,
                json.JSONDecodeError,
            ):
                if payload.get("state") != "queued":
                    payload["study_progress"] = {
                        "state": "invalid_progress",
                        "analysis_withheld": True,
                    }
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
    .workspace-nav { grid-column:1/-1; display:flex; flex-wrap:wrap; gap:.55rem;
      align-items:center; padding:.75rem; border:1px solid var(--line); border-radius:14px;
      background:#101713; position:sticky; top:.5rem; z-index:2 }
    .workspace-nav button { background:#17211d; color:var(--ink); border-color:var(--line) }
    .workspace-nav button[aria-pressed="true"] { background:var(--accent); color:#07100c;
      border-color:transparent }
    .workspace-nav .status { flex:1 1 18rem; margin:0; text-align:right }
    .workspace-area-hidden { display:none!important }
    .backup-list { display:grid; gap:.7rem; margin:1rem 0 }
    .season-list { display:grid; gap:.7rem; margin-top:1rem }
    .manager-profile { margin:1rem 0; padding:1rem; border:1px solid var(--line);
      border-radius:14px; background:#101713 }
    .journal-list { display:grid; gap:.55rem; margin:.75rem 0 }
    .matchday-command { margin:1rem 0; padding:1rem; border:1px solid #49695b;
      border-radius:14px; background:#0b1511 }
    .matchday-command h3 { margin:.1rem 0 .7rem }
    .matchday-journey { display:grid; grid-template-columns:repeat(4,1fr); gap:.45rem;
      padding:0; list-style:none }
    .manager-product-journey { display:grid; grid-template-columns:repeat(5,1fr);
      gap:.45rem; padding:0; margin:1rem 0; list-style:none }
    .manager-product-step { padding:.65rem; border:1px solid var(--line);
      border-radius:10px; color:var(--muted); text-align:center; background:#0c1210 }
    .manager-product-step[data-status="complete"],
    .manager-product-step[data-status="available"] { color:var(--accent) }
    .manager-product-step[data-status="action_required"] {
      color:var(--warn); border-color:var(--warn) }
    .matchday-step { padding:.55rem; border:1px solid var(--line); border-radius:9px;
      color:var(--muted); text-align:center }
    .matchday-step[data-status="complete"],.matchday-step[data-status="available"] { color:var(--accent) }
    .matchday-step[data-status="action_required"],.matchday-step[data-status="blocked"] { color:var(--warn); border-color:var(--warn) }
    .briefing-signal { margin:.55rem 0; padding:.75rem; border-left:4px solid var(--line);
      background:#0c1210; border-radius:0 9px 9px 0 }
    .briefing-signal[data-severity="high"] { border-left-color:var(--danger) }
    .briefing-signal[data-severity="moderate"] { border-left-color:var(--warn) }
    .briefing-signal[data-severity="informational"] { border-left-color:var(--accent) }
    .briefing-signal strong,.briefing-signal span { display:block }
    .decision-preview { margin:1rem 0; padding:1rem; border:1px solid #49695b;
      border-radius:12px; background:#0c1511 }
    .decision-preview[aria-busy="true"] { opacity:.72 }
    .standings { width:100%; border-collapse:collapse; margin-top:1rem }
    .standings th,.standings td { padding:.55rem; border-bottom:1px solid var(--line); text-align:right }
    .standings th:nth-child(2),.standings td:nth-child(2) { text-align:left }
    .squad-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:.55rem; margin:.75rem 0 }
    .squad-player { display:grid; grid-template-columns:1fr auto; align-items:center; gap:.6rem; padding:.55rem; border:1px solid var(--line); border-radius:9px }
    .squad-player.unavailable { opacity:.65 }
    .squad-player select { min-width:7.5rem }
    .danger-button { background:var(--danger) }
    code { overflow-wrap:anywhere }
    pre { max-height:340px; overflow:auto; padding:1rem; border-radius:10px; background:#070a09;
      color:#cbd5d0; white-space:pre-wrap; overflow-wrap:anywhere }
    [hidden] { display:none!important }
    @media(max-width:520px){ .row,.matchday-journey,.manager-product-journey { grid-template-columns:1fr } header { padding-top:2rem }
      .workspace-nav { position:static } .workspace-nav button { flex:1 1 45% }
      .workspace-nav .status { flex-basis:100%; text-align:left } }
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
  <nav id="workspace-nav" class="workspace-nav" aria-label="&#24037;&#20316;&#21488;&#21306;&#22495;" hidden>
    <button type="button" data-workspace-view="career" aria-pressed="true">&#32463;&#29702;&#29983;&#28079;</button>
    <button type="button" data-workspace-view="lab" aria-pressed="false">&#27604;&#36187;&#23454;&#39564;&#23460;</button>
    <button type="button" data-workspace-view="evidence" aria-pressed="false">&#35777;&#25454;&#20013;&#24515;</button>
    <button type="button" data-workspace-view="operations" aria-pressed="false">&#36816;&#32500;&#19982;&#24674;&#22797;</button>
    <p id="workspace-view-description" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
  </nav>
  <section id="overview-panel" class="overview" aria-labelledby="overview-title"><h2 id="overview-title">系统状态</h2>
    <div id="cards" class="cards" aria-live="polite" aria-atomic="true"></div><p id="workflow" class="status" role="status" aria-live="polite" aria-atomic="true"></p><button id="workflow-action" type="button" hidden>&#21069;&#24448;&#19979;&#19968;&#27493;</button>
    <button id="logout-button" type="button"__REMOTE_LOGOUT_HIDDEN__>安全退出</button>
    <p id="message" class="status" role="status" aria-live="polite" aria-atomic="true"></p><p id="report-link" hidden></p></section>
  <section id="action-panel" class="action" data-workspace-area="lab" aria-labelledby="action-title"><h2 id="action-title">工作区与单场比赛</h2>
    <form id="setup-form"><label>工作区名称<input name="name" maxlength="80" required value="My GFS Studio"></label>
      <div class="row"><label>模式<select name="mode"><option value="stable">稳定</option><option value="research">研究</option><option value="cognitive">认知</option></select></label>
      <label>随机种子<input name="seed" type="number" min="0" max="2147483647" value="42" required></label></div>
      <button type="submit">创建工作区</button></form>
    <form id="match-form" hidden><div class="row"><label>主队<input name="home" maxlength="80" value="Brazil" required></label>
      <label>客队<input name="away" maxlength="80" value="Argentina" required></label></div>
      <label>比赛体验<select name="experience"><option value="observational">原生观赛</option><option value="tactical_lab">战术实验室</option></select></label>
      <div id="tactical-options" class="row" hidden><label>主队战术<select name="home_tactic"></select></label><label>客队战术<select name="away_tactic"></select></label></div>
      <label class="check"><input name="reuse_last_seed" type="checkbox" disabled>复用上一场相同对阵的随机条件，用于配对比较</label>
      <label class="check"><input name="fast" type="checkbox" checked>快速模式</label>
      <p id="match-plan-guidance" class="status">原生观赛保留球队自身体系和稳定比分路径。</p><button type="submit">运行比赛</button></form>
    </section>
  <section id="pair-panel" class="wide" data-workspace-area="lab" aria-labelledby="pair-title" hidden><h2 id="pair-title">一键配对战术对决</h2>
    <p>自动连续运行基线与处理两场比赛，锁定相同对阵、快速配置和 seed，仅改变一侧战术，并生成可审计配对比较。</p>
    <form id="pair-form"><div class="row"><label>主队<input name="home" maxlength="80" value="Brazil" required></label><label>客队<input name="away" maxlength="80" value="Argentina" required></label></div>
      <div class="row"><label>干预侧<select name="focus_side"><option value="home">主队</option><option value="away">客队</option></select></label><label>共享 seed<input name="seed" type="number" min="0" max="2147483647" value="42" required></label></div>
      <div class="row"><label>基线战术<select name="baseline_tactic"></select></label><label>处理战术<select name="treatment_tactic"></select></label></div>
      <label>对手固定战术<select name="opponent_tactic"></select></label>
      <label class="check"><input name="fast" type="checkbox" checked>快速模式</label>
      <p id="pair-guidance" class="status">研究模式下，合格单个配对只支持这一固定对阵与 seed 的局部归因，不是总体效应或显著性检验。</p>
      <button type="submit">运行基线与处理配对</button></form></section>
  <section id="fork-panel" class="wide" data-workspace-area="lab" aria-labelledby="fork-title" hidden><h2 id="fork-title">世界模型因果分叉</h2>
    <p>自动创建两个严格配对的模拟世界：基线世界加载同一模型但禁止预测进入动作策略；干预世界只打开经过质量门控的动作策略。球队、战术、检查点、快速配置与 seed 全部固定。</p>
    <form id="fork-form"><div class="row"><label>主队<input name="home" maxlength="80" value="Brazil" required></label><label>客队<input name="away" maxlength="80" value="Argentina" required></label></div>
      <div class="row"><label>主队固定战术<select name="home_tactic"></select></label><label>客队固定战术<select name="away_tactic"></select></label></div>
      <label>共享 seed<input name="seed" type="number" min="0" max="2147483647" value="42" required></label>
      <label class="check"><input name="fast" type="checkbox" checked>快速模式</label>
      <p id="fork-guidance" class="status">仅研究模式可用。单个分叉证明这一固定模拟局面中策略开关造成的差异，不代表总体效应、真实足球因果关系或模型晋级。</p>
      <button type="submit">生成两个未来并比较</button></form></section>
  <section id="study-panel" class="wide" data-workspace-area="lab" aria-labelledby="study-title" hidden><h2 id="study-title">固定预算战术研究</h2>
    <p>只在研究模式运行：固定同一对阵与完整种子预算，仅改变一侧战术。预算完成前隐藏全部效应，避免可选停止和挑选结果。</p>
    <form id="study-form"><div class="row">
      <label>研究 ID<input name="study_id" pattern="[a-z0-9][a-z0-9_-]{2,63}" maxlength="64" value="pressing-study-01" required></label>
      <label>干预侧<select name="focus_side"><option value="home">主队</option><option value="away">客队</option></select></label></div>
      <div class="row"><label>主队<input name="home" maxlength="80" value="Brazil" required></label><label>客队<input name="away" maxlength="80" value="Argentina" required></label></div>
      <div class="row"><label>基线战术<select name="baseline_tactic"></select></label><label>处理战术<select name="treatment_tactic"></select></label></div>
      <label>对手固定战术<select name="opponent_tactic"></select></label>
      <div class="row"><label>固定配对预算<select name="pair_budget"><option value="4">4（代码验收）</option><option value="8" selected>8</option><option value="12">12</option><option value="16">16</option><option value="24">24</option><option value="32">32</option></select></label>
      <label>起始种子<input name="seed_start" type="number" min="0" max="2147483616" value="1001" required></label></div>
      <label class="check"><input name="fast" type="checkbox" checked>快速模式</label>
      <p id="study-guidance" class="status">主要指标预注册为干预侧 xG 差；结论仅适用于固定对阵与该种子集合，不自动授权产品推广。</p>
      <button type="submit">冻结协议并运行研究</button></form></section>
  <section id="season-panel" class="wide" data-workspace-area="career" aria-labelledby="season-title" hidden><h2 id="season-title">赛季中心</h2>
    <p id="season-summary" class="status" role="status" aria-live="polite" aria-atomic="true">创建工作区后可建立持续赛季。</p>
    <ol id="manager-product-journey" class="manager-product-journey" aria-label="&#32463;&#29702;&#20135;&#21697;&#23436;&#25972;&#20027;&#27969;&#31243;"></ol>
    <div id="matchday-command-center" class="matchday-command" aria-labelledby="matchday-command-title" hidden>
      <h3 id="matchday-command-title">比赛日指挥台</h3>
      <ol id="matchday-journey" class="matchday-journey" aria-label="比赛日流程"></ol>
      <div id="matchday-briefing" class="cards" role="list" aria-label="本轮赛前情报"></div>
      <div id="matchday-intelligence"></div>
      <div id="matchday-debrief" aria-live="polite" aria-atomic="true"></div>
      <div id="matchday-attribution"></div>
      <section id="manager-decision-ledger" aria-labelledby="manager-decision-ledger-title" hidden>
        <h3 id="manager-decision-ledger-title">经理决策闭环</h3>
        <p id="manager-decision-ledger-summary" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
        <p id="manager-advisor-evidence-summary" class="status"></p>
        <div id="manager-decision-ledger-list" class="journal-list" role="list" aria-label="经理决策、执行证据与长期影响"></div>
      </section>
    </div>
    <form id="season-form"><label>参赛球队（英文逗号分隔，4–16 队）<input name="teams" maxlength="1000" value="Brazil, Argentina, France, Germany" required></label>
      <div class="row"><label>循环次数<select name="legs"><option value="1">单循环</option><option value="2">双循环</option></select></label>
      <label>关注球队（可选）<input name="manager_team" maxlength="80" value="Brazil"></label></div>
      <div class="row"><label>赛季经理目标<select name="manager_objective"><option value="top_half">进入上半区</option><option value="champion">赢得冠军</option><option value="points_target">达到指定积分</option></select></label>
      <label>目标积分<input name="manager_points_target" type="number" min="1" max="90" value="6" disabled></label></div>
      <fieldset id="season-commitment-fieldset"><legend>&#36187;&#23395;&#25215;&#35834;&#21512;&#21516;</legend>
        <p class="status">&#36873;&#25321;&#36328;&#21608;&#21487;&#32467;&#31639;&#30340;&#25112;&#26415;&#19982;&#38453;&#23481;&#25215;&#35834;&#65307;&#36829;&#32422;&#21482;&#36827;&#20837;&#36879;&#26126;&#30340;&#33891;&#20107;&#20250;&#35780;&#23457;&#65292;&#19981;&#20250;&#26263;&#20013;&#20462;&#25913;&#27604;&#36187;&#32467;&#26524;&#12290;</p>
        <div class="row"><label>&#25112;&#26415;&#25215;&#35834;<select name="commitment_tactic_policy"><option value="club_identity">&#22362;&#23432;&#20465;&#20048;&#37096;&#36523;&#20221;&#65288;&#33267;&#23569; 60%&#65289;</option><option value="adaptive">&#20027;&#21160;&#36866;&#24212;&#65288;&#33267;&#23569; 2 &#31181;&#25112;&#26415;&#65289;</option></select></label>
        <label>&#38453;&#23481;&#25215;&#35834;<select name="commitment_rotation_policy"><option value="share_load">&#20998;&#25285;&#36127;&#33655;&#65288;&#33267;&#23569; 50% &#38750;&#26368;&#24378;&#38453;&#65289;</option><option value="trust_core">&#20449;&#20219;&#26680;&#24515;&#65288;&#33267;&#23569; 60% &#26368;&#24378;&#38453;&#65289;</option></select></label></div>
      </fieldset>
      <fieldset><legend>俱乐部长期资源（固定 6 点，单项 0–4）</legend><div class="row">
        <label>恢复设施<input name="resource_recovery" type="number" min="0" max="4" value="2" required></label>
        <label>医疗康复<input name="resource_medical" type="number" min="0" max="4" value="2" required></label>
        <label>运动科学<input name="resource_sports_science" type="number" min="0" max="4" value="2" required></label></div>
        <p id="club-resource-summary" class="status" role="status" aria-live="polite" aria-atomic="true">已分配 6/6 点。</p></fieldset>
      <fieldset id="sporting-director-fieldset" hidden><legend>体育总监赛季规划</legend>
        <p class="status">先冻结建队理念、风险边界和最多三个优先位置；同一计划同时约束续约、普通招募与自由签约。</p>
        <div class="row"><label>建队理念<select id="sporting-philosophy"><option value="balanced">均衡建设</option><option value="win_now">即战竞争</option><option value="youth_pathway">青年通道</option><option value="financial_control">财政控制</option></select></label>
        <label>风险边界<select id="sporting-risk"><option value="low">低风险</option><option value="balanced" selected>平衡</option><option value="high">高风险</option></select></label></div>
        <label>优先位置（可多选，最多 3 个）<select id="sporting-priority-roles" multiple size="5" aria-describedby="sporting-director-summary"></select></label>
        <div id="sporting-role-diagnostics" class="cards" role="list" aria-label="阵容位置诊断"></div>
        <p id="sporting-director-summary" class="status" role="status" aria-live="polite" aria-atomic="true">赛季完成后生成统一规划简报。</p></fieldset>
      <fieldset id="lifecycle-fieldset" hidden><legend>合同、退役与青训</legend>
        <p class="status">每季最多续约 4 名到期球员；退休不可撤销，离队空缺由同位置虚构青训球员补齐。</p>
        <div id="lifecycle-list" role="list" aria-label="到期合同与退休预览"></div>
        <p id="lifecycle-summary" class="status" role="status" aria-live="polite" aria-atomic="true">赛季完成后加载生命周期预览。</p></fieldset>
      <fieldset id="recruitment-fieldset" hidden><legend>下一赛季阵容建设（可选，窗口上限 8 点并受现金余额限制）</legend>
        <p class="status">最多完成两笔“签入 + 放走”交易；候选人为确定性虚构球员，签约会真实改变下一赛季名单。</p>
        <div class="recruitment-move row"><label>签入球员 1<select class="recruitment-candidate"><option value="">不签入</option></select></label><label>放走球员 1<select class="recruitment-outgoing"><option value="">不放走</option></select></label></div>
        <div class="recruitment-move row"><label>签入球员 2<select class="recruitment-candidate"><option value="">不签入</option></select></label><label>放走球员 2<select class="recruitment-outgoing"><option value="">不放走</option></select></label></div>
        <p id="recruitment-summary" class="status" role="status" aria-live="polite" aria-atomic="true">赛季完成后加载候选市场。</p></fieldset>
      <fieldset id="free-agent-fieldset" hidden><legend>全局自由球员市场（可选，最多签约 1 人）</legend>
        <p class="status">球员保留跨俱乐部身份；签约无转会费，但必须放走一名同位置球员，工资从新阵容继续结算。</p>
        <div class="row"><label>签入自由球员<select id="free-agent-candidate"><option value="">不签约</option></select></label>
        <label>同位置替换球员<select id="free-agent-outgoing"><option value="">不放走</option></select></label></div>
        <button id="scout-free-agent" type="button">侦察所选球员（每窗口 2 次）</button>
        <p id="free-agent-summary" class="status" role="status" aria-live="polite" aria-atomic="true">赛季完成后加载全局市场。</p></fieldset>
      <label class="check"><input name="fast" type="checkbox" checked>快速模式</label>
      <button type="submit">创建可恢复赛季</button></form>
    <form id="player-promise-form" hidden><fieldset><legend>&#20855;&#21517;&#29699;&#21592;&#35282;&#33394;&#25215;&#35834;</legend>
      <p class="status">&#26368;&#22810;&#19977;&#20154;&#65292;&#27599;&#31181;&#35282;&#33394;&#26368;&#22810;&#19968;&#20154;&#12290;&#33891;&#20107;&#20250;&#21482;&#32467;&#31639;&#32463;&#29702;&#21487;&#25511;&#30340;&#39318;&#21457;&#23433;&#25490;&#65307;&#23454;&#38469;&#20986;&#22330;&#20998;&#38047;&#21333;&#29420;&#35760;&#24405;&#12290;</p>
      <div class="player-promise-row row"><label>&#29699;&#21592; 1<select class="player-promise-player" required></select></label><label>&#35282;&#33394;<select class="player-promise-role"><option value="core">&#26680;&#24515;</option><option value="rotation">&#36718;&#25442;</option><option value="development">&#22521;&#20859;</option></select></label></div>
      <div class="player-promise-row row"><label>&#29699;&#21592; 2<select class="player-promise-player"><option value="">&#19981;&#25215;&#35834;</option></select></label><label>&#35282;&#33394;<select class="player-promise-role"><option value="rotation">&#36718;&#25442;</option><option value="core">&#26680;&#24515;</option><option value="development">&#22521;&#20859;</option></select></label></div>
      <div class="player-promise-row row"><label>&#29699;&#21592; 3<select class="player-promise-player"><option value="">&#19981;&#25215;&#35834;</option></select></label><label>&#35282;&#33394;<select class="player-promise-role"><option value="development">&#22521;&#20859;</option><option value="rotation">&#36718;&#25442;</option><option value="core">&#26680;&#24515;</option></select></label></div>
      <p id="player-promise-summary" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
      <button type="submit">&#20923;&#32467;&#29699;&#21592;&#25215;&#35834;</button></fieldset></form>
    <form id="manager-decision-form" hidden><p id="manager-fixture" class="status"></p>
      <fieldset id="club-situation-fieldset" hidden><legend>&#20465;&#20048;&#37096;&#24773;&#22659;&#20915;&#31574;</legend><p id="club-situation-summary" class="status"></p><label>&#32463;&#29702;&#21709;&#24212;<select id="club-situation-choice" aria-describedby="club-situation-tradeoff" required></select></label><p id="club-situation-tradeoff" class="status" role="status" aria-live="polite" aria-atomic="true"></p></fieldset>
      <div class="row"><label>本场战术<select name="tactic"></select></label><label>轮换策略<select name="rotation"><option value="strongest">最强阵容 · 体能代价高</option><option value="balanced" selected>平衡轮换</option><option value="rotate">大幅轮换 · 即战力下降</option></select></label></div>
      <fieldset><legend>比赛名单</legend><label class="check"><input id="manager-manual-lineup" type="checkbox">手动选择首发与替补</label>
        <p id="manager-squad-status" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
        <div id="manager-squad" class="squad-grid"></div></fieldset>
      <fieldset id="manager-in-match-plan"><legend>In-match response plan</legend>
        <p class="status">Each rule is evaluated once at its minute against the live score. An unmet rule is skipped.</p>
        <div class="row manager-rule"><label class="check"><input class="manager-rule-enabled" type="checkbox">Enable rule 1</label><label>Minute<input class="manager-rule-minute" type="number" min="1" max="89" value="60"></label><label>Condition<select class="manager-rule-condition"><option value="trailing">Trailing</option><option value="drawing">Drawing</option><option value="leading">Leading</option><option value="always">Always</option></select></label><label>Switch tactic<select class="manager-rule-tactic"><option value="gegenpress">Gegenpress</option><option value="balanced">Balanced</option><option value="tiki_taka">Tiki-taka</option><option value="counter_attack">Counter attack</option><option value="low_block_counter">Low block counter</option><option value="direct_vertical">Direct vertical</option></select></label></div>
        <div class="row manager-rule"><label class="check"><input class="manager-rule-enabled" type="checkbox">Enable rule 2</label><label>Minute<input class="manager-rule-minute" type="number" min="1" max="89" value="70"></label><label>Condition<select class="manager-rule-condition"><option value="trailing">Trailing</option><option value="drawing" selected>Drawing</option><option value="leading">Leading</option><option value="always">Always</option></select></label><label>Switch tactic<select class="manager-rule-tactic"><option value="direct_vertical">Direct vertical</option><option value="gegenpress">Gegenpress</option><option value="balanced">Balanced</option><option value="tiki_taka">Tiki-taka</option><option value="counter_attack">Counter attack</option><option value="low_block_counter">Low block counter</option></select></label></div>
        <div class="row manager-rule"><label class="check"><input class="manager-rule-enabled" type="checkbox">Enable rule 3</label><label>Minute<input class="manager-rule-minute" type="number" min="1" max="89" value="80"></label><label>Condition<select class="manager-rule-condition"><option value="trailing">Trailing</option><option value="drawing">Drawing</option><option value="leading" selected>Leading</option><option value="always">Always</option></select></label><label>Switch tactic<select class="manager-rule-tactic"><option value="low_block_counter">Low block counter</option><option value="balanced">Balanced</option><option value="gegenpress">Gegenpress</option><option value="tiki_taka">Tiki-taka</option><option value="counter_attack">Counter attack</option><option value="direct_vertical">Direct vertical</option></select></label></div>
      </fieldset>
      <section id="manager-world-model-advice" class="decision-preview" aria-labelledby="manager-world-model-advice-title" aria-live="polite" aria-atomic="true">
        <h3 id="manager-world-model-advice-title">世界模型赛前顾问</h3>
        <p id="manager-world-model-advice-summary" class="status">研究模式可显式请求一次绑定当前赛前状态的七战术比较；模型不会自动替你提交。</p>
        <div class="toolbar"><button id="request-manager-advice" type="button">请求世界模型建议</button><button id="adopt-manager-advice" type="button" disabled>采用模型建议</button></div>
        <div id="manager-world-model-advice-candidates" class="cards" role="list" aria-label="世界模型战术候选比较"></div>
        <p class="status">短视野策略代理，不预测比分或胜率；建议、经理选择和赛果分别取证。</p>
      </section>
      <section id="manager-decision-preview" class="decision-preview" aria-labelledby="manager-decision-preview-title" aria-live="polite" aria-atomic="true">
        <h3 id="manager-decision-preview-title">提交前影响预览</h3>
        <p class="status">调整方案后，这里显示将被正式冻结的可控机械后果；不会预测比分或胜率。</p>
      </section>
      <button type="submit">确认本场决策</button></form>
    <div id="season-actions" class="toolbar" hidden><button id="play-matchday" type="button">推进下一比赛日</button></div>
    <div id="manager-career-contract" class="manager-profile" hidden></div>
    <div id="manager-profile" class="manager-profile" hidden></div>
    <p id="season-history-summary" class="status" hidden></p>
    <div id="season-history" class="season-list" role="list" aria-label="已完成赛季历史"></div>
    <div id="season-standings"></div><div id="season-fixtures" class="season-list" role="list" aria-label="赛季赛程与比赛复盘"></div></section>
  <section id="action-adoption-panel" class="wide" data-workspace-area="evidence" aria-labelledby="action-adoption-title"><h2 id="action-adoption-title">世界模型动作采用</h2>
    <p id="action-adoption-summary" class="status" role="status" aria-live="polite" aria-atomic="true">正在读取动作采用状态……</p>
    <div id="action-adoption-metrics" class="cards" role="list" aria-label="世界模型动作采用指标"></div>
    <p id="action-adoption-evidence" class="status"></p>
    <p id="manager-advisor-protocol-evidence" class="status"></p></section>
  <section id="library-panel" class="wide" data-workspace-area="evidence" aria-labelledby="library-title"><div class="panel-head"><div><h2 id="library-title">证据资料库</h2><p id="library-summary" class="status" aria-live="polite" aria-atomic="true">正在读取历史工件……</p></div>
    <label>筛选<select id="library-filter"><option value="all">全部</option><option value="matches">比赛</option><option value="forks">因果分叉</option><option value="paired">战术配对</option><option value="studies">固定预算研究</option></select></label></div>
    <div id="library-list" class="cards" role="list" aria-label="历史比赛、世界模型分叉、配对比较和固定预算研究"></div></section>
  <section id="recovery-panel" class="wide" data-workspace-area="operations" aria-labelledby="recovery-title"><h2 id="recovery-title">恢复中心</h2>
    <p>备份保存在受控部署存储中。恢复前必须校验，并输入完整备份 ID 确认替换当前工作区。</p>
    <div class="toolbar"><button id="create-backup" type="button">创建完整备份</button><span id="recovery-message" class="status" role="status" aria-live="polite" aria-atomic="true"></span></div>
    <div id="backup-list" class="backup-list" role="list" aria-label="托管备份" aria-busy="true"></div>
    <form id="restore-form" hidden aria-labelledby="restore-title"><h3 id="restore-title">确认恢复</h3><p>准备恢复：<code id="selected-backup"></code></p>
      <label>输入完整备份 ID<input name="confirmation" maxlength="64" autocomplete="off" required></label>
      <label class="check"><input name="replace" type="checkbox" required>我确认使用该备份替换当前工作区并重置任务历史</label>
      <div class="toolbar"><button class="danger-button" type="submit">确认恢复</button><button id="cancel-restore" type="button">取消</button></div>
    </form></section>
  <section id="release-panel" class="wide" data-workspace-area="evidence" aria-labelledby="release-title"><h2 id="release-title">产品与论文发布门禁</h2>
    <p id="release-summary" class="status" role="status" aria-live="polite" aria-atomic="true">正在读取发布状态……</p>
    <div id="release-gates" class="cards" role="list" aria-label="发布门禁列表"></div></section>
  <section id="operations-detail-panel" class="wide" data-workspace-area="operations" aria-labelledby="evidence-title"><h2 id="evidence-title">证据与运行详情</h2><pre id="details" tabindex="0" aria-label="当前工作区和任务的 JSON 证据">正在读取……</pre></section>
</main>
<noscript><p role="alert" aria-live="assertive" aria-atomic="true">GFS Studio 需要 JavaScript 才能执行受控工作流。</p></noscript>
<script nonce="__NONCE__">
const csrf=document.querySelector('meta[name="gfs-csrf"]').content;
const workspaceNav=document.querySelector('#workspace-nav'),workspaceViewDescription=document.querySelector('#workspace-view-description'),workspaceViewButtons=[...document.querySelectorAll('[data-workspace-view]')],workspaceAreaSections=[...document.querySelectorAll('[data-workspace-area]')],actionPanel=document.querySelector('#action-panel');let currentWorkspaceArea='career',workspaceAreaExplicit=false;
const cards=document.querySelector('#cards'),setup=document.querySelector('#setup-form'),match=document.querySelector('#match-form'),pairPanel=document.querySelector('#pair-panel'),pairForm=document.querySelector('#pair-form'),forkPanel=document.querySelector('#fork-panel'),forkForm=document.querySelector('#fork-form'),studyPanel=document.querySelector('#study-panel'),studyForm=document.querySelector('#study-form'),seasonPanel=document.querySelector('#season-panel'),seasonForm=document.querySelector('#season-form'),seasonSummary=document.querySelector('#season-summary'),matchdayCommand=document.querySelector('#matchday-command-center'),matchdayJourney=document.querySelector('#matchday-journey'),matchdayBriefing=document.querySelector('#matchday-briefing'),matchdayIntelligence=document.querySelector('#matchday-intelligence'),matchdayDebrief=document.querySelector('#matchday-debrief'),matchdayAttribution=document.querySelector('#matchday-attribution'),managerDecisionLedger=document.querySelector('#manager-decision-ledger'),managerDecisionLedgerSummary=document.querySelector('#manager-decision-ledger-summary'),managerDecisionLedgerList=document.querySelector('#manager-decision-ledger-list'),managerDecisionForm=document.querySelector('#manager-decision-form'),managerFixture=document.querySelector('#manager-fixture'),managerTactic=managerDecisionForm.querySelector('[name="tactic"]'),managerRotation=managerDecisionForm.querySelector('[name="rotation"]'),managerManual=document.querySelector('#manager-manual-lineup'),managerSquadStatus=document.querySelector('#manager-squad-status'),managerSquad=document.querySelector('#manager-squad'),managerRules=[...managerDecisionForm.querySelectorAll('.manager-rule')],managerDecisionPreview=document.querySelector('#manager-decision-preview'),managerDecisionSubmit=managerDecisionForm.querySelector('button[type="submit"]'),managerWorldModelAdvice=document.querySelector('#manager-world-model-advice'),managerWorldModelAdviceSummary=document.querySelector('#manager-world-model-advice-summary'),managerWorldModelAdviceCandidates=document.querySelector('#manager-world-model-advice-candidates'),requestManagerAdvice=document.querySelector('#request-manager-advice'),adoptManagerAdvice=document.querySelector('#adopt-manager-advice'),seasonActions=document.querySelector('#season-actions'),playMatchday=document.querySelector('#play-matchday'),seasonStandings=document.querySelector('#season-standings'),seasonFixtures=document.querySelector('#season-fixtures');let activePoll='',restoreTrigger=null,currentSeason=null,managerPreviewTimer=0,managerPreviewSequence=0,managerPreviewBaseRevision=null,currentManagerAdvice=null,managerAdviceIntent=null;
const managerCareerContract=document.querySelector('#manager-career-contract'),managerProfilePanel=document.querySelector('#manager-profile'),seasonHistorySummary=document.querySelector('#season-history-summary'),seasonHistory=document.querySelector('#season-history'),seasonObjective=seasonForm.querySelector('[name="manager_objective"]'),seasonPointsTarget=seasonForm.querySelector('[name="manager_points_target"]'),clubResourceSummary=document.querySelector('#club-resource-summary'),clubResourceInputs=[...seasonForm.querySelectorAll('[name^="resource_"]')];let currentCareer=null;
const managerTeamInput=seasonForm.querySelector('[name="manager_team"]'),recruitmentFieldset=document.querySelector('#recruitment-fieldset'),recruitmentSummary=document.querySelector('#recruitment-summary'),recruitmentRows=[...seasonForm.querySelectorAll('.recruitment-move')];let currentRecruitmentMarket=null,recruitmentRequestTeam='';
const lifecycleFieldset=document.querySelector('#lifecycle-fieldset'),lifecycleList=document.querySelector('#lifecycle-list'),lifecycleSummary=document.querySelector('#lifecycle-summary');let currentLifecyclePreview=null,lifecycleRequestTeam='';
const freeAgentFieldset=document.querySelector('#free-agent-fieldset'),freeAgentCandidate=document.querySelector('#free-agent-candidate'),freeAgentOutgoing=document.querySelector('#free-agent-outgoing'),freeAgentSummary=document.querySelector('#free-agent-summary'),scoutFreeAgent=document.querySelector('#scout-free-agent');let currentFreeAgentMarket=null,freeAgentRequestTeam='';
const sportingDirectorFieldset=document.querySelector('#sporting-director-fieldset'),sportingPhilosophy=document.querySelector('#sporting-philosophy'),sportingRisk=document.querySelector('#sporting-risk'),sportingPriorityRoles=document.querySelector('#sporting-priority-roles'),sportingRoleDiagnostics=document.querySelector('#sporting-role-diagnostics'),sportingDirectorSummary=document.querySelector('#sporting-director-summary');let currentSportingPlan=null,sportingRequestTeam='';
const clubSituationFieldset=document.querySelector('#club-situation-fieldset'),clubSituationSummary=document.querySelector('#club-situation-summary'),clubSituationChoice=document.querySelector('#club-situation-choice'),clubSituationTradeoff=document.querySelector('#club-situation-tradeoff');let currentClubSituation=null;
const playerPromiseForm=document.querySelector('#player-promise-form'),playerPromiseRows=[...playerPromiseForm.querySelectorAll('.player-promise-row')],playerPromiseSummary=document.querySelector('#player-promise-summary');
const releaseSummary=document.querySelector('#release-summary'),releaseGates=document.querySelector('#release-gates');
const evidenceKitDownload=document.createElement('a'),evidenceKitRow=document.createElement('p');evidenceKitDownload.id='evidence-kit-download';evidenceKitDownload.href='/api/v1/excellence/evidence-kit.zip';evidenceKitDownload.download='gfs-excellence-evidence-kit-v1.zip';evidenceKitDownload.setAttribute('aria-describedby','release-summary');evidenceKitDownload.textContent='Download template-only evidence kit';evidenceKitRow.append(evidenceKitDownload);releaseSummary.insertAdjacentElement('afterend',evidenceKitRow);
const message=document.querySelector('#message'),details=document.querySelector('#details'),workflow=document.querySelector('#workflow'),workflowAction=document.querySelector('#workflow-action'),report=document.querySelector('#report-link'),logoutButton=document.querySelector('#logout-button');
const adoptionSummary=document.querySelector('#action-adoption-summary'),adoptionMetrics=document.querySelector('#action-adoption-metrics'),adoptionEvidence=document.querySelector('#action-adoption-evidence');
const librarySummary=document.querySelector('#library-summary'),libraryList=document.querySelector('#library-list'),libraryFilter=document.querySelector('#library-filter');let currentLibrary={matches:[],pairs:[],forks:[],studies:[]};
const experienceSelect=match.querySelector('[name="experience"]'),tacticalOptions=document.querySelector('#tactical-options'),homeTactic=match.querySelector('[name="home_tactic"]'),awayTactic=match.querySelector('[name="away_tactic"]'),reuseSeed=match.querySelector('[name="reuse_last_seed"]'),planGuidance=document.querySelector('#match-plan-guidance');let currentMatchCapabilities=null,currentStudioMode='stable';
const baselineTactic=studyForm.querySelector('[name="baseline_tactic"]'),treatmentTactic=studyForm.querySelector('[name="treatment_tactic"]'),opponentTactic=studyForm.querySelector('[name="opponent_tactic"]');
const pairBaselineTactic=pairForm.querySelector('[name="baseline_tactic"]'),pairTreatmentTactic=pairForm.querySelector('[name="treatment_tactic"]'),pairOpponentTactic=pairForm.querySelector('[name="opponent_tactic"]');
const pairGuidance=document.querySelector('#pair-guidance');
const forkHomeTactic=forkForm.querySelector('[name="home_tactic"]'),forkAwayTactic=forkForm.querySelector('[name="away_tactic"]'),forkGuidance=document.querySelector('#fork-guidance');
const createBackupButton=document.querySelector('#create-backup'),backupList=document.querySelector('#backup-list'),recoveryMessage=document.querySelector('#recovery-message'),restoreForm=document.querySelector('#restore-form'),selectedBackup=document.querySelector('#selected-backup'),cancelRestore=document.querySelector('#cancel-restore');
const esc=v=>String(v??'—');
const workspaceAreaDescriptions={career:'经理生涯 · 赛季、比赛日、阵容与长期俱乐部后果',lab:'比赛实验室 · 隔离的单场、配对与固定预算战术研究',evidence:'证据中心 · 世界模型采用、历史工件与发布门禁',operations:'运维与恢复 · 备份、恢复和原始运行状态'};
const workflowAreaByAction={start_season:'career',freeze_player_promises:'career',submit_manager_decision:'career',advance_season_matchday:'career',resume_season_matchday:'career',review_sporting_plan:'career',review_season:'career',run_standalone_match:'lab',open_dashboard:'evidence',resolve_readiness:'operations',inspect_season_state:'operations'};
try{const storedArea=sessionStorage.getItem('gfs-workspace-area');if(workspaceAreaDescriptions[storedArea]){currentWorkspaceArea=storedArea;workspaceAreaExplicit=true}}catch(_error){}
function applyWorkspaceArea(area,{remember=false,configured=true}={}){const resolved=workspaceAreaDescriptions[area]?area:'career';currentWorkspaceArea=resolved;workspaceNav.hidden=!configured;for(const section of workspaceAreaSections){const visible=configured?section.dataset.workspaceArea===resolved:section===actionPanel;section.classList.toggle('workspace-area-hidden',!visible)}for(const button of workspaceViewButtons)button.setAttribute('aria-pressed',button.dataset.workspaceView===resolved?'true':'false');workspaceViewDescription.textContent=workspaceAreaDescriptions[resolved];if(remember){workspaceAreaExplicit=true;try{sessionStorage.setItem('gfs-workspace-area',resolved)}catch(_error){}}}
function workflowWorkspaceArea(data){const action=data.studio?.workflow?.next_action?.id;if(workflowAreaByAction[action])return workflowAreaByAction[action];if(data.studio?.season)return'career';if(Number(data.studio?.matches_played||0)>0)return'evidence';return'career'}
function renderWorkspaceAreas(data){const configured=Boolean(data.configured),active=(data.tasks||[]).some(task=>['queued','running'].includes(task?.state));if(!workspaceAreaExplicit&&!active)currentWorkspaceArea=workflowWorkspaceArea(data);applyWorkspaceArea(currentWorkspaceArea,{configured})}
for(const button of workspaceViewButtons)button.addEventListener('click',()=>applyWorkspaceArea(button.dataset.workspaceView,{remember:true,configured:true}));
function announce(node,text,kind='status',focus=false){node.className='status '+(kind==='error'?'error':kind==='success'?'ok':'');node.setAttribute('role',kind==='error'?'alert':'status');node.setAttribute('aria-live',kind==='error'?'assertive':'polite');node.setAttribute('aria-atomic','true');node.textContent=text;if(focus){node.setAttribute('tabindex','-1');node.focus()}}
function setBusy(node,busy){node.setAttribute('aria-busy',busy?'true':'false')}
function card(label,value){const el=document.createElement('div');el.className='card';const a=document.createElement('span');a.className='label';a.textContent=label;const b=document.createElement('strong');b.className='value';b.textContent=esc(value);el.append(a,b);return el}
function renderRelease(data){releaseGates.replaceChildren();const cp=data.control_plane||data.studio?.control_plane||{},release=cp.release,ex=cp.excellence||{};if(!release){releaseSummary.textContent='发布状态尚不可用。';return}const product=release.scores?.product??ex.tracks?.product?.score??'—',academic=release.scores?.academic??ex.tracks?.academic?.score??'—',plan=release.completion_plan||{},actions=new Map((plan.steps||[]).map(step=>[step.gate_id,step])),kit=plan.evidence_kit?.ready?'就绪':'待检查';releaseSummary.textContent=`产品 ${product}/100 · 学术 ${academic}/100 · 代码契约${release.code_ready?'通过':'未通过'} · 最终发布${release.release_ready?'就绪':'未就绪'} · 证据包${kit} · ${release.open_gate_count} 个开放门禁 · 下一步 ${plan.recommended_gate_id||'无'}`;for(const gate of release.gates||[]){const action=actions.get(gate.id),state=gate.passed?'通过':action?.state||'待完成',detail=action&&!gate.passed?`${state} · 责任角色 ${action.operator}`:state,item=card(gate.label,detail);item.setAttribute('role','listitem');item.dataset.gateId=gate.id;if(action)item.dataset.actionState=action.state;releaseGates.append(item)}}
function showReport(url,comparisonUrl,studyUrl,baselineUrl){report.replaceChildren();if(!url&&!comparisonUrl&&!studyUrl&&!baselineUrl){report.hidden=true;return}const links=[[baselineUrl,'打开配对基线（新窗口）'],[url,'打开比赛/处理场仪表板（新窗口）'],[comparisonUrl,'打开双世界/战术配对比较（新窗口）'],[studyUrl,'打开固定预算战术研究（新窗口）']];for(const [href,label] of links){if(!href)continue;if(report.childNodes.length)report.append(document.createTextNode(' · '));const link=document.createElement('a');link.href=href;link.target='_blank';link.rel='noopener';link.textContent=label;report.append(link)}report.hidden=false}
function populateTactics(select,tactics){const selected=select.value;select.replaceChildren();for(const tactic of tactics||[]){const option=document.createElement('option');option.value=tactic.id;option.textContent=tactic.label;option.title=tactic.description||'';select.append(option)}if([...select.options].some(option=>option.value===selected))select.value=selected}
function configureMatchPlan(capabilities,mode){currentMatchCapabilities=capabilities||currentMatchCapabilities||{};currentStudioMode=mode||currentStudioMode;populateTactics(homeTactic,currentMatchCapabilities.tactics);populateTactics(awayTactic,currentMatchCapabilities.tactics);const labOption=[...experienceSelect.options].find(option=>option.value==='tactical_lab'),labAllowed=['research','cognitive'].includes(currentStudioMode);labOption.disabled=!labAllowed;if(!labAllowed&&experienceSelect.value==='tactical_lab')experienceSelect.value='observational';const lab=experienceSelect.value==='tactical_lab';if(lab&&homeTactic.value==='team_identity'&&awayTactic.value==='team_identity')homeTactic.value='balanced';tacticalOptions.hidden=!lab;homeTactic.disabled=!lab;awayTactic.disabled=!lab;reuseSeed.disabled=!lab;if(!lab)reuseSeed.checked=false;planGuidance.textContent=lab?'战术实验使用物理比分。单场结果只作描述；复用上一场随机条件后才能进行配对归因。':'原生观赛保留球队自身体系和稳定比分路径。'}
function configureTacticalStudy(capabilities,mode){const tactics=capabilities?.tactics||[];for(const select of [baselineTactic,treatmentTactic,opponentTactic])populateTactics(select,tactics);if(!baselineTactic.dataset.initialized){baselineTactic.value='balanced';treatmentTactic.value='gegenpress';opponentTactic.value='low_block_counter';baselineTactic.dataset.initialized='true'}studyPanel.hidden=mode!=='research'}
function configurePairedMatch(capabilities,mode,seed){const tactics=capabilities?.tactics||[];for(const select of [pairBaselineTactic,pairTreatmentTactic,pairOpponentTactic])populateTactics(select,tactics);if(!pairBaselineTactic.dataset.initialized){pairBaselineTactic.value='balanced';pairTreatmentTactic.value='gegenpress';pairOpponentTactic.value='low_block_counter';pairBaselineTactic.dataset.initialized='true'}if(!pairForm.dataset.seedInitialized&&Number.isInteger(Number(seed))){pairForm.querySelector('[name="seed"]').value=String(seed);pairForm.dataset.seedInitialized='true'}pairPanel.hidden=!['research','cognitive'].includes(mode);pairGuidance.textContent=mode==='cognitive'?'认知模式含不受共享 seed 完全控制的供应商输出；配对仅作描述，自动撤销战术归因资格。':'研究模式下，资格检查全部通过的单个配对只支持这一固定对阵与 seed 的局部归因，不是总体效应或显著性检验。'}
function configureWorldModelFork(capabilities,mode,seed){const tactics=capabilities?.tactics||[];for(const select of [forkHomeTactic,forkAwayTactic])populateTactics(select,tactics);if(!forkForm.dataset.tacticsInitialized){forkHomeTactic.value='balanced';forkAwayTactic.value='low_block_counter';forkForm.dataset.tacticsInitialized='true'}if(!forkForm.dataset.seedInitialized&&Number.isInteger(Number(seed))){forkForm.querySelector('[name="seed"]').value=String(seed);forkForm.dataset.seedInitialized='true'}forkPanel.hidden=mode!=='research';forkGuidance.textContent=mode==='research'?'基线与干预使用同一世界模型检查点；唯一计划差异是 MATCH_WM_PLAN=0 → 1。结果只解释这一固定模拟局面。':'请创建研究模式工作区；稳定模式不加载世界模型，认知模式含无法由共享 seed 完全控制的供应商输出。'}
function renderActionAdoption(studio){adoptionMetrics.replaceChildren();if(!studio){adoptionSummary.textContent='创建工作区后，这里会显示世界模型对动作选择的实际影响。';adoptionEvidence.textContent='运行观测和机制证据将分别呈现。';return}const mechanism=studio.evidence?.action_adoption_mechanism||{},research=studio.mode!=='stable',latest=studio.last_match||{},adoption=latest.world_model_action_adoption||{};if(!research){adoptionSummary.textContent='稳定模式不启用研究型世界模型动作策略，比赛由稳定模拟器决策。';adoptionMetrics.append(card('策略状态','未启用'),card('最近动作影响','不适用'))}else if(!latest.match_id){adoptionSummary.textContent='动作策略已配置并受质量门控；运行一场比赛后可观察实际影响。';adoptionMetrics.append(card('策略状态','已配置·质量门控'),card('决策机会','尚无比赛'),card('实际动作改变','尚无比赛'))}else{const opportunities=Number(adoption.opportunities||0),influenced=Number(adoption.influenced_opportunities||0),eligible=Number(adoption.attribution_eligible_opportunities||0),changed=Number(adoption.counterfactual_action_changes||0),expected=Number(adoption.expected_counterfactual_action_changes||0),shift=Number(adoption.mean_recommended_probability_shift||0);adoptionSummary.textContent=`最近比赛 ${latest.home} vs ${latest.away}：世界模型在 ${influenced}/${opportunities} 个决策机会中产生非零概率影响。`;adoptionMetrics.append(card('非零影响',`${influenced}/${opportunities}`),card('可归因机会',`${eligible}/${opportunities}`),card('实际动作改变',changed),card('期望动作改变',expected.toFixed(3)),card('平均推荐概率偏移',(shift*100).toFixed(3)+' pp'))}if(mechanism.available){const done=mechanism.runs_executed??0,total=mechanism.fixed_run_budget??'—',state=mechanism.execution_state||mechanism.protocol_state||'unknown',promotion=mechanism.promotion_authorized?'已授权推广':'未授权推广';adoptionEvidence.textContent=`机制证据：${state} · ${done}/${total} 次固定预算运行 · ${promotion}。单场运行观测不等于机制证明。`}else{adoptionEvidence.textContent='机制证据协议缺失；当前只能查看运行观测，不能形成机制结论。'}}
function artifactLink(label,url){if(!url)return null;const link=document.createElement('a');link.href=url;link.target='_blank';link.rel='noopener';link.textContent=label;return link}
async function resumeInterruptedTask(taskId,button){button.disabled=true;button.setAttribute('aria-busy','true');announce(message,'正在恢复已持久化的配对事务……');try{const data=await api('/api/v1/tasks/'+encodeURIComponent(taskId)+'/requeue',{method:'POST',body:JSON.stringify({reason:'studio_pair_transaction_resume'})});announce(message,'配对事务已使用原任务身份重新排队。','success');activePoll=data.task.task_id;await pollTask(data.task.task_id)}catch(error){announce(message,error.message,'error',true)}finally{button.disabled=false;button.setAttribute('aria-busy','false')}}
function renderLibrary(library){
  currentLibrary=library||{matches:[],pairs:[],forks:[],studies:[]};libraryList.replaceChildren();
  const filter=libraryFilter.value,matches=currentLibrary.matches||[],pairs=currentLibrary.pairs||[],forks=currentLibrary.forks||[],studies=currentLibrary.studies||[];
  const visibleMatches=['paired','forks','studies'].includes(filter)?[]:matches,visiblePairs=['matches','forks','studies'].includes(filter)?[]:pairs,visibleForks=['matches','paired','studies'].includes(filter)?[]:forks,visibleStudies=['matches','paired','forks'].includes(filter)?[]:studies;
  librarySummary.textContent=`${matches.length} 场比赛 · ${forks.length} 个世界模型分叉 · ${pairs.length} 个战术配对 · ${studies.length} 项固定预算研究${currentLibrary.truncated?' · 仅显示最近 50 项':''}`;
  for(const item of visibleMatches){const node=card(`${item.home} vs ${item.away}`,`${item.score?.home??'—'}–${item.score?.away??'—'}`);node.setAttribute('role','listitem');const meta=document.createElement('p');meta.className='status';meta.textContent=`${item.match_id} · seed ${item.seed} · ${item.experience} · 完整性 ${item.integrity}`;const links=document.createElement('p'),matchLink=artifactLink('比赛复盘',item.dashboard_url),pairLink=artifactLink('配对比较',item.comparison_url);if(matchLink)links.append(matchLink);if(matchLink&&pairLink)links.append(document.createTextNode(' · '));if(pairLink)links.append(pairLink);node.append(meta,links);libraryList.append(node)}
  for(const item of visibleForks){const node=card(`${item.home} vs ${item.away}`,item.state);node.setAttribute('role','listitem');const meta=document.createElement('p');meta.className='status';meta.textContent=`共享 seed ${item.seed} · predict_only → action_policy · 战术固定 ${item.home_tactic}/${item.away_tactic}`;const propagation=document.createElement('p'),evidence=item.propagation||{};propagation.className='status';propagation.textContent=evidence.available?`传播证据：${Number(evidence.changed_decisions||0)} 次动作改变 · ${Number(evidence.directly_observed_changes||0)} 次直接观察 · ${Number(evidence.locally_attributable_changes||0)} 次局部归因；下游窗口仅描述`:(item.state==='completed'?`传播证据不可用：${evidence.status||'unknown'}`:'传播证据将在双世界完成后生成');const links=document.createElement('p'),base=artifactLink('基线世界',item.baseline_url),treatment=artifactLink('干预世界',item.treatment_url),comparison=artifactLink('分叉传播与对比',item.comparison_url);for(const link of [base,treatment,comparison]){if(!link)continue;if(links.childNodes.length)links.append(document.createTextNode(' · '));links.append(link)}node.append(meta,propagation,links);if(item.state==='interrupted'){const resume=document.createElement('button');resume.type='button';resume.textContent='安全恢复因果分叉';resume.addEventListener('click',()=>resumeInterruptedTask(item.task_id,resume));node.append(resume)}libraryList.append(node)}
  for(const item of visiblePairs){const node=card(`${item.home} vs ${item.away}`,item.state);node.setAttribute('role','listitem');const meta=document.createElement('p');meta.className='status';meta.textContent=`共享 seed ${item.seed} · ${item.focus_side} 侧 · ${item.baseline_tactic} → ${item.treatment_tactic}`;const links=document.createElement('p'),base=artifactLink('基线复盘',item.baseline_url),treatment=artifactLink('处理复盘',item.treatment_url),comparison=artifactLink('配对比较',item.comparison_url);for(const link of [base,treatment,comparison]){if(!link)continue;if(links.childNodes.length)links.append(document.createTextNode(' · '));links.append(link)}node.append(meta,links);if(item.state==='interrupted'){const resume=document.createElement('button');resume.type='button';resume.textContent='安全恢复配对事务';resume.addEventListener('click',()=>resumeInterruptedTask(item.task_id,resume));node.append(resume)}libraryList.append(node)}
  for(const item of visibleStudies){const node=card(item.study_id,item.state);node.setAttribute('role','listitem');const meta=document.createElement('p');meta.className='status';meta.textContent=`${item.home} vs ${item.away} · ${item.baseline_tactic} → ${item.treatment_tactic} · ${item.pairs_completed}/${item.fixed_pair_budget} 对${item.analysis_withheld?' · 分析隐藏':''}`;const links=document.createElement('p'),studyLink=artifactLink('研究复盘',item.study_url);if(studyLink)links.append(studyLink);node.append(meta,links);libraryList.append(node)}
  if(!libraryList.childNodes.length){const empty=document.createElement('p');empty.className='status';empty.textContent='当前筛选下还没有可用证据。';libraryList.append(empty)}
}
function lineupRole(lineup,playerId){if((lineup?.starters||[]).includes(playerId))return 'starter';if((lineup?.bench||[]).includes(playerId))return 'bench';return 'reserve'}
function renderManagerSquad(season,managed){managerSquad.replaceChildren();const squad=season?.manager_squad;if(!squad?.available){managerManual.checked=false;managerManual.disabled=true;managerSquadStatus.textContent=squad?.reason==='roster_unavailable'?'该球队没有可用的球员 roster；本场只能使用球队级自动轮换。':squad?.reason==='roster_missing_goalkeeper'?'该 roster 缺少可信门将数据；为避免伪造角色，本场只启用球队级管理。':'当前伤停与位置覆盖无法组成合法 11 人名单；请先恢复阵容状态。';return}managerManual.disabled=false;const frozen=managed?.manager_decision?.lineup,matchingFrozen=frozen&&managed.manager_decision.rotation===managerRotation.value?frozen:null,suggested=matchingFrozen||squad.automatic?.[managerRotation.value],manual=managerManual.checked;let starters=0,bench=0;for(const player of squad.players||[]){const label=document.createElement('label');label.className='squad-player'+(player.selectable?'':' unavailable');const info=document.createElement('span');info.textContent=`${player.shirt_number??'—'} · ${player.name} · ${player.role}${player.injured?' · 伤病':''}${player.suspended?' · 停赛':''}`;const select=document.createElement('select');select.dataset.playerId=player.player_id;select.setAttribute('aria-label',`${player.name} 的名单角色`);for(const [value,text] of [['reserve','未入选'],['starter','首发'],['bench','替补']]){const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option)}select.value=player.selectable?lineupRole(suggested,player.player_id):'reserve';select.disabled=!manual||!player.selectable;if(select.value==='starter')starters+=1;if(select.value==='bench')bench+=1;select.addEventListener('change',()=>updateManagerSquadCount());label.append(info,select);managerSquad.append(label)}managerSquadStatus.textContent=`${manual?'手动名单':'自动建议'} · ${starters} 人首发 · ${bench} 人替补 · 当前阵型 ${squad.formation||'—'}`}
function updateManagerSquadCount(){let starters=0,bench=0;for(const select of managerSquad.querySelectorAll('select')){if(select.value==='starter')starters+=1;if(select.value==='bench')bench+=1}managerSquadStatus.textContent=`手动名单 · ${starters} 人首发 · ${bench} 人替补${starters===11&&bench<=12?'':' · 需要恰好 11 名首发且替补不超过 12 人'}`}
function manualLineupPayload(){if(!managerManual.checked)return null;const starters=[],bench=[];for(const select of managerSquad.querySelectorAll('select')){if(select.value==='starter')starters.push(select.dataset.playerId);if(select.value==='bench')bench.push(select.dataset.playerId)}return{starters,bench,source:'manual',roster_fingerprint:currentSeason?.manager_squad?.roster_fingerprint||''}}
function setPlayerOptions(select,ids,players,placeholder,selected){select.replaceChildren();const blank=document.createElement('option');blank.value='';blank.textContent=placeholder;select.append(blank);for(const id of ids||[]){const player=players.get(id);if(!player)continue;const option=document.createElement('option');option.value=id;option.textContent=`${player.name} · ${player.role}`;select.append(option)}if([...select.options].some(option=>option.value===selected))select.value=selected}
function ruleLineup(managed){return manualLineupPayload()||managed?.manager_decision?.lineup||currentSeason?.manager_squad?.automatic?.[managerRotation.value]||{starters:[],bench:[]}}
function populateRulePlayers(row,managed,substitution){const squad=currentSeason?.manager_squad,players=new Map((squad?.players||[]).map(player=>[player.player_id,player])),lineup=ruleLineup(managed);setPlayerOptions(row.querySelector('.manager-sub-off'),lineup.starters,players,'No player off',substitution?.off||'');setPlayerOptions(row.querySelector('.manager-sub-on'),lineup.bench,players,'No player on',substitution?.on||'')}
function renderManagerRules(managed){const frozen=managed?.manager_decision?.in_match_plan?.instructions||[];for(const [index,row] of managerRules.entries()){const rule=frozen[index],enabled=row.querySelector('.manager-rule-enabled'),minute=row.querySelector('.manager-rule-minute'),condition=row.querySelector('.manager-rule-condition'),tactic=row.querySelector('.manager-rule-tactic');enabled.checked=Boolean(rule);if(rule){minute.value=String(rule.minute);condition.value=rule.condition;tactic.value=rule.tactic||''}populateRulePlayers(row,managed,rule?.substitution);for(const control of row.querySelectorAll('input:not(.manager-rule-enabled),select'))control.disabled=!enabled.checked}}
function inMatchPlanPayload(){const instructions=[];for(const [index,row] of managerRules.entries()){if(!row.querySelector('.manager-rule-enabled').checked)continue;const off=row.querySelector('.manager-sub-off').value,on=row.querySelector('.manager-sub-on').value,tactic=row.querySelector('.manager-rule-tactic').value||null;if(Boolean(off)!==Boolean(on))throw new Error('A planned substitution requires both an off and an on player.');if(!tactic&&!off)throw new Error('Each enabled rule needs a tactic or a substitution.');instructions.push({rule_id:`studio-rule-${index+1}`,minute:Number(row.querySelector('.manager-rule-minute').value),condition:row.querySelector('.manager-rule-condition').value,tactic,substitution:off?{off,on}:null})}instructions.sort((left,right)=>left.minute-right.minute);if(new Set(instructions.map(rule=>rule.minute)).size!==instructions.length)throw new Error('In-match rule minutes must be unique.');return instructions.length?{team:managerDecisionForm.dataset.team,instructions}:null}
function renderManagerWorldModelAdvice(advice,season,managed){managerWorldModelAdviceCandidates.replaceChildren();currentManagerAdvice=advice?.advice_identity?advice:null;const researchMode=['research','cognitive'].includes(currentStudioMode),current=Boolean(currentManagerAdvice&&currentManagerAdvice.issued_revision===season?.revision&&currentManagerAdvice.fixture_id===managed?.fixture_id);requestManagerAdvice.disabled=!researchMode||!managed||managed.state!=='scheduled'||Number(managed.attempts||0)!==0;adoptManagerAdvice.disabled=!current;if(!currentManagerAdvice){managerWorldModelAdviceSummary.textContent=researchMode?'尚未请求本场建议。请求会运行本地已校准候选，对全部七种可玩战术作一次有界比较。':'稳定模式不加载或模仿研究世界模型；经理决策继续使用确定性机械预览。';return}const reliability=currentManagerAdvice.reliability||{},adoption=managed?.manager_advice_adoption,intentLabels={adopt_recommendation:'明确采用建议',reviewed_then_selected:'查看后自行选择'};managerWorldModelAdviceSummary.textContent=`建议 ${currentManagerAdvice.recommended_tactic} · 置信 ${Math.round(Number(currentManagerAdvice.recommendation_confidence||0)*100)}% · 前两名差值 ${Number(currentManagerAdvice.recommendation_margin||0).toFixed(3)} · 历史信任 ${Math.round(Number(reliability.adjusted_recommendation_trust||0)*100)}% · ${reliability.evidence_tier||'证据不足'}${current?'':' · 已过期，请重新请求'}${adoption?' · '+(intentLabels[adoption.intent]||adoption.intent):''}`;for(const row of currentManagerAdvice.candidates||[]){const node=card(`#${row.rank} · ${row.tactic}`,`风险调整值 ${Number(row.risk_adjusted_value).toFixed(3)} · 置信 ${Math.round(Number(row.effective_confidence||0)*100)}% · 不确定性 ${Math.round(Number(row.uncertainty||0)*100)}% · 疲劳代理 ${Math.round(Number(row.fatigue_cost_proxy||0)*100)}%`);node.setAttribute('role','listitem');managerWorldModelAdviceCandidates.append(node)}}
async function requestManagerWorldModelAdvice(){if(!managerDecisionForm.dataset.fixtureId)return;requestManagerAdvice.disabled=true;adoptManagerAdvice.disabled=true;managerAdviceIntent=null;setBusy(managerWorldModelAdvice,true);managerWorldModelAdviceSummary.textContent='正在用本地候选模型比较七种战术；不会运行比赛……';try{const data=await api('/api/v1/seasons/decision-advice',{method:'POST',body:JSON.stringify({fixture_id:managerDecisionForm.dataset.fixtureId})});if(!data.advice?.available){currentManagerAdvice=null;managerWorldModelAdviceSummary.textContent=`世界模型建议不可用：${data.advice?.reason||'质量门禁关闭'}`;return}await refresh()}catch(error){managerWorldModelAdviceSummary.className='status error';managerWorldModelAdviceSummary.textContent=`世界模型建议失败：${error.message}`}finally{setBusy(managerWorldModelAdvice,false);if(!currentManagerAdvice)requestManagerAdvice.disabled=!['research','cognitive'].includes(currentStudioMode)}}
function adoptCurrentManagerAdvice(){if(!currentManagerAdvice||adoptManagerAdvice.disabled)return;managerTactic.value=currentManagerAdvice.recommended_tactic;managerAdviceIntent='adopt_recommendation';managerWorldModelAdviceSummary.textContent+=` · 已选择采用 ${currentManagerAdvice.recommended_tactic}，仍需完成影响预览并正式提交。`;scheduleManagerDecisionPreview()}
function managerDecisionPayload(includeRevision=false){const lineup=manualLineupPayload();if(lineup&&(lineup.starters.length!==11||lineup.bench.length>12))throw new Error('手动名单需要恰好 11 名首发，且替补不能超过 12 人。');const inMatchPlan=inMatchPlanPayload(),decision={team:managerDecisionForm.dataset.team,tactic:managerTactic.value,rotation:managerRotation.value};if(lineup)decision.lineup=lineup;if(inMatchPlan)decision.in_match_plan=inMatchPlan;if(currentClubSituation)decision.club_event_choice={schema_version:1,event_id:currentClubSituation.event_id,event_identity:currentClubSituation.event_identity,choice_id:clubSituationChoice.value};const payload={fixture_id:managerDecisionForm.dataset.fixtureId,decision};if(includeRevision){if(!Number.isInteger(managerPreviewBaseRevision))throw new Error('请等待当前方案的影响预览完成后再提交。');payload.expected_revision=managerPreviewBaseRevision;if(currentManagerAdvice&&managerAdviceIntent&&currentManagerAdvice.issued_revision===managerPreviewBaseRevision){payload.advice_adoption={schema_version:1,advice_identity:currentManagerAdvice.advice_identity,intent:managerAdviceIntent}}}return payload}
function renderManagerDecisionPreview(preview){managerDecisionPreview.replaceChildren();const title=document.createElement('h3');title.id='manager-decision-preview-title';title.textContent='提交前影响预览';const facts=document.createElement('div');facts.className='cards';facts.setAttribute('role','list');facts.setAttribute('aria-label','正式提交将冻结的影响');const decision=preview.normalized_decision||{},lineup=preview.lineup||{},tradeoff=preview.engine_tradeoff||{},inMatch=preview.in_match_plan||{};for(const [label,value] of [['冻结方案',`${decision.tactic} · ${decision.rotation}`],['比赛名单',lineup.available?`${lineup.source} · ${lineup.starters} 首发 / ${lineup.bench} 替补`:'当前无可冻结名单'],['固定机械权衡',`状态代价 ${tradeoff.status_penalty??'—'} · 轮换等级 ${tradeoff.rotation_level??'—'}`],['临场预案',`${inMatch.rule_count??0} 条规则${inMatch.controls_substitutions?' · 包含换人控制':''}`]]){const node=card(label,value);node.setAttribute('role','listitem');facts.append(node)}managerDecisionPreview.append(title,facts);const commitment=preview.season_commitment_projection;if(commitment?.entries?.length){const heading=document.createElement('h4');heading.textContent=`若本次决策计入后的赛季承诺 · ${commitment.recorded_matches} 场`;const list=document.createElement('ul');const labels={board_objective:'董事会目标',tactical_identity:'战术身份',squad_stewardship:'阵容管理'};for(const row of commitment.entries){const item=document.createElement('li');const metric=row.metric||{},value=row.id==='tactical_identity'?`身份占比 ${Math.round((metric.identity_share||0)*100)}% · 战术种类 ${metric.distinct_tactics||0}`:row.id==='squad_stewardship'?`计数占比 ${Math.round((metric.counted_share||0)*100)}%`:`当前 ${metric.points??0} 分`;item.textContent=`${labels[row.id]||row.id}：${row.status} · ${value}`;list.append(item)}managerDecisionPreview.append(heading,list)}const players=preview.player_promise_projection;if(players?.entries?.length){const heading=document.createElement('h4');heading.textContent=`若本次名单计入后的具名球员承诺 · ${players.completed_matches} 场`;const list=document.createElement('ul');for(const row of players.entries){const item=document.createElement('li');item.textContent=`${row.name}：${row.status} · 首发 ${row.starts}/${row.known_lineups} · ${Math.round(row.start_share*100)}% / 要求 ${Math.round(row.minimum_start_share*100)}%${row.excused_unavailable?' · 豁免 '+row.excused_unavailable+' 场':''}`;list.append(item)}managerDecisionPreview.append(heading,list)}const situation=preview.club_situation?.resolution;if(situation){const detail=document.createElement('p');detail.className='status';detail.textContent=`俱乐部情境：${situation.choice?.choice_id||'—'} · ${situation.control==='manager'?'经理明确选择':'确定性兼容选择'}`;managerDecisionPreview.append(detail)}const prep=preview.opponent_preparation||{},boundary=document.createElement('p');boundary.className='status';boundary.textContent=`对手准备：${prep.state||'不可用'} · ${prep.selected_tactic||'team_identity'}。这是零写入的可控机械预览，不预测比分、胜率，也不作因果声明。`;managerDecisionPreview.append(boundary)}
async function refreshManagerDecisionPreview(){const sequence=++managerPreviewSequence;if(managerDecisionForm.hidden||!managerDecisionForm.dataset.fixtureId)return;let payload;try{payload=managerDecisionPayload()}catch(error){managerDecisionPreview.replaceChildren();const title=document.createElement('h3');title.id='manager-decision-preview-title';title.textContent='提交前影响预览';const status=document.createElement('p');status.className='status error';status.textContent=error.message;managerDecisionPreview.append(title,status);return}managerDecisionPreview.setAttribute('aria-busy','true');try{const data=await api('/api/v1/seasons/decision-preview',{method:'POST',body:JSON.stringify(payload)});if(sequence===managerPreviewSequence){managerPreviewBaseRevision=data.preview.base_revision;renderManagerDecisionPreview(data.preview);managerDecisionSubmit.disabled=false}}catch(error){if(sequence===managerPreviewSequence){managerDecisionPreview.replaceChildren();const title=document.createElement('h3');title.id='manager-decision-preview-title';title.textContent='提交前影响预览';const status=document.createElement('p');status.className='status error';status.textContent=error.message;managerDecisionPreview.append(title,status)}}finally{if(sequence===managerPreviewSequence)managerDecisionPreview.setAttribute('aria-busy','false')}}
function scheduleManagerDecisionPreview(){clearTimeout(managerPreviewTimer);managerPreviewBaseRevision=null;managerDecisionSubmit.disabled=true;managerPreviewTimer=setTimeout(()=>void refreshManagerDecisionPreview(),160)}
for(const row of managerRules){const tactic=row.querySelector('.manager-rule-tactic'),none=document.createElement('option');none.value='';none.textContent='No tactical switch';tactic.insertBefore(none,tactic.firstChild);for(const [labelText,className] of [['Player off','manager-sub-off'],['Player on','manager-sub-on']]){const label=document.createElement('label'),select=document.createElement('select');label.textContent=labelText;select.className=className;label.append(select);row.append(label)}}
const renderManagerSquadWithoutRules=renderManagerSquad;
renderManagerSquad=(season,managed)=>{renderManagerSquadWithoutRules(season,managed);renderManagerRules(managed)};
function renderSeason(season,configured){currentSeason=season||null;seasonPanel.hidden=!configured;seasonStandings.replaceChildren();seasonFixtures.replaceChildren();if(!configured)return;if(!season){managerSquad.replaceChildren();seasonForm.hidden=false;managerDecisionForm.hidden=true;seasonActions.hidden=true;seasonSummary.textContent='尚未创建赛季。赛季比赛会结算疲劳、士气与伤停，单场实验不会污染长期状态。';return}seasonForm.hidden=true;const managed=season.next_manager_fixture,managerTeam=season.plan?.manager_team,needsDecision=Boolean(managed&&managerTeam&&!managed.manager_decision);managerDecisionForm.hidden=!managed||!managerTeam;populateTactics(managerTactic,currentMatchCapabilities?.tactics||[]);if(managed&&managerTeam){const fixtureChanged=managerDecisionForm.dataset.fixtureId!==managed.fixture_id;managerDecisionForm.dataset.fixtureId=managed.fixture_id;managerDecisionForm.dataset.team=managerTeam;if(fixtureChanged)managerManual.checked=false;managerFixture.textContent=`下一场：第 ${managed.matchday} 轮 · ${managed.home} vs ${managed.away}`;if(managed.manager_decision){managerTactic.value=managed.manager_decision.tactic;managerRotation.value=managed.manager_decision.rotation;managerManual.checked=managed.manager_decision.lineup?.source==='manual'}}renderManagerSquad(season,managed);seasonActions.hidden=season.next_matchday===null;playMatchday.disabled=season.next_matchday===null||needsDecision;seasonSummary.textContent=`${season.season_id} · ${season.state} · ${season.progress?.completed??0}/${season.progress?.total??0} 场 · ${season.next_matchday===null?'赛季完成':'下一比赛日 '+season.next_matchday}${needsDecision?' · 请先提交本场决策':''}`;const table=document.createElement('table');table.className='standings';const caption=document.createElement('caption');caption.textContent='实时积分榜';const head=document.createElement('thead'),headRow=document.createElement('tr');for(const label of ['排名','球队','赛','胜','平','负','净胜','积分']){const th=document.createElement('th');th.scope='col';th.textContent=label;headRow.append(th)}head.append(headRow);const body=document.createElement('tbody');for(const row of season.standings||[]){const tr=document.createElement('tr');for(const value of [row.position,row.team,row.played,row.won,row.drawn,row.lost,row.goal_difference,row.points]){const td=document.createElement('td');td.textContent=esc(value);tr.append(td)}body.append(tr)}table.append(caption,head,body);seasonStandings.append(table);for(const fixture of season.fixtures||[]){const score=fixture.score?`${fixture.score.home}–${fixture.score.away}`:fixture.state,node=card(`第 ${fixture.matchday} 轮 · ${fixture.home} vs ${fixture.away}`,score);node.setAttribute('role','listitem');const meta=document.createElement('p');meta.className='status';const choice=fixture.manager_decision?` · ${fixture.manager_decision.tactic} / ${fixture.manager_decision.rotation}${fixture.manager_decision.lineup?' · 11 人首发':''}`:'';meta.textContent=`${fixture.fixture_id} · ${fixture.state}${fixture.attempts?' · 尝试 '+fixture.attempts+' 次':''}${choice}`;const link=artifactLink('打开比赛复盘',fixture.dashboard_url);node.append(meta);if(link)node.append(link);seasonFixtures.append(node)}}
function renderMatchdayCommand(season,configured){matchdayJourney.replaceChildren();matchdayBriefing.replaceChildren();matchdayDebrief.replaceChildren();const command=season?.matchday_command_center;if(!configured||!command?.enabled){matchdayCommand.hidden=true;return}matchdayCommand.hidden=false;const journeyLabels={briefing:'赛前情报',decision:'阵容与预案',matchday:'比赛日',debrief:'赛后复盘'},statusLabels={complete:'已完成',available:'可复盘',action_required:'需要决策',blocked:'等待决策',ready:'已就绪',pending:'待比赛',not_applicable:'本轮不适用'};for(const step of command.journey||[]){const item=document.createElement('li');item.className='matchday-step';item.dataset.status=step.status;item.textContent=`${journeyLabels[step.id]||step.id} · ${statusLabels[step.status]||step.status}`;matchdayJourney.append(item)}const resources=command.club_resources||{},resourceEffects=resources.effects||{},phaseLabels={decision_required:'需要提交本场决策',ready_to_advance:'本比赛日已就绪',ready_to_resume:'恢复未完成比赛日',season_complete:'赛季已完成'},fixture=command.current_fixture||command.next_fixture,cardsToShow=[['当前阶段',phaseLabels[command.phase]||command.phase],['球队',command.manager_team],['近期状态',command.recent_form],['俱乐部资源',`恢复 ${resources.recovery??'—'} · 医疗 ${resources.medical??'—'} · 运动科学 ${resources.sports_science??'—'}`],['长期机械效果',`轮后恢复 ${resourceEffects.manager_rest_units??'—'} · 医疗进度 ${resourceEffects.medical_recovery_credit_per_matchday??'—'} · 疲劳负荷 ×${resourceEffects.fatigue_load_factor??'—'} · 当场实力 +0`]];if(fixture){const decisionState=fixture.state==='completed'?'本轮比赛已完成':fixture.decision?'已冻结，可重新提交直至开赛':command.is_bye_matchday?'本轮轮空，无需决策':'尚未提交',fixtureLabel=command.current_fixture?'本轮对阵':'下一场对手',preparation=fixture.opponent_preparation||{},evidence=preparation.evidence||{},prepState={adapted:'启用针对性准备',insufficient_evidence:'证据不足，保持原生体系',ambiguous_pattern:'模式不明确，保持原生体系',native_pattern:'观察到原生体系，不覆盖',unsupported_pattern:'无响应规则，保持原生体系'}[preparation.state]||preparation.state||'尚无';cardsToShow.push([fixtureLabel,`${fixture.venue==='home'?'主场':'客场'} vs ${fixture.opponent}`],['积分形势',`第 ${fixture.manager_position} · ${fixture.manager_points} 分 / 对手第 ${fixture.opponent_position} · ${fixture.opponent_points} 分`],['决策状态',decisionState],['对手准备',`${prepState} · ${evidence.observation_count??0}/${preparation.thresholds?.minimum_observations??2} 场 · ${preparation.selected_tactic||'team_identity'}${fixture.decision?' · 已冻结':' · 提交后冻结'}`])}for(const [label,value] of cardsToShow){const node=card(label,value);node.setAttribute('role','listitem');matchdayBriefing.append(node)}const last=command.last_result;if(last){const title=document.createElement('h4');title.textContent='上一场执教复盘';const summary=document.createElement('p');const outcome={win:'胜',draw:'平',loss:'负'}[last.outcome]||last.outcome;summary.textContent=`第 ${last.matchday} 轮 · ${last.home} ${last.score.home}–${last.score.away} ${last.away} · ${outcome} · 获得 ${last.points_earned} 分`;matchdayDebrief.append(title,summary);const prep=last.opponent_preparation;if(prep?.available){const audit=document.createElement('p');audit.className='status';audit.textContent=`对手准备审计：${prep.state} · 使用 ${prep.selected_tactic} · 仅基于此前 ${prep.evidence?.observation_count??0} 场已完成决策。`;matchdayDebrief.append(audit)}const safeFixture=(season.fixtures||[]).find(item=>item.fixture_id===last.fixture_id),link=artifactLink('打开完整比赛复盘',safeFixture?.dashboard_url);if(link)matchdayDebrief.append(link)}else{const empty=document.createElement('p');empty.className='status';empty.textContent='首场比赛完成后，这里会显示结果、积分收益与审计复盘入口。';matchdayDebrief.append(empty)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='俱乐部资源只改变后续持久状态，不直接增加当场实力；对手响应也只是固定游戏规则，不是学习结果或已证明的克制关系。';matchdayDebrief.append(boundary);playMatchday.disabled=command.phase==='season_complete'||command.decision_required;playMatchday.textContent=command.is_bye_matchday?`推进第 ${command.current_matchday} 比赛日（本队轮空）`:command.phase==='season_complete'?'赛季已完成':command.is_resume_matchday?`恢复第 ${command.current_matchday} 比赛日`:`进行第 ${command.current_matchday} 比赛日`;if(command.is_bye_matchday)managerDecisionForm.hidden=true;seasonSummary.textContent=`${season.season_id} · ${season.progress?.completed??0}/${season.progress?.total??0} 场 · ${phaseLabels[command.phase]||command.phase}${command.decision_required?' · 下一步：提交阵容与临场预案':''}`}
function renderManagerIntelligence(command,configured){matchdayIntelligence.replaceChildren();matchdayAttribution.replaceChildren();if(!configured||!command?.enabled)return;const briefing=command.prematch_intelligence;if(briefing?.available){const title=document.createElement('h4');title.textContent='有证据的赛前决策情报';const coverage=document.createElement('p');coverage.className='status';const coverageLabels={direct_persisted_state:'双方持久化比赛状态 + 当前 roster',mixed_persisted_and_baseline:'一方持久化状态 + 一方默认基线',roster_plus_default_baseline:'当前 roster + 默认赛季基线',partial_team_state:'部分球队状态'};coverage.textContent=`证据覆盖：${coverageLabels[briefing.evidence_coverage]||briefing.evidence_coverage}`;const facts=document.createElement('div');facts.className='cards';facts.setAttribute('role','list');facts.setAttribute('aria-label','球队状态证据');for(const [label,value] of [['本队状态',`疲劳 ${(Number(briefing.manager?.condition?.fatigue||0)*100).toFixed(0)}% · 士气 ${(Number(briefing.manager?.condition?.morale||0)*100).toFixed(0)}% · 缺阵 ${briefing.manager?.condition?.unavailable_players??0}`],['可用阵容',`${briefing.manager?.selectable_players??0} 人 · ${briefing.manager?.formation||'未知'} · 高负荷 ${briefing.manager?.high_load_players??0} 人`],['对手状态',`${briefing.opponent?.formation||'未知阵型'} · 可用 ${briefing.opponent?.selectable_players??0} 人 · 缺阵 ${briefing.opponent?.condition?.unavailable_players??0}`]]){const node=card(label,value);node.setAttribute('role','listitem');facts.append(node)}const signals=document.createElement('div');signals.setAttribute('role','list');signals.setAttribute('aria-label','赛前风险与决策提示');for(const signal of briefing.signals||[]){const item=document.createElement('div');item.className='briefing-signal';item.dataset.severity=signal.severity;item.setAttribute('role','listitem');const heading=document.createElement('strong'),evidence=document.createElement('span'),prompt=document.createElement('span');heading.textContent=signal.title;evidence.className='status';evidence.textContent=`证据：${signal.evidence}`;prompt.textContent=`决策提示：${signal.decision_prompt}`;item.append(heading,evidence,prompt);signals.append(item)}const tradeoff=document.createElement('details'),summary=document.createElement('summary'),list=document.createElement('ul');summary.textContent='查看轮换的固定引擎权衡';for(const [label,key] of [['最强阵容','strongest'],['平衡轮换','balanced'],['大幅轮换','rotate']]){const row=briefing.rotation_tradeoff?.[key]||{},item=document.createElement('li');item.textContent=`${label}：状态代价 ${row.status_penalty??'—'}，轮换等级 ${row.rotation_level??'—'}`;list.append(item)}const caveat=document.createElement('p');caveat.className='status';caveat.textContent='这些是模拟器机械权衡，不是胜率预测或现实建议。';tradeoff.append(summary,list,caveat);matchdayIntelligence.append(title,coverage,facts,signals,tradeoff)}const debrief=command.postmatch_debrief;if(!debrief)return;const title=document.createElement('h4');title.textContent='经理决策执行证据';matchdayAttribution.append(title);if(!debrief.available){const unavailable=document.createElement('p');unavailable.className='status';unavailable.textContent=`直接执行证据不可用：${debrief.reason||'unknown'}`;matchdayAttribution.append(unavailable);return}const effects=debrief.direct_engine_effects||{},execution=debrief.in_match_execution||{},facts=document.createElement('div');facts.className='cards';facts.setAttribute('role','list');facts.setAttribute('aria-label','经理决策直接执行事实');for(const [label,value] of [['冻结方案',`${debrief.decision?.tactic} · ${debrief.decision?.rotation} · ${debrief.decision?.lineup_source}`],['机械状态变化',`${Number(effects.combined_status_delta||0)>=0?'+':''}${Number(effects.combined_status_delta||0).toFixed(2)}（轮换代价 ${effects.rotation_status_penalty??'—'}）`],['疲劳负荷倍率',Number(effects.fatigue_load_multiplier??1).toFixed(2)],['临场指令',`${execution.applied??0} 执行 / ${execution.skipped??0} 跳过 / ${execution.failed??0} 失败`]]){const node=card(label,value);node.setAttribute('role','listitem');facts.append(node)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='机械效果和指令执行是直接运行证据；单场比分仅作描述，不能证明该决策导致赛果。';matchdayAttribution.append(facts,boundary)}
const renderManagerIntelligenceWithoutClubSupport=renderManagerIntelligence;
renderManagerIntelligence=(command,configured)=>{renderManagerIntelligenceWithoutClubSupport(command,configured);const debrief=command?.postmatch_debrief,support=debrief?.club_support;if(!configured||!debrief?.available||!support)return;const allocation=support.plan||{},effects=debrief.direct_engine_effects||{},value=support.available?`恢复 ${allocation.recovery} · 医疗 ${allocation.medical} · 运动科学 ${allocation.sports_science} · 疲劳结算 ×${Number(effects.club_fatigue_load_factor??1).toFixed(2)}`:'旧报告未记录资源证据',node=card('俱乐部长期支持',value);node.setAttribute('role','listitem');matchdayAttribution.append(node)};
const renderMatchdayCommandWithoutIntelligence=renderMatchdayCommand;
function renderClubStrategyBriefing(briefing){if(!briefing?.available)return;const manager=briefing.manager||{},opponent=briefing.opponent||{},preview=briefing.applied_preview,title=document.createElement('h4'),facts=document.createElement('div');title.textContent='赛季身份与对手打法';facts.className='cards';facts.setAttribute('role','list');facts.setAttribute('aria-label','俱乐部赛季身份情报');const postureLabels={proactive:'主动进取',balanced:'稳定平衡',conservative:'谨慎务实'},actual=preview?(preview.home.team+' '+preview.home_tactic+' / '+preview.away.team+' '+preview.away_tactic):'提交本场决策后生成';for(const [label,value] of [['本队身份',manager.primary_tactic+' · '+(postureLabels[manager.risk_posture]||manager.risk_posture)],['对手身份',opponent.primary_tactic+' · '+(postureLabels[opponent.risk_posture]||opponent.risk_posture)],['对手本场基线',briefing.opponent_identity_tactic],['冻结后实际打法',actual]]){const node=card(label,value);node.setAttribute('role','listitem');facts.append(node)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='身份来自当前阵容、上季排名和有限惯性；它是可重放游戏策略，不是胜率预测。';matchdayIntelligence.append(title,facts,boundary)}
renderMatchdayCommand=(season,configured)=>{renderMatchdayCommandWithoutIntelligence(season,configured);renderManagerIntelligence(season?.matchday_command_center,configured);renderClubStrategyBriefing(season?.matchday_command_center?.club_strategy)};
function renderCareerContract(career){currentCareer=career||null;managerCareerContract.replaceChildren();managerCareerContract.hidden=true;if(!career?.available)return;managerCareerContract.hidden=false;const title=document.createElement('h3');title.textContent='经理合同与职业声望';const projected=career.projected_after_review||null,status=projected?.employment_status||career.employment_status,statusLabels={secure:'地位稳固',stable:'正常续任',under_review:'董事会观察',dismissed:'已被解雇',unappointed:'未任职'},summary=document.createElement('div');summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','经理长期合同状态');for(const [label,value] of [['当前俱乐部',career.current_team],['董事会信任',projected?.board_confidence??career.board_confidence],['职业声望',projected?.reputation??career.reputation],['任职状态',statusLabels[status]||status],['已完成赛季',career.completed_seasons],['达成目标',career.objectives_achieved]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}managerCareerContract.append(title,summary);if(career.pending_review){const review=document.createElement('p');review.className='status';review.textContent=`待归档董事会评价：${career.pending_review.grade} · 信任 ${career.pending_review.confidence_delta>=0?'+':''}${career.pending_review.confidence_delta} · 声望 ${career.pending_review.reputation_delta>=0?'+':''}${career.pending_review.reputation_delta}`;managerCareerContract.append(review)}const contract=career.next_contract||{},notice=document.createElement('p');notice.className=status==='dismissed'?'status error':'status';if(!contract.same_club_allowed)notice.textContent=`已失去 ${career.current_team} 的续任资格；下一赛季必须选择另一支参赛球队。`;else if(contract.same_club_points_growth_required)notice.textContent=`观察期续任：若选择积分目标，不得低于上季 ${contract.baseline_points} 分再加 ${contract.same_club_points_growth_required} 分；夺冠或上半区目标仍可用。`;else notice.textContent='当前没有额外续任约束。';const boundary=document.createElement('p');boundary.className='status';boundary.textContent=`信任与声望是最近 ${career.history_scope?.retained_seasons??0} 份保留赛季折算的游戏分数，不是现实教练评价或因果证据。`;managerCareerContract.append(notice,boundary)}
function objectiveText(objective){if(!objective)return'未设置';if(objective.kind==='champion')return'赢得冠军';if(objective.kind==='top_half')return`进入前 ${objective.target_position} 名`;return`达到 ${objective.target_points} 分`}
function renderClubFinance(finance){if(!finance?.available)return;managerCareerContract.hidden=false;const title=document.createElement('h3');title.textContent='俱乐部财政与工资';const wage=finance.current_wage||{},projection=finance.projected_after_current_season,summary=document.createElement('div');summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','俱乐部财政状态');for(const [label,value] of [['当前现金',`${finance.balance} 点`],['下窗可用',`${finance.recruitment_allowance}/${finance.recruitment_window_cap} 点`],['本季工资',`${wage.seasonal_wage_expense??'—'} 点${wage.source?' · 球队级基线':''}`],['赛季后预测',projection?`${projection.settlement.delta>=0?'+':''}${projection.settlement.delta} · 余额 ${projection.balance_after}`:'赛季进行中']]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}managerCareerContract.append(title,summary);if((finance.ledger_entries||[]).length){const detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div');detailsTitle.textContent=`查看财政账本（${finance.ledger_entry_count} 条）`;list.setAttribute('role','list');for(const entry of (finance.ledger_entries||[]).slice(-8).reverse()){const label=entry.type==='season_settlement'?'赛季结算':'引援支出',node=card(`${entry.season_id} · ${label}`,`${entry.delta>=0?'+':''}${entry.delta} · ${entry.balance_before} → ${entry.balance_after}`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);managerCareerContract.append(detailsNode)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='财政为有界游戏积分账本；收入、工资和估值不代表现实俱乐部财务数据。';managerCareerContract.append(boundary)}

function renderLeagueEcosystem(ecosystem){const event=ecosystem?.latest_transition;if(!ecosystem?.available||!event)return;seasonHistory.hidden=false;const clubs=event.ai_clubs||[],moves=clubs.reduce((total,club)=>total+(club.decision?.selected_moves||[]).length,0),title=document.createElement('h3'),summary=document.createElement('div');title.textContent='动态联赛生态';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','非玩家俱乐部演化摘要');for(const [label,value] of [['赛季演化',`${event.from_season_id} → ${event.to_season_id}`],['AI 俱乐部',`${clubs.length} 支`],['实际补强',`${moves} 笔`],['玩家控制',event.manager_controlled_team||'无']]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}seasonHistory.append(title,summary);const detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div');detailsTitle.textContent='查看各俱乐部决策与证据';list.setAttribute('role','list');for(const club of clubs){const decision=club.decision||{},selected=decision.selected_moves||[],strategy=decision.strategy==='rebuild'?'重建':'选择性补强',description=!decision.available?'缺少阵容证据，未执行':selected.length?`${strategy} · 第 ${decision.position}/${decision.league_size} 名 · 现金上限 ${decision.spending_cap} · ${selected.map(move=>`${move.outgoing_name||move.outgoing_player_id} → ${move.candidate_name||move.candidate_id}（+${Number(move.quality_improvement).toFixed(3)}，${move.cost} 点）`).join('；')}`:`${strategy} · 第 ${decision.position}/${decision.league_size} 名 · 没有达到 +${Number(decision.minimum_quality_improvement).toFixed(3)} 门槛的可负担升级`;const node=card(club.team,description);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);seasonHistory.append(detailsNode);const boundary=document.createElement('p');boundary.className='status';boundary.textContent='AI 俱乐部使用可重放的游戏策略、现金和虚构市场；这不是现实转会或竞技因果判断。';seasonHistory.append(boundary)}
function renderPlayerDevelopment(development){const transactions=development?.latest_transactions||[];if(!development?.available||!transactions.length)return;seasonHistory.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div');title.textContent='球员成长与生涯';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','球员成长结算摘要');const changed=transactions.reduce((total,item)=>total+Number(item.summary?.changed_players||0),0),minutes=transactions.reduce((total,item)=>total+Number(item.summary?.total_minutes_observed||0),0),complete=transactions.filter(item=>item.summary?.evidence_coverage==='complete').length;for(const [label,value] of [['结算俱乐部',`${transactions.length} 支`],['能力变化球员',`${changed} 人`],['精确上场分钟',minutes.toFixed(0)],['完整证据',`${complete}/${transactions.length}`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看各俱乐部成长、年龄曲线与证据';list.setAttribute('role','list');for(const transaction of transactions){const records=(transaction.players||[]).filter(player=>player.status==='developed').sort((left,right)=>Math.abs(Number(right.season_delta||0))-Math.abs(Number(left.season_delta||0))),leaders=records.slice(0,3).map(player=>`${player.name} ${Number(player.season_delta||0)>=0?'+':''}${Number(player.season_delta||0).toFixed(3)}（${Number(player.minutes||0).toFixed(0)} 分钟）`).join('；')||'没有具备年龄证据的留队球员',node=card(`${transaction.team} · ${transaction.source_season_id} → ${transaction.target_season_id}`,`${transaction.summary?.changed_players||0} 人变化 · 平均能力 ${Number(transaction.summary?.mean_ability_delta||0)>=0?'+':''}${Number(transaction.summary?.mean_ability_delta||0).toFixed(4)} · 体育科学 ${transaction.sports_science_points} 点`);node.setAttribute('role','listitem');const evidence=document.createElement('p');evidence.className='status';evidence.textContent=`证据 ${transaction.summary?.evidence_coverage||'unknown'} · 观察 ${Number(transaction.summary?.total_minutes_observed||0).toFixed(0)} 分钟 · ${leaders}`;node.append(evidence);list.append(node)}detailsNode.append(detailsTitle,list);const boundary=document.createElement('p');boundary.className='status';boundary.textContent='成长只使用冻结年龄、可用比赛报告中的精确分钟和赛季开始前分配的体育科学资源；缺失分钟不会被推测。';seasonHistory.append(title,summary,detailsNode,boundary)}
function renderManagerCareer(season,history,historySummary,configured){managerProfilePanel.replaceChildren();seasonHistory.replaceChildren();managerProfilePanel.hidden=true;seasonHistorySummary.hidden=true;if(!configured)return;const profile=season?.manager_profile;if(profile?.available){managerProfilePanel.hidden=false;const title=document.createElement('h3');title.textContent=`${profile.team} · 赛季经理档案`;const objective=profile.objective||{},resources=season.plan?.manager_resources||{},statusLabels={achieved:'已达成',missed:'未达成',currently_meeting:'当前达到',in_progress:'进行中',unreachable:'数学上已不可达'},summary=document.createElement('div');summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','经理目标与赛季记录');for(const [label,value] of [['冻结目标',objectiveText(objective)],['目标状态',statusLabels[objective.status]||objective.status],['当前排名与积分',`第 ${objective.current_position} · ${objective.current_points} 分`],['执教战绩',`${profile.record?.win??0} 胜 ${profile.record?.draw??0} 平 ${profile.record?.loss??0} 负`],['俱乐部资源',`恢复 ${resources.recovery??2} · 医疗 ${resources.medical??2} · 运动科学 ${resources.sports_science??2}`],['主要战术',profile.decision_identity?.dominant_tactic||'尚无'],['主要轮换',profile.decision_identity?.dominant_rotation||'尚无']]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}managerProfilePanel.append(title,summary);const journal=profile.journal||[];if(journal.length){const detailsNode=document.createElement('details'),journalTitle=document.createElement('summary'),list=document.createElement('div');journalTitle.textContent=`查看决策日志（${journal.length} 场）`;list.className='journal-list';list.setAttribute('role','list');for(const entry of journal.slice(-8).reverse()){const node=card(`第 ${entry.matchday} 轮 · ${entry.venue==='home'?'主场':'客场'} vs ${entry.opponent}`,`${entry.goals_for}–${entry.goals_against} · ${entry.outcome} · ${entry.tactic} / ${entry.rotation}`);node.setAttribute('role','listitem');const meta=document.createElement('p');meta.className='status';const prep=entry.opponent_preparation;meta.textContent=`${entry.lineup_source} 名单 · ${entry.planned_instruction_count} 条临场预案 · ${entry.points_earned} 分${prep?.available?` · 对手 ${prep.selected_tactic}（${prep.state}）`:''}`;node.append(meta);const link=artifactLink('打开本场复盘',entry.dashboard_url);if(link)node.append(link);list.append(node)}detailsNode.append(journalTitle,list);managerProfilePanel.append(detailsNode)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='目标与日志只描述当前模拟赛季；资源效果是长期游戏机制，战术频率和对手响应不代表因果效果。';managerProfilePanel.append(boundary)}if(historySummary?.total){seasonHistorySummary.hidden=false;seasonHistorySummary.textContent=`已完成 ${historySummary.total} 个赛季 · 保留最近 ${historySummary.retained}/${historySummary.limit} 季${historySummary.truncated?' · 更早归档已按保留策略移出当前会话':''}`}for(const archived of history||[]){const profile=archived.manager_profile;if(!profile?.available)continue;const objective=profile.objective||{},node=card(`${archived.season_id} · ${profile.team}`,`${objectiveText(objective)} · ${objective.status}`);node.setAttribute('role','listitem');const meta=document.createElement('p');meta.className='status';meta.textContent=`最终第 ${objective.current_position} · ${objective.current_points} 分 · ${profile.record?.win??0}胜${profile.record?.draw??0}平${profile.record?.loss??0}负 · ${profile.decision_identity?.dominant_tactic||'无主要战术'}`;const last=(profile.journal||[]).slice(-1)[0],link=artifactLink('打开该季最后一场复盘',last?.dashboard_url);node.append(meta);if(link)node.append(link);seasonHistory.append(node)}const complete=season?.state==='complete';if(!season){seasonForm.hidden=false;seasonForm.dataset.startNext='false';seasonForm.querySelector('button[type="submit"]').textContent='创建可恢复赛季'}else if(complete){seasonForm.hidden=false;seasonForm.dataset.startNext='true';const marker=season.season_id;if(seasonForm.dataset.preparedFor!==marker){seasonForm.querySelector('[name="teams"]').value=(season.plan?.teams||[]).join(', ');seasonForm.querySelector('[name="legs"]').value=String(season.plan?.legs||1);seasonForm.querySelector('[name="manager_team"]').value=season.plan?.manager_team||'';seasonObjective.value=season.plan?.manager_objective||'top_half';seasonPointsTarget.value=String(season.plan?.manager_points_target||6);const priorResources=season.plan?.manager_resources||{recovery:2,medical:2,sports_science:2};seasonForm.querySelector('[name="resource_recovery"]').value=String(priorResources.recovery);seasonForm.querySelector('[name="resource_medical"]').value=String(priorResources.medical);seasonForm.querySelector('[name="resource_sports_science"]').value=String(priorResources.sports_science);seasonForm.dataset.preparedFor=marker}seasonForm.querySelector('button[type="submit"]').textContent='归档本季并开始下一赛季'}else{seasonForm.dataset.startNext='false'}syncSeasonObjective();syncClubResources()}
function syncSeasonObjective(){const enabled=seasonObjective.value==='points_target';seasonPointsTarget.disabled=!enabled;seasonPointsTarget.required=enabled}
function syncClubResources(){const total=clubResourceInputs.reduce((sum,input)=>sum+Number(input.value||0),0),remaining=6-total;clubResourceSummary.textContent=remaining===0?'已分配 6/6 点。':remaining>0?`还需分配 ${remaining} 点（当前 ${total}/6）。`:`超出预算 ${-remaining} 点（当前 ${total}/6）。`;clubResourceSummary.className='status'+(remaining===0?'':' error');return total}
function populateSportingPlan(plan){currentSportingPlan=plan||null;sportingPriorityRoles.replaceChildren();sportingRoleDiagnostics.replaceChildren();if(!plan){sportingDirectorSummary.className='status';sportingDirectorSummary.textContent='赛季完成后生成统一规划简报。';return}const recommended=new Set(plan.recommendation?.priority_roles||[]);for(const row of plan.role_diagnostics||[]){const option=new Option(`${row.role} · 深度 ${row.depth} · 待离队 ${Number(row.expiring||0)+Number(row.retiring||0)} · 市场 ${Number(row.recruitment_options||0)+Number(row.free_agent_options||0)} · 需求 ${Number(row.need_score).toFixed(2)}`,row.role);option.selected=recommended.has(row.role);sportingPriorityRoles.append(option);if(row.depth<2||row.expiring||row.retiring){const node=card(`${row.role} · 深度 ${row.depth}`,`平均质量 ${row.mean_quality===null?'无':Number(row.mean_quality).toFixed(3)} · 平均年龄 ${row.mean_age??'未知'} · 到期 ${row.expiring} · 退休 ${row.retiring} · 可选 ${Number(row.recruitment_options||0)+Number(row.free_agent_options||0)}`);node.setAttribute('role','listitem');sportingRoleDiagnostics.append(node)}}sportingPhilosophy.value=plan.recommendation?.philosophy||'balanced';sportingRisk.value=plan.recommendation?.risk_level||'balanced';syncSportingPlan()}
function syncSportingPlan(){if(!currentSportingPlan)return 0;const roles=[...sportingPriorityRoles.selectedOptions].map(option=>option.value),invalid=roles.length>3,philosophyLabels={balanced:'均衡建设',win_now:'即战竞争',youth_pathway:'青年通道',financial_control:'财政控制'},riskText=sportingRisk.value==='low'?'自由球员必须已侦察且区间下界优于被替换者':sportingRisk.value==='balanced'?'自由球员必须先完成侦察':'允许依据基线观察承担不确定性';sportingDirectorSummary.className='status'+(invalid?' error':'');sportingDirectorSummary.textContent=invalid?'优先位置最多选择 3 个。':`${currentSportingPlan.source_season_id} → ${currentSportingPlan.target_season_id} · ${philosophyLabels[sportingPhilosophy.value]} · ${riskText} · 优先 ${roles.join('/')||'不限制位置'} · 现金 ${currentSportingPlan.club_balance} · 可用 ${currentSportingPlan.available_budget}`;return invalid?-1:roles.length}
async function loadSportingPlan(team){const requested=String(team||'').trim(),preserved=currentSportingPlan?.team===requested?{philosophy:sportingPhilosophy.value,risk:sportingRisk.value,roles:[...sportingPriorityRoles.selectedOptions].map(option=>option.value)}:null;currentSportingPlan=null;sportingRequestTeam=requested;populateSportingPlan(null);if(!requested)return;sportingDirectorSummary.textContent='正在汇总阵容、合同、财政、市场与历史签约……';try{const data=await api('/api/v1/sporting-plans/'+encodeURIComponent(requested));if(sportingRequestTeam!==requested)return;populateSportingPlan(data.plan);if(preserved){sportingPhilosophy.value=preserved.philosophy;sportingRisk.value=preserved.risk;for(const option of sportingPriorityRoles.options)option.selected=preserved.roles.includes(option.value);syncSportingPlan()}}catch(error){if(sportingRequestTeam!==requested)return;currentSportingPlan=null;sportingDirectorSummary.className='status error';sportingDirectorSummary.textContent=`体育规划不可用：${error.message}`}}
function sportingDirectivePayload(){if(seasonForm.dataset.startNext!=='true'||!managerTeamInput.value.trim())return null;if(!currentSportingPlan||currentSportingPlan.team!==managerTeamInput.value.trim())throw new Error('Sporting plan is not loaded for the selected club.');if(syncSportingPlan()<0)throw new Error('Sporting plan has too many priority roles.');return{schema_version:1,planning_id:currentSportingPlan.planning_id,philosophy:sportingPhilosophy.value,risk_level:sportingRisk.value,priority_roles:[...sportingPriorityRoles.selectedOptions].map(option=>option.value)}}
function populateRecruitmentMarket(market){const available=Number(market?.available_budget??market?.budget??0);for(const row of recruitmentRows){const candidateSelect=row.querySelector('.recruitment-candidate'),outgoingSelect=row.querySelector('.recruitment-outgoing');candidateSelect.replaceChildren();outgoingSelect.replaceChildren();const noCandidate=document.createElement('option'),noOutgoing=document.createElement('option');noCandidate.value='';noCandidate.textContent='不签入';noOutgoing.value='';noOutgoing.textContent='不放走';candidateSelect.append(noCandidate);outgoingSelect.append(noOutgoing);for(const candidate of market?.candidates||[]){const option=document.createElement('option'),cost=Number(candidate.recruitment_cost);option.value=candidate.player_id;option.textContent=`${candidate.name} · ${candidate.role} · 能力 ${Number(candidate.quality).toFixed(3)} · ${cost} 点${cost>available?' · 当前余额不足':''}`;option.dataset.cost=String(cost);option.disabled=cost>available;candidateSelect.append(option)}for(const player of market?.outgoing_players||[]){const option=document.createElement('option');option.value=player.player_id;option.textContent=`${player.name} · ${player.role} · 能力 ${Number(player.quality).toFixed(3)}`;outgoingSelect.append(option)}}syncRecruitmentSummary()}
async function loadRecruitmentMarket(team){const requested=String(team||'').trim();if(!requested){currentRecruitmentMarket=null;recruitmentRequestTeam='';populateRecruitmentMarket(null);recruitmentSummary.textContent='先选择下一赛季关注球队。';return}if(currentRecruitmentMarket?.team===requested)return;currentRecruitmentMarket=null;populateRecruitmentMarket(null);recruitmentRequestTeam=requested;recruitmentSummary.className='status';recruitmentSummary.textContent='正在加载固定候选市场……';try{const data=await api('/api/v1/recruitment-markets/'+encodeURIComponent(requested));if(recruitmentRequestTeam!==requested)return;currentRecruitmentMarket=data.market;populateRecruitmentMarket(currentRecruitmentMarket)}catch(error){if(recruitmentRequestTeam!==requested)return;currentRecruitmentMarket=null;populateRecruitmentMarket(null);recruitmentSummary.className='status error';recruitmentSummary.textContent=`候选市场不可用：${error.message}`}}
function syncRecruitmentSummary(){if(!currentRecruitmentMarket){if(!recruitmentSummary.classList.contains('error'))recruitmentSummary.textContent='赛季完成后加载候选市场。';return 0}let spent=0,moves=0,incomplete=false;for(const row of recruitmentRows){const candidate=row.querySelector('.recruitment-candidate'),outgoing=row.querySelector('.recruitment-outgoing');if(candidate.value)spent+=Number(candidate.selectedOptions[0]?.dataset.cost||0);if(candidate.value||outgoing.value)moves+=1;if(Boolean(candidate.value)!==Boolean(outgoing.value))incomplete=true}const budget=Number(currentRecruitmentMarket.available_budget??currentRecruitmentMarket.budget),remaining=budget-spent,repeatedCandidates=new Set(recruitmentRows.map(row=>row.querySelector('.recruitment-candidate').value).filter(Boolean)).size!==recruitmentRows.filter(row=>row.querySelector('.recruitment-candidate').value).length,repeatedOutgoing=new Set(recruitmentRows.map(row=>row.querySelector('.recruitment-outgoing').value).filter(Boolean)).size!==recruitmentRows.filter(row=>row.querySelector('.recruitment-outgoing').value).length,invalid=incomplete||remaining<0||repeatedCandidates||repeatedOutgoing,settlement=Number(currentRecruitmentMarket.settlement_preview?.delta||0);recruitmentSummary.className='status'+(invalid?' error':'');recruitmentSummary.textContent=incomplete?'每笔交易必须同时选择签入和放走球员。':repeatedCandidates||repeatedOutgoing?'同一球员不能出现在两笔交易中。':remaining<0?`超出俱乐部可用现金 ${-remaining} 点。`:`赛季结算 ${settlement>=0?'+':''}${settlement} · 现金 ${currentRecruitmentMarket.club_balance} · 本窗可用 ${budget}/${currentRecruitmentMarket.fixed_window_cap??8} · 已选 ${moves} 笔 ${spent} 点 · 剩余 ${remaining} 点。`;return invalid?-1:spent}
function recruitmentPayload(){if(seasonForm.dataset.startNext!=='true'||!currentRecruitmentMarket||currentRecruitmentMarket.team!==managerTeamInput.value.trim())return null;const moves=[];for(const row of recruitmentRows){const candidate_id=row.querySelector('.recruitment-candidate').value,outgoing_player_id=row.querySelector('.recruitment-outgoing').value;if(!candidate_id&&!outgoing_player_id)continue;if(!candidate_id||!outgoing_player_id)throw new Error('Each recruitment move requires both an incoming and outgoing player.');moves.push({candidate_id,outgoing_player_id})}if(!moves.length)return null;if(syncRecruitmentSummary()<0)throw new Error('Recruitment selection violates the fixed market contract.');return{schema_version:1,market_id:currentRecruitmentMarket.market_id,moves}}
function configureRecruitmentWindow(season){const complete=season?.state==='complete'&&Boolean(managerTeamInput.value.trim());recruitmentFieldset.hidden=!complete;if(!complete){currentRecruitmentMarket=null;recruitmentRequestTeam='';populateRecruitmentMarket(null)}else void loadRecruitmentMarket(managerTeamInput.value);const transaction=season?.recruitment_transaction;if(transaction&&managerProfilePanel){const names=(transaction.moves||[]).map(move=>`${move.incoming?.name||move.candidate_id} ← ${move.outgoing?.name||move.outgoing_player_id}`).join(' · '),node=card('本季阵容建设',`${names} · 使用 ${transaction.spent}/${transaction.budget} 点`);node.setAttribute('role','listitem');managerProfilePanel.append(node)}}
function renderClubStrategies(snapshot){if(!snapshot?.profiles?.length)return;seasonHistory.hidden=false;const title=document.createElement('h3'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),postureLabels={proactive:'主动进取',balanced:'稳定平衡',conservative:'谨慎务实'};title.textContent='俱乐部赛季身份';detailsTitle.textContent='查看 '+snapshot.profiles.length+' 支球队的冻结打法';list.setAttribute('role','list');for(const profile of snapshot.profiles){const control=profile.control==='manager'?'玩家控制':'AI 控制',evidence=profile.evidence_quality==='roster_derived'?'阵容证据':'球队级基线',node=card(profile.team+' · '+control,profile.primary_tactic+' · '+(postureLabels[profile.risk_posture]||profile.risk_posture)+' · 主场 '+profile.home_tactic+' / 客场 '+profile.away_tactic+' · '+profile.recruitment_move_count+' 笔引援');node.setAttribute('role','listitem');const reason=document.createElement('p');reason.className='status';reason.textContent=evidence+' · '+(profile.reasons||[]).join(' · ');node.append(reason);list.append(node)}detailsNode.append(detailsTitle,list);const boundary=document.createElement('p');boundary.className='status';boundary.textContent='身份是冻结的游戏策略快照；阵容和赛季表现会影响下一季，但不构成现实教练评价。';seasonHistory.append(title,detailsNode,boundary)}
function populateLifecyclePreview(preview){lifecycleList.replaceChildren();if(!preview){lifecycleSummary.textContent='赛季完成后加载生命周期预览。';return}const rows=new Map((preview.players||[]).map(player=>[player.player_id,player]));for(const playerId of preview.retiring_player_ids||[]){const player=rows.get(playerId),node=card(`${player?.name||playerId} · ${player?.role||'未知位置'}`,`赛季结束后退休 · 当前 ${player?.age??'年龄未知'} 岁 · 冻结退休年龄 ${player?.retirement_age}`);node.setAttribute('role','listitem');lifecycleList.append(node)}for(const playerId of preview.expiring_player_ids||[]){const player=rows.get(playerId),label=document.createElement('label'),input=document.createElement('input'),text=document.createElement('span');label.className='check lifecycle-renewal';label.setAttribute('role','listitem');input.type='checkbox';input.value=playerId;input.checked=(preview.recommended_renew_player_ids||[]).includes(playerId);text.textContent=`续约 ${player?.name||playerId} · ${player?.role||'未知位置'} · 质量 ${Number(player?.quality||0).toFixed(3)}`;input.addEventListener('change',syncLifecycleSummary);label.append(input,text);lifecycleList.append(label)}syncLifecycleSummary()}
async function loadLifecyclePreview(team){const requested=String(team||'').trim();currentLifecyclePreview=null;lifecycleRequestTeam=requested;populateLifecyclePreview(null);if(!requested)return;lifecycleSummary.className='status';lifecycleSummary.textContent='正在加载合同与退休预览……';try{const data=await api('/api/v1/lifecycle-previews/'+encodeURIComponent(requested));if(lifecycleRequestTeam!==requested)return;currentLifecyclePreview=data.preview;populateLifecyclePreview(currentLifecyclePreview)}catch(error){if(lifecycleRequestTeam!==requested)return;currentLifecyclePreview=null;lifecycleList.replaceChildren();lifecycleSummary.className='status';lifecycleSummary.textContent=`生命周期计划不适用：${error.message}`}}
function syncLifecycleSummary(){if(!currentLifecyclePreview)return 0;const selected=[...lifecycleList.querySelectorAll('.lifecycle-renewal input:checked')],limit=Number(currentLifecyclePreview.renewal_limit||4),invalid=selected.length>limit;lifecycleSummary.className='status'+(invalid?' error':'');lifecycleSummary.textContent=`到期 ${currentLifecyclePreview.expiring_player_ids?.length||0} 人 · 退休 ${currentLifecyclePreview.retiring_player_ids?.length||0} 人 · 已选择续约 ${selected.length}/${limit} 人${invalid?' · 超出续约上限':''}`;return invalid?-1:selected.length}
function retentionPayload(){if(!currentLifecyclePreview||currentLifecyclePreview.team!==managerTeamInput.value.trim())return null;if(syncLifecycleSummary()<0)throw new Error('Retention selection exceeds the fixed renewal limit.');return{schema_version:1,cycle_id:currentLifecyclePreview.cycle_id,renew_player_ids:[...lifecycleList.querySelectorAll('.lifecycle-renewal input:checked')].map(input=>input.value)}}
function renderPlayerLifecycle(lifecycle){const transactions=lifecycle?.latest_transactions||[];if(!lifecycle?.available||!transactions.length)return;seasonHistory.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div');title.textContent='合同、退役与青训';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','球员生命周期摘要');const total=key=>transactions.reduce((sum,item)=>sum+Number(item.summary?.[key]||0),0);for(const [label,value] of [['续约',`${total('renewed')} 人`],['合同离队',`${total('contract_releases')} 人`],['退休',`${total('retirements')} 人`],['青训晋升',`${total('academy_promotions')} 人`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看各俱乐部生命周期交易';list.setAttribute('role','list');for(const transaction of transactions){const exits=(transaction.exits||[]).map(item=>`${item.name}（${item.reason==='retired'?'退休':'合同到期'}）`).join('；')||'无人离队',promotions=(transaction.academy_promotions||[]).map(item=>`${item.name} · ${item.role}`).join('；')||'无青训晋升',node=card(`${transaction.team} · ${transaction.source_season_id} → ${transaction.target_season_id}`,`续约 ${transaction.summary?.renewed||0} · 离队 ${transaction.summary?.contract_releases||0} · 退休 ${transaction.summary?.retirements||0} · 青训 ${transaction.summary?.academy_promotions||0}`);node.setAttribute('role','listitem');const detail=document.createElement('p');detail.className='status';detail.textContent=`${exits} · ${promotions} · ${transaction.selection_source==='manager_plan'?'经理选择':'确定性推荐'}`;node.append(detail);list.append(node)}detailsNode.append(detailsTitle,list);const boundary=document.createElement('p');boundary.className='status';boundary.textContent='合同年限、退休年龄和青训能力均为可重放的虚构游戏规则，不代表现实雇佣、健康、潜力或估值判断。';seasonHistory.append(title,summary,detailsNode,boundary)}
function populateFreeAgentMarket(market){freeAgentCandidate.replaceChildren(new Option('不签约',''));freeAgentOutgoing.replaceChildren(new Option('不放走',''));if(!market){scoutFreeAgent.disabled=true;freeAgentSummary.textContent='赛季完成后加载全局市场。';return}for(const player of market.candidates||[]){const observation=player.observation||{},estimate=Number(observation.estimated_quality),low=Number(observation.quality_low),high=Number(observation.quality_high),level=player.scouted?'已侦察':'基线观察',option=new Option(`${player.name} · ${player.role} · ${player.age??'年龄未知'} 岁 · 估计 ${estimate.toFixed(3)} · 区间 ${low.toFixed(3)}–${high.toFixed(3)} · ${level} · 原 ${player.origin_team}`,player.player_id);option.dataset.role=player.role;option.dataset.scouted=player.scouted?'true':'false';freeAgentCandidate.append(option)}syncFreeAgentSelection()}
function syncFreeAgentSelection(){freeAgentOutgoing.replaceChildren(new Option('不放走',''));if(!currentFreeAgentMarket){scoutFreeAgent.disabled=true;return}const selectedOption=freeAgentCandidate.selectedOptions[0],role=selectedOption?.dataset.role||'';for(const player of currentFreeAgentMarket.outgoing_players||[]){if(role&&player.role!==role)continue;freeAgentOutgoing.append(new Option(`${player.name} · ${player.role} · 已知质量 ${Number(player.quality).toFixed(3)}`,player.player_id))}const selected=Boolean(freeAgentCandidate.value),complete=selected&&Boolean(freeAgentOutgoing.value),incomplete=selected!==Boolean(freeAgentOutgoing.value),budget=currentFreeAgentMarket.scouting_budget||{used:0,limit:2,remaining:0},alreadyScouted=selectedOption?.dataset.scouted==='true';scoutFreeAgent.disabled=!selected||alreadyScouted||Number(budget.remaining)<=0;freeAgentSummary.className='status'+(incomplete?' error':'');freeAgentSummary.textContent=!currentFreeAgentMarket.candidate_count?'当前窗口没有可签自由球员。':incomplete?`签约必须同时选择同位置替换球员。球探 ${budget.used}/${budget.limit}`:complete?`已冻结 1 笔零转会费自由签约；新阵容工资将在赛季结算。球探 ${budget.used}/${budget.limit}`:`全球池 ${currentFreeAgentMarket.candidate_count} 人 · 本窗最多签约 1 人 · 球探 ${budget.used}/${budget.limit} · 转会费 0`;return incomplete?-1:complete?1:0}
async function loadFreeAgentMarket(team){const requested=String(team||'').trim();currentFreeAgentMarket=null;freeAgentRequestTeam=requested;populateFreeAgentMarket(null);if(!requested)return;freeAgentSummary.className='status';freeAgentSummary.textContent='正在加载身份连续的全局市场……';try{const data=await api('/api/v1/free-agent-markets/'+encodeURIComponent(requested));if(freeAgentRequestTeam!==requested)return;currentFreeAgentMarket=data.market;populateFreeAgentMarket(currentFreeAgentMarket)}catch(error){if(freeAgentRequestTeam!==requested)return;currentFreeAgentMarket=null;freeAgentSummary.className='status';freeAgentSummary.textContent=`全局市场不适用：${error.message}`}}
async function scoutSelectedFreeAgent(){const team=managerTeamInput.value.trim(),playerId=freeAgentCandidate.value;if(!currentFreeAgentMarket||!team||!playerId)return;const outgoingId=freeAgentOutgoing.value;scoutFreeAgent.disabled=true;freeAgentSummary.className='status';freeAgentSummary.textContent='正在生成可重放的区间球探报告……';try{const data=await api('/api/v1/scouting-reports',{method:'POST',body:JSON.stringify({team,player_id:playerId})});currentFreeAgentMarket=data.market;populateFreeAgentMarket(currentFreeAgentMarket);freeAgentCandidate.value=playerId;syncFreeAgentSelection();if(outgoingId)freeAgentOutgoing.value=outgoingId;syncFreeAgentSelection();freeAgentSummary.textContent=`球探报告已持久化；候选人的区间已收窄。球探 ${currentFreeAgentMarket.scouting_budget.used}/${currentFreeAgentMarket.scouting_budget.limit}`;await loadSportingPlan(team)}catch(error){freeAgentSummary.className='status error';freeAgentSummary.textContent=`球探失败：${error.message}`;syncFreeAgentSelection()}}
function freeAgentPayload(){if(!currentFreeAgentMarket||currentFreeAgentMarket.team!==managerTeamInput.value.trim()||!freeAgentCandidate.value)return null;if(syncFreeAgentSelection()<0||!freeAgentOutgoing.value)throw new Error('Free-agent signing requires a complete same-role replacement.');const recruitedOutgoing=new Set(recruitmentRows.map(row=>row.querySelector('.recruitment-outgoing').value).filter(Boolean));if(recruitedOutgoing.has(freeAgentOutgoing.value))throw new Error('The same outgoing player cannot be used in recruitment and free-agent signing.');return{schema_version:1,market_id:currentFreeAgentMarket.market_id,free_agent_id:freeAgentCandidate.value,outgoing_player_id:freeAgentOutgoing.value}}
const configureRecruitmentWindowWithoutLifecycle=configureRecruitmentWindow;
configureRecruitmentWindow=season=>{configureRecruitmentWindowWithoutLifecycle(season);const complete=season?.state==='complete'&&Boolean(managerTeamInput.value.trim());lifecycleFieldset.hidden=!complete;if(!complete){currentLifecyclePreview=null;lifecycleRequestTeam='';populateLifecyclePreview(null)}else void loadLifecyclePreview(managerTeamInput.value)};
const configureRecruitmentWindowWithoutGlobalMarket=configureRecruitmentWindow;
configureRecruitmentWindow=season=>{configureRecruitmentWindowWithoutGlobalMarket(season);const complete=season?.state==='complete'&&Boolean(managerTeamInput.value.trim());freeAgentFieldset.hidden=!complete;if(!complete){currentFreeAgentMarket=null;freeAgentRequestTeam='';populateFreeAgentMarket(null)}else void loadFreeAgentMarket(managerTeamInput.value)};
const configureRecruitmentWindowWithoutSportingPlan=configureRecruitmentWindow;
configureRecruitmentWindow=season=>{configureRecruitmentWindowWithoutSportingPlan(season);const complete=season?.state==='complete'&&Boolean(managerTeamInput.value.trim());sportingDirectorFieldset.hidden=!complete;if(!complete){currentSportingPlan=null;sportingRequestTeam='';populateSportingPlan(null)}else void loadSportingPlan(managerTeamInput.value)};
const renderManagerCareerWithoutRecruitment=renderManagerCareer;
renderManagerCareer=(season,history,historySummary,configured)=>{renderManagerCareerWithoutRecruitment(season,history,historySummary,configured);configureRecruitmentWindow(season);for(const archived of history||[]){const transaction=archived?.recruitment_transaction;if(!transaction)continue;const names=(transaction.moves||[]).map(move=>`${move.incoming?.name||move.candidate_id} ← ${move.outgoing?.name||move.outgoing_player_id}`).join(' · '),node=card(`${archived.season_id} · 阵容建设`,`${names} · 使用 ${transaction.spent}/${transaction.budget} 点`);node.setAttribute('role','listitem');seasonHistory.append(node)}};
const renderSeasonWithoutCommandCenter=renderSeason;
renderSeason=(season,configured,history=[],historySummary={})=>{renderSeasonWithoutCommandCenter(season,configured);renderMatchdayCommand(season,configured);renderManagerCareer(season,history,historySummary,configured)};
function render(data){createBackupButton.disabled=!data.configured;cards.replaceChildren();const s=data.studio,tasks=data.tasks||[],latest=tasks[0];if(!data.configured){cards.append(card('工作区','未配置'),card('API 调用','0'),card('任务',tasks.length));setup.hidden=false;match.hidden=true;pairPanel.hidden=true;studyPanel.hidden=true;workflow.textContent='创建工作区后，系统会先执行证据就绪检查。'}else{const r=s.readiness||{},w=s.workflow||{};cards.append(card('模式',s.mode),card('就绪',r.ready?'是':'否'),card('已完成比赛',s.matches_played),card('最近任务',latest?.state||'无'));setup.hidden=true;match.hidden=false;pairPanel.hidden=!['research','cognitive'].includes(s.mode);studyPanel.hidden=s.mode!=='research';workflow.textContent=r.ready?'证据门禁通过，可以提交后台比赛、赛季、配对对决或固定预算研究任务。':'阻塞项：'+(r.blockers||[]).join(', ')}renderSeason(s?.season,data.configured,s?.season_history||[],s?.season_history_summary||{});renderCareerContract(s?.manager_career);renderActionAdoption(data.configured?s:null);renderLibrary(data.evidence_library);const isCompleted=latest?.state==='completed',dashboardUrl=isCompleted?(latest.result?.treatment_url||latest.result?.dashboard_url):s?.last_match?.dashboard_url,comparisonUrl=isCompleted?latest.result?.comparison_url:s?.last_match?.comparison_url,studyUrl=isCompleted?latest.result?.study_url:null,baselineUrl=isCompleted?latest.result?.baseline_url:null;showReport(dashboardUrl,comparisonUrl,studyUrl,baselineUrl);if(['queued','running'].includes(latest?.state)&&activePoll!==latest.task_id){activePoll=latest.task_id;void pollTask(latest.task_id)}details.textContent=JSON.stringify(data,null,2)}
function renderGlobalPlayerMarket(market){if(!market?.available)return;seasonHistory.hidden=false;const transition=market.latest_transition||{},title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div');title.textContent='全局自由球员流动';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','全局自由球员市场摘要');for(const [label,value] of [['当前球员池',`${market.pool_size||0} 人`],['本窗签约',`${transition.summary?.signings||0} 人`],['新入市场',`${transition.summary?.entries||0} 人`],['池内退休',`${transition.summary?.pool_retirements||0} 人`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看跨俱乐部签约与市场来源';list.setAttribute('role','list');for(const signing of transition.signings||[]){const node=card(`${signing.incoming?.name||signing.free_agent_id} → ${signing.team}`,`${signing.incoming?.role||''} · 原俱乐部 ${signing.origin_team||'未知'} · 替换 ${signing.outgoing?.name||signing.outgoing?.player_id} · ${signing.control==='manager'?'经理签约':'AI 签约'}`);node.setAttribute('role','listitem');list.append(node)}for(const entry of transition.entries||[]){const node=card(`${entry.player?.name||entry.player_id} · 进入市场`,`${entry.entry_reason==='contract_released'?'合同到期离队':'阵容替换离队'} · 原 ${entry.origin_team} · ${entry.available_from_season_id} 起可签`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);const boundary=document.createElement('p');boundary.className='status';boundary.textContent='该市场只公开身份、位置、年龄与流动来源；候选能力真值留在权威模拟账本中，经理与 AI 均依据有限球探观察决策。';seasonHistory.append(title,summary,detailsNode,boundary)}
const renderWithoutClubFinance=render;
function renderScouting(scouting){if(!scouting?.report_count)return;seasonHistory.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),boundary=document.createElement('p');title.textContent='球探信息账本';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','球探信息预算摘要');for(const [label,value] of [['持久化报告',`${scouting.report_count} 份`],['覆盖窗口',`${scouting.window_count} 个`],['每队每窗口预算',`${scouting.reports_per_window} 次`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看最近的区间报告';list.setAttribute('role','list');for(const report of scouting.recent_reports||[]){const observation=report.observation||{},node=card(`${report.team} · ${report.player_id}`,`${report.target_season_id} · 估计 ${Number(observation.estimated_quality).toFixed(3)} · 区间 ${Number(observation.quality_low).toFixed(3)}–${Number(observation.quality_high).toFixed(3)} · 宽度 ${Number(observation.interval_width).toFixed(3)}`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);boundary.className='status';boundary.textContent='候选球员真值始终隐藏；区间按构造覆盖模拟真值，球探只收窄信息范围，不承诺现实校准或竞技预测。';seasonHistory.append(title,summary,detailsNode,boundary)}
function renderScoutingOutcomes(outcomes){if(!outcomes?.outcome_count)return;seasonHistory.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),boundary=document.createElement('p'),coverage=outcomes.coverage_counts||{};title.textContent='签约结果与球探复盘';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','签约结果账本摘要');for(const [label,value] of [['已结算签约',`${outcomes.outcome_count} 笔`],['经理签约',`${outcomes.manager_outcome_count} 笔`],['AI 签约',`${outcomes.ai_outcome_count} 笔`],['完整出场证据',`${coverage.complete||0} 笔`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看我的已结算签约';list.setAttribute('role','list');const usageLabels={core:'核心',rotation:'轮换',fringe:'边缘',unused:'未使用',evidence_partial:'证据不完整',evidence_unavailable:'证据不足'};for(const outcome of outcomes.manager_outcomes||[]){const observation=outcome.observation||{},realized=outcome.realized||{},node=card(`${outcome.player_name||outcome.player_id} · ${outcome.team}`,`${outcome.season_id} · ${outcome.observation_source==='baseline_observation'?'未追加侦察':'已侦察'} · 估计 ${Number(observation.estimated_quality).toFixed(3)} → 到队已知 ${Number(realized.actual_quality_at_signing).toFixed(3)} · 误差 ${Number(realized.absolute_estimation_error).toFixed(3)} · ${Number(realized.minutes).toFixed(0)} 分钟/${realized.appearances||0} 场 · ${usageLabels[realized.usage_band]||realized.usage_band} · 工资档 ${realized.wage_tier}`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);boundary.className='status';boundary.textContent='使用率、积分和排名只描述模拟赛季中的后续事实，不证明签约造成球队成绩；AI 球员真值仅计入汇总，不向经理公开。';seasonHistory.append(title,summary,detailsNode,boundary)}
function renderSportingDirection(season){const brief=season?.sporting_brief,evaluation=season?.sporting_evaluation,directive=evaluation?.directive;if(!brief||!directive)return;managerProfilePanel.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),boundary=document.createElement('p'),philosophyLabels={balanced:'均衡建设',win_now:'即战竞争',youth_pathway:'青年通道',financial_control:'财政控制'},riskLabels={low:'低风险',balanced:'平衡风险',high:'高风险'};title.textContent='冻结体育总监计划';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','本赛季体育计划摘要');for(const [label,value] of [['建队理念',philosophyLabels[directive.philosophy]||directive.philosophy],['风险边界',riskLabels[directive.risk_level]||directive.risk_level],['优先位置',(directive.priority_roles||[]).join(' / ')||'不限制'],['窗口执行',`${evaluation.selected_recruitment_moves||0} 笔招募 · ${evaluation.selected_free_agent?'1 笔自由签约':'无自由签约'} · ${evaluation.selected_renewals||0} 人续约`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看冻结位置诊断';list.setAttribute('role','list');for(const row of (brief.role_diagnostics||[]).filter(row=>row.need_score>0)){const node=card(`${row.role} · 需求 ${Number(row.need_score).toFixed(2)}`,`深度 ${row.depth} · 平均质量 ${row.mean_quality===null?'无':Number(row.mean_quality).toFixed(3)} · 到期 ${row.expiring} · 退休 ${row.retiring} · 窗口可选 ${Number(row.recruitment_options||0)+Number(row.free_agent_options||0)}`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);boundary.className='status';boundary.textContent='该计划是赛季开始前冻结的游戏约束；诊断与选择不构成现实体育建议，也不预测赛季成绩。';managerProfilePanel.append(title,summary,detailsNode,boundary)}
function renderSportingReview(reviews){const review=reviews?.latest_review;if(!review)return;seasonHistory.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),boundary=document.createElement('p'),result=review.season_result||{},execution=review.execution||{},directive=review.directive||{},coverageLabels={complete:'\u5b8c\u6574',partial:'\u90e8\u5206',unavailable:'\u4e0d\u53ef\u7528'};title.textContent='\u8de8\u8d5b\u5b63\u4f53\u80b2\u6218\u7565\u590d\u76d8';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','\u4f53\u80b2\u6218\u7565\u590d\u76d8\u6458\u8981');for(const [label,value] of [['\u8d5b\u5b63',review.season_id],['\u51bb\u7ed3\u7406\u5ff5',directive.philosophy],['\u7a97\u53e3\u6267\u884c',`${execution.selected_recruitment_moves||0} \u7b14\u5f15\u63f4 \u00b7 ${execution.selected_free_agent?'1 \u7b14\u81ea\u7531\u7b7e\u7ea6':'\u65e0\u81ea\u7531\u7b7e\u7ea6'} \u00b7 ${execution.selected_renewals||0} \u4eba\u7eed\u7ea6`],['\u8d5b\u5b63\u76ee\u6807',result.objective_status],['\u6392\u540d / \u79ef\u5206',`${result.position} / ${result.points}`],['\u8d22\u653f\u53d8\u5316',result.finance_delta],['\u51fa\u573a\u8bc1\u636e',coverageLabels[review.evidence_coverage]||review.evidence_coverage]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='\u67e5\u770b\u7403\u5458\u4f7f\u7528\u3001\u4f18\u5148\u4f4d\u7f6e\u4e0e\u540e\u7eed\u63d0\u793a';list.setAttribute('role','list');for(const player of review.incoming_usage||[]){const node=card(`${player.name} \u00b7 ${player.role}`,`${player.source} \u00b7 ${Number(player.minutes).toFixed(0)} \u5206\u949f / ${player.appearances} \u573a \u00b7 \u4f7f\u7528\u7387 ${(Number(player.utilization_share)*100).toFixed(1)}% \u00b7 ${player.usage_band}`);node.setAttribute('role','listitem');list.append(node)}for(const row of review.priority_role_delivery||[]){const node=card(`\u4f18\u5148\u4f4d\u7f6e ${row.role}`,`\u539f\u6df1\u5ea6 ${row.source_depth} \u00b7 \u9884\u8ba1\u79bb\u961f ${row.source_pending_exits} \u00b7 \u65b0\u589e ${row.incoming_count} \u00b7 \u8d5b\u540e\u6df1\u5ea6 ${row.post_depth}`);node.setAttribute('role','listitem');list.append(node)}for(const signal of review.learning_signals||[]){const node=card(`\u590d\u76d8\u63d0\u793a \u00b7 ${signal.role}`,signal.interpretation);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);boundary.className='status';boundary.textContent='\u590d\u76d8\u53ea\u5c06\u51bb\u7ed3\u8ba1\u5212\u4e0e\u540e\u7eed\u6a21\u62df\u4e8b\u5b9e\u5bf9\u9f50\uff1b\u5b83\u4e0d\u58f0\u79f0\u7b7e\u7ea6\u3001\u7eed\u7ea6\u6216\u5efa\u961f\u7406\u5ff5\u5bfc\u81f4\u4e86\u6210\u7ee9\u3002';seasonHistory.append(title,summary,detailsNode,boundary)}
const populateSportingPlanWithoutContinuity=populateSportingPlan;populateSportingPlan=plan=>{populateSportingPlanWithoutContinuity(plan);if(!plan?.previous_strategy_review)return;for(const option of sportingPriorityRoles.options){const row=(plan.role_diagnostics||[]).find(item=>item.role===option.value);if(row?.continuity_signal)option.textContent+=' \u00b7 \u4e0a\u5b63\u590d\u76d8\u5ef6\u7eed';}const feedback=plan.previous_strategy_review,node=card(`\u4e0a\u5b63\u6218\u7565\u590d\u76d8 \u00b7 ${feedback.season_id}`,`\u672a\u89e3\u51b3\u4f18\u5148\u4f4d\u7f6e ${(feedback.priority_unaddressed_roles||[]).join('/')||'\u65e0'} \u00b7 \u4f4e\u4f7f\u7528\u4f4d\u7f6e ${(feedback.incoming_low_usage_roles||[]).join('/')||'\u65e0'} \u00b7 \u76ee\u6807 ${feedback.objective_status} \u00b7 \u8d22\u653f ${feedback.finance_delta>=0?'+':''}${feedback.finance_delta}`);node.setAttribute('role','listitem');sportingRoleDiagnostics.prepend(node)};
function syncClubSituation(){if(!currentClubSituation)return;const selected=(currentClubSituation.choices||[]).find(row=>row.choice_id===clubSituationChoice.value),constraint=selected?.constraint||{};clubSituationTradeoff.textContent=selected?.tradeoff||'';if(constraint.rotation_in&&!constraint.rotation_in.includes(managerRotation.value))managerRotation.value=constraint.rotation_in[0];if(constraint.tactic_equals)managerTactic.value=constraint.tactic_equals;if(constraint.tactic_not_equals&&managerTactic.value===constraint.tactic_not_equals){const alternative=[...managerTactic.options].find(option=>option.value!==constraint.tactic_not_equals);if(alternative)managerTactic.value=alternative.value}renderManagerSquad(currentSeason,currentSeason?.next_manager_fixture)}
function renderManagerDecisionLedger(season){managerDecisionLedgerList.replaceChildren();const ledger=season?.manager_decision_ledger,entries=ledger?.entries||[];managerDecisionLedger.hidden=!ledger?.available||!entries.length;if(managerDecisionLedger.hidden){managerDecisionLedgerSummary.textContent='';return}const summary=ledger.summary||{};managerDecisionLedgerSummary.textContent=`${summary.decisions||0} 次冻结决策 · ${summary.executed_with_direct_evidence||0} 场有直接执行证据 · 执行覆盖 ${Math.round((summary.direct_execution_coverage||0)*100)}% · 账本 ${String(ledger.ledger_identity||'').slice(0,12)}`;const states={frozen_awaiting_execution:'已冻结，等待比赛',executed_with_direct_evidence:'已执行，有直接证据',executed_evidence_unavailable:'已完成，直接证据不可用'},statuses={on_track:'按计划',at_risk:'有风险',pending:'待证据',fulfilled:'已兑现',missed:'已违约',excused:'已豁免',evidence_unavailable:'证据不可用'},signed=value=>`${Number(value||0)>=0?'+':''}${Number(value||0).toFixed(2)}`;for(const row of entries.slice(0,6)){const node=document.createElement('article');node.className='card';node.setAttribute('role','listitem');const heading=document.createElement('strong');heading.textContent=`第 ${row.matchday} 轮 · ${row.venue==='home'?'主场':'客场'} vs ${row.opponent}`;const lifecycle=document.createElement('p');lifecycle.className='status';lifecycle.textContent=`${states[row.lifecycle_state]||row.lifecycle_state} · ${row.decision?.tactic} / ${row.decision?.rotation} · 决策 ${String(row.decision_identity||'').slice(0,10)}`;const execution=document.createElement('p'),direct=row.execution||{},effects=direct.direct_engine_effects||{},instructions=direct.in_match_execution||{};execution.textContent=direct.available?`直接执行：机械状态 ${Number(effects.combined_status_delta||0)>=0?'+':''}${Number(effects.combined_status_delta||0).toFixed(2)} · 临场 ${instructions.applied||0} 执行 / ${instructions.skipped||0} 跳过 / ${instructions.failed||0} 失败`:`直接执行：${direct.reason||'证据不可用'}`;const advisor=document.createElement('p'),support=row.world_model_decision_support||{};if(support.available){const adoption=support.adoption,interaction=adoption?(adoption.intent==='adopt_recommendation'?'明确采用':'查看后改选'):'未绑定采用意图';advisor.textContent=`模型顾问：建议 ${support.recommended_tactic} · 选择 ${support.selected_tactic} · ${support.aligned_with_recommendation?'一致':'不一致'} · ${interaction} · 置信 ${Math.round(Number(support.recommendation_confidence||0)*100)}%`}else{advisor.textContent='模型顾问：本场未请求或证据不可用'}const result=document.createElement('p');result.textContent=row.observed_result?`观察赛果：${row.observed_result.score.home}–${row.observed_result.score.away} · ${row.observed_result.outcome} · 仅作描述`:'观察赛果：尚未产生';const accounting=document.createElement('p'),policy=row.long_term_accounting?.season_commitments?.after?.entries||[],players=row.long_term_accounting?.player_role_promises?.after?.entries||[],evidence=row.long_term_accounting?.evidence_state==='hypothetical_if_counted'?'若本场计入':'已计入完成证据',policyText=policy.map(item=>`${item.id==='tactical_identity'?'战术身份':'阵容管理'} ${statuses[item.status]||item.status}`).join(' / ')||'无赛季承诺',playerText=players.map(item=>`${item.name} ${statuses[item.status]||item.status}`).join(' / ')||'无具名承诺';accounting.textContent=`长期记账（${evidence}）：${policyText} · ${playerText}`;const persistent=document.createElement('p'),world=row.long_term_accounting?.persistent_team_state_delta;if(world?.available){const match=world.match_delta||{},matchMetrics=match.metrics_delta||{},matchSummary=match.summary||{},recovery=world.recovery_delta||{},recoveryMetrics=recovery.metrics_delta||{},recoverySummary=recovery.summary||{};persistent.textContent=`世界状态：比赛后疲劳 ${signed(matchMetrics.team_fatigue_ema)} · 士气 ${signed(matchMetrics.squad_morale_ema)} · 新伤 ${matchSummary.new_injuries||0} · 新停赛 ${matchSummary.new_suspensions||0} · 球员变化 ${matchSummary.changed_players||0}${world.recovery_complete?`；恢复后疲劳 ${signed(recoveryMetrics.team_fatigue_ema)} · 伤愈 ${recoverySummary.injuries_cleared||0} · 解禁 ${recoverySummary.suspensions_cleared||0}`:'；比赛日恢复尚未结算'}`}else{persistent.textContent=`世界状态：${world?.reason||'快照证据不可用'}`};const boundary=document.createElement('p');boundary.className='status';boundary.textContent='模型建议、经理选择、直接执行、世界状态、赛果和长期记账分别取证；伤停为模拟状态，单场赛果不证明决策效果。';node.append(heading,lifecycle,advisor,execution,result,persistent,accounting,boundary);const link=artifactLink('打开该场完整复盘',row.dashboard);if(link)node.append(link);managerDecisionLedgerList.append(node)}}
function renderClubSituation(season){const fixture=season?.next_manager_fixture,resolutions=season?.club_timeline?.events||[],frozen=resolutions.find(row=>row.event?.fixture_id===fixture?.fixture_id),situation=season?.club_timeline_view?.current_situation||frozen?.event||null;currentClubSituation=situation;clubSituationChoice.replaceChildren();clubSituationFieldset.hidden=!situation;if(!situation)return;clubSituationSummary.textContent=`${situation.title} \u00b7 \u8bc1\u636e ${situation.trigger?.source||'persisted_season'}`;for(const row of situation.choices||[]){const option=new Option(row.label,row.choice_id);option.selected=(fixture?.manager_decision?.club_event_choice?.choice_id||situation.recommended_choice)===row.choice_id;clubSituationChoice.append(option)}syncClubSituation();if(frozen){managerProfilePanel.hidden=false;const node=card('\u4ff1\u4e50\u90e8\u65f6\u95f4\u7ebf',`\u5df2\u89e3\u51b3 ${season.club_timeline_view?.resolved_count||0} \u4e2a\u60c5\u5883 \u00b7 \u6700\u8fd1 ${frozen.event.kind} / ${frozen.choice.choice_id} \u00b7 ${frozen.control}`);node.setAttribute('role','listitem');managerProfilePanel.append(node)}}
clubSituationChoice.addEventListener('change',syncClubSituation);
function renderCompletedClubTimeline(season){if(season?.next_manager_fixture)return;const latest=(season?.club_timeline?.events||[]).slice(-1)[0];if(!latest)return;for(const item of matchdayJourney.children){if(item.textContent.startsWith('club_situation'))item.textContent=item.textContent.replace('club_situation','\u4ff1\u4e50\u90e8\u60c5\u5883')}managerProfilePanel.hidden=false;const node=card('\u4ff1\u4e50\u90e8\u65f6\u95f4\u7ebf',`\u672c\u5b63\u5df2\u89e3\u51b3 ${season.club_timeline_view?.resolved_count||0} \u4e2a\u60c5\u5883 \u00b7 \u6700\u8fd1 ${latest.event.kind} / ${latest.choice.choice_id} \u00b7 ${latest.control}`);node.setAttribute('role','listitem');managerProfilePanel.append(node)}
function renderSeasonCommitments(season){
  const progress=season?.season_commitment_view;if(!progress)return;managerProfilePanel.hidden=false;
  const labels={board_objective:'\u8463\u4e8b\u4f1a\u7ed3\u679c',tactical_identity:'\u6218\u672f\u8eab\u4efd',squad_stewardship:'\u9635\u5bb9\u4f7f\u7528'},statuses={fulfilled:'\u5df2\u5151\u73b0',missed:'\u5df2\u8fdd\u7ea6',unreachable:'\u5df2\u4e0d\u53ef\u8fbe',on_track:'\u6309\u8ba1\u5212',at_risk:'\u6709\u98ce\u9669',pending:'\u5f85\u5efa\u7acb\u8bc1\u636e'};
  const detailsNode=document.createElement('details'),title=document.createElement('summary'),list=document.createElement('div');title.textContent=`\u8d5b\u5b63\u627f\u8bfa \u00b7 ${progress.recorded_matches}/${progress.recorded_matches+progress.remaining_matches} \u573a\u8bc1\u636e`;list.className='cards';list.setAttribute('role','list');
  for(const entry of progress.entries||[]){const metric=entry.metric||{};let value;if(entry.id==='tactical_identity'){value=metric.policy==='adaptive'?`${metric.policy} \u00b7 ${metric.distinct_tactics||0} \u79cd \u00b7 \u4e3b\u5bfc ${Math.round((metric.dominant_share||0)*100)}%`:`${metric.policy} \u00b7 ${metric.identity_tactic||'\u2014'} \u00b7 ${Math.round((metric.identity_share||0)*100)}%`}else if(entry.id==='squad_stewardship'){value=`${metric.policy} \u00b7 ${Math.round((metric.counted_share||0)*100)}%`}else{value=`${metric.objective} \u00b7 ${metric.points??0} \u5206`}const node=card(labels[entry.id]||entry.id,`${statuses[entry.status]||entry.status} \u00b7 ${value}`);node.setAttribute('role','listitem');list.append(node)}
  const boundary=document.createElement('p');boundary.className='status';boundary.textContent='\u627f\u8bfa\u8fdb\u5ea6\u53ea\u91cd\u653e\u5df2\u5b8c\u6210\u51b3\u7b56\uff1b\u8d5b\u5b63\u672b\u5151\u73b0\u6216\u8fdd\u7ea6\u624d\u8fdb\u5165\u900f\u660e\u8463\u4e8b\u4f1a\u7ed3\u7b97\u3002';detailsNode.append(title,list,boundary);managerProfilePanel.append(detailsNode)
}
function syncPlayerPromiseForm(){const players=new Map((currentSeason?.manager_squad?.players||[]).map(row=>[row.player_id,row])),chosen=[];for(const row of playerPromiseRows){const playerId=row.querySelector('.player-promise-player').value,role=row.querySelector('.player-promise-role'),player=players.get(playerId),development=[...role.options].find(option=>option.value==='development');if(development)development.disabled=!player||!Number.isInteger(player.age)||player.age>23;if(role.value==='development'&&development?.disabled)role.value='rotation';if(playerId)chosen.push({player_id:playerId,role:role.value})}const duplicatePlayers=new Set(chosen.map(row=>row.player_id)).size!==chosen.length,duplicateRoles=new Set(chosen.map(row=>row.role)).size!==chosen.length;playerPromiseSummary.textContent=duplicatePlayers?'\u540c\u4e00\u7403\u5458\u4e0d\u80fd\u627f\u62c5\u591a\u4e2a\u627f\u8bfa\u3002':duplicateRoles?'\u6bcf\u79cd\u627f\u8bfa\u89d2\u8272\u6700\u591a\u4e00\u4eba\u3002':`\u5df2\u9009 ${chosen.length}/3 \u4eba \u00b7 \u8d5b\u5b63\u5f00\u59cb\u540e\u4e0d\u53ef\u66ff\u6362`;return duplicatePlayers||duplicateRoles?-1:chosen.length}
function populatePlayerPromiseForm(season){const players=season?.manager_squad?.players||[];for(const [index,row] of playerPromiseRows.entries()){const select=row.querySelector('.player-promise-player');select.replaceChildren();if(index>0)select.add(new Option('\u4e0d\u627f\u8bfa',''));for(const player of players){const age=Number.isInteger(player.age)?` \u00b7 ${player.age}`:' \u00b7 \u5e74\u9f84\u672a\u77e5';select.add(new Option(`${player.name||player.player_id} \u00b7 ${player.role}${age}`,player.player_id))}}syncPlayerPromiseForm()}
function renderPlayerPromises(season,history=[]){const contract=season?.player_role_promises,progress=season?.player_promise_view,squad=season?.manager_squad,needsSetup=Boolean(season?.next_manager_fixture&&!contract&&squad?.available);playerPromiseForm.hidden=!needsSetup;if(needsSetup){populatePlayerPromiseForm(season);managerDecisionForm.hidden=true;playMatchday.disabled=true}if(progress?.entries?.length){managerProfilePanel.hidden=false;const statuses={fulfilled:'\u5df2\u5151\u73b0',missed:'\u5df2\u8fdd\u7ea6',excused:'\u56e0\u4e0d\u53ef\u7528\u8c41\u514d',on_track:'\u6309\u8ba1\u5212',at_risk:'\u6709\u98ce\u9669',pending:'\u5f85\u8bc1\u636e',evidence_unavailable:'\u8bc1\u636e\u4e0d\u53ef\u7528'},roles={core:'\u6838\u5fc3',rotation:'\u8f6e\u6362',development:'\u57f9\u517b'},detailsNode=document.createElement('details'),title=document.createElement('summary'),list=document.createElement('div');title.textContent=`\u5177\u540d\u7403\u5458\u627f\u8bfa \u00b7 ${progress.completed_matches}/${progress.completed_matches+progress.remaining_matches} \u573a`;list.className='cards';list.setAttribute('role','list');for(const row of progress.entries){const node=card(`${row.name} \u00b7 ${roles[row.promised_role]||row.promised_role}`,`${statuses[row.status]||row.status} \u00b7 \u9996\u53d1 ${row.starts}/${row.known_lineups} \u00b7 \u8c41\u514d ${row.excused_unavailable||0} \u573a \u00b7 ${Math.round(row.start_share*100)}% / ${Math.round(row.minimum_start_share*100)}%`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(title,list);managerProfilePanel.append(detailsNode)}const latest=[...(history||[])].reverse().find(item=>item.player_promise_outcomes)?.player_promise_outcomes;if(latest?.players?.length){managerProfilePanel.hidden=false;const node=card('\u4e0a\u5b63\u7403\u5458\u627f\u8bfa\u7ed3\u7b97',`${latest.coverage} \u00b7 ${latest.players.map(row=>`${row.name} ${row.observed_status} ${Math.round(row.minute_share*100)}%`).join(' / ')}`);node.setAttribute('role','listitem');managerProfilePanel.append(node)}}
const renderSeasonWithoutClubTimeline=renderSeason;renderSeason=(season,configured,history=[],historySummary={})=>{renderSeasonWithoutClubTimeline(season,configured,history,historySummary);renderClubSituation(season);for(const item of matchdayJourney.children){if(item.textContent.startsWith('club_situation'))item.textContent=item.textContent.replace('club_situation','\u4ff1\u4e50\u90e8\u60c5\u5883')}renderCompletedClubTimeline(season);renderSeasonCommitments(season);renderPlayerPromises(season,history);renderManagerDecisionLedger(season);const managed=season?.next_manager_fixture;if(!currentManagerAdvice||currentManagerAdvice.fixture_id!==managed?.fixture_id)managerAdviceIntent=null;renderManagerWorldModelAdvice(managed?.manager_decision_advice,season,managed);if(!managerDecisionForm.hidden)scheduleManagerDecisionPreview()};
const submitWithoutClubSituation=submit;submit=(form,path,payload,headers={})=>{if(form===managerDecisionForm&&path==='/api/v1/seasons/decision'&&currentClubSituation){payload.decision.club_event_choice={schema_version:1,event_id:currentClubSituation.event_id,event_identity:currentClubSituation.event_identity,choice_id:clubSituationChoice.value}}return submitWithoutClubSituation(form,path,payload,headers)};
for(const row of playerPromiseRows){row.querySelector('.player-promise-player').addEventListener('change',syncPlayerPromiseForm);row.querySelector('.player-promise-role').addEventListener('change',syncPlayerPromiseForm)}
playerPromiseForm.addEventListener('submit',event=>{event.preventDefault();const count=syncPlayerPromiseForm();if(count<1){announce(playerPromiseSummary,count<0?playerPromiseSummary.textContent:'\u81f3\u5c11\u9009\u62e9\u4e00\u540d\u7403\u5458\u3002','error',true);return}const promises=playerPromiseRows.map(row=>({player_id:row.querySelector('.player-promise-player').value,role:row.querySelector('.player-promise-role').value})).filter(row=>row.player_id);submit(playerPromiseForm,'/api/v1/seasons/player-promises',{schema_version:1,promises})});
function renderUnifiedWorkflow(data){if(!data.configured){workflowAction.hidden=true;workflowAction._target=null;return}const latest=(data.tasks||[])[0];if(['queued','running'].includes(latest?.state)){workflow.textContent=`\u540e\u53f0\u4efb\u52a1\u6b63\u5728${latest.state==='queued'?'\u6392\u961f':'\u8fd0\u884c'} \u00b7 ${latest.kind||'task'} \u00b7 \u5b8c\u6210\u540e\u81ea\u52a8\u6062\u590d\u4e3b\u7ebf\u4e0b\u4e00\u6b65\u3002`;workflow.dataset.action='wait_for_task';workflowAction.hidden=true;workflowAction._target=null;return}const studio=data.studio,flow=studio?.workflow||{},action=flow.next_action||{},labels={resolve_readiness:'\u8bc1\u636e\u95e8\u7981\u672a\u901a\u8fc7\uff0c\u8bf7\u5148\u68c0\u67e5\u963b\u585e\u9879\u3002',wait_for_match:'\u6bd4\u8d5b\u4e8b\u52a1\u6b63\u5728\u8fd0\u884c\uff0c\u5b8c\u6210\u540e\u5c06\u81ea\u52a8\u8fdb\u5165\u4e0b\u4e00\u6b65\u3002',start_season:'\u8bc1\u636e\u95e8\u7981\u901a\u8fc7 \u00b7 \u4e0b\u4e00\u6b65\uff1a\u5efa\u7acb\u6301\u4e45\u8d5b\u5b63\uff0c\u8fdb\u5165\u4ff1\u4e50\u90e8\u3001\u6bd4\u8d5b\u65e5\u4e0e\u751f\u6daf\u4e3b\u5faa\u73af\u3002',freeze_player_promises:'\u4e0b\u4e00\u6b65\uff1a\u51bb\u7ed3\u672c\u8d5b\u5b63\u7684\u5177\u540d\u7403\u5458\u627f\u8bfa\u3002',submit_manager_decision:'\u4e0b\u4e00\u6b65\uff1a\u63d0\u4ea4\u672c\u8f6e\u9635\u5bb9\u3001\u6218\u672f\u4e0e\u4e34\u573a\u9884\u6848\u3002',advance_season_matchday:'\u4e0b\u4e00\u6b65\uff1a\u6240\u6709\u5fc5\u8981\u51b3\u7b56\u5df2\u51bb\u7ed3\uff0c\u63a8\u8fdb\u5f53\u524d\u6bd4\u8d5b\u65e5\u3002',resume_season_matchday:'\u4e0b\u4e00\u6b65\uff1a\u6062\u590d\u672a\u5b8c\u6210\u7684\u5f53\u524d\u6bd4\u8d5b\u65e5\u3002',review_sporting_plan:'\u8d5b\u5b63\u5df2\u5b8c\u6210 \u00b7 \u4e0b\u4e00\u6b65\uff1a\u5ba1\u9605\u4f53\u80b2\u603b\u76d1\u89c4\u5212\uff0c\u51bb\u7ed3\u4e0b\u8d5b\u5b63\u7b56\u7565\u3002',review_season:'\u8d5b\u5b63\u5df2\u5b8c\u6210 \u00b7 \u4e0b\u4e00\u6b65\uff1a\u590d\u76d8\u79ef\u5206\u699c\u5e76\u51b3\u5b9a\u662f\u5426\u5f00\u59cb\u4e0b\u8d5b\u5b63\u3002',open_dashboard:'\u4e0b\u4e00\u6b65\uff1a\u6253\u5f00\u6700\u8fd1\u7684\u53ef\u5ba1\u8ba1\u6bd4\u8d5b\u590d\u76d8\u3002',inspect_season_state:'\u8d5b\u5b63\u72b6\u6001\u65e0\u6cd5\u6620\u5c04\u5230\u53d7\u652f\u6301\u7684\u4e0b\u4e00\u6b65\uff0c\u8bf7\u68c0\u67e5\u8be6\u7ec6\u8bc1\u636e\u3002'};const context=Number.isInteger(action.matchday)?` \u00b7 \u7b2c ${action.matchday} \u8f6e${action.opponent?' \u00b7 vs '+action.opponent:''}`:'';workflow.textContent=labels[action.id]||action.reason||'\u5f53\u524d\u6ca1\u6709\u53ef\u7528\u7684\u4e0b\u4e00\u6b65\u3002';if(context)workflow.textContent+=context;workflow.dataset.action=action.id||'';const targets={resolve_readiness:details,wait_for_match:message,start_season:seasonForm,freeze_player_promises:playerPromiseForm,submit_manager_decision:managerDecisionForm,advance_season_matchday:playMatchday,resume_season_matchday:playMatchday,review_sporting_plan:sportingDirectorFieldset,review_season:seasonPanel,open_dashboard:report,inspect_season_state:details};workflowAction._target=targets[action.id]||null;workflowAction.hidden=!workflowAction._target||['wait_for_match'].includes(action.id)}
const renderUnifiedWorkflowWithoutAreas=renderUnifiedWorkflow;renderUnifiedWorkflow=data=>{renderUnifiedWorkflowWithoutAreas(data);const action=data.studio?.workflow?.next_action?.id;workflowAction._area=workflowAreaByAction[action]||workflowWorkspaceArea(data)};
workflowAction.addEventListener('click',()=>{const target=workflowAction._target;if(!target)return;if(workflowAction._area)applyWorkspaceArea(workflowAction._area,{remember:true,configured:true});target.scrollIntoView({block:'center'});const focusable=target.matches('button,input,select,a[href]')?target:target.querySelector('button:not([disabled]),input:not([disabled]),select:not([disabled]),a[href]');if(focusable)focusable.focus();else{target.setAttribute('tabindex','-1');target.focus()}});
render=data=>{renderWithoutClubFinance(data);renderClubFinance(data.studio?.club_finance);renderSportingDirection(data.studio?.season);renderSportingReview(data.studio?.sporting_reviews);renderLeagueEcosystem(data.studio?.league_ecosystem);renderClubStrategies(data.studio?.season?.club_strategies);renderPlayerDevelopment(data.studio?.player_development);renderPlayerLifecycle(data.studio?.player_lifecycle);renderGlobalPlayerMarket(data.studio?.player_market);renderScouting(data.studio?.scouting);renderScoutingOutcomes(data.studio?.scouting_outcomes);renderUnifiedWorkflow(data);renderWorkspaceAreas(data)};
function renderManagerProductJourney(season,configured){
  const journey=document.querySelector('#manager-product-journey');journey.replaceChildren();journey.hidden=!configured;if(!configured)return;
  const command=season?.matchday_command_center||{},phase=command.phase||'',hasResult=Boolean(command.last_result),complete=season?.state==='complete';
  const preparation=!season?'pending':command.decision_required?'action_required':['ready_to_advance','ready_to_resume','season_complete'].includes(phase)?'complete':'available';
  const match=!season?'pending':['ready_to_advance','ready_to_resume'].includes(phase)?'action_required':phase==='decision_required'?'pending':hasResult||complete?'complete':'pending';
  const steps=[
    ['club_plan','\u4ff1\u4e50\u90e8\u89c4\u5212',season?'complete':'action_required'],
    ['preparation','\u8d5b\u524d\u51b3\u7b56',preparation],
    ['match','\u6bd4\u8d5b\u6267\u884c',match],
    ['evidence','\u8d5b\u540e\u8bc1\u636e',hasResult?'available':'pending'],
    ['career','\u957f\u671f\u8fd0\u8425',!season?'pending':complete?'action_required':'available'],
  ],states={complete:'\u5df2\u8fde\u63a5',available:'\u53ef\u8fdb\u5165',action_required:'\u5f53\u524d\u4e3b\u4efb\u52a1',pending:'\u5f85\u524d\u7f6e'};
  for(const [id,label,status]of steps){const item=document.createElement('li');item.className='manager-product-step';item.dataset.stage=id;item.dataset.status=status;if(status==='action_required')item.setAttribute('aria-current','step');item.textContent=label+' \u00b7 '+states[status];journey.append(item)}
}
const renderUnifiedWorkflowWithoutManagerJourney=renderUnifiedWorkflow;
renderUnifiedWorkflow=data=>{renderUnifiedWorkflowWithoutManagerJourney(data);renderManagerProductJourney(data.studio?.season,data.configured)};
function setRecoveryMessage(text,isError=false,focus=false){announce(recoveryMessage,text,isError?'error':'success',focus)}
function closeRestore(returnFocus=true){restoreForm.reset();restoreForm.hidden=true;delete restoreForm.dataset.backupId;if(returnFocus&&restoreTrigger?.isConnected)restoreTrigger.focus();restoreTrigger=null}
function prepareRestore(id,trigger){restoreTrigger=trigger;selectedBackup.textContent=id;restoreForm.dataset.backupId=id;restoreForm.reset();restoreForm.hidden=false;restoreForm.querySelector('input[name="confirmation"]').focus()}
function renderBackups(data){backupList.replaceChildren();if(!data.backups.length){const empty=document.createElement('p');empty.className='status';empty.textContent='尚无托管备份。';backupList.append(empty);return}for(const backup of data.backups){const item=document.createElement('div');item.className='card';item.setAttribute('role','listitem');const title=document.createElement('strong');title.textContent=backup.backup_id;const meta=document.createElement('p');meta.className='status';meta.textContent=new Date(backup.modified_at).toLocaleString()+' · '+backup.size_bytes+' bytes';const actions=document.createElement('div');actions.className='toolbar';const verify=document.createElement('button');verify.type='button';verify.textContent='校验';verify.setAttribute('aria-label','校验备份 '+backup.backup_id);verify.addEventListener('click',async()=>{verify.disabled=true;setBusy(item,true);try{await api('/api/v1/backups/'+encodeURIComponent(backup.backup_id)+'/verify',{method:'POST'});setRecoveryMessage('备份完整性与语义校验通过。')}catch(e){setRecoveryMessage(e.message,true,true)}finally{verify.disabled=false;setBusy(item,false)}});const restore=document.createElement('button');restore.type='button';restore.className='danger-button';restore.textContent='准备恢复';restore.setAttribute('aria-label','准备恢复备份 '+backup.backup_id);restore.addEventListener('click',()=>prepareRestore(backup.backup_id,restore));actions.append(verify,restore);item.append(title,meta,actions);backupList.append(item)}}
async function refreshRecovery(){setBusy(backupList,true);try{renderBackups(await api('/api/v1/recovery'))}catch(e){setRecoveryMessage(e.message,true,true)}finally{setBusy(backupList,false)}}
async function api(path,options={}){const response=await fetch(path,{...options,headers:{'Content-Type':'application/json','X-GFS-CSRF':csrf,...options.headers}});const data=await response.json();if(!response.ok)throw new Error(data.error?.message||'请求失败');return data}
async function refresh(){try{const data=await api('/api/v1/studio');const mode=data.studio?.mode||'stable';configureMatchPlan(data.match_capabilities,mode);configureTacticalStudy(data.match_capabilities,mode);configurePairedMatch(data.match_capabilities,mode,data.studio?.seed);configureWorldModelFork(data.match_capabilities,mode,data.studio?.seed);render(data);renderRelease(data)}catch(e){announce(message,e.message,'error',true)}}
async function pollTask(id){try{while(true){const data=await api('/api/v1/tasks/'+encodeURIComponent(id)),task=data.task,isStudy=task.kind==='tactical_study',isPair=task.kind==='paired_match',isFork=task.kind==='world_model_fork',isSeason=task.kind==='season_matchday',progress=task.study_progress;details.textContent=JSON.stringify(data,null,2);if(isStudy&&progress?.pairs_completed!==undefined){announce(message,`战术研究 ${progress.pairs_completed}/${progress.fixed_pair_budget} 对已完成；中期效应保持隐藏。`)}else{announce(message,task.state==='queued'?(isStudy?'战术研究已排队，固定预算尚未开始……':isFork?'因果分叉已排队，将依次生成预测-only与动作策略世界……':isPair?'配对对决已排队，将连续运行基线与处理场……':isSeason?'赛季比赛日已持久化排队……':'比赛任务已排队……'):(isStudy?'战术研究正在后台运行；预算完成前不展示效应……':isFork?'正在用共享 seed 生成两个世界并核验唯一策略干预……':isPair?'正在运行共享 seed 的基线与处理场……':isSeason?'正在推进赛季比赛日并结算长期状态……':'比赛正在后台运行……'))}if(task.state==='completed'){announce(message,isStudy?'固定预算研究完成，最终分析现已开放。':isFork?'世界模型因果分叉完成，双世界回放与差异链已开放。':isPair?'配对对决完成，三层复盘已开放。':isSeason?'赛季比赛日完成，积分与球队状态已结算。':'比赛完成。','success',true);showReport(task.result?.treatment_url||task.result?.dashboard_url,task.result?.comparison_url,task.result?.study_url,task.result?.baseline_url);await refresh();return}if(['failed','interrupted'].includes(task.state)){announce(message,task.error?.message||'任务中断，可检查状态后重新提交。','error',true);await refresh();return}await new Promise(resolve=>setTimeout(resolve,750))}}catch(e){announce(message,e.message,'error',true)}finally{activePoll=''}}
async function submit(form,path,payload,headers={}){const button=form.querySelector('button[type="submit"]')||form.querySelector('button');button.disabled=true;setBusy(form,true);announce(message,'正在提交……');report.hidden=true;try{const data=await api(path,{method:'POST',body:JSON.stringify(payload),headers});if(data.task){announce(message,data.created?'任务已持久化排队。':'已返回同一幂等任务。');activePoll=data.task.task_id;await pollTask(data.task.task_id)}else{announce(message,'操作成功。','success',true);await refresh()}}catch(e){announce(message,e.message,'error',true)}finally{button.disabled=false;setBusy(form,false)}}
const submitWithoutRecruitment=submit;
submit=(form,path,payload,headers={})=>{if(form===seasonForm&&path==='/api/v1/seasons'){const commitmentForm=new FormData(seasonForm);payload.plan.manager_commitments=payload.plan.manager_team?{schema_version:1,tactic_policy:String(commitmentForm.get('commitment_tactic_policy')||'club_identity'),rotation_policy:String(commitmentForm.get('commitment_rotation_policy')||'share_load')}:null;if(payload?.start_next){try{payload.plan.manager_recruitment=recruitmentPayload();payload.plan.manager_retention=retentionPayload();payload.plan.manager_free_agent=freeAgentPayload();payload.plan.manager_sporting_directive=sportingDirectivePayload()}catch(error){announce(sportingDirectorSummary,error.message,'error',true);return Promise.resolve()}}}return submitWithoutRecruitment(form,path,payload,headers)};
setup.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(setup);submit(setup,'/api/v1/studio',{name:f.get('name'),mode:f.get('mode'),seed:Number(f.get('seed'))})});
seasonForm.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(seasonForm),teams=String(f.get('teams')||'').split(',').map(value=>value.trim()).filter(Boolean),manager=String(f.get('manager_team')||'').trim(),objective=seasonObjective.value,points=objective==='points_target'?Number(seasonPointsTarget.value):null,startNext=seasonForm.dataset.startNext==='true',contract=currentCareer?.next_contract||{},sameClub=startNext&&manager===currentCareer?.current_team,resources={recovery:Number(f.get('resource_recovery')),medical:Number(f.get('resource_medical')),sports_science:Number(f.get('resource_sports_science'))};if(manager&&syncClubResources()!==6){announce(message,'俱乐部资源必须恰好分配 6 点。','error',true);clubResourceInputs[0].focus();return}if(sameClub&&!contract.same_club_allowed){announce(message,`你已失去 ${manager} 的续任资格，请选择另一支参赛球队。`,'error',true);seasonForm.querySelector('[name="manager_team"]').focus();return}if(sameClub&&objective==='points_target'&&contract.same_club_points_growth_required){const maximum=(teams.length-1)*Number(f.get('legs'))*3,required=Math.min(maximum,Number(contract.baseline_points||0)+Number(contract.same_club_points_growth_required));if(points<required){announce(message,`观察期积分目标至少为 ${required} 分。`,'error',true);seasonPointsTarget.focus();return}}submit(seasonForm,'/api/v1/seasons',{start_next:startNext,plan:{teams,legs:Number(f.get('legs')),fast:f.get('fast')==='on',manager_team:manager||null,manager_objective:manager?objective:null,manager_points_target:manager?points:null,manager_resources:manager?resources:null}})});
seasonObjective.addEventListener('change',syncSeasonObjective);
for(const input of clubResourceInputs)input.addEventListener('input',syncClubResources);syncClubResources();
for(const row of recruitmentRows){row.querySelector('.recruitment-candidate').addEventListener('change',syncRecruitmentSummary);row.querySelector('.recruitment-outgoing').addEventListener('change',syncRecruitmentSummary)}
managerTeamInput.addEventListener('change',()=>{currentRecruitmentMarket=null;currentLifecyclePreview=null;currentFreeAgentMarket=null;currentSportingPlan=null;if(seasonForm.dataset.startNext==='true'){void loadRecruitmentMarket(managerTeamInput.value);void loadLifecyclePreview(managerTeamInput.value);void loadFreeAgentMarket(managerTeamInput.value);void loadSportingPlan(managerTeamInput.value)}});
freeAgentCandidate.addEventListener('change',syncFreeAgentSelection);
freeAgentOutgoing.addEventListener('change',syncFreeAgentSelection);
scoutFreeAgent.addEventListener('click',()=>void scoutSelectedFreeAgent());
sportingPhilosophy.addEventListener('change',syncSportingPlan);
sportingRisk.addEventListener('change',syncSportingPlan);
sportingPriorityRoles.addEventListener('change',syncSportingPlan);
const renderActionAdoptionWithoutFormalEvidence=renderActionAdoption;
renderActionAdoption=studio=>{renderActionAdoptionWithoutFormalEvidence(studio);if(!studio)return;const evidence=studio.evidence||{},mechanism=evidence.action_adoption_mechanism||{},outcome=evidence.action_outcome_study||{},parts=[];if(mechanism.available){const label=mechanism.result_status==='mechanism_confirmed'?'动作采纳机制已确认':mechanism.execution_state||mechanism.protocol_state||'unknown',formal=mechanism.mechanism||{},changed=Number(formal.realized_counterfactual_action_changes||0),opportunities=Number(formal.action_opportunities||0);parts.push(`机制：${label} · ${mechanism.runs_executed??0}/${mechanism.fixed_run_budget??'—'} 次运行 · ${changed}/${opportunities} 次实际反事实动作变化`)}if(outcome.available){const state=outcome.result_status||outcome.execution_state||'not_started',promotion=outcome.promotion_supported?'结果门支持晋级':'结果门未支持晋级',primary=outcome.primary||{},behavior=outcome.behavior||{},delta=Number(primary.point_delta),low=Number(primary.ci95_low),high=Number(primary.ci95_high),effect=Number.isFinite(delta)&&Number.isFinite(low)&&Number.isFinite(high)?` · 损失差 ${delta.toFixed(3)} · 95% CI [${low.toFixed(3)}, ${high.toFixed(3)}]`:'',changed=Number.isFinite(Number(behavior.changed_pairs))?` · 行为变化 ${behavior.changed_pairs}/${outcome.pairs_total??'—'} 对`:'';parts.push(`全长度结果：${state} · ${outcome.runs_executed??0}/${outcome.fixed_run_budget??'—'} 次运行${effect}${changed} · ${promotion}`)}if(parts.length)adoptionEvidence.textContent=parts.join('；')+'。世界模型已改变完整比赛行为，但本次结果不确定，未证明赛果改善或真实足球因果效应。'};
const renderManagerDecisionPreviewWithoutWorldModelComparison=renderManagerDecisionPreview;
renderManagerDecisionPreview=preview=>{renderManagerDecisionPreviewWithoutWorldModelComparison(preview);adoptManagerAdvice.textContent='\u91c7\u7528\u6a21\u578b\u5efa\u8bae';const advice=preview?.world_model_advice,comparison=advice?.comparison;if(!comparison)return;const authority=comparison.authority||{},exploratory=authority.level==='exploratory_only',heading=document.createElement('h4');heading.textContent='\u4e16\u754c\u6a21\u578b\u5efa\u8bae vs \u5f53\u524d\u9009\u62e9';if(exploratory)adoptManagerAdvice.textContent='\u5ba1\u9605\u540e\u91c7\u7528\u63a2\u7d22\u5efa\u8bae';const signed=(value,digits=3)=>`${Number(value||0)>=0?'+':''}${Number(value||0).toFixed(digits)}`,deltas=comparison.recommended_minus_selected||{},events=deltas.event_probabilities||{},facts=document.createElement('div');facts.className='cards';facts.setAttribute('role','list');facts.setAttribute('aria-label','\u4e16\u754c\u6a21\u578b\u5efa\u8bae\u4e0e\u5f53\u524d\u6218\u672f\u7684\u77ed\u89c6\u91ce\u4ee3\u7406\u5dee\u5f02');const reasonLabels={low_model_confidence:'\u6a21\u578b\u7f6e\u4fe1\u5ea6\u4f4e',low_historical_trust:'\u5386\u53f2\u4fe1\u4efb\u4f4e',narrow_top_two_margin:'\u524d\u4e24\u540d\u5dee\u8ddd\u5c0f',insufficient_history:'\u5386\u53f2\u6837\u672c\u4e0d\u8db3',low_authority_guidance:'\u4f4e\u6743\u5a01\u5f15\u5bfc'},reasons=(authority.reasons||[]).map(reason=>reasonLabels[reason]||reason).join(' / ')||'\u5df2\u8fbe\u6709\u754c\u5ba1\u9605\u663e\u793a\u95e8\u69db';for(const [label,value] of [['\u663e\u793a\u6743\u5a01',`${exploratory?'\u4ec5\u63a2\u7d22':'\u6709\u754c\u5ba1\u9605'} \u00b7 ${reasons}`],['\u6218\u672f\u6392\u540d',`\u5efa\u8bae ${comparison.recommended_tactic} #${comparison.recommended_rank} / \u5f53\u524d ${comparison.selected_tactic} #${comparison.selected_rank}`],['\u98ce\u9669\u8c03\u6574\u503c\u5dee',signed(deltas.risk_adjusted_value)],['\u7f6e\u4fe1 / \u4e0d\u786e\u5b9a\u6027',`${signed(deltas.effective_confidence)} / ${signed(deltas.uncertainty)}`],['\u75b2\u52b3 / \u7ed3\u6784\u98ce\u9669',`${signed(deltas.fatigue_cost_proxy)} / ${signed(deltas.structural_risk_proxy)}`],['\u4e8b\u4ef6\u5206\u5e03\u5dee',`\u4fdd\u6301 ${signed(Number(events.retain||0)*100,1)}pp / \u4e22\u5931 ${signed(Number(events.turnover||0)*100,1)}pp / \u5c04\u95e8 ${signed(Number(events.shot||0)*100,1)}pp`]]){const node=card(label,value);node.setAttribute('role','listitem');facts.append(node)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent=`${advice.current?'\u5f53\u524d\u5efa\u8bae\u8eab\u4efd\u6709\u6548':'\u5efa\u8bae\u5df2\u8fc7\u671f\uff0c\u4e0d\u53ef\u7ed1\u5b9a\u91c7\u7528'}\u3002\u6b63\u503c\u8868\u793a\u63a8\u8350\u6218\u672f\u51cf\u53bb\u5f53\u524d\u6218\u672f\uff1b\u8fd9\u4e9b\u662f\u77ed\u89c6\u91ce\u6a21\u62df\u5668\u4ee3\u7406\u5dee\u5f02\uff0c\u4e0d\u662f\u6bd4\u5206\u3001\u80dc\u7387\u6216\u56e0\u679c\u4f30\u8ba1\u3002`;managerDecisionPreview.append(heading,facts,boundary)};
const forkBranchLabel=document.createElement('label'),forkBranchInput=document.createElement('input');forkBranchLabel.textContent='分叉时间（比赛分钟）';forkBranchInput.name='branch_minute';forkBranchInput.type='number';forkBranchInput.min='0';forkBranchInput.max='90';forkBranchInput.step='0.5';forkBranchInput.value='45';forkBranchInput.required=true;forkBranchLabel.append(forkBranchInput);forkForm.querySelector('.check').before(forkBranchLabel);
const renderActionAdoptionWithoutManagerProtocol=renderActionAdoption;
renderActionAdoption=studio=>{renderActionAdoptionWithoutManagerProtocol(studio);const node=document.querySelector('#manager-advisor-protocol-evidence'),protocol=studio?.evidence?.manager_advisor_adoption;if(!protocol?.available){node.textContent=studio?'\u7ecf\u7406\u987e\u95ee\u91c7\u7eb3\u534f\u8bae\u7f3a\u5931\uff1b\u4e0d\u80fd\u5f62\u6210\u91c7\u7eb3\u53d6\u8bc1\u7ed3\u8bba\u3002':'';return}node.textContent=`\u7ecf\u7406\u7ea7\u91c7\u7eb3\u534f\u8bae\uff1a${protocol.protocol_state} \u00b7 \u56fa\u5b9a\u4fe1\u606f\u7a97\u53e3 ${(protocol.fixed_information_windows||[]).join('/')} \u00b7 ${protocol.results_available?'\u5df2\u6709\u7ed3\u679c':'\u5c1a\u65e0\u7ed3\u679c'} \u00b7 \u4e0d\u6388\u6743\u8d5b\u679c\u56e0\u679c\u6216\u4ea7\u54c1/\u8bba\u6587\u664b\u7ea7\u3002`};
const renderManagerDecisionLedgerWithoutAdvisorEvidence=renderManagerDecisionLedger;
renderManagerDecisionLedger=season=>{renderManagerDecisionLedgerWithoutAdvisorEvidence(season);const node=document.querySelector('#manager-advisor-evidence-summary'),evidence=season?.manager_decision_ledger?.summary?.world_model_advisor;if(!evidence||!evidence.advised_decisions){node.textContent='';return}node.textContent=`\u4e16\u754c\u6a21\u578b\u987e\u95ee\uff1a${evidence.advised_decisions}/${evidence.decisions} \u6b21\u51b3\u7b56\u83b7\u5f97\u5efa\u8bae \u00b7 ${evidence.adopted_recommendation} \u6b21\u660e\u786e\u91c7\u7528 \u00b7 ${evidence.reviewed_then_selected} \u6b21\u67e5\u770b\u540e\u81ea\u4e3b\u9009\u62e9 \u00b7 ${evidence.unlinked_advice} \u6b21\u672a\u7ed1\u5b9a\u610f\u56fe \u00b7 \u76f4\u63a5\u6267\u884c\u8bc1\u636e ${Math.round(Number(evidence.direct_execution_coverage||0)*100)}% \u00b7 \u53ea\u63cf\u8ff0\u91c7\u7eb3\u548c\u6267\u884c\uff0c\u4e0d\u4f30\u8ba1\u8d5b\u679c\u56e0\u679c\u6548\u5e94\u3002`};
const renderManagerDecisionLedgerWithoutExecutionTrace=renderManagerDecisionLedger;
renderManagerDecisionLedger=season=>{renderManagerDecisionLedgerWithoutExecutionTrace(season);const ledger=season?.manager_decision_ledger,entries=(ledger?.entries||[]).slice(0,6),stateLabels={awaiting_execution:'\u7b49\u5f85\u6bd4\u8d5b\u6267\u884c',evidence_unavailable:'\u8fd0\u884c\u65f6\u7ed1\u5b9a\u8bc1\u636e\u4e0d\u53ef\u7528',recommendation_executed:'\u5efa\u8bae\u5df2\u7ecf\u8fd0\u884c\u65f6\u6267\u884c',reviewed_alternative_executed:'\u5ba1\u9605\u540e\u7684\u66ff\u4ee3\u9009\u62e9\u5df2\u6267\u884c',advised_selection_executed_unlinked:'\u6240\u9009\u6218\u672f\u5df2\u6267\u884c\uff0c\u672a\u7ed1\u5b9a\u91c7\u7528\u610f\u56fe'},controlLabels={pressing_intensity:'\u903c\u62a2',line_height:'\u9632\u7ebf\u9ad8\u5ea6',verticality:'\u7eb5\u5411\u6027',possession_orientation:'\u63a7\u7403',counter_attack:'\u53cd\u51fb',compactness:'\u7d27\u51d1\u5ea6'};for(const [index,row] of entries.entries()){const trace=row.advisor_execution_trace;if(!trace?.available)continue;const host=managerDecisionLedgerList.children[index];if(!host)continue;const details=document.createElement('details'),summary=document.createElement('summary'),runtime=trace.runtime_binding||{};summary.textContent=`\u987e\u95ee\u6267\u884c\u94fe\uff1a${stateLabels[trace.end_to_end_state]||trace.end_to_end_state}`;const chain=document.createElement('p');chain.textContent=`\u5efa\u8bae ${trace.recommendation?.tactic} \u2192 ${trace.interaction?.status==='explicit'?(trace.interaction.intent==='adopt_recommendation'?'\u660e\u786e\u91c7\u7528':'\u5ba1\u9605\u540e\u6539\u9009'):'\u672a\u7ed1\u5b9a\u610f\u56fe'} \u2192 \u51bb\u7ed3 ${trace.selection?.tactic} \u2192 ${runtime.status==='verified'?'\u5f15\u64ce\u5df2\u9a8c\u8bc1':'\u5f15\u64ce\u672a\u9a8c\u8bc1'}`;details.append(summary,chain);const binding=row.execution?.tactical_binding;if(binding?.available){const vector=binding.initial_vector||{},facts=document.createElement('p');facts.textContent=Object.entries(controlLabels).map(([key,label])=>`${label} ${Math.round(Number(vector[key]||0)*100)}%`).join(' \u00b7 ');const drift=document.createElement('p');drift.className='status';drift.textContent=`${binding.binding_kind==='native_team_vector'?'\u7403\u961f\u539f\u751f\u5411\u91cf':'\u9501\u5b9a\u9884\u8bbe'} \u00b7 \u7ec8\u573a\u53d8\u5316 ${binding.changed_controls?.length||0}/22 \u7ef4 \u00b7 \u7ed1\u5b9a ${String(binding.binding_identity||'').slice(0,12)}`;details.append(facts,drift)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='\u8fd9\u91cc\u8bc1\u660e\u6218\u672f\u8f93\u5165\u786e\u5b9e\u8fdb\u5165\u5f15\u64ce\uff1b\u4e0d\u8bc1\u660e\u5b83\u5bfc\u81f4\u4e86\u6bd4\u5206\u6216\u80dc\u8d1f\u3002';details.append(boundary);host.append(details)}const evidence=ledger?.summary?.world_model_advisor,node=document.querySelector('#manager-advisor-evidence-summary');if(evidence?.advised_decisions)node.textContent+=` \u00b7 \u6218\u672f\u8fd0\u884c\u65f6\u7ed1\u5b9a ${evidence.verified_tactical_bindings||0}/${evidence.executed_advised_decisions||0} \u00b7 \u5efa\u8bae\u76f4\u63a5\u6267\u884c ${evidence.direct_recommendation_executions||0} \u6b21`};
const renderManagerIntelligenceWithoutTacticalBinding=renderManagerIntelligence;
renderManagerIntelligence=(command,configured)=>{renderManagerIntelligenceWithoutTacticalBinding(command,configured);const debrief=command?.postmatch_debrief,binding=debrief?.tactical_binding;if(!configured||!debrief?.available||!binding)return;if(!binding.available){const unavailable=document.createElement('p');unavailable.className='status';unavailable.textContent=`\u6218\u672f\u8fd0\u884c\u65f6\u7ed1\u5b9a\u4e0d\u53ef\u7528\uff1a${binding.reason||'unknown'}`;matchdayAttribution.append(unavailable);return}const vector=binding.initial_vector||{},value=`${binding.applied_tactic} \u00b7 ${binding.binding_kind==='native_team_vector'?'\u539f\u751f\u7403\u961f\u5411\u91cf':'\u9501\u5b9a\u6218\u672f\u9884\u8bbe'} \u00b7 \u903c\u62a2 ${Math.round(Number(vector.pressing_intensity||0)*100)}% \u00b7 \u9632\u7ebf ${Math.round(Number(vector.line_height||0)*100)}% \u00b7 \u7eb5\u5411 ${Math.round(Number(vector.verticality||0)*100)}% \u00b7 \u53d8\u5316 ${binding.changed_controls?.length||0}/22 \u7ef4`,node=card('\u5f15\u64ce\u6218\u672f\u7ed1\u5b9a',value);node.setAttribute('role','listitem');matchdayAttribution.append(node)};
const renderManagerWorldModelAdviceWithoutStatusReset=renderManagerWorldModelAdvice;
renderManagerWorldModelAdvice=(...args)=>{managerWorldModelAdviceSummary.className='status';return renderManagerWorldModelAdviceWithoutStatusReset(...args)};
managerManual.addEventListener('change',()=>renderManagerSquad(currentSeason,currentSeason?.next_manager_fixture));
managerRotation.addEventListener('change',()=>renderManagerSquad(currentSeason,currentSeason?.next_manager_fixture));
requestManagerAdvice.addEventListener('click',()=>void requestManagerWorldModelAdvice());
adoptManagerAdvice.addEventListener('click',adoptCurrentManagerAdvice);
managerTactic.addEventListener('change',()=>{if(currentManagerAdvice)managerAdviceIntent='reviewed_then_selected'});
managerDecisionForm.addEventListener('submit',event=>{event.preventDefault();let payload;try{payload=managerDecisionPayload(true)}catch(error){announce(message,error.message,'error',true);return}submit(managerDecisionForm,'/api/v1/seasons/decision',payload)});
managerDecisionForm.addEventListener('input',scheduleManagerDecisionPreview);
managerDecisionForm.addEventListener('change',scheduleManagerDecisionPreview);
for(const row of managerRules){row.querySelector('.manager-rule-enabled').addEventListener('change',event=>{for(const control of row.querySelectorAll('input:not(.manager-rule-enabled),select'))control.disabled=!event.target.checked})}
managerSquad.addEventListener('change',()=>{for(const row of managerRules)populateRulePlayers(row,currentSeason?.next_manager_fixture,{off:row.querySelector('.manager-sub-off').value,on:row.querySelector('.manager-sub-on').value})});
playMatchday.addEventListener('click',()=>submit(seasonActions,'/api/v1/seasons/next-matchday',{}));
experienceSelect.addEventListener('change',()=>configureMatchPlan(currentMatchCapabilities,currentStudioMode));
libraryFilter.addEventListener('change',()=>renderLibrary(currentLibrary));
match.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(match),key=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random(),experience=f.get('experience'),plan={experience,home_tactic:experience==='tactical_lab'?f.get('home_tactic'):'team_identity',away_tactic:experience==='tactical_lab'?f.get('away_tactic'):'team_identity',reuse_last_seed:f.get('reuse_last_seed')==='on'};submit(match,'/api/v1/matches',{home:f.get('home'),away:f.get('away'),fast:f.get('fast')==='on',plan},{'Idempotency-Key':key})});
pairForm.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(pairForm),focus=f.get('focus_side'),baseline=f.get('baseline_tactic'),treatment=f.get('treatment_tactic'),opponent=f.get('opponent_tactic'),plan={seed:Number(f.get('seed')),baseline:{home_tactic:focus==='home'?baseline:opponent,away_tactic:focus==='away'?baseline:opponent},treatment:{home_tactic:focus==='home'?treatment:opponent,away_tactic:focus==='away'?treatment:opponent}},key=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random();submit(pairForm,'/api/v1/paired-matches',{home:f.get('home'),away:f.get('away'),fast:f.get('fast')==='on',plan},{'Idempotency-Key':key})});
forkForm.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(forkForm),plan={seed:Number(f.get('seed')),home_tactic:f.get('home_tactic'),away_tactic:f.get('away_tactic'),branch_at_sec:Number(f.get('branch_minute'))*60},key=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random();submit(forkForm,'/api/v1/world-model-forks',{home:f.get('home'),away:f.get('away'),fast:f.get('fast')==='on',plan},{'Idempotency-Key':key})});
studyForm.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(studyForm),focus=f.get('focus_side'),baseline=f.get('baseline_tactic'),treatment=f.get('treatment_tactic'),opponent=f.get('opponent_tactic'),budget=Number(f.get('pair_budget')),start=Number(f.get('seed_start')),seeds=Array.from({length:budget},(_,index)=>start+index),plan={study_id:f.get('study_id'),fixture:{home:f.get('home'),away:f.get('away')},baseline:{home_tactic:focus==='home'?baseline:opponent,away_tactic:focus==='away'?baseline:opponent},treatment:{home_tactic:focus==='home'?treatment:opponent,away_tactic:focus==='away'?treatment:opponent},seeds,fast:f.get('fast')==='on',analysis_plan:{smallest_effect_size:0.05}},key=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random();submit(studyForm,'/api/v1/tactical-studies',{plan},{'Idempotency-Key':key})});
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
