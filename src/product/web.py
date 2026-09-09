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
from src.product.manager_intervention_workspace import (
    build_manager_intervention_workspace,
    validate_manager_intervention_workspace,
)
from src.product.manager_world_navigator import (
    build_manager_world_navigator,
    validate_manager_world_navigator,
)
from src.product.manager_world_story import (
    build_manager_world_story,
    validate_manager_world_story,
)
from src.product.match_plan import (
    MatchPlan, PairedMatchPlan, WorldModelForkPlan, WorldModelForkSetPlan,
    playable_tactic_catalog,
)
from src.product.season import ManagerDecision, SeasonPlan
from src.product.player_promises import PlayerPromisePlan
from src.product.tactical_study import TacticalStudyPlan
from src.product.recovery import ProductRecovery
from src.product.study_delivery import ProductValueStudyDelivery
from src.product.tasks import (
    COUNTERFACTUAL_FUTURE_STATUSES,
    BackgroundMatchWorker, ProductTaskQueue, TaskConflict,
)
from src.product.telemetry import ProductTelemetry, route_template
from src.product.web_security import WebAccessPolicy, is_loopback_host
from src.product.workspace import ProductWorkspace, StudioConfig
from src.product.world_model_fork_set import (
    validate_fork_set_scenario_evidence,
)
from src.simulation.llm_gateway import provider_preflight


LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 64 * 1024
JSON_HEADERS = [("Content-Type", "application/json; charset=utf-8")]
_WebResponse = tuple[int, list[tuple[str, str]], bytes]
_METHOD_RESTRICTED_EXACT_PATHS = frozenset({
    "/api/v1/studio",
    "/api/v1/matches",
    "/api/v1/paired-matches",
    "/api/v1/world-model-forks",
    "/api/v1/world-model-fork-sets",
    "/api/v1/seasons/world-model-future-set",
    "/api/v1/seasons/world-model-future-review",
    "/api/v1/tactical-studies",
    "/api/v1/seasons",
    "/api/v1/seasons/next-matchday",
    "/api/v1/seasons/decision",
    "/api/v1/tasks",
    "/api/v1/recovery",
    "/api/v1/backups",
    "/api/v1/excellence/evidence-kit.zip",
    "/api/v1/studies/product-value",
    "/api/v1/studies/product-value/registrations",
})
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
        "simulation_clock": {
            "contract": "authoritative_tick_v2",
            "legacy_contract": "legacy_affective_offset_v1",
            "studio_requires_authoritative": True,
        },
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
            "clock_contract": "authoritative_tick_v2",
            "result_contract": "evidence_graded_counterfactual_future_v1",
            "fixed_controls": [
                "fixture", "tactics", "fast_configuration", "checkpoint",
                "clock_contract",
            ],
            "score_path": "physics_official",
            "evidence_chain": [
                "isolated_policy_assignment",
                "identical_pre_intervention_branch_anchor",
                "realized_action_change",
                "direct_runtime_identity",
                "descriptive_downstream_windows",
                "unified_counterfactual_future_summary",
            ],
            "downstream_causal_attribution": False,
            "claim_boundary": "single_fixture_seed_simulator_contrast_only",
        },
        "world_model_fork_set": {
            "modes": ["research"],
            "scenario_budget": {"min": 2, "max": 4, "default": 3},
            "default_branch_times_seconds": [1800, 2700, 3600],
            "fixed_controls": [
                "fixture", "seed", "tactics", "fast_configuration",
                "checkpoint", "policy_pair",
            ],
            "interim_ranking": "withheld_and_never_performed",
            "result_contract": "descriptive_timing_sensitivity_v1",
            "scenario_evidence_contract": (
                "identity_bound_non_ranked_per_timepoint_v1"
            ),
            "mechanism_example_contract": (
                "max_three_actions_with_30s_120s_descriptive_windows_v1"
            ),
            "claim_boundary": "fixed_simulator_timing_set_no_best_time",
            "manager_bridge": {
                "endpoint": "/api/v1/seasons/world-model-future-set",
                "review_endpoint": "/api/v1/seasons/world-model-future-review",
                "requires": "frozen_unstarted_manager_decision",
                "identity_checks": "before_and_after_task_execution",
                "review_intents": [
                    "keep_after_review", "revise_after_review",
                ],
                "review_receipt": (
                    "v3_preserves_bounded_action_mechanism_examples"
                ),
                "second_persisted_season_state": False,
            },
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
    ) -> _WebResponse:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))

        health = self._health_response(method, path)
        if health is not None:
            return health
        self._require_allowed_access(environ)
        authentication = self._authentication_response(environ, method, path)
        if authentication is not None:
            return authentication
        if not self.access_policy.authenticated(environ):
            if method == "GET" and path == "/":
                return self._redirect_response("/login")
            raise WebRequestError(
                401, "authentication_required", "Authentication required",
            )

        response = None
        if method == "GET":
            response = self._get_response(path)
        elif method == "POST":
            response = self._post_response(environ, path)
        if response is not None:
            return response
        if self._is_method_restricted_path(path):
            raise WebRequestError(405, "method_not_allowed", "Method not allowed")
        raise WebRequestError(404, "not_found", "Resource not found")

    def _health_response(
        self, method: str, path: str,
    ) -> _WebResponse | None:
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
        return None

    def _require_allowed_access(self, environ: dict[str, Any]) -> None:
        if not self.access_policy.host_allowed(environ):
            raise WebRequestError(400, "invalid_host", "Host is not allowed")
        if not self.access_policy.secure_transport(environ):
            raise WebRequestError(
                426, "https_required", "Remote Web access requires the trusted HTTPS proxy",
            )

    def _authentication_response(
        self,
        environ: dict[str, Any],
        method: str,
        path: str,
    ) -> _WebResponse | None:
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
        return None

    def _ready_response(self) -> _WebResponse:
        payload = self._studio_status()
        ready = bool(
            payload.get("configured")
            and payload.get("studio", {}).get("readiness", {}).get("ready")
        )
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
            "schema_version": 1,
            "ready": ready,
            "configured": payload["configured"],
            "blockers": blockers,
            "telemetry_ready": telemetry_ready,
        })

    def _studio_status_response(self) -> _WebResponse:
        return self._json_response(200, self._studio_status(), csrf=True)

    def _tasks_response(self) -> _WebResponse:
        return self._json_response(200, {
            "schema_version": 1,
            "tasks": [
                self._task_for_web(task) for task in self.task_queue.list_tasks()
            ],
        })

    def _get_response(self, path: str) -> _WebResponse | None:
        exact_routes: dict[str, Callable[[], _WebResponse]] = {
            "/": self._html_response,
            "/readyz": self._ready_response,
            "/api/v1/studio": self._studio_status_response,
            "/api/v1/operations": lambda: self._json_response(
                200, self.telemetry.snapshot(),
            ),
            "/api/v1/excellence/evidence-kit.zip": self._evidence_kit_response,
            "/api/v1/studies/product-value": self._product_value_study_status,
            "/api/v1/recovery": self._recovery_status,
            "/api/v1/tasks": self._tasks_response,
        }
        handler = exact_routes.get(path)
        if handler is not None:
            return handler()

        resource_routes = (
            (r"/api/v1/sporting-plans/([^/]+)", self._sporting_plan),
            (r"/api/v1/recruitment-markets/([^/]+)", self._recruitment_market),
            (r"/api/v1/lifecycle-previews/([^/]+)", self._lifecycle_preview),
            (r"/api/v1/free-agent-markets/([^/]+)", self._free_agent_market),
        )
        for pattern, resource_handler in resource_routes:
            match = re.fullmatch(pattern, path)
            if match is not None:
                return resource_handler(unquote(match.group(1)))

        value_packet = re.fullmatch(
            r"/api/v1/studies/product-value/packets/(REG-[0-9A-F]{24})\.json",
            path,
        )
        if value_packet is not None:
            return self._product_value_packet_response(value_packet.group(1))
        if path.startswith("/api/v1/tasks/"):
            return self._task_response(path.removeprefix("/api/v1/tasks/"))
        if path.startswith("/artifacts/"):
            return self._artifact_response(path.removeprefix("/artifacts/"))
        return None

    def _task_response(self, task_id: str) -> _WebResponse:
        if "/" in task_id:
            raise WebRequestError(404, "task_not_found", "Task not found")
        try:
            task = self.task_queue.get_task(task_id)
        except (FileNotFoundError, ValueError) as exc:
            raise WebRequestError(404, "task_not_found", "Task not found") from exc
        return self._json_response(200, {"task": self._task_for_web(task)})

    def _post_response(
        self, environ: dict[str, Any], path: str,
    ) -> _WebResponse | None:
        payload_routes: dict[str, Callable[[dict[str, Any]], _WebResponse]] = {
            "/api/v1/studies/product-value/registrations": (
                self._register_product_value_participant
            ),
            "/api/v1/studio": self._create_studio,
            "/api/v1/seasons": self._create_season,
            "/api/v1/scouting-reports": self._scout_free_agent,
            "/api/v1/seasons/decision": self._set_season_decision,
            "/api/v1/seasons/decision-preview": self._preview_season_decision,
            "/api/v1/seasons/decision-advice": self._request_season_decision_advice,
            "/api/v1/seasons/player-promises": self._set_player_promises,
            "/api/v1/seasons/world-model-future-review": (
                self._review_manager_world_model_future_set
            ),
        }
        payload_handler = payload_routes.get(path)
        if payload_handler is not None:
            self._require_csrf(environ)
            return payload_handler(self._read_json(environ))

        queued_routes: dict[
            str, Callable[[dict[str, Any], dict[str, Any]], _WebResponse]
        ] = {
            "/api/v1/matches": self._queue_match,
            "/api/v1/paired-matches": self._queue_paired_match,
            "/api/v1/world-model-forks": self._queue_world_model_fork,
            "/api/v1/world-model-fork-sets": self._queue_world_model_fork_set,
            "/api/v1/seasons/world-model-future-set": (
                self._queue_manager_world_model_future_set
            ),
            "/api/v1/tactical-studies": self._queue_tactical_study,
        }
        queued_handler = queued_routes.get(path)
        if queued_handler is not None:
            self._require_csrf(environ)
            return queued_handler(environ, self._read_json(environ))
        if path == "/api/v1/seasons/next-matchday":
            self._require_csrf(environ)
            return self._queue_season_matchday(environ)
        if path == "/api/v1/backups":
            self._require_csrf(environ)
            return self._create_managed_backup()

        backup_action = re.fullmatch(
            r"/api/v1/backups/([^/]+)/(verify|restore)", path,
        )
        if backup_action is not None:
            self._require_csrf(environ)
            backup_id, action = backup_action.groups()
            if action == "verify":
                return self._verify_managed_backup(backup_id)
            return self._restore_managed_backup(backup_id, self._read_json(environ))
        task_requeue = re.fullmatch(
            r"/api/v1/tasks/([a-zA-Z0-9]+)/requeue", path,
        )
        if task_requeue is not None:
            self._require_csrf(environ)
            return self._requeue_task(
                task_requeue.group(1), self._read_json(environ),
            )
        return None

    @staticmethod
    def _is_method_restricted_path(path: str) -> bool:
        return bool(
            path in _METHOD_RESTRICTED_EXACT_PATHS
            or re.fullmatch(r"/api/v1/backups/([^/]+)/(verify|restore)", path)
            or re.fullmatch(r"/api/v1/tasks/([a-zA-Z0-9]+)/requeue", path)
            or re.fullmatch(
                r"/api/v1/studies/product-value/packets/(REG-[0-9A-F]{24})\.json",
                path,
            )
        )

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

    def _product_value_study_status(
        self,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        try:
            status = ProductValueStudyDelivery(self.root).status()
        except (OSError, ValueError) as exc:
            raise WebRequestError(
                503,
                "product_value_study_unavailable",
                "The blinded product-value study authority failed validation",
            ) from exc
        return self._json_response(200, status, csrf=True)

    def _register_product_value_participant(
        self,
        payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        if set(payload) != {
            "participant_id", "target_role", "moderator_id", "consent_recorded",
        }:
            raise WebRequestError(
                422,
                "invalid_study_registration",
                "Study registration fields are not exact",
            )
        participant_id = self._text_field(payload, "participant_id", maximum=52)
        target_role = self._text_field(payload, "target_role", maximum=32)
        moderator_id = self._text_field(payload, "moderator_id", maximum=52)
        if payload.get("consent_recorded") is not True:
            raise WebRequestError(
                422,
                "consent_required",
                "Recorded informed consent is required before registration",
            )
        if not self._mutation_lock.acquire(blocking=False):
            raise WebRequestError(
                409, "operation_in_progress", "Another mutation is running",
            )
        try:
            try:
                result = ProductValueStudyDelivery(self.root).register(
                    participant_id=participant_id,
                    target_role=target_role,
                    moderator_id=moderator_id,
                    consent_recorded=True,
                )
            except (OSError, ValueError) as exc:
                raise WebRequestError(
                    422,
                    "study_registration_rejected",
                    "Registration was rejected by the frozen study authority",
                ) from exc
            return self._json_response(201 if result["created"] else 200, {
                "schema_version": 1,
                "status": "registered" if result["created"] else "already_registered",
                "created": result["created"],
                "registration": result["registration"],
                "packet_url": (
                    "/api/v1/studies/product-value/packets/"
                    f"{result['registration']['registration_id']}.json"
                ),
                "scoring_material_exposed": False,
                "external_calls_made": False,
                "matches_executed": 0,
                "training_executed": False,
            })
        finally:
            self._mutation_lock.release()

    def _product_value_packet_response(
        self,
        registration_id: str,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        import hashlib

        try:
            content = ProductValueStudyDelivery(self.root).packet_bytes(
                registration_id
            )
        except (OSError, ValueError) as exc:
            raise WebRequestError(
                404,
                "participant_packet_unavailable",
                "No valid blinded packet exists for this registration",
            ) from exc
        digest = hashlib.sha256(content).hexdigest()
        return 200, [
            ("Content-Type", "application/json; charset=utf-8"),
            (
                "Content-Disposition",
                f'attachment; filename="{registration_id}.json"',
            ),
            ("X-GFS-Artifact-SHA256", digest),
            ("X-GFS-Blinded-Study-Packet", "true"),
        ], content

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

    def _unconfigured_studio_status(
        self,
        *,
        provider: dict[str, Any],
        web_tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "configured": False,
            "studio": None,
            "match_capabilities": _match_capabilities(),
            "control_plane": ProductControlPlane(self.root).snapshot(),
            "provider": provider,
            "access": self.access_policy.public_summary(),
            "operations": self.telemetry.snapshot(),
            "tasks": web_tasks[:20],
            "evidence_library": {
                "schema_version": 1,
                "limit": 50,
                "matches": [],
                "pairs": [],
                "forks": [],
                "fork_sets": [],
                "studies": [],
                "truncated": False,
            },
        }

    def _attach_active_season_artifact_urls(
        self, season: dict[str, Any],
    ) -> None:
        for fixture in season.get("fixtures") or []:
            if not isinstance(fixture, dict):
                continue
            dashboard_url = self._safe_artifact_url(fixture.get("dashboard"))
            if dashboard_url:
                fixture["dashboard_url"] = dashboard_url
        profile = season.get("manager_profile") or {}
        journal = profile.get("journal") if isinstance(profile, dict) else None
        if not isinstance(journal, list):
            return
        safe_by_fixture = {
            fixture.get("fixture_id"): fixture.get("dashboard_url")
            for fixture in season.get("fixtures") or []
            if isinstance(fixture, dict) and fixture.get("dashboard_url")
        }
        for entry in journal:
            if isinstance(entry, dict):
                entry["dashboard_url"] = safe_by_fixture.get(entry.get("fixture_id"))

    def _attach_archived_season_artifact_urls(
        self, season_history: Any,
    ) -> None:
        for archived in season_history or []:
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

    def _attach_status_artifact_urls(
        self, status: dict[str, Any],
    ) -> dict[str, Any] | None:
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
            self._attach_active_season_artifact_urls(season)
        self._attach_archived_season_artifact_urls(status.get("season_history"))
        return season if isinstance(season, dict) else None

    def _library_matches(
        self, match_history: Any,
    ) -> list[dict[str, Any]]:
        library_matches = []
        for match in match_history or []:
            if not isinstance(match, Mapping):
                continue
            item = {
                key: match.get(key) for key in (
                    "match_id", "home", "away", "seed", "fast", "score",
                    "integrity", "experience", "home_tactic", "away_tactic",
                )
            }
            item["dashboard_url"] = self._safe_artifact_url(match.get("dashboard"))
            item["comparison_url"] = self._safe_artifact_url(
                match.get("comparison_dashboard")
            )
            library_matches.append(item)
        return library_matches

    @staticmethod
    def _manager_context_for_web(raw_context: Any) -> dict[str, Any] | None:
        if not isinstance(raw_context, Mapping):
            return None
        raw_fixture = raw_context.get("fixture") or {}
        return {
            "season_id": raw_context.get("season_id"),
            "season_revision": raw_context.get("season_revision"),
            "fixture_id": (
                raw_context.get("fixture_id") or raw_fixture.get("fixture_id")
            ),
            "matchday": raw_context.get("matchday") or raw_fixture.get("matchday"),
            "manager_team": raw_context.get("manager_team"),
            "decision_identity": raw_context.get("decision_identity"),
            "context_identity": raw_context.get("context_identity"),
            "claim_boundary": raw_context.get("claim_boundary"),
        }

    def _fork_set_library_item(
        self, task: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        request = task.get("request") or {}
        try:
            plan = WorldModelForkSetPlan.from_payload(request.get("plan") or {})
        except ValueError:
            return None
        result = task.get("result") or {}
        progress = task.get("fork_set_progress") or {}
        aggregate = result.get("aggregate") or {}
        try:
            scenario_evidence = (
                validate_fork_set_scenario_evidence(
                    result.get("scenario_evidence"), plan.branch_times_sec,
                )
                if result.get("scenario_evidence") is not None
                else []
            )
        except ValueError:
            scenario_evidence = []
        manager_context = self._manager_context_for_web(
            result.get("manager_context") or request.get("manager_context")
        )
        return {
            "task_id": task.get("task_id"),
            "home": request.get("home"),
            "away": request.get("away"),
            "seed": plan.seed,
            "branch_times_sec": list(plan.branch_times_sec),
            "home_tactic": plan.home_tactic,
            "away_tactic": plan.away_tactic,
            "state": task.get("state"),
            "scenarios_completed": progress.get("scenarios_completed", 0),
            "fixed_scenario_budget": len(plan.branch_times_sec),
            "timing_sensitivity_observed": (
                aggregate.get("timing_sensitivity_observed") is True
            ),
            "action_divergence_scenarios": _bounded_public_count(
                aggregate.get("action_divergence_scenarios", 0)
            ),
            "local_attribution_scenarios": _bounded_public_count(
                aggregate.get("local_attribution_scenarios", 0)
            ),
            "descriptive_future_difference_scenarios": _bounded_public_count(
                aggregate.get("descriptive_future_difference_scenarios", 0)
            ),
            "scenario_evidence": scenario_evidence,
            "scenario_evidence_available": bool(scenario_evidence),
            "ranking_performed": False,
            "fork_set_url": result.get("fork_set_url"),
            "manager_context": manager_context,
        }

    @staticmethod
    def _fork_propagation_for_web(result: Mapping[str, Any]) -> dict[str, Any]:
        raw_propagation = result.get("propagation") or {}
        if not isinstance(raw_propagation, Mapping):
            return {"available": False, "status": "unavailable"}
        raw_future = raw_propagation.get("future_summary") or {}
        if not isinstance(raw_future, Mapping):
            raw_future = {}
        future_status = str(raw_future.get("status") or "")
        if future_status not in COUNTERFACTUAL_FUTURE_STATUSES:
            future_status = "unknown_future_status"
        propagation = {
            "available": bool(raw_propagation.get("available")),
            "status": str(raw_propagation.get("status") or "unknown")[:80],
            "changed_decisions": _bounded_public_count(
                raw_propagation.get("changed_decisions")
            ),
            "directly_observed_changes": _bounded_public_count(
                raw_propagation.get("directly_observed_changes")
            ),
            "locally_attributable_changes": _bounded_public_count(
                raw_propagation.get("locally_attributable_changes")
            ),
            "replay_windows_available": (
                raw_propagation.get("replay_windows_available") is True
            ),
            "downstream_causal_attribution_authorized": False,
            "future_summary": {
                "available": raw_future.get("available") is True,
                "status": future_status,
                "changed_actions": _bounded_public_count(
                    raw_future.get("changed_actions")
                ),
                "descriptive_outcome_difference_count": _bounded_public_count(
                    raw_future.get("descriptive_outcome_difference_count")
                ),
                "simulator_local_action_attribution": (
                    raw_future.get("simulator_local_action_attribution") is True
                ),
                "match_outcome_causality": False,
                "real_football_causality": False,
            },
        }
        if raw_propagation.get("branch_at_sec") is not None:
            propagation.update({
                "branch_at_sec": raw_propagation.get("branch_at_sec"),
                "branch_anchor_verified": bool(
                    raw_propagation.get("branch_anchor_verified")
                ),
                "branch_state_identity": str(
                    raw_propagation.get("branch_state_identity") or ""
                )[:64],
            })
        return propagation

    def _fork_library_item(
        self, task: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        request = task.get("request") or {}
        try:
            plan = WorldModelForkPlan.from_payload(request.get("plan") or {})
        except ValueError:
            return None
        result = task.get("result") or {}
        return {
            "task_id": task.get("task_id"),
            "home": request.get("home"),
            "away": request.get("away"),
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
            "propagation": self._fork_propagation_for_web(result),
        }

    @staticmethod
    def _pair_library_item(
        task: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        request = task.get("request") or {}
        try:
            plan = PairedMatchPlan.from_payload(request.get("plan") or {})
        except ValueError:
            return None
        result = task.get("result") or {}
        return {
            "task_id": task.get("task_id"),
            "home": request.get("home"),
            "away": request.get("away"),
            "seed": plan.seed,
            "focus_side": plan.focus_side,
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
        }

    @staticmethod
    def _study_library_item(
        task: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        raw_plan = (task.get("request") or {}).get("plan") or {}
        try:
            plan = TacticalStudyPlan.from_payload(raw_plan)
        except ValueError:
            return None
        progress = task.get("study_progress") or {}
        result = task.get("result") or {}
        return {
            "task_id": task.get("task_id"),
            "study_id": plan.study_id,
            "home": plan.home,
            "away": plan.away,
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
                task.get("state") != "completed" or not result.get("study_url")
            ),
            "study_url": result.get("study_url"),
        }

    def _library_task_groups(
        self, web_tasks: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {
            "studies": [],
            "pairs": [],
            "forks": [],
            "fork_sets": [],
        }
        builders = {
            "world_model_fork_set": ("fork_sets", self._fork_set_library_item),
            "world_model_fork": ("forks", self._fork_library_item),
            "paired_match": ("pairs", self._pair_library_item),
            "tactical_study": ("studies", self._study_library_item),
        }
        for task in web_tasks:
            kind = task.get("kind")
            selected = builders.get(kind) if isinstance(kind, str) else None
            if selected is None:
                continue
            group, builder = selected
            item = builder(task)
            if item is not None:
                groups[group].append(item)
        return groups

    @staticmethod
    def _manager_future_sets_for_web(
        season: Mapping[str, Any],
        fork_sets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        season_id = str(season.get("season_id") or "")
        season_revision = int(season.get("revision", -1))
        next_fixture = season.get("next_manager_fixture") or {}
        review_by_task = {
            review.get("task_id"): review
            for fixture in season.get("fixtures") or []
            for review in fixture.get("manager_future_reviews") or []
            if isinstance(fixture, Mapping)
            and isinstance(review, Mapping)
            and isinstance(review.get("task_id"), str)
        }
        manager_sets = []
        for item in fork_sets:
            context = item.get("manager_context")
            if (
                not isinstance(context, Mapping)
                or context.get("season_id") != season_id
            ):
                continue
            manager_sets.append({
                **item,
                "review_receipt": review_by_task.get(item.get("task_id")),
                "binding_current": bool(
                    context.get("season_revision") == season_revision
                    and context.get("fixture_id")
                    == next_fixture.get("fixture_id")
                ),
                "claim_boundary": (
                    "simulator exploration bound to one frozen manager decision; "
                    "never score prediction or causal coaching evidence"
                ),
            })
        return manager_sets

    @staticmethod
    def _normalize_manager_set_bindings(
        manager_sets: list[dict[str, Any]],
        intervention_workspace: Mapping[str, Any],
    ) -> bool:
        canonical_binding = {
            row["task_id"]: row["current_binding"]
            for row in intervention_workspace["explorations"]
        }
        normalized = False
        for item in manager_sets:
            task_id = item.get("task_id")
            if (
                task_id in canonical_binding
                and item.get("binding_current") is not canonical_binding[task_id]
            ):
                item["binding_current"] = canonical_binding[task_id]
                normalized = True
        return normalized

    def _attach_manager_world_views(
        self,
        season: dict[str, Any],
        fork_sets: list[dict[str, Any]],
    ) -> None:
        manager_sets = self._manager_future_sets_for_web(season, fork_sets)
        season["manager_future_sets"] = manager_sets
        intervention_workspace = build_manager_intervention_workspace(
            season, manager_sets,
        )
        if self._normalize_manager_set_bindings(
            manager_sets, intervention_workspace,
        ):
            intervention_workspace = build_manager_intervention_workspace(
                season, manager_sets,
            )
        validate_manager_intervention_workspace(
            intervention_workspace,
            season=season,
            manager_future_sets=manager_sets,
        )
        season["manager_intervention_workspace"] = intervention_workspace
        world_navigator = build_manager_world_navigator(season)
        validate_manager_world_navigator(world_navigator, season=season)
        season["manager_world_navigator"] = world_navigator
        world_story = build_manager_world_story(world_navigator)
        validate_manager_world_story(world_story, navigator=world_navigator)
        season["manager_world_story"] = world_story
        command = season.get("matchday_command_center")
        if isinstance(command, dict):
            command["manager_future_sets"] = manager_sets
            command["manager_intervention_workspace"] = intervention_workspace
            command["manager_world_navigator"] = world_navigator
            command["manager_world_story"] = world_story

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
            return self._unconfigured_studio_status(
                provider=provider, web_tasks=web_tasks,
            )
        try:
            status = ProductWorkspace.load(self.root).status()
        except (OSError, ValueError) as exc:
            raise WebRequestError(
                409, "invalid_session", "The persisted Studio session is invalid",
            ) from exc
        season = self._attach_status_artifact_urls(status)
        library_matches = self._library_matches(status.get("match_history"))
        library_tasks = self._library_task_groups(web_tasks)
        library_studies = library_tasks["studies"]
        library_pairs = library_tasks["pairs"]
        library_forks = library_tasks["forks"]
        library_fork_sets = library_tasks["fork_sets"]
        if season is not None:
            self._attach_manager_world_views(season, library_fork_sets)
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
                "fork_sets": library_fork_sets,
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

    def _fixture_request_fields(
        self,
        payload: dict[str, Any],
        *,
        same_team_message: str,
    ) -> tuple[str, str, bool]:
        home = self._text_field(payload, "home", maximum=80)
        away = self._text_field(payload, "away", maximum=80)
        if home.casefold() == away.casefold():
            raise WebRequestError(422, "same_team", same_team_message)
        fast = payload.get("fast", False)
        if not isinstance(fast, bool):
            raise WebRequestError(422, "invalid_fast", "fast must be boolean")
        return home, away, fast

    def _queue_match(
        self, environ: dict[str, Any], payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        home, away, fast = self._fixture_request_fields(
            payload, same_team_message="Home and away teams must differ",
        )
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
        home, away, fast = self._fixture_request_fields(
            payload, same_team_message="Teams must differ",
        )
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
        home, away, fast = self._fixture_request_fields(
            payload, same_team_message="Teams must differ",
        )
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

    def _queue_world_model_fork_set(
        self, environ: dict[str, Any], payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        home, away, fast = self._fixture_request_fields(
            payload, same_team_message="Teams must differ",
        )
        try:
            plan = WorldModelForkSetPlan.from_payload(payload.get("plan"))
        except ValueError as exc:
            raise WebRequestError(
                422, "invalid_world_model_fork_set", str(exc),
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
                    422, "invalid_world_model_fork_set", str(exc),
                ) from exc
            if not workspace.readiness()["ready"]:
                raise WebRequestError(
                    409, "world_model_fork_set_blocked",
                    "World-model fork set blocked by readiness gates",
                )
            try:
                task, created = self.task_queue.submit_world_model_fork_set(
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

    def _queue_manager_world_model_future_set(
        self, environ: dict[str, Any], payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        raw_times = payload.get("branch_times_sec")
        if not isinstance(raw_times, list):
            raise WebRequestError(
                422, "invalid_manager_future_set",
                "branch_times_sec must be a fixed list",
            )
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
            if not workspace.readiness()["ready"]:
                raise WebRequestError(
                    409, "manager_future_set_blocked",
                    "Manager future set blocked by readiness gates",
                )
            try:
                context = workspace.manager_future_set_context(
                    fixture_id=payload.get("fixture_id"),
                )
                plan = WorldModelForkSetPlan(
                    home_tactic=context["home_tactic"],
                    away_tactic=context["away_tactic"],
                    seed=context["match_seed"],
                    branch_times_sec=tuple(raw_times),
                )
                plan.validate_for_mode(workspace.config.mode)
                fixture = context["fixture"]
                task, created = self.task_queue.submit_world_model_fork_set(
                    fixture["home"], fixture["away"],
                    fast=context["fast"], plan=plan,
                    manager_context=context,
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
                    422, "invalid_manager_future_set", str(exc),
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

    def _review_manager_world_model_future_set(
        self, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        allowed = {
            "task_id", "fixture_id", "expected_revision", "intent",
            "decision",
        }
        if set(payload) - allowed:
            raise WebRequestError(
                422, "invalid_manager_future_review",
                "manager future review has unsupported fields",
            )
        try:
            task_id = self._text_field(payload, "task_id", maximum=64)
            fixture_id = self._text_field(payload, "fixture_id", maximum=32)
            intent = self._text_field(payload, "intent", maximum=32)
            expected_revision = payload.get("expected_revision")
            if (
                isinstance(expected_revision, bool)
                or not isinstance(expected_revision, int)
                or expected_revision < 0
            ):
                raise ValueError("manager future review revision is invalid")
            task = self.task_queue.get_task(task_id)
            if (
                task.get("kind") != "world_model_fork_set"
                or task.get("state") != "completed"
                or not isinstance(task.get("request"), Mapping)
                or not isinstance(task.get("result"), Mapping)
                or not isinstance(
                    (task.get("request") or {}).get("manager_context"),
                    Mapping,
                )
            ):
                raise ValueError(
                    "manager future review requires a completed bound future set"
                )
            decision_payload = payload.get("decision")
            final_decision = None
            if intent == "revise_after_review":
                final_decision = ManagerDecision.from_payload(
                    decision_payload or {}
                )
            elif intent == "keep_after_review":
                if decision_payload is not None:
                    raise ValueError(
                        "keep-after-review does not accept a decision payload"
                    )
            else:
                raise ValueError("manager future review intent is invalid")
            season = ProductWorkspace.load(self.root).review_manager_future_set(
                task_id=task_id,
                task_request=task["request"],
                task_result=task["result"],
                intent=intent,
                fixture_id=fixture_id,
                expected_revision=expected_revision,
                final_decision=final_decision,
            )
        except FileNotFoundError as exc:
            raise WebRequestError(
                404, "manager_future_review_source_missing",
                "Studio or future-set task was not found",
            ) from exc
        except ValueError as exc:
            raise WebRequestError(
                422, "invalid_manager_future_review", str(exc),
            ) from exc
        return self._json_response(200, {
            "schema_version": 1, "season": season,
        })

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

    @staticmethod
    def _attach_task_result_urls(payload: dict[str, Any]) -> None:
        result = payload.get("result") or {}
        for field, url_field in (
            ("dashboard", "dashboard_url"),
            ("baseline_dashboard", "baseline_url"),
            ("treatment_dashboard", "treatment_url"),
            ("comparison_dashboard", "comparison_url"),
            ("study_dashboard", "study_url"),
            ("fork_set_dashboard", "fork_set_url"),
        ):
            dashboard_url = ProductWebApp._safe_artifact_url(result.get(field))
            if dashboard_url:
                result[url_field] = dashboard_url

    def _attach_study_progress(self, payload: dict[str, Any]) -> None:
        plan_payload = (payload.get("request") or {}).get("plan") or {}
        try:
            validated_plan = TacticalStudyPlan.from_payload(plan_payload)
            if payload.get("state") == "queued":
                payload["study_progress"] = {
                    "state": "queued",
                    "pairs_completed": 0,
                    "fixed_pair_budget": len(validated_plan.seeds),
                    "analysis_withheld": True,
                }
                return
            workspace = ProductWorkspace.load(self.root)
            progress_path = (
                workspace.output_root
                / "studies"
                / validated_plan.study_id
                / "progress.json"
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
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
        ):
            if payload.get("state") != "queued":
                payload["study_progress"] = {
                    "state": "invalid_progress",
                    "analysis_withheld": True,
                }

    def _attach_fork_set_progress(self, payload: dict[str, Any]) -> None:
        plan_payload = (payload.get("request") or {}).get("plan") or {}
        try:
            validated_plan = WorldModelForkSetPlan.from_payload(plan_payload)
            budget = len(validated_plan.branch_times_sec)
            if payload.get("state") == "queued":
                payload["fork_set_progress"] = {
                    "state": "queued",
                    "scenarios_completed": 0,
                    "fixed_scenario_budget": budget,
                    "ranking_withheld": True,
                }
                return
            workspace = ProductWorkspace.load(self.root)
            progress_path = (
                workspace.output_root
                / "fork_sets"
                / str(payload.get("task_id"))
                / "progress.json"
            )
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed = int(progress.get("scenarios_completed", 0))
            reported_budget = int(progress.get("fixed_scenario_budget", 0))
            if (
                progress.get("analysis") is not None
                or progress.get("interim_ranking_disclosed") is not False
                or not 0 <= completed <= reported_budget
                or reported_budget != budget
            ):
                raise ValueError("invalid public fork-set progress")
            payload["fork_set_progress"] = {
                "state": str(progress.get("state") or "unknown"),
                "scenarios_completed": completed,
                "fixed_scenario_budget": reported_budget,
                "ranking_withheld": True,
            }
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            json.JSONDecodeError,
        ):
            if payload.get("state") != "queued":
                payload["fork_set_progress"] = {
                    "state": "invalid_progress",
                    "ranking_withheld": True,
                }

    def _task_for_web(self, task: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(task, ensure_ascii=False, default=str))
        self._attach_task_result_urls(payload)
        kind = payload.get("kind")
        if kind == "tactical_study":
            self._attach_study_progress(payload)
        elif kind == "world_model_fork_set":
            self._attach_fork_set_progress(payload)
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
    try:
        ProductRecovery(resolved_root).recover_interrupted_restore()
        queue = ProductTaskQueue(resolved_root)
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
    .world-story { margin:.8rem 0 1rem; padding:1rem; border:1px solid #49695b;
      border-radius:12px; background:#0a1210 }
    .world-story h5 { margin:.1rem 0 .35rem }
    .world-story-rail { display:grid; grid-template-columns:repeat(5,1fr);
      gap:.45rem; padding:0; margin:.8rem 0; list-style:none }
    .world-story-step { padding:.65rem; border:1px solid var(--line);
      border-radius:9px; color:var(--muted); text-align:center; background:#0c1210 }
    .world-story-step[data-status="complete"],
    .world-story-step[data-status="selected"],
    .world-story-step[data-status="changed"],
    .world-story-step[data-status="world_persisted"] { color:var(--accent) }
    .world-story-step[data-status="action_required"],
    .world-story-step[data-status="available"],
    .world-story-step[data-status="influenced_only"] {
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
    .transition-map-panel { margin:1rem 0; padding:1rem; border:1px solid var(--line);
      border-radius:12px; background:#0c1210 }
    .transition-map-panel h6 { margin:.1rem 0 .45rem; font-size:1rem }
    .transition-map-scroll { max-width:100%; overflow-x:auto; border:1px solid var(--line);
      border-radius:10px; background:#090d0c }
    .transition-map-scroll:focus-visible { outline:3px solid var(--warn); outline-offset:3px }
    .transition-map { width:100%; min-width:610px; border-collapse:collapse; font-variant-numeric:tabular-nums }
    .transition-map caption { padding:.7rem; color:var(--muted); text-align:left }
    .transition-map th,.transition-map td { padding:.65rem .75rem; border:1px solid var(--line); text-align:center }
    .transition-map thead th,.transition-map tbody th { color:var(--muted); background:#111916 }
    .transition-map td[data-active="true"] { color:#07100c; background:var(--accent); font-weight:800 }
    .transition-map td[data-diagonal="true"] { color:var(--muted); background:#101513 }
    .transition-cell-button { width:100%; min-height:2.75rem; padding:.35rem; color:inherit;
      background:transparent; border:0; border-radius:7px; font:inherit; font-weight:inherit }
    .transition-cell-button:disabled { cursor:not-allowed; opacity:.58 }
    .transition-cell-button[aria-pressed="true"] { outline:3px solid var(--warn); outline-offset:-3px }
    .transition-detail { margin-top:.75rem; padding:.85rem; border:1px solid #49695b;
      border-radius:10px; background:#0c1511 }
    .transition-detail h6 { margin:.1rem 0 .45rem }
    .transition-detail:focus-visible { outline:3px solid var(--warn); outline-offset:3px }
    .squad-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:.55rem; margin:.75rem 0 }
    .squad-player { display:grid; grid-template-columns:1fr auto; align-items:center; gap:.6rem; padding:.55rem; border:1px solid var(--line); border-radius:9px }
    .squad-player.unavailable { opacity:.65 }
    .squad-player select { min-width:7.5rem }
    .danger-button { background:var(--danger) }
    code { overflow-wrap:anywhere }
    pre { max-height:340px; overflow:auto; padding:1rem; border-radius:10px; background:#070a09;
      color:#cbd5d0; white-space:pre-wrap; overflow-wrap:anywhere }
    [hidden] { display:none!important }
    @media(max-width:520px){ .row,.matchday-journey,.manager-product-journey,
      .world-story-rail { grid-template-columns:1fr } header { padding-top:2rem }
      .workspace-nav { position:static } .workspace-nav button { flex:1 1 45% }
      .workspace-nav .status { flex-basis:100%; text-align:left } }
    @media(prefers-reduced-motion:no-preference){ section { animation:rise .45s ease both }
      @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}} }
    @media(forced-colors:active){ section,.card,input,select,button,.transition-map-panel,
      .transition-map-scroll,.transition-map th,.transition-map td { border:1px solid CanvasText }
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
    <p>在同一对阵、seed、战术、检查点与快速配置下，按预注册的多个比赛时点分别创建 predict-only 与 action-policy 两个未来，观察动作采用何时真正传导为可见差异。</p>
    <form id="fork-form" data-single-fork-endpoint="/api/v1/world-model-forks"><div class="row"><label>主队<input name="home" maxlength="80" value="Brazil" required></label><label>客队<input name="away" maxlength="80" value="Argentina" required></label></div>
      <div class="row"><label>主队固定战术<select name="home_tactic"></select></label><label>客队固定战术<select name="away_tactic"></select></label></div>
      <label>共享 seed<input name="seed" type="number" min="0" max="2147483647" value="42" required></label>
      <label class="check"><input name="fast" type="checkbox" checked>快速模式</label>
      <p id="fork-guidance" class="status">仅研究模式可用。完整固定预算完成前不做排序；最终只报告模拟器内时间敏感性，不推荐最佳时点。</p>
      <button type="submit">生成多时点未来集</button></form></section>
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
      <section id="manager-world-navigator" aria-labelledby="manager-world-navigator-title" hidden>
        <h4 id="manager-world-navigator-title">赛季世界导航</h4>
        <p id="manager-world-navigator-summary" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
        <section id="manager-world-story" class="world-story" aria-labelledby="manager-world-story-title" hidden>
          <h5 id="manager-world-story-title">&#26412;&#36718;&#19990;&#30028;&#27169;&#22411;&#36129;&#29486;&#38142;</h5>
          <p id="manager-world-story-summary" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
          <ol id="manager-world-story-stages" class="world-story-rail" aria-label="&#26412;&#36718;&#19990;&#30028;&#27169;&#22411;&#36129;&#29486;&#38142;"></ol>
          <button id="manager-world-story-open" type="button" class="secondary" hidden>&#25171;&#24320;&#26412;&#36718;&#23436;&#25972;&#19990;&#30028;&#35777;&#25454;</button>
          <p class="status">&#23558;&#21516;&#19968;&#22330;&#24050;&#39564;&#35777;&#35777;&#25454;&#21387;&#32553;&#25104;&#19968;&#26465;&#21487;&#35835;&#38142;&#36335;&#65292;&#19981;&#25226;&#20849;&#29616;&#21319;&#32423;&#20026;&#36187;&#26524;&#22240;&#26524;&#12290;</p>
        </section>
        <div id="manager-world-navigator-current"></div>
        <div class="decision-preview" aria-labelledby="manager-world-influence-title">
          <h5 id="manager-world-influence-title">世界模型进入正式世界</h5>
          <ol id="manager-world-influence-path" class="journal-list" aria-label="世界模型证据传导覆盖"></ol>
          <p class="status">各项是独立证据覆盖，不是递减漏斗；局部动作变化和持久状态同场出现不构成赛果因果。</p>
          <h5 id="manager-world-action-adoption-title">整赛季动作采纳账本</h5>
          <div id="manager-world-action-adoption-metrics" class="cards" role="list" aria-labelledby="manager-world-action-adoption-title"></div>
          <p id="manager-world-action-adoption-summary" class="status"></p>
          <div id="manager-world-action-transition-map" class="transition-map-panel" aria-labelledby="manager-world-action-transition-map-title" hidden>
            <h6 id="manager-world-action-transition-map-title">世界模型动作转移地图</h6>
            <p id="manager-world-action-transition-map-summary" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
            <div id="manager-world-action-transition-map-scroll" class="transition-map-scroll" role="region" aria-labelledby="manager-world-action-transition-map-title" tabindex="0">
              <table id="manager-world-action-transition-table" class="transition-map">
                <caption>局部可归因动作转移次数；行是反事实基准动作，列是正式采用动作。</caption>
                <thead><tr><th scope="col">基准 → 采用</th><th scope="col">持球</th><th scope="col">传球</th><th scope="col">传中</th><th scope="col">射门</th><th scope="col">无动作</th></tr></thead>
                <tbody id="manager-world-action-transition-table-body"></tbody>
              </table>
            </div>
            <div id="manager-world-action-transition-detail" class="transition-detail" role="region" aria-labelledby="manager-world-action-transition-detail-title" aria-live="polite" aria-atomic="true" tabindex="-1" hidden>
              <h6 id="manager-world-action-transition-detail-title">动作转移后的世界证据</h6>
              <p id="manager-world-action-transition-selection" class="status"></p>
              <p id="manager-world-action-transition-world" class="status"></p>
              <p id="manager-world-action-transition-boundary" class="status"></p>
              <button id="manager-world-action-transition-open" type="button" class="secondary" disabled>打开最近相关世界章节</button>
            </div>
            <p class="status">矩阵只呈现共享随机数下的模拟器局部动作改变；零表示在已有 V3 证据中未观察到该转移，不代表真实足球中不存在。它不授权动作排名、跨格效果比较或赛果因果。</p>
          </div>
          <details id="manager-world-action-adoption-diagnostics" hidden>
            <summary>按采纳状态检查世界章节</summary>
            <div id="manager-world-action-adoption-states" class="cards" role="list" aria-label="世界模型动作采纳状态"></div>
          </details>
          <p id="manager-world-action-propagation-boundary" class="status">账本只统计正式比赛中的动作决策及同章后续事实；按采纳状态分层只描述共现，不授权状态间效果比较、自动排序、赛果归因或现实足球因果。</p>
          <details id="manager-world-trajectory" hidden>
            <summary id="manager-world-trajectory-summary">逐轮世界轨迹</summary>
            <div id="manager-world-trajectory-list" class="journal-list" role="list" aria-label="逐轮世界轨迹"></div>
            <p id="manager-world-trajectory-boundary" class="status">轨迹只按时间连接正式世界事实；缺失证据不插值，变化标记不是转折点推断、效果估计或现实足球因果。</p>
            <p id="manager-world-future-continuity-boundary" class="status">赛前分叉证据只说明经理看过什么、是否修改选择以及该选择是否进入正式运行；不会把模拟未来与实际赛果配对为预测准确率或干预效果。</p>
          </details>
          <details id="manager-world-gap-diagnostics" hidden>
            <summary id="manager-world-gap-summary">赛季证据断点诊断</summary>
            <div id="manager-world-gap-list" class="cards" role="list" aria-label="赛季世界证据断点"></div>
          </details>
        </div>
        <button id="manager-world-navigator-action" type="button" hidden>查看当前下一步</button>
        <div id="manager-world-navigator-secondary" class="toolbar" role="group" aria-label="可选的经理世界动作"></div>
        <p id="manager-world-navigator-gaps" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
        <div id="manager-world-navigator-history" class="journal-list" role="list" aria-label="已完成的足球世界章节"></div>
      </section>
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
        <div id="manager-future-set" class="decision-preview" hidden>
          <h4>经理反事实干预工作台</h4>
          <div id="manager-intervention-workspace" aria-live="polite" aria-atomic="true"></div>
          <p id="manager-future-set-summary" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
          <label>分叉分钟（2–4 个，逗号分隔）<input id="manager-future-set-minutes" type="text" inputmode="decimal" pattern="[0-9., ]+" value="30,45,60"></label>
          <button id="request-manager-future-set" type="button">运行赛前未来实验</button>
          <div id="manager-future-set-list" class="cards" role="list" aria-label="当前赛季经理未来实验"></div>
        </div>
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
    <section id="m2-research-control" class="decision-preview" aria-labelledby="m2-research-control-title" hidden>
      <h3 id="m2-research-control-title">M2 结果对齐研究控制面</h3>
      <p id="m2-research-control-summary" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
      <ol id="m2-research-control-stages" class="journal-list" aria-label="M2 五阶段研究流程"></ol>
      <p id="m2-research-control-next" class="status"></p>
      <details id="m2-research-control-commands"><summary>查看严格下一步命令</summary><div id="m2-research-control-command-list"></div></details>
      <p id="m2-research-control-boundary" class="status"></p>
    </section>
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
  <section id="product-value-study-panel" class="wide" data-workspace-area="evidence" aria-labelledby="product-value-study-title">
    <h2 id="product-value-study-title">Product-value study operations</h2>
    <p>Moderator-only registration and scoring-key-free packet delivery. This panel does not run participants, matches, training, or analysis.</p>
    <p id="product-value-study-summary" class="status" role="status" aria-live="polite" aria-atomic="true">Loading blinded study authority...</p>
    <div id="product-value-study-counts" class="cards" role="list" aria-label="Aggregate product-value study registration counts"></div>
    <form id="product-value-study-form" aria-labelledby="product-value-study-register-title">
      <h3 id="product-value-study-register-title">Register a consented pseudonymous participant</h3>
      <div class="row"><label>Participant pseudonym<input name="participant_id" pattern="participant-[a-z0-9]{8,40}" maxlength="52" autocomplete="off" required></label><label>Target role<select name="target_role"><option value="football_analyst">Football analyst</option><option value="research_engineer">Research engineer</option><option value="product_operator">Product operator</option></select></label></div>
      <label>Moderator pseudonym<input name="moderator_id" pattern="moderator-[a-z0-9]{8,40}" maxlength="52" autocomplete="off" required></label>
      <label class="check"><input name="consent_recorded" type="checkbox" required>Recorded informed consent was obtained before this registration</label>
      <div class="toolbar"><button type="submit">Register and prepare blinded packet</button><a id="product-value-study-packet" hidden>Download blinded participant packet</a></div>
    </form>
    <p id="product-value-study-message" class="status" role="status" aria-live="polite" aria-atomic="true"></p>
  </section>
</main>
<noscript><p role="alert" aria-live="assertive" aria-atomic="true">GFS Studio 需要 JavaScript 才能执行受控工作流。</p></noscript>
<script nonce="__NONCE__">
const csrf=document.querySelector('meta[name="gfs-csrf"]').content;
const workspaceNav=document.querySelector('#workspace-nav'),workspaceViewDescription=document.querySelector('#workspace-view-description'),workspaceViewButtons=[...document.querySelectorAll('[data-workspace-view]')],workspaceAreaSections=[...document.querySelectorAll('[data-workspace-area]')],actionPanel=document.querySelector('#action-panel');let currentWorkspaceArea='career',workspaceAreaExplicit=false;
const cards=document.querySelector('#cards'),setup=document.querySelector('#setup-form'),match=document.querySelector('#match-form'),pairPanel=document.querySelector('#pair-panel'),pairForm=document.querySelector('#pair-form'),forkPanel=document.querySelector('#fork-panel'),forkForm=document.querySelector('#fork-form'),studyPanel=document.querySelector('#study-panel'),studyForm=document.querySelector('#study-form'),seasonPanel=document.querySelector('#season-panel'),seasonForm=document.querySelector('#season-form'),seasonSummary=document.querySelector('#season-summary'),matchdayCommand=document.querySelector('#matchday-command-center'),matchdayJourney=document.querySelector('#matchday-journey'),matchdayBriefing=document.querySelector('#matchday-briefing'),matchdayIntelligence=document.querySelector('#matchday-intelligence'),matchdayDebrief=document.querySelector('#matchday-debrief'),matchdayAttribution=document.querySelector('#matchday-attribution'),managerDecisionLedger=document.querySelector('#manager-decision-ledger'),managerDecisionLedgerSummary=document.querySelector('#manager-decision-ledger-summary'),managerDecisionLedgerList=document.querySelector('#manager-decision-ledger-list'),managerDecisionForm=document.querySelector('#manager-decision-form'),managerFixture=document.querySelector('#manager-fixture'),managerTactic=managerDecisionForm.querySelector('[name="tactic"]'),managerRotation=managerDecisionForm.querySelector('[name="rotation"]'),managerManual=document.querySelector('#manager-manual-lineup'),managerSquadStatus=document.querySelector('#manager-squad-status'),managerSquad=document.querySelector('#manager-squad'),managerRules=[...managerDecisionForm.querySelectorAll('.manager-rule')],managerDecisionPreview=document.querySelector('#manager-decision-preview'),managerDecisionSubmit=managerDecisionForm.querySelector('button[type="submit"]'),managerWorldModelAdvice=document.querySelector('#manager-world-model-advice'),managerWorldModelAdviceSummary=document.querySelector('#manager-world-model-advice-summary'),managerWorldModelAdviceCandidates=document.querySelector('#manager-world-model-advice-candidates'),requestManagerAdvice=document.querySelector('#request-manager-advice'),adoptManagerAdvice=document.querySelector('#adopt-manager-advice'),seasonActions=document.querySelector('#season-actions'),playMatchday=document.querySelector('#play-matchday'),seasonStandings=document.querySelector('#season-standings'),seasonFixtures=document.querySelector('#season-fixtures');let activePoll='',restoreTrigger=null,currentSeason=null,managerPreviewTimer=0,managerPreviewSequence=0,managerPreviewBaseRevision=null,currentManagerAdvice=null,managerAdviceIntent=null;
const managerCareerContract=document.querySelector('#manager-career-contract'),managerProfilePanel=document.querySelector('#manager-profile'),seasonHistorySummary=document.querySelector('#season-history-summary'),seasonHistory=document.querySelector('#season-history'),seasonObjective=seasonForm.querySelector('[name="manager_objective"]'),seasonPointsTarget=seasonForm.querySelector('[name="manager_points_target"]'),clubResourceSummary=document.querySelector('#club-resource-summary'),clubResourceInputs=[...seasonForm.querySelectorAll('[name^="resource_"]')];let currentCareer=null;
const managerWorldNavigator=document.querySelector('#manager-world-navigator'),managerWorldNavigatorSummary=document.querySelector('#manager-world-navigator-summary'),managerWorldNavigatorCurrent=document.querySelector('#manager-world-navigator-current'),managerWorldInfluencePath=document.querySelector('#manager-world-influence-path'),managerWorldGapDiagnostics=document.querySelector('#manager-world-gap-diagnostics'),managerWorldGapSummary=document.querySelector('#manager-world-gap-summary'),managerWorldGapList=document.querySelector('#manager-world-gap-list'),managerWorldNavigatorAction=document.querySelector('#manager-world-navigator-action'),managerWorldNavigatorSecondary=document.querySelector('#manager-world-navigator-secondary'),managerWorldNavigatorGaps=document.querySelector('#manager-world-navigator-gaps'),managerWorldNavigatorHistory=document.querySelector('#manager-world-navigator-history');
const managerWorldStory=document.querySelector('#manager-world-story'),managerWorldStorySummary=document.querySelector('#manager-world-story-summary'),managerWorldStoryStages=document.querySelector('#manager-world-story-stages'),managerWorldStoryOpen=document.querySelector('#manager-world-story-open');
const managerWorldActionAdoptionMetrics=document.querySelector('#manager-world-action-adoption-metrics'),managerWorldActionAdoptionSummary=document.querySelector('#manager-world-action-adoption-summary'),managerWorldActionAdoptionDiagnostics=document.querySelector('#manager-world-action-adoption-diagnostics'),managerWorldActionAdoptionStates=document.querySelector('#manager-world-action-adoption-states');
const managerWorldActionTransitionMap=document.querySelector('#manager-world-action-transition-map'),managerWorldActionTransitionMapSummary=document.querySelector('#manager-world-action-transition-map-summary'),managerWorldActionTransitionTable=document.querySelector('#manager-world-action-transition-table'),managerWorldActionTransitionTableBody=document.querySelector('#manager-world-action-transition-table-body'),managerWorldActionTransitionDetail=document.querySelector('#manager-world-action-transition-detail'),managerWorldActionTransitionSelection=document.querySelector('#manager-world-action-transition-selection'),managerWorldActionTransitionWorld=document.querySelector('#manager-world-action-transition-world'),managerWorldActionTransitionBoundary=document.querySelector('#manager-world-action-transition-boundary'),managerWorldActionTransitionOpen=document.querySelector('#manager-world-action-transition-open');
const managerWorldTrajectory=document.querySelector('#manager-world-trajectory'),managerWorldTrajectorySummary=document.querySelector('#manager-world-trajectory-summary'),managerWorldTrajectoryList=document.querySelector('#manager-world-trajectory-list');
const managerTeamInput=seasonForm.querySelector('[name="manager_team"]'),recruitmentFieldset=document.querySelector('#recruitment-fieldset'),recruitmentSummary=document.querySelector('#recruitment-summary'),recruitmentRows=[...seasonForm.querySelectorAll('.recruitment-move')];let currentRecruitmentMarket=null,recruitmentRequestTeam='';
const lifecycleFieldset=document.querySelector('#lifecycle-fieldset'),lifecycleList=document.querySelector('#lifecycle-list'),lifecycleSummary=document.querySelector('#lifecycle-summary');let currentLifecyclePreview=null,lifecycleRequestTeam='';
const freeAgentFieldset=document.querySelector('#free-agent-fieldset'),freeAgentCandidate=document.querySelector('#free-agent-candidate'),freeAgentOutgoing=document.querySelector('#free-agent-outgoing'),freeAgentSummary=document.querySelector('#free-agent-summary'),scoutFreeAgent=document.querySelector('#scout-free-agent');let currentFreeAgentMarket=null,freeAgentRequestTeam='';
const sportingDirectorFieldset=document.querySelector('#sporting-director-fieldset'),sportingPhilosophy=document.querySelector('#sporting-philosophy'),sportingRisk=document.querySelector('#sporting-risk'),sportingPriorityRoles=document.querySelector('#sporting-priority-roles'),sportingRoleDiagnostics=document.querySelector('#sporting-role-diagnostics'),sportingDirectorSummary=document.querySelector('#sporting-director-summary');let currentSportingPlan=null,sportingRequestTeam='';
const clubSituationFieldset=document.querySelector('#club-situation-fieldset'),clubSituationSummary=document.querySelector('#club-situation-summary'),clubSituationChoice=document.querySelector('#club-situation-choice'),clubSituationTradeoff=document.querySelector('#club-situation-tradeoff');let currentClubSituation=null;
const playerPromiseForm=document.querySelector('#player-promise-form'),playerPromiseRows=[...playerPromiseForm.querySelectorAll('.player-promise-row')],playerPromiseSummary=document.querySelector('#player-promise-summary');
const managerFutureSet=document.querySelector('#manager-future-set'),managerInterventionWorkspace=document.querySelector('#manager-intervention-workspace'),managerFutureSetSummary=document.querySelector('#manager-future-set-summary'),managerFutureSetMinutes=document.querySelector('#manager-future-set-minutes'),requestManagerFutureSet=document.querySelector('#request-manager-future-set'),managerFutureSetList=document.querySelector('#manager-future-set-list');
const releaseSummary=document.querySelector('#release-summary'),releaseGates=document.querySelector('#release-gates');
const productValueStudySummary=document.querySelector('#product-value-study-summary'),productValueStudyCounts=document.querySelector('#product-value-study-counts'),productValueStudyForm=document.querySelector('#product-value-study-form'),productValueStudyMessage=document.querySelector('#product-value-study-message'),productValueStudyPacket=document.querySelector('#product-value-study-packet');
const evidenceKitDownload=document.createElement('a'),evidenceKitRow=document.createElement('p');evidenceKitDownload.id='evidence-kit-download';evidenceKitDownload.href='/api/v1/excellence/evidence-kit.zip';evidenceKitDownload.download='gfs-excellence-evidence-kit-v1.zip';evidenceKitDownload.setAttribute('aria-describedby','release-summary');evidenceKitDownload.textContent='Download template-only evidence kit';evidenceKitRow.append(evidenceKitDownload);releaseSummary.insertAdjacentElement('afterend',evidenceKitRow);
const message=document.querySelector('#message'),details=document.querySelector('#details'),workflow=document.querySelector('#workflow'),workflowAction=document.querySelector('#workflow-action'),report=document.querySelector('#report-link'),logoutButton=document.querySelector('#logout-button');
const adoptionSummary=document.querySelector('#action-adoption-summary'),adoptionMetrics=document.querySelector('#action-adoption-metrics'),adoptionEvidence=document.querySelector('#action-adoption-evidence');
const m2ResearchControl=document.querySelector('#m2-research-control'),m2ResearchControlSummary=document.querySelector('#m2-research-control-summary'),m2ResearchControlStages=document.querySelector('#m2-research-control-stages'),m2ResearchControlNext=document.querySelector('#m2-research-control-next'),m2ResearchControlCommands=document.querySelector('#m2-research-control-commands'),m2ResearchControlCommandList=document.querySelector('#m2-research-control-command-list'),m2ResearchControlBoundary=document.querySelector('#m2-research-control-boundary');
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
function renderProductValueStudy(data){productValueStudyCounts.replaceChildren();const measurement=data.measurement_records_present?'measurement records present; analysis required':'no measurement records';productValueStudySummary.textContent=`Frozen blinded delivery ready - ${Number(data.registration_count||0)}/${Number(data.minimum_valid_participants||24)} registrations - ${measurement}`;productValueStudyCounts.append(card('Registrations',Number(data.registration_count||0)),card('AB / BA',`${Number(data.sequence_counts?.AB||0)} / ${Number(data.sequence_counts?.BA||0)}`),card('Registration',data.registration_open?'Open':'Closed after measurement'),card('Scoring material','Not exposed'));for(const item of productValueStudyCounts.children)item.setAttribute('role','listitem')}
async function refreshProductValueStudy(){try{renderProductValueStudy(await api('/api/v1/studies/product-value'))}catch(error){productValueStudyCounts.replaceChildren();productValueStudySummary.textContent=`Blinded study delivery unavailable: ${error.message}`}}
productValueStudyForm.addEventListener('submit',async event=>{event.preventDefault();const button=productValueStudyForm.querySelector('button[type="submit"]'),form=new FormData(productValueStudyForm);button.disabled=true;productValueStudyForm.setAttribute('aria-busy','true');productValueStudyPacket.hidden=true;announce(productValueStudyMessage,'Registering against the frozen authority...');try{const data=await api('/api/v1/studies/product-value/registrations',{method:'POST',body:JSON.stringify({participant_id:String(form.get('participant_id')||''),target_role:String(form.get('target_role')||''),moderator_id:String(form.get('moderator_id')||''),consent_recorded:form.get('consent_recorded')==='on'})});productValueStudyPacket.href=data.packet_url;productValueStudyPacket.download=data.registration.registration_id+'.json';productValueStudyPacket.hidden=false;announce(productValueStudyMessage,`${data.status}: ${data.registration.registration_id}. Download contains no moderator identity or scoring key.`,'success',true);await refreshProductValueStudy()}catch(error){announce(productValueStudyMessage,error.message,'error',true)}finally{button.disabled=false;productValueStudyForm.setAttribute('aria-busy','false')}});
function populateTactics(select,tactics){const selected=select.value;select.replaceChildren();for(const tactic of tactics||[]){const option=document.createElement('option');option.value=tactic.id;option.textContent=tactic.label;option.title=tactic.description||'';select.append(option)}if([...select.options].some(option=>option.value===selected))select.value=selected}
function configureMatchPlan(capabilities,mode){currentMatchCapabilities=capabilities||currentMatchCapabilities||{};currentStudioMode=mode||currentStudioMode;populateTactics(homeTactic,currentMatchCapabilities.tactics);populateTactics(awayTactic,currentMatchCapabilities.tactics);const labOption=[...experienceSelect.options].find(option=>option.value==='tactical_lab'),labAllowed=['research','cognitive'].includes(currentStudioMode);labOption.disabled=!labAllowed;if(!labAllowed&&experienceSelect.value==='tactical_lab')experienceSelect.value='observational';const lab=experienceSelect.value==='tactical_lab';if(lab&&homeTactic.value==='team_identity'&&awayTactic.value==='team_identity')homeTactic.value='balanced';tacticalOptions.hidden=!lab;homeTactic.disabled=!lab;awayTactic.disabled=!lab;reuseSeed.disabled=!lab;if(!lab)reuseSeed.checked=false;planGuidance.textContent=lab?'战术实验使用物理比分。单场结果只作描述；复用上一场随机条件后才能进行配对归因。':'原生观赛保留球队自身体系和稳定比分路径。'}
function configureTacticalStudy(capabilities,mode){const tactics=capabilities?.tactics||[];for(const select of [baselineTactic,treatmentTactic,opponentTactic])populateTactics(select,tactics);if(!baselineTactic.dataset.initialized){baselineTactic.value='balanced';treatmentTactic.value='gegenpress';opponentTactic.value='low_block_counter';baselineTactic.dataset.initialized='true'}studyPanel.hidden=mode!=='research'}
function configurePairedMatch(capabilities,mode,seed){const tactics=capabilities?.tactics||[];for(const select of [pairBaselineTactic,pairTreatmentTactic,pairOpponentTactic])populateTactics(select,tactics);if(!pairBaselineTactic.dataset.initialized){pairBaselineTactic.value='balanced';pairTreatmentTactic.value='gegenpress';pairOpponentTactic.value='low_block_counter';pairBaselineTactic.dataset.initialized='true'}if(!pairForm.dataset.seedInitialized&&Number.isInteger(Number(seed))){pairForm.querySelector('[name="seed"]').value=String(seed);pairForm.dataset.seedInitialized='true'}pairPanel.hidden=!['research','cognitive'].includes(mode);pairGuidance.textContent=mode==='cognitive'?'认知模式含不受共享 seed 完全控制的供应商输出；配对仅作描述，自动撤销战术归因资格。':'研究模式下，资格检查全部通过的单个配对只支持这一固定对阵与 seed 的局部归因，不是总体效应或显著性检验。'}
function configureWorldModelFork(capabilities,mode,seed){const tactics=capabilities?.tactics||[];for(const select of [forkHomeTactic,forkAwayTactic])populateTactics(select,tactics);if(!forkForm.dataset.tacticsInitialized){forkHomeTactic.value='balanced';forkAwayTactic.value='low_block_counter';forkForm.dataset.tacticsInitialized='true'}if(!forkForm.dataset.seedInitialized&&Number.isInteger(Number(seed))){forkForm.querySelector('[name="seed"]').value=String(seed);forkForm.dataset.seedInitialized='true'}forkPanel.hidden=mode!=='research';forkGuidance.textContent=mode==='research'?'固定多个分叉时点；每个时点只改变 MATCH_WM_PLAN=0 → 1。完整预算后仅报告模拟器内时间敏感性，不挑选最佳时点。':'请创建研究模式工作区；稳定模式不加载世界模型，认知模式含无法由共享 seed 完全控制的供应商输出。'}
function renderActionAdoptionBase(studio){adoptionMetrics.replaceChildren();if(!studio){adoptionSummary.textContent='创建工作区后，这里会显示世界模型对动作选择的实际影响。';adoptionEvidence.textContent='运行观测和机制证据将分别呈现。';return}const mechanism=studio.evidence?.action_adoption_mechanism||{},research=studio.mode!=='stable',latest=studio.last_match||{},adoption=latest.world_model_action_adoption||{};if(!research){adoptionSummary.textContent='稳定模式不启用研究型世界模型动作策略，比赛由稳定模拟器决策。';adoptionMetrics.append(card('策略状态','未启用'),card('最近动作影响','不适用'))}else if(!latest.match_id){adoptionSummary.textContent='动作策略已配置并受质量门控；运行一场比赛后可观察实际影响。';adoptionMetrics.append(card('策略状态','已配置·质量门控'),card('决策机会','尚无比赛'),card('实际动作改变','尚无比赛'))}else{const opportunities=Number(adoption.opportunities||0),influenced=Number(adoption.influenced_opportunities||0),eligible=Number(adoption.attribution_eligible_opportunities||0),changed=Number(adoption.counterfactual_action_changes||0),expected=Number(adoption.expected_counterfactual_action_changes||0),shift=Number(adoption.mean_recommended_probability_shift||0),exact=adoption.expected_change_estimator==='shared_uniform_inverse_cdf_overlap_v1';adoptionSummary.textContent=`最近比赛 ${latest.home} vs ${latest.away}：世界模型在 ${influenced}/${opportunities} 个决策机会中产生非零概率影响。`;adoptionMetrics.append(card('非零影响',`${influenced}/${opportunities}`),card('可归因机会',`${eligible}/${opportunities}`),card('实际动作改变',changed),card(exact?'共享采样期望改变':'旧版期望改变',expected.toFixed(3)),card('平均推荐概率偏移',(shift*100).toFixed(3)+' pp'))}if(mechanism.available){const done=mechanism.runs_executed??0,total=mechanism.fixed_run_budget??'—',state=mechanism.execution_state||mechanism.protocol_state||'unknown',promotion=mechanism.promotion_authorized?'已授权推广':'未授权推广';adoptionEvidence.textContent=`机制证据：${state} · ${done}/${total} 次固定预算运行 · ${promotion}。单场运行观测不等于机制证明。`}else{adoptionEvidence.textContent='机制证据协议缺失；当前只能查看运行观测，不能形成机制结论。'}}
function renderActionAdoptionActionSignals(studio){const latest=studio?.last_match||{},adoption=latest.world_model_action_adoption||{},breakdown=adoption.action_signal_breakdown||{},cross=studio?.evidence?.cross_action_validation||{};if(cross.available){const authorized=Boolean(cross.cross_planning_authorized),quality=Number(cross.runtime_cross_quality||0),state=authorized?'\u5df2\u9a8c\u8bc1\u00b7\u6709\u754c\u6743\u9650':'\u672a\u9a8c\u8bc1\u00b7\u96f6\u6743\u9650';adoptionMetrics.append(card('\u4f20\u4e2d\u4e13\u9879\u9a8c\u8bc1',state+' \u00b7 \u8d28\u91cf '+quality.toFixed(3)));adoptionEvidence.textContent+=' \u4f20\u4e2d\u6743\u9650\uff1a'+String(cross.status||'unknown')+'\uff1b\u5f53\u524d\u68c0\u67e5\u70b9\u539f\u56e0 '+String(cross.checkpoint_reason||'unknown')+'\u3002'}if(studio?.mode==='stable'||!latest.match_id)return;const labels={pass:'\u4f20\u7403',shot:'\u5c04\u95e8',cross:'\u4f20\u4e2d',hold:'\u6301\u7403'};for(const action of ['pass','shot','cross','hold']){const row=breakdown[action]||{},count=Number(row.signal_opportunities||0);if(!count)continue;const positive=Number(row.positive_guidance||0),negative=Number(row.negative_guidance||0),shift=Number(row.mean_probability_delta||0),authority=Number(row.mean_applied_authority||0);adoptionMetrics.append(card(labels[action]+'\u4e13\u9879\u4fe1\u53f7',count+' \u6b21 \u00b7 \u63d0\u5347 '+positive+' \u00b7 \u6291\u5236 '+negative+' \u00b7 \u5e73\u5747 '+(shift*100).toFixed(2)+' pp \u00b7 \u6743\u9650 '+(authority*100).toFixed(1)+'%'))}if(Object.keys(breakdown).length)adoptionEvidence.textContent+=' \u52a8\u4f5c\u7ea7\u7edf\u8ba1\u662f\u8fd0\u884c\u89c2\u6d4b\uff0c\u4e0d\u4ee3\u8868\u8d5b\u679c\u6539\u5584\u3002'}
function renderActionAdoptionShotValidation(studio){const shot=studio?.evidence?.shot_action_validation||{};if(!shot.available)return;const authorized=Boolean(shot.frozen_head_authorized),skill=Number(shot.skill_vs_physics_xg_prior||0),state=authorized?'\u5df2\u5c01\u5b58\u664b\u7ea7':'\u7269\u7406 xG \u56de\u9000';adoptionMetrics.append(card('\u5c04\u95e8\u4e13\u9879\u9a8c\u8bc1',state+' \u00b7 '+Number(shot.sealed_samples||0)+' \u6837\u672c \u00b7 skill '+skill.toFixed(4)));adoptionEvidence.textContent+=' \u5c04\u95e8\u6982\u7387\u6e90\uff1a'+String(shot.probability_source||'physics_xg_prior')+'\uff1b\u8054\u5408\u5c04\u95e8\u5934\u4e0d\u5177\u5907\u76f4\u63a5\u52a8\u4f5c\u6743\u9650\u3002'}
function renderActionAdoptionHoldReference(studio){const reference=studio?.last_match?.world_model_action_adoption?.reference_action_breakdown||{};if(!Object.keys(reference).length)return;const redistributed=Number(reference.redistribution_opportunities||0),realized=Number(reference.realized_actions||0),changed=Number(reference.counterfactual_changes||0),gain=Number(reference.mean_probability_gain||0);adoptionMetrics.append(card('\u6301\u7403\u53c2\u8003\u6548\u5e94',redistributed+' \u6b21\u518d\u5206\u914d \u00b7 \u5b9e\u9645\u6301\u7403 '+realized+' \u00b7 \u5c40\u90e8\u6539\u53d8 '+changed+' \u00b7 \u5e73\u5747 +'+(gain*100).toFixed(2)+' pp'));adoptionEvidence.textContent+=' \u6301\u7403\u4ec5\u662f\u53cd\u4e8b\u5b9e\u53c2\u8003\u52a8\u4f5c\uff1b\u5b83\u83b7\u5f97\u7684\u6982\u7387\u662f\u5176\u4ed6\u5df2\u9a8c\u8bc1\u52a8\u4f5c\u88ab\u6291\u5236\u540e\u7684\u95f4\u63a5\u518d\u5206\u914d\uff0c\u4e0d\u662f\u72ec\u7acb\u6301\u7403\u6a21\u578b\u5efa\u8bae\uff0c\u4e0d\u8ba1\u5165\u76f4\u63a5\u91c7\u7528\u7387\u3002'}
function artifactLink(label,url){if(!url)return null;const link=document.createElement('a');link.href=url;link.target='_blank';link.rel='noopener';link.textContent=label;return link}
async function resumeInterruptedTask(taskId,button){button.disabled=true;button.setAttribute('aria-busy','true');announce(message,'正在恢复已持久化的配对事务……');try{const data=await api('/api/v1/tasks/'+encodeURIComponent(taskId)+'/requeue',{method:'POST',body:JSON.stringify({reason:'studio_pair_transaction_resume'})});announce(message,'配对事务已使用原任务身份重新排队。','success');activePoll=data.task.task_id;await pollTask(data.task.task_id)}catch(error){announce(message,error.message,'error',true)}finally{button.disabled=false;button.setAttribute('aria-busy','false')}}
function renderLibraryBase(library){
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
function renderManagerIntelligenceBase(command,configured){matchdayIntelligence.replaceChildren();matchdayAttribution.replaceChildren();if(!configured||!command?.enabled)return;const briefing=command.prematch_intelligence;if(briefing?.available){const title=document.createElement('h4');title.textContent='有证据的赛前决策情报';const coverage=document.createElement('p');coverage.className='status';const coverageLabels={direct_persisted_state:'双方持久化比赛状态 + 当前 roster',mixed_persisted_and_baseline:'一方持久化状态 + 一方默认基线',roster_plus_default_baseline:'当前 roster + 默认赛季基线',partial_team_state:'部分球队状态'};coverage.textContent=`证据覆盖：${coverageLabels[briefing.evidence_coverage]||briefing.evidence_coverage}`;const facts=document.createElement('div');facts.className='cards';facts.setAttribute('role','list');facts.setAttribute('aria-label','球队状态证据');for(const [label,value] of [['本队状态',`疲劳 ${(Number(briefing.manager?.condition?.fatigue||0)*100).toFixed(0)}% · 士气 ${(Number(briefing.manager?.condition?.morale||0)*100).toFixed(0)}% · 缺阵 ${briefing.manager?.condition?.unavailable_players??0}`],['可用阵容',`${briefing.manager?.selectable_players??0} 人 · ${briefing.manager?.formation||'未知'} · 高负荷 ${briefing.manager?.high_load_players??0} 人`],['对手状态',`${briefing.opponent?.formation||'未知阵型'} · 可用 ${briefing.opponent?.selectable_players??0} 人 · 缺阵 ${briefing.opponent?.condition?.unavailable_players??0}`]]){const node=card(label,value);node.setAttribute('role','listitem');facts.append(node)}const signals=document.createElement('div');signals.setAttribute('role','list');signals.setAttribute('aria-label','赛前风险与决策提示');for(const signal of briefing.signals||[]){const item=document.createElement('div');item.className='briefing-signal';item.dataset.severity=signal.severity;item.setAttribute('role','listitem');const heading=document.createElement('strong'),evidence=document.createElement('span'),prompt=document.createElement('span');heading.textContent=signal.title;evidence.className='status';evidence.textContent=`证据：${signal.evidence}`;prompt.textContent=`决策提示：${signal.decision_prompt}`;item.append(heading,evidence,prompt);signals.append(item)}const tradeoff=document.createElement('details'),summary=document.createElement('summary'),list=document.createElement('ul');summary.textContent='查看轮换的固定引擎权衡';for(const [label,key] of [['最强阵容','strongest'],['平衡轮换','balanced'],['大幅轮换','rotate']]){const row=briefing.rotation_tradeoff?.[key]||{},item=document.createElement('li');item.textContent=`${label}：状态代价 ${row.status_penalty??'—'}，轮换等级 ${row.rotation_level??'—'}`;list.append(item)}const caveat=document.createElement('p');caveat.className='status';caveat.textContent='这些是模拟器机械权衡，不是胜率预测或现实建议。';tradeoff.append(summary,list,caveat);matchdayIntelligence.append(title,coverage,facts,signals,tradeoff)}const debrief=command.postmatch_debrief;if(!debrief)return;const title=document.createElement('h4');title.textContent='经理决策执行证据';matchdayAttribution.append(title);if(!debrief.available){const unavailable=document.createElement('p');unavailable.className='status';unavailable.textContent=`直接执行证据不可用：${debrief.reason||'unknown'}`;matchdayAttribution.append(unavailable);return}const effects=debrief.direct_engine_effects||{},execution=debrief.in_match_execution||{},facts=document.createElement('div');facts.className='cards';facts.setAttribute('role','list');facts.setAttribute('aria-label','经理决策直接执行事实');for(const [label,value] of [['冻结方案',`${debrief.decision?.tactic} · ${debrief.decision?.rotation} · ${debrief.decision?.lineup_source}`],['机械状态变化',`${Number(effects.combined_status_delta||0)>=0?'+':''}${Number(effects.combined_status_delta||0).toFixed(2)}（轮换代价 ${effects.rotation_status_penalty??'—'}）`],['疲劳负荷倍率',Number(effects.fatigue_load_multiplier??1).toFixed(2)],['临场指令',`${execution.applied??0} 执行 / ${execution.skipped??0} 跳过 / ${execution.failed??0} 失败`]]){const node=card(label,value);node.setAttribute('role','listitem');facts.append(node)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='机械效果和指令执行是直接运行证据；单场比分仅作描述，不能证明该决策导致赛果。';matchdayAttribution.append(facts,boundary)}
function renderManagerIntelligenceClubSupport(command,configured){const debrief=command?.postmatch_debrief,support=debrief?.club_support;if(!configured||!debrief?.available||!support)return;const allocation=support.plan||{},effects=debrief.direct_engine_effects||{},value=support.available?`恢复 ${allocation.recovery} · 医疗 ${allocation.medical} · 运动科学 ${allocation.sports_science} · 疲劳结算 ×${Number(effects.club_fatigue_load_factor??1).toFixed(2)}`:'旧报告未记录资源证据',node=card('俱乐部长期支持',value);node.setAttribute('role','listitem');matchdayAttribution.append(node)}
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
const configureRecruitmentWindowBase=configureRecruitmentWindow;
function configureRecruitmentWindowLifecycle(season){const complete=season?.state==='complete'&&Boolean(managerTeamInput.value.trim());lifecycleFieldset.hidden=!complete;if(!complete){currentLifecyclePreview=null;lifecycleRequestTeam='';populateLifecyclePreview(null)}else void loadLifecyclePreview(managerTeamInput.value)}
function configureRecruitmentWindowGlobalMarket(season){const complete=season?.state==='complete'&&Boolean(managerTeamInput.value.trim());freeAgentFieldset.hidden=!complete;if(!complete){currentFreeAgentMarket=null;freeAgentRequestTeam='';populateFreeAgentMarket(null)}else void loadFreeAgentMarket(managerTeamInput.value)}
function configureRecruitmentWindowSportingPlan(season){const complete=season?.state==='complete'&&Boolean(managerTeamInput.value.trim());sportingDirectorFieldset.hidden=!complete;if(!complete){currentSportingPlan=null;sportingRequestTeam='';populateSportingPlan(null)}else void loadSportingPlan(managerTeamInput.value)}
const RECRUITMENT_WINDOW_CONFIGURE_STAGES=Object.freeze([configureRecruitmentWindowBase,configureRecruitmentWindowLifecycle,configureRecruitmentWindowGlobalMarket,configureRecruitmentWindowSportingPlan]);
configureRecruitmentWindow=season=>{for(const configureStage of RECRUITMENT_WINDOW_CONFIGURE_STAGES)configureStage(season)};
const renderManagerCareerWithoutRecruitment=renderManagerCareer;
renderManagerCareer=(season,history,historySummary,configured)=>{renderManagerCareerWithoutRecruitment(season,history,historySummary,configured);configureRecruitmentWindow(season);for(const archived of history||[]){const transaction=archived?.recruitment_transaction;if(!transaction)continue;const names=(transaction.moves||[]).map(move=>`${move.incoming?.name||move.candidate_id} ← ${move.outgoing?.name||move.outgoing_player_id}`).join(' · '),node=card(`${archived.season_id} · 阵容建设`,`${names} · 使用 ${transaction.spent}/${transaction.budget} 点`);node.setAttribute('role','listitem');seasonHistory.append(node)}};
const renderSeasonBase=renderSeason;
function renderSeasonCommandCenter(season,configured,history=[],historySummary={}){renderMatchdayCommand(season,configured);renderManagerCareer(season,history,historySummary,configured)}
function render(data){createBackupButton.disabled=!data.configured;cards.replaceChildren();const s=data.studio,tasks=data.tasks||[],latest=tasks[0];if(!data.configured){cards.append(card('工作区','未配置'),card('API 调用','0'),card('任务',tasks.length));setup.hidden=false;match.hidden=true;pairPanel.hidden=true;studyPanel.hidden=true;workflow.textContent='创建工作区后，系统会先执行证据就绪检查。'}else{const r=s.readiness||{},w=s.workflow||{};cards.append(card('模式',s.mode),card('就绪',r.ready?'是':'否'),card('已完成比赛',s.matches_played),card('最近任务',latest?.state||'无'));setup.hidden=true;match.hidden=false;pairPanel.hidden=!['research','cognitive'].includes(s.mode);studyPanel.hidden=s.mode!=='research';workflow.textContent=r.ready?'证据门禁通过，可以提交后台比赛、赛季、配对对决或固定预算研究任务。':'阻塞项：'+(r.blockers||[]).join(', ')}renderSeason(s?.season,data.configured,s?.season_history||[],s?.season_history_summary||{});renderCareerContract(s?.manager_career);renderActionAdoption(data.configured?s:null);renderLibrary(data.evidence_library);const isCompleted=latest?.state==='completed',dashboardUrl=isCompleted?(latest.result?.treatment_url||latest.result?.dashboard_url):s?.last_match?.dashboard_url,comparisonUrl=isCompleted?latest.result?.comparison_url:s?.last_match?.comparison_url,studyUrl=isCompleted?latest.result?.study_url:null,baselineUrl=isCompleted?latest.result?.baseline_url:null;showReport(dashboardUrl,comparisonUrl,studyUrl,baselineUrl);if(['queued','running'].includes(latest?.state)&&activePoll!==latest.task_id){activePoll=latest.task_id;void pollTask(latest.task_id)}details.textContent=JSON.stringify(data,null,2)}
function renderGlobalPlayerMarket(market){if(!market?.available)return;seasonHistory.hidden=false;const transition=market.latest_transition||{},title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div');title.textContent='全局自由球员流动';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','全局自由球员市场摘要');for(const [label,value] of [['当前球员池',`${market.pool_size||0} 人`],['本窗签约',`${transition.summary?.signings||0} 人`],['新入市场',`${transition.summary?.entries||0} 人`],['池内退休',`${transition.summary?.pool_retirements||0} 人`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看跨俱乐部签约与市场来源';list.setAttribute('role','list');for(const signing of transition.signings||[]){const node=card(`${signing.incoming?.name||signing.free_agent_id} → ${signing.team}`,`${signing.incoming?.role||''} · 原俱乐部 ${signing.origin_team||'未知'} · 替换 ${signing.outgoing?.name||signing.outgoing?.player_id} · ${signing.control==='manager'?'经理签约':'AI 签约'}`);node.setAttribute('role','listitem');list.append(node)}for(const entry of transition.entries||[]){const node=card(`${entry.player?.name||entry.player_id} · 进入市场`,`${entry.entry_reason==='contract_released'?'合同到期离队':'阵容替换离队'} · 原 ${entry.origin_team} · ${entry.available_from_season_id} 起可签`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);const boundary=document.createElement('p');boundary.className='status';boundary.textContent='该市场只公开身份、位置、年龄与流动来源；候选能力真值留在权威模拟账本中，经理与 AI 均依据有限球探观察决策。';seasonHistory.append(title,summary,detailsNode,boundary)}
const renderWithoutClubFinance=render;
function renderScouting(scouting){if(!scouting?.report_count)return;seasonHistory.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),boundary=document.createElement('p');title.textContent='球探信息账本';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','球探信息预算摘要');for(const [label,value] of [['持久化报告',`${scouting.report_count} 份`],['覆盖窗口',`${scouting.window_count} 个`],['每队每窗口预算',`${scouting.reports_per_window} 次`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看最近的区间报告';list.setAttribute('role','list');for(const report of scouting.recent_reports||[]){const observation=report.observation||{},node=card(`${report.team} · ${report.player_id}`,`${report.target_season_id} · 估计 ${Number(observation.estimated_quality).toFixed(3)} · 区间 ${Number(observation.quality_low).toFixed(3)}–${Number(observation.quality_high).toFixed(3)} · 宽度 ${Number(observation.interval_width).toFixed(3)}`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);boundary.className='status';boundary.textContent='候选球员真值始终隐藏；区间按构造覆盖模拟真值，球探只收窄信息范围，不承诺现实校准或竞技预测。';seasonHistory.append(title,summary,detailsNode,boundary)}
function renderScoutingOutcomes(outcomes){if(!outcomes?.outcome_count)return;seasonHistory.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),boundary=document.createElement('p'),coverage=outcomes.coverage_counts||{};title.textContent='签约结果与球探复盘';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','签约结果账本摘要');for(const [label,value] of [['已结算签约',`${outcomes.outcome_count} 笔`],['经理签约',`${outcomes.manager_outcome_count} 笔`],['AI 签约',`${outcomes.ai_outcome_count} 笔`],['完整出场证据',`${coverage.complete||0} 笔`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看我的已结算签约';list.setAttribute('role','list');const usageLabels={core:'核心',rotation:'轮换',fringe:'边缘',unused:'未使用',evidence_partial:'证据不完整',evidence_unavailable:'证据不足'};for(const outcome of outcomes.manager_outcomes||[]){const observation=outcome.observation||{},realized=outcome.realized||{},node=card(`${outcome.player_name||outcome.player_id} · ${outcome.team}`,`${outcome.season_id} · ${outcome.observation_source==='baseline_observation'?'未追加侦察':'已侦察'} · 估计 ${Number(observation.estimated_quality).toFixed(3)} → 到队已知 ${Number(realized.actual_quality_at_signing).toFixed(3)} · 误差 ${Number(realized.absolute_estimation_error).toFixed(3)} · ${Number(realized.minutes).toFixed(0)} 分钟/${realized.appearances||0} 场 · ${usageLabels[realized.usage_band]||realized.usage_band} · 工资档 ${realized.wage_tier}`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);boundary.className='status';boundary.textContent='使用率、积分和排名只描述模拟赛季中的后续事实，不证明签约造成球队成绩；AI 球员真值仅计入汇总，不向经理公开。';seasonHistory.append(title,summary,detailsNode,boundary)}
function renderSportingDirection(season){const brief=season?.sporting_brief,evaluation=season?.sporting_evaluation,directive=evaluation?.directive;if(!brief||!directive)return;managerProfilePanel.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),boundary=document.createElement('p'),philosophyLabels={balanced:'均衡建设',win_now:'即战竞争',youth_pathway:'青年通道',financial_control:'财政控制'},riskLabels={low:'低风险',balanced:'平衡风险',high:'高风险'};title.textContent='冻结体育总监计划';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','本赛季体育计划摘要');for(const [label,value] of [['建队理念',philosophyLabels[directive.philosophy]||directive.philosophy],['风险边界',riskLabels[directive.risk_level]||directive.risk_level],['优先位置',(directive.priority_roles||[]).join(' / ')||'不限制'],['窗口执行',`${evaluation.selected_recruitment_moves||0} 笔招募 · ${evaluation.selected_free_agent?'1 笔自由签约':'无自由签约'} · ${evaluation.selected_renewals||0} 人续约`]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='查看冻结位置诊断';list.setAttribute('role','list');for(const row of (brief.role_diagnostics||[]).filter(row=>row.need_score>0)){const node=card(`${row.role} · 需求 ${Number(row.need_score).toFixed(2)}`,`深度 ${row.depth} · 平均质量 ${row.mean_quality===null?'无':Number(row.mean_quality).toFixed(3)} · 到期 ${row.expiring} · 退休 ${row.retiring} · 窗口可选 ${Number(row.recruitment_options||0)+Number(row.free_agent_options||0)}`);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);boundary.className='status';boundary.textContent='该计划是赛季开始前冻结的游戏约束；诊断与选择不构成现实体育建议，也不预测赛季成绩。';managerProfilePanel.append(title,summary,detailsNode,boundary)}
function renderSportingReview(reviews){const review=reviews?.latest_review;if(!review)return;seasonHistory.hidden=false;const title=document.createElement('h3'),summary=document.createElement('div'),detailsNode=document.createElement('details'),detailsTitle=document.createElement('summary'),list=document.createElement('div'),boundary=document.createElement('p'),result=review.season_result||{},execution=review.execution||{},directive=review.directive||{},coverageLabels={complete:'\u5b8c\u6574',partial:'\u90e8\u5206',unavailable:'\u4e0d\u53ef\u7528'};title.textContent='\u8de8\u8d5b\u5b63\u4f53\u80b2\u6218\u7565\u590d\u76d8';summary.className='cards';summary.setAttribute('role','list');summary.setAttribute('aria-label','\u4f53\u80b2\u6218\u7565\u590d\u76d8\u6458\u8981');for(const [label,value] of [['\u8d5b\u5b63',review.season_id],['\u51bb\u7ed3\u7406\u5ff5',directive.philosophy],['\u7a97\u53e3\u6267\u884c',`${execution.selected_recruitment_moves||0} \u7b14\u5f15\u63f4 \u00b7 ${execution.selected_free_agent?'1 \u7b14\u81ea\u7531\u7b7e\u7ea6':'\u65e0\u81ea\u7531\u7b7e\u7ea6'} \u00b7 ${execution.selected_renewals||0} \u4eba\u7eed\u7ea6`],['\u8d5b\u5b63\u76ee\u6807',result.objective_status],['\u6392\u540d / \u79ef\u5206',`${result.position} / ${result.points}`],['\u8d22\u653f\u53d8\u5316',result.finance_delta],['\u51fa\u573a\u8bc1\u636e',coverageLabels[review.evidence_coverage]||review.evidence_coverage]]){const node=card(label,value);node.setAttribute('role','listitem');summary.append(node)}detailsTitle.textContent='\u67e5\u770b\u7403\u5458\u4f7f\u7528\u3001\u4f18\u5148\u4f4d\u7f6e\u4e0e\u540e\u7eed\u63d0\u793a';list.setAttribute('role','list');for(const player of review.incoming_usage||[]){const node=card(`${player.name} \u00b7 ${player.role}`,`${player.source} \u00b7 ${Number(player.minutes).toFixed(0)} \u5206\u949f / ${player.appearances} \u573a \u00b7 \u4f7f\u7528\u7387 ${(Number(player.utilization_share)*100).toFixed(1)}% \u00b7 ${player.usage_band}`);node.setAttribute('role','listitem');list.append(node)}for(const row of review.priority_role_delivery||[]){const node=card(`\u4f18\u5148\u4f4d\u7f6e ${row.role}`,`\u539f\u6df1\u5ea6 ${row.source_depth} \u00b7 \u9884\u8ba1\u79bb\u961f ${row.source_pending_exits} \u00b7 \u65b0\u589e ${row.incoming_count} \u00b7 \u8d5b\u540e\u6df1\u5ea6 ${row.post_depth}`);node.setAttribute('role','listitem');list.append(node)}for(const signal of review.learning_signals||[]){const node=card(`\u590d\u76d8\u63d0\u793a \u00b7 ${signal.role}`,signal.interpretation);node.setAttribute('role','listitem');list.append(node)}detailsNode.append(detailsTitle,list);boundary.className='status';boundary.textContent='\u590d\u76d8\u53ea\u5c06\u51bb\u7ed3\u8ba1\u5212\u4e0e\u540e\u7eed\u6a21\u62df\u4e8b\u5b9e\u5bf9\u9f50\uff1b\u5b83\u4e0d\u58f0\u79f0\u7b7e\u7ea6\u3001\u7eed\u7ea6\u6216\u5efa\u961f\u7406\u5ff5\u5bfc\u81f4\u4e86\u6210\u7ee9\u3002';seasonHistory.append(title,summary,detailsNode,boundary)}
const populateSportingPlanWithoutContinuity=populateSportingPlan;populateSportingPlan=plan=>{populateSportingPlanWithoutContinuity(plan);if(!plan?.previous_strategy_review)return;for(const option of sportingPriorityRoles.options){const row=(plan.role_diagnostics||[]).find(item=>item.role===option.value);if(row?.continuity_signal)option.textContent+=' \u00b7 \u4e0a\u5b63\u590d\u76d8\u5ef6\u7eed';}const feedback=plan.previous_strategy_review,node=card(`\u4e0a\u5b63\u6218\u7565\u590d\u76d8 \u00b7 ${feedback.season_id}`,`\u672a\u89e3\u51b3\u4f18\u5148\u4f4d\u7f6e ${(feedback.priority_unaddressed_roles||[]).join('/')||'\u65e0'} \u00b7 \u4f4e\u4f7f\u7528\u4f4d\u7f6e ${(feedback.incoming_low_usage_roles||[]).join('/')||'\u65e0'} \u00b7 \u76ee\u6807 ${feedback.objective_status} \u00b7 \u8d22\u653f ${feedback.finance_delta>=0?'+':''}${feedback.finance_delta}`);node.setAttribute('role','listitem');sportingRoleDiagnostics.prepend(node)};
function syncClubSituation(){if(!currentClubSituation)return;const selected=(currentClubSituation.choices||[]).find(row=>row.choice_id===clubSituationChoice.value),constraint=selected?.constraint||{};clubSituationTradeoff.textContent=selected?.tradeoff||'';if(constraint.rotation_in&&!constraint.rotation_in.includes(managerRotation.value))managerRotation.value=constraint.rotation_in[0];if(constraint.tactic_equals)managerTactic.value=constraint.tactic_equals;if(constraint.tactic_not_equals&&managerTactic.value===constraint.tactic_not_equals){const alternative=[...managerTactic.options].find(option=>option.value!==constraint.tactic_not_equals);if(alternative)managerTactic.value=alternative.value}renderManagerSquad(currentSeason,currentSeason?.next_manager_fixture)}
const managerWorldChapterHashPrefix='#world-chapter=';
function managerWorldChapterFromLocation(){const hash=String(location.hash||'');if(!hash.startsWith(managerWorldChapterHashPrefix))return null;try{const parts=hash.slice(managerWorldChapterHashPrefix.length).split('/');if(parts.length!==2)return null;const seasonId=decodeURIComponent(parts[0]),fixtureId=decodeURIComponent(parts[1]);return seasonId&&fixtureId&&seasonId.length<=160&&fixtureId.length<=160?{seasonId,fixtureId}:null}catch{return null}}
function visibleManagerLedgerEntries(season){const entries=season?.manager_decision_ledger?.entries||[],current=entries.filter(row=>row.lifecycle_state==='frozen_awaiting_execution').slice(-1),completed=entries.filter(row=>row.lifecycle_state!=='frozen_awaiting_execution').slice(-6).reverse(),selectedRef=managerWorldChapterFromLocation(),selected=selectedRef?.seasonId===season?.season_id?entries.find(row=>row.fixture_id===selectedRef.fixtureId&&row.lifecycle_state!=='frozen_awaiting_execution'):null;if(selected&&!completed.some(row=>row.fixture_id===selected.fixture_id))completed.push(selected);return [...current,...completed]}
function renderManagerDecisionLedgerBase(season){managerDecisionLedgerList.replaceChildren();const ledger=season?.manager_decision_ledger,entries=ledger?.entries||[];managerDecisionLedger.hidden=!ledger?.available||!entries.length;if(managerDecisionLedger.hidden){managerDecisionLedgerSummary.textContent='';return}const summary=ledger.summary||{};managerDecisionLedgerSummary.textContent=`${summary.decisions||0} 次冻结决策 · ${summary.executed_with_direct_evidence||0} 场有直接执行证据 · 执行覆盖 ${Math.round((summary.direct_execution_coverage||0)*100)}% · 账本 ${String(ledger.ledger_identity||'').slice(0,12)}`;const states={frozen_awaiting_execution:'已冻结，等待比赛',executed_with_direct_evidence:'已执行，有直接证据',executed_evidence_unavailable:'已完成，直接证据不可用'},statuses={on_track:'按计划',at_risk:'有风险',pending:'待证据',fulfilled:'已兑现',missed:'已违约',excused:'已豁免',evidence_unavailable:'证据不可用'},signed=value=>`${Number(value||0)>=0?'+':''}${Number(value||0).toFixed(2)}`;for(const row of visibleManagerLedgerEntries(season)){const node=document.createElement('article');node.className='card';node.setAttribute('role','listitem');node.dataset.fixtureId=String(row.fixture_id||'');node.dataset.chapterComplete=String(row.lifecycle_state!=='frozen_awaiting_execution');node.tabIndex=-1;const heading=document.createElement('strong');heading.textContent=`第 ${row.matchday} 轮 · ${row.venue==='home'?'主场':'客场'} vs ${row.opponent}`;const lifecycle=document.createElement('p');lifecycle.className='status';lifecycle.textContent=`${states[row.lifecycle_state]||row.lifecycle_state} · ${row.decision?.tactic} / ${row.decision?.rotation} · 决策 ${String(row.decision_identity||'').slice(0,10)}`;const execution=document.createElement('p'),direct=row.execution||{},effects=direct.direct_engine_effects||{},instructions=direct.in_match_execution||{};execution.textContent=direct.available?`直接执行：机械状态 ${Number(effects.combined_status_delta||0)>=0?'+':''}${Number(effects.combined_status_delta||0).toFixed(2)} · 临场 ${instructions.applied||0} 执行 / ${instructions.skipped||0} 跳过 / ${instructions.failed||0} 失败`:`直接执行：${direct.reason||'证据不可用'}`;const advisor=document.createElement('p'),support=row.world_model_decision_support||{};if(support.available){const adoption=support.adoption,interaction=adoption?(adoption.intent==='adopt_recommendation'?'明确采用':'查看后改选'):'未绑定采用意图';advisor.textContent=`模型顾问：建议 ${support.recommended_tactic} · 选择 ${support.selected_tactic} · ${support.aligned_with_recommendation?'一致':'不一致'} · ${interaction} · 置信 ${Math.round(Number(support.recommendation_confidence||0)*100)}%`}else{advisor.textContent='模型顾问：本场未请求或证据不可用'}const result=document.createElement('p');result.textContent=row.observed_result?`观察赛果：${row.observed_result.score.home}–${row.observed_result.score.away} · ${row.observed_result.outcome} · 仅作描述`:'观察赛果：尚未产生';const accounting=document.createElement('p'),policy=row.long_term_accounting?.season_commitments?.after?.entries||[],players=row.long_term_accounting?.player_role_promises?.after?.entries||[],evidence=row.long_term_accounting?.evidence_state==='hypothetical_if_counted'?'若本场计入':'已计入完成证据',policyText=policy.map(item=>`${item.id==='tactical_identity'?'战术身份':'阵容管理'} ${statuses[item.status]||item.status}`).join(' / ')||'无赛季承诺',playerText=players.map(item=>`${item.name} ${statuses[item.status]||item.status}`).join(' / ')||'无具名承诺';accounting.textContent=`长期记账（${evidence}）：${policyText} · ${playerText}`;const persistent=document.createElement('p'),world=row.long_term_accounting?.persistent_team_state_delta;if(world?.available){const match=world.match_delta||{},matchMetrics=match.metrics_delta||{},matchSummary=match.summary||{},recovery=world.recovery_delta||{},recoveryMetrics=recovery.metrics_delta||{},recoverySummary=recovery.summary||{};persistent.textContent=`世界状态：比赛后疲劳 ${signed(matchMetrics.team_fatigue_ema)} · 士气 ${signed(matchMetrics.squad_morale_ema)} · 新伤 ${matchSummary.new_injuries||0} · 新停赛 ${matchSummary.new_suspensions||0} · 球员变化 ${matchSummary.changed_players||0}${world.recovery_complete?`；恢复后疲劳 ${signed(recoveryMetrics.team_fatigue_ema)} · 伤愈 ${recoverySummary.injuries_cleared||0} · 解禁 ${recoverySummary.suspensions_cleared||0}`:'；比赛日恢复尚未结算'}`}else{persistent.textContent=`世界状态：${world?.reason||'快照证据不可用'}`};const boundary=document.createElement('p');boundary.className='status';boundary.textContent='模型建议、经理选择、直接执行、世界状态、赛果和长期记账分别取证；伤停为模拟状态，单场赛果不证明决策效果。';node.append(heading,lifecycle,advisor,execution,result,persistent,accounting,boundary);const link=artifactLink('打开该场完整复盘',row.dashboard);if(link)node.append(link);managerDecisionLedgerList.append(node)}}
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
function renderSeasonClubTimeline(season,configured,history=[],historySummary={}){renderClubSituation(season);for(const item of matchdayJourney.children){if(item.textContent.startsWith('club_situation'))item.textContent=item.textContent.replace('club_situation','\u4ff1\u4e50\u90e8\u60c5\u5883')}renderCompletedClubTimeline(season);renderSeasonCommitments(season);renderPlayerPromises(season,history);renderManagerDecisionLedger(season);const managed=season?.next_manager_fixture;if(!currentManagerAdvice||currentManagerAdvice.fixture_id!==managed?.fixture_id)managerAdviceIntent=null;renderManagerWorldModelAdvice(managed?.manager_decision_advice,season,managed);if(!managerDecisionForm.hidden)scheduleManagerDecisionPreview()}
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
async function pollTask(id){try{while(true){const data=await api('/api/v1/tasks/'+encodeURIComponent(id)),task=data.task,isStudy=task.kind==='tactical_study',isPair=task.kind==='paired_match',isFork=task.kind==='world_model_fork',isForkSet=task.kind==='world_model_fork_set',isSeason=task.kind==='season_matchday',progress=task.study_progress||task.fork_set_progress;details.textContent=JSON.stringify(data,null,2);if(isForkSet&&progress?.scenarios_completed!==undefined){announce(message,`未来集 ${progress.scenarios_completed}/${progress.fixed_scenario_budget} 个场景已完成；不披露中期排名。`)}else if(isStudy&&progress?.pairs_completed!==undefined){announce(message,`战术研究 ${progress.pairs_completed}/${progress.fixed_pair_budget} 对已完成；中期效应保持隐藏。`)}else{announce(message,task.state==='queued'?(isStudy?'战术研究已排队，固定预算尚未开始……':isForkSet?'多时点未来集已排队，固定场景预算尚未开始……':isFork?'因果分叉已排队，将依次生成预测-only与动作策略世界……':isPair?'配对对决已排队，将连续运行基线与处理场……':isSeason?'赛季比赛日已持久化排队……':'比赛任务已排队……'):(isStudy?'战术研究正在后台运行；预算完成前不展示效应……':isForkSet?'正在生成固定多时点未来集；完成前不做排名……':isFork?'正在用共享 seed 生成两个世界并核验唯一策略干预……':isPair?'正在运行共享 seed 的基线与处理场……':isSeason?'正在推进赛季比赛日并结算长期状态……':'比赛正在后台运行……'))}if(task.state==='completed'){announce(message,isStudy?'固定预算研究完成，最终分析现已开放。':isForkSet?'多时点未来集完成，时间敏感性报告已开放。':isFork?'世界模型因果分叉完成，双世界回放与差异链已开放。':isPair?'配对对决完成，三层复盘已开放。':isSeason?'赛季比赛日完成，积分与球队状态已结算。':'比赛完成。','success',true);showReport(task.result?.treatment_url||task.result?.dashboard_url,task.result?.comparison_url,task.result?.study_url||task.result?.fork_set_url,task.result?.baseline_url);await refresh();return}if(['failed','interrupted'].includes(task.state)){announce(message,task.error?.message||'任务中断，可检查状态后重新提交。','error',true);await refresh();return}await new Promise(resolve=>setTimeout(resolve,750))}}catch(e){announce(message,e.message,'error',true)}finally{activePoll=''}}
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
function renderActionAdoptionFormalEvidence(studio){if(!studio)return;const evidence=studio.evidence||{},mechanism=evidence.action_adoption_mechanism||{},outcome=evidence.action_outcome_study||{},parts=[];if(mechanism.available){const label=mechanism.result_status==='mechanism_confirmed'?'动作采纳机制已确认':mechanism.execution_state||mechanism.protocol_state||'unknown',formal=mechanism.mechanism||{},changed=Number(formal.realized_counterfactual_action_changes||0),opportunities=Number(formal.action_opportunities||0);parts.push(`机制：${label} · ${mechanism.runs_executed??0}/${mechanism.fixed_run_budget??'—'} 次运行 · ${changed}/${opportunities} 次实际反事实动作变化`)}if(outcome.available){const state=outcome.result_status||outcome.execution_state||'not_started',promotion=outcome.promotion_supported?'结果门支持晋级':'结果门未支持晋级',primary=outcome.primary||{},behavior=outcome.behavior||{},delta=Number(primary.point_delta),low=Number(primary.ci95_low),high=Number(primary.ci95_high),effect=Number.isFinite(delta)&&Number.isFinite(low)&&Number.isFinite(high)?` · 损失差 ${delta.toFixed(3)} · 95% CI [${low.toFixed(3)}, ${high.toFixed(3)}]`:'',changed=Number.isFinite(Number(behavior.changed_pairs))?` · 行为变化 ${behavior.changed_pairs}/${outcome.pairs_total??'—'} 对`:'';parts.push(`全长度结果：${state} · ${outcome.runs_executed??0}/${outcome.fixed_run_budget??'—'} 次运行${effect}${changed} · ${promotion}`)}if(parts.length)adoptionEvidence.textContent=parts.join('；')+'。世界模型已改变完整比赛行为，但本次结果不确定，未证明赛果改善或真实足球因果效应。'}
const renderManagerDecisionPreviewWithoutWorldModelComparison=renderManagerDecisionPreview;
renderManagerDecisionPreview=preview=>{renderManagerDecisionPreviewWithoutWorldModelComparison(preview);adoptManagerAdvice.textContent='\u91c7\u7528\u6a21\u578b\u5efa\u8bae';const advice=preview?.world_model_advice,comparison=advice?.comparison;if(!comparison)return;const authority=comparison.authority||{},exploratory=authority.level==='exploratory_only',heading=document.createElement('h4');heading.textContent='\u4e16\u754c\u6a21\u578b\u5efa\u8bae vs \u5f53\u524d\u9009\u62e9';if(exploratory)adoptManagerAdvice.textContent='\u5ba1\u9605\u540e\u91c7\u7528\u63a2\u7d22\u5efa\u8bae';const signed=(value,digits=3)=>`${Number(value||0)>=0?'+':''}${Number(value||0).toFixed(digits)}`,deltas=comparison.recommended_minus_selected||{},events=deltas.event_probabilities||{},facts=document.createElement('div');facts.className='cards';facts.setAttribute('role','list');facts.setAttribute('aria-label','\u4e16\u754c\u6a21\u578b\u5efa\u8bae\u4e0e\u5f53\u524d\u6218\u672f\u7684\u77ed\u89c6\u91ce\u4ee3\u7406\u5dee\u5f02');const reasonLabels={low_model_confidence:'\u6a21\u578b\u7f6e\u4fe1\u5ea6\u4f4e',low_historical_trust:'\u5386\u53f2\u4fe1\u4efb\u4f4e',narrow_top_two_margin:'\u524d\u4e24\u540d\u5dee\u8ddd\u5c0f',insufficient_history:'\u5386\u53f2\u6837\u672c\u4e0d\u8db3',low_authority_guidance:'\u4f4e\u6743\u5a01\u5f15\u5bfc'},reasons=(authority.reasons||[]).map(reason=>reasonLabels[reason]||reason).join(' / ')||'\u5df2\u8fbe\u6709\u754c\u5ba1\u9605\u663e\u793a\u95e8\u69db';for(const [label,value] of [['\u663e\u793a\u6743\u5a01',`${exploratory?'\u4ec5\u63a2\u7d22':'\u6709\u754c\u5ba1\u9605'} \u00b7 ${reasons}`],['\u6218\u672f\u6392\u540d',`\u5efa\u8bae ${comparison.recommended_tactic} #${comparison.recommended_rank} / \u5f53\u524d ${comparison.selected_tactic} #${comparison.selected_rank}`],['\u98ce\u9669\u8c03\u6574\u503c\u5dee',signed(deltas.risk_adjusted_value)],['\u7f6e\u4fe1 / \u4e0d\u786e\u5b9a\u6027',`${signed(deltas.effective_confidence)} / ${signed(deltas.uncertainty)}`],['\u75b2\u52b3 / \u7ed3\u6784\u98ce\u9669',`${signed(deltas.fatigue_cost_proxy)} / ${signed(deltas.structural_risk_proxy)}`],['\u4e8b\u4ef6\u5206\u5e03\u5dee',`\u4fdd\u6301 ${signed(Number(events.retain||0)*100,1)}pp / \u4e22\u5931 ${signed(Number(events.turnover||0)*100,1)}pp / \u5c04\u95e8 ${signed(Number(events.shot||0)*100,1)}pp`]]){const node=card(label,value);node.setAttribute('role','listitem');facts.append(node)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent=`${advice.current?'\u5f53\u524d\u5efa\u8bae\u8eab\u4efd\u6709\u6548':'\u5efa\u8bae\u5df2\u8fc7\u671f\uff0c\u4e0d\u53ef\u7ed1\u5b9a\u91c7\u7528'}\u3002\u6b63\u503c\u8868\u793a\u63a8\u8350\u6218\u672f\u51cf\u53bb\u5f53\u524d\u6218\u672f\uff1b\u8fd9\u4e9b\u662f\u77ed\u89c6\u91ce\u6a21\u62df\u5668\u4ee3\u7406\u5dee\u5f02\uff0c\u4e0d\u662f\u6bd4\u5206\u3001\u80dc\u7387\u6216\u56e0\u679c\u4f30\u8ba1\u3002`;managerDecisionPreview.append(heading,facts,boundary)};
const forkBranchLabel=document.createElement('label'),forkBranchInput=document.createElement('input');forkBranchLabel.textContent='预注册分叉分钟（2–4 个，逗号分隔）';forkBranchInput.name='branch_minutes';forkBranchInput.type='text';forkBranchInput.inputMode='decimal';forkBranchInput.pattern='[0-9., ]+';forkBranchInput.value='30,45,60';forkBranchInput.required=true;forkBranchLabel.append(forkBranchInput);forkForm.querySelector('.check').before(forkBranchLabel);
function renderLibraryBranchIdentity(library){const source=library||{matches:[],forks:[]},filter=libraryFilter.value,visibleMatches=['paired','forks','studies'].includes(filter)?[]:(source.matches||[]),visibleForks=['matches','paired','studies'].includes(filter)?[]:(source.forks||[]),hosts=[...libraryList.children].slice(visibleMatches.length,visibleMatches.length+visibleForks.length);for(const [index,item] of visibleForks.entries()){const host=hosts[index];if(!host)continue;const evidence=item.propagation||{},minute=Number(item.branch_at_sec)/60,identity=String(evidence.branch_state_identity||'').slice(0,16),row=document.createElement('p');row.className='status';row.textContent='分叉 '+(Number.isFinite(minute)?minute.toFixed(1):'—')+' 分钟 · 前缀锚点 '+(evidence.branch_anchor_verified?'已验证':'未验证')+' · '+(identity||'无 identity')+' · 确定性重放（非进程快照）';host.append(row)}}
function renderLibraryUnifiedFuture(library){const source=library||{matches:[],forks:[]},filter=libraryFilter.value,visibleMatches=['paired','forks','studies'].includes(filter)?[]:(source.matches||[]),visibleForks=['matches','paired','studies'].includes(filter)?[]:(source.forks||[]),hosts=[...libraryList.children].slice(visibleMatches.length,visibleMatches.length+visibleForks.length),labels={descriptive_only_ineligible:'仅描述：资格未通过',no_realized_action_divergence:'策略介入但动作未分叉',action_divergence_without_local_attribution:'动作分叉但局部归因不足',local_action_divergence_with_descriptive_future_difference:'局部动作已归因，未来出现描述性差异',local_action_divergence_without_measured_future_difference:'局部动作已归因，已测未来未变化'};for(const [index,item] of visibleForks.entries()){const host=hosts[index],future=item.propagation?.future_summary;if(!host||!future?.available)continue;const row=document.createElement('p'),local=future.simulator_local_action_attribution?'局部动作归因成立':'不授予局部动作归因';row.className='status';row.textContent='反事实未来：'+(labels[future.status]||future.status)+' · 动作变化 '+Number(future.changed_actions||0)+' · 后续描述差异 '+Number(future.descriptive_outcome_difference_count||0)+' · '+local+' · 不授予赛果因果';host.append(row)}}
const LIBRARY_RENDER_STAGES=Object.freeze([renderLibraryBase,renderLibraryBranchIdentity,renderLibraryUnifiedFuture,renderLibraryForkSets]);
function renderLibrary(library){for(const renderStage of LIBRARY_RENDER_STAGES)renderStage(library)}
function renderLibraryForkSets(library){const source=library||{},sets=source.fork_sets||[],filter=libraryFilter.value;if(!['all','forks'].includes(filter))return;if(sets.length)librarySummary.textContent+=' \u00b7 '+sets.length+' \u4e2a\u591a\u65f6\u70b9\u672a\u6765\u96c6';for(const item of sets){const node=card(`${item.home} vs ${item.away}`,item.state);node.setAttribute('role','listitem');const minutes=(item.branch_times_sec||[]).map(value=>(Number(value)/60).toFixed(1)).join(' / '),progress=document.createElement('p'),boundary=document.createElement('p'),links=document.createElement('p');progress.className='status';progress.textContent=`\u56fa\u5b9a seed ${item.seed} \u00b7 \u5206\u53c9\u5206\u949f ${minutes} \u00b7 ${item.scenarios_completed||0}/${item.fixed_scenario_budget||0} \u4e2a\u573a\u666f\u5b8c\u6210`;boundary.className='status';boundary.textContent=item.state==='completed'?`\u65f6\u95f4\u654f\u611f\u6027 ${item.timing_sensitivity_observed?'\u5df2\u89c2\u5bdf\u5230':'\u672a\u89c2\u5bdf\u5230'} \u00b7 \u672a\u6267\u884c\u6700\u4f73\u65f6\u70b9\u6392\u540d \u00b7 \u4e0d\u6388\u4e88\u8d5b\u679c\u6216\u73b0\u5b9e\u8db3\u7403\u56e0\u679c`:'\u5b8c\u6574\u9884\u7b97\u5b8c\u6210\u524d\u4e0d\u62ab\u9732\u573a\u666f\u6392\u540d\u6216\u6700\u4f73\u65f6\u70b9';const link=artifactLink('\u6253\u5f00\u591a\u65f6\u70b9\u672a\u6765\u96c6',item.fork_set_url);if(link)links.append(link);if(item.state==='interrupted'){const resume=document.createElement('button');resume.type='button';resume.textContent='\u5b89\u5168\u6062\u590d\u672a\u6765\u96c6';resume.addEventListener('click',()=>resumeInterruptedTask(item.task_id,resume));node.append(progress,boundary,links,resume)}else node.append(progress,boundary,links);libraryList.append(node)}}
function renderActionAdoptionCurrentCodeEvidence(studio){if(!studio)return;const mechanism=studio.evidence?.action_adoption_mechanism||{},outcome=studio.evidence?.action_outcome_study||{},parts=[];if(mechanism.historical_result_available&&!mechanism.result_identity_verified)parts.push('\u5386\u53f2\u52a8\u4f5c\u91c7\u7eb3\u7ed3\u679c\u4e0e\u5f53\u524d\u4ee3\u7801\u8eab\u4efd\u4e0d\u4e00\u81f4\uff0c\u4e0d\u9002\u7528\u4e8e\u5f53\u524d\u63a7\u5236\u5668');else if(mechanism.result_identity_verified)parts.push('\u52a8\u4f5c\u91c7\u7eb3\u673a\u5236\u7ed3\u679c\u5df2\u9a8c\u8bc1\u5f53\u524d\u4ee3\u7801\u8eab\u4efd');if(outcome.historical_result_available&&!outcome.result_identity_verified)parts.push('\u5386\u53f2\u5168\u573a\u7ed3\u679c\u4e0e\u5f53\u524d\u4ee3\u7801\u8eab\u4efd\u4e0d\u4e00\u81f4\uff0c\u4e0d\u80fd\u4f5c\u4e3a\u5f53\u524d\u6548\u679c\u8bc1\u636e');else if(outcome.result_identity_verified)parts.push('\u5168\u573a\u7ed3\u679c\u5df2\u9a8c\u8bc1\u5f53\u524d\u4ee3\u7801\u8eab\u4efd');if(parts.some(value=>value.includes('\u4e0d\u4e00\u81f4')))adoptionEvidence.textContent=parts.join('\uff1b')+'\u3002\u5f53\u524d\u9875\u9762\u7684\u52a8\u4f5c\u7ea7\u6570\u636e\u4ec5\u662f\u8fd0\u884c\u89c2\u6d4b\uff1b\u9700\u8981\u65b0\u7684\u9884\u6ce8\u518c\u5b9e\u9a8c\u624d\u80fd\u66f4\u65b0\u673a\u5236\u6216\u8d5b\u679c\u7ed3\u8bba\u3002'}
function renderActionAdoptionManagerProtocol(studio){const node=document.querySelector('#manager-advisor-protocol-evidence'),protocol=studio?.evidence?.manager_advisor_adoption;if(!protocol?.available){node.textContent=studio?'\u7ecf\u7406\u987e\u95ee\u91c7\u7eb3\u534f\u8bae\u7f3a\u5931\uff1b\u4e0d\u80fd\u5f62\u6210\u91c7\u7eb3\u53d6\u8bc1\u7ed3\u8bba\u3002':'';return}node.textContent=`\u7ecf\u7406\u7ea7\u91c7\u7eb3\u534f\u8bae\uff1a${protocol.protocol_state} \u00b7 \u56fa\u5b9a\u4fe1\u606f\u7a97\u53e3 ${(protocol.fixed_information_windows||[]).join('/')} \u00b7 ${protocol.results_available?'\u5df2\u6709\u7ed3\u679c':'\u5c1a\u65e0\u7ed3\u679c'} \u00b7 \u4e0d\u6388\u6743\u8d5b\u679c\u56e0\u679c\u6216\u4ea7\u54c1/\u8bba\u6587\u664b\u7ea7\u3002`}
function renderActionAdoptionM2Outcome(studio){const m2=studio?.evidence?.outcome_aligned_m2_study;if(!m2?.available)return;const done=Number(m2.runs_executed||0),total=m2.fixed_run_budget??'\u2014',state=String(m2.result_status||m2.execution_state||m2.protocol_state||'unknown'),promotion=m2.promotion_supported?'\u652f\u6301\u664b\u7ea7':'\u672a\u652f\u6301\u664b\u7ea7';adoptionMetrics.append(card('M2 \u7ed3\u679c\u5bf9\u9f50\u7814\u7a76',`${state} \u00b7 ${done}/${total} \u00b7 ${promotion}`));if(m2.result_applicable_to_current_code){const primary=m2.primary||{},point=Number(primary.point_estimate),low=Number(primary.ci95_low),high=Number(primary.ci95_high),effect=Number.isFinite(point)&&Number.isFinite(low)&&Number.isFinite(high)?`\u63a7\u5236\u961f xG \u51c0\u503c ${point.toFixed(3)}\uff0c95% CI [${low.toFixed(3)}, ${high.toFixed(3)}]`:'\u5df2\u5b8c\u6210\u5f53\u524d\u8eab\u4efd\u9a8c\u8bc1';adoptionEvidence.textContent+=` M2\uff1a${effect}\uff0c${promotion}\u3002`}else if(m2.historical_result_available){adoptionEvidence.textContent+=' M2 \u5386\u53f2\u7ed3\u679c\u4e0e\u5f53\u524d\u534f\u8bae\u3001\u68c0\u67e5\u70b9\u3001\u4ee3\u7801\u6216\u8f93\u5165\u8eab\u4efd\u4e0d\u4e00\u81f4\uff0c\u5df2\u5931\u6548\u3002'}else{adoptionEvidence.textContent+=` M2 \u5df2\u9884\u6ce8\u518c ${total} \u573a\u955c\u50cf\u5bf9\u7167\uff1b\u5c1a\u672a\u7ed1\u5b9a\u5408\u683c\u68c0\u67e5\u70b9\u6216\u6267\u884c\uff0c\u4e0d\u6539\u5199 M1 \u5386\u53f2\u7ed3\u8bba\u3002`}}
function renderM2ResearchControl(m2){m2ResearchControlStages.replaceChildren();m2ResearchControlCommandList.replaceChildren();const control=m2?.research_control;if(!m2?.available||!control){m2ResearchControl.hidden=true;m2ResearchControlSummary.textContent='';m2ResearchControlNext.textContent='';m2ResearchControlBoundary.textContent='';return}m2ResearchControl.hidden=false;const statusLabels={blocked_protocol:'协议阻塞',awaiting_current_preflight:'等待当前前检',blocked_before_training:'训练前阻塞',candidate_identity_drift:'候选身份漂移',candidate_rejected:'候选已拒绝',awaiting_candidate:'等待合格候选',ready_for_authorized_execution:'可申请正式执行',formal_execution_interrupted:'正式执行已中断',awaiting_final_analysis:'等待完整分析',completed_promotion_supported:'完成并支持晋级',completed_no_promotion:'完成但不支持晋级'},stageLabels={freeze_protocol:'1 · 冻结协议',verify_training_readiness:'2 · 验证训练准备',qualify_sealed_candidate:'3 · 资格审查候选',execute_mirrored_study:'4 · 执行镜像研究',decide_promotion:'5 · 决定是否晋级'},stageStatusLabels={completed:'已完成',pending:'待开始',blocked:'被前序证据阻塞',failed:'失败',stale:'身份已过期',ready:'可执行',running:'执行中',completed_supported:'完成并支持',completed_not_supported:'完成但不支持'},actionLabels={repair_or_restore_frozen_protocol:'恢复并验证冻结协议',record_training_preflight:'记录身份绑定的零训练前检',resolve_preflight_failures:'解决全部训练前检查失败项',requalify_candidate_identity:'按当前身份重新审查候选',preregister_new_candidate_after_rejection:'保留拒绝并预注册新候选',train_or_supply_and_qualify_candidate:'训练或提供一个候选并完成资格审查',authorize_fixed_m2_study:'明确授权固定360场研究',resume_identity_matched_m2_study:'仅恢复同一身份的正式研究',request_independent_reproduction:'申请独立复现',retain_research_only_and_diagnose:'保持研究模式并诊断'};const preflight=m2.training_preflight||{},candidate=m2.candidate_qualification||{};m2ResearchControlSummary.textContent=(statusLabels[control.status]||control.status)+' · 阶段 '+Number(control.completed_stage_count||0)+'/'+Number(control.stage_count||0)+' · 正式比赛 '+Number(control.runs_executed||0)+'/'+Number(control.fixed_run_budget||0)+' · 前检 '+(preflight.ready?'有效':preflight.available?'无效或阻塞':'未记录')+' · 候选 '+(candidate.eligible?'合格':candidate.available?'不合格或过期':'未提供');for(const stage of control.stages||[]){const item=document.createElement('li'),title=document.createElement('strong'),stageState=document.createElement('p');title.textContent=stageLabels[stage.stage_id]||stage.stage_id;stageState.className='status';stageState.textContent=(stageStatusLabels[stage.status]||stage.status)+' · '+String(stage.evidence||'无证据');item.append(title,stageState);m2ResearchControlStages.append(item)}const action=control.next_action||{},authorization=action.requires_explicit_authorization?' · 需要明确授权':' · 无需训练或正式比赛授权';m2ResearchControlNext.textContent='唯一下一步：'+(actionLabels[action.id]||action.label||action.id||'无')+authorization;const commands=action.commands||[];m2ResearchControlCommands.hidden=!commands.length;for(const command of commands){const line=document.createElement('p'),code=document.createElement('code');code.textContent=String(command);line.append(code);m2ResearchControlCommandList.append(line)}m2ResearchControlBoundary.textContent='控制面只展示身份绑定的研究准备与执行状态，不会从网页启动训练或正式实验。'+String(control.claim_boundary||'')}
function renderActionAdoptionM2ResearchControl(studio){renderM2ResearchControl(studio?.evidence?.outcome_aligned_m2_study)}
const ACTION_ADOPTION_RENDER_STAGES=Object.freeze([renderActionAdoptionBase,renderActionAdoptionActionSignals,renderActionAdoptionShotValidation,renderActionAdoptionHoldReference,renderActionAdoptionFormalEvidence,renderActionAdoptionCurrentCodeEvidence,renderActionAdoptionManagerProtocol,renderActionAdoptionM2Outcome,renderActionAdoptionM2ResearchControl]);
function renderActionAdoption(studio){for(const renderStage of ACTION_ADOPTION_RENDER_STAGES)renderStage(studio)}
function renderManagerDecisionLedgerAdvisorEvidence(season){const node=document.querySelector('#manager-advisor-evidence-summary'),evidence=season?.manager_decision_ledger?.summary?.world_model_advisor;if(!evidence||!evidence.advised_decisions){node.textContent='';return}node.textContent=`\u4e16\u754c\u6a21\u578b\u987e\u95ee\uff1a${evidence.advised_decisions}/${evidence.decisions} \u6b21\u51b3\u7b56\u83b7\u5f97\u5efa\u8bae \u00b7 ${evidence.adopted_recommendation} \u6b21\u660e\u786e\u91c7\u7528 \u00b7 ${evidence.reviewed_then_selected} \u6b21\u67e5\u770b\u540e\u81ea\u4e3b\u9009\u62e9 \u00b7 ${evidence.unlinked_advice} \u6b21\u672a\u7ed1\u5b9a\u610f\u56fe \u00b7 \u76f4\u63a5\u6267\u884c\u8bc1\u636e ${Math.round(Number(evidence.direct_execution_coverage||0)*100)}% \u00b7 \u53ea\u63cf\u8ff0\u91c7\u7eb3\u548c\u6267\u884c\uff0c\u4e0d\u4f30\u8ba1\u8d5b\u679c\u56e0\u679c\u6548\u5e94\u3002`}
function renderManagerDecisionLedgerExecutionTrace(season){const ledger=season?.manager_decision_ledger,entries=visibleManagerLedgerEntries(season),stateLabels={awaiting_execution:'\u7b49\u5f85\u6bd4\u8d5b\u6267\u884c',evidence_unavailable:'\u8fd0\u884c\u65f6\u7ed1\u5b9a\u8bc1\u636e\u4e0d\u53ef\u7528',recommendation_executed:'\u5efa\u8bae\u5df2\u7ecf\u8fd0\u884c\u65f6\u6267\u884c',reviewed_alternative_executed:'\u5ba1\u9605\u540e\u7684\u66ff\u4ee3\u9009\u62e9\u5df2\u6267\u884c',advised_selection_executed_unlinked:'\u6240\u9009\u6218\u672f\u5df2\u6267\u884c\uff0c\u672a\u7ed1\u5b9a\u91c7\u7528\u610f\u56fe'},controlLabels={pressing_intensity:'\u903c\u62a2',line_height:'\u9632\u7ebf\u9ad8\u5ea6',verticality:'\u7eb5\u5411\u6027',possession_orientation:'\u63a7\u7403',counter_attack:'\u53cd\u51fb',compactness:'\u7d27\u51d1\u5ea6'};for(const [index,row] of entries.entries()){const trace=row.advisor_execution_trace;if(!trace?.available)continue;const host=managerDecisionLedgerList.children[index];if(!host)continue;const details=document.createElement('details'),summary=document.createElement('summary'),runtime=trace.runtime_binding||{};summary.textContent=`\u987e\u95ee\u6267\u884c\u94fe\uff1a${stateLabels[trace.end_to_end_state]||trace.end_to_end_state}`;const chain=document.createElement('p');chain.textContent=`\u5efa\u8bae ${trace.recommendation?.tactic} \u2192 ${trace.interaction?.status==='explicit'?(trace.interaction.intent==='adopt_recommendation'?'\u660e\u786e\u91c7\u7528':'\u5ba1\u9605\u540e\u6539\u9009'):'\u672a\u7ed1\u5b9a\u610f\u56fe'} \u2192 \u51bb\u7ed3 ${trace.selection?.tactic} \u2192 ${runtime.status==='verified'?'\u5f15\u64ce\u5df2\u9a8c\u8bc1':'\u5f15\u64ce\u672a\u9a8c\u8bc1'}`;details.append(summary,chain);const binding=row.execution?.tactical_binding;if(binding?.available){const vector=binding.initial_vector||{},facts=document.createElement('p');facts.textContent=Object.entries(controlLabels).map(([key,label])=>`${label} ${Math.round(Number(vector[key]||0)*100)}%`).join(' \u00b7 ');const drift=document.createElement('p');drift.className='status';drift.textContent=`${binding.binding_kind==='native_team_vector'?'\u7403\u961f\u539f\u751f\u5411\u91cf':'\u9501\u5b9a\u9884\u8bbe'} \u00b7 \u7ec8\u573a\u53d8\u5316 ${binding.changed_controls?.length||0}/22 \u7ef4 \u00b7 \u7ed1\u5b9a ${String(binding.binding_identity||'').slice(0,12)}`;details.append(facts,drift)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='\u8fd9\u91cc\u8bc1\u660e\u6218\u672f\u8f93\u5165\u786e\u5b9e\u8fdb\u5165\u5f15\u64ce\uff1b\u4e0d\u8bc1\u660e\u5b83\u5bfc\u81f4\u4e86\u6bd4\u5206\u6216\u80dc\u8d1f\u3002';details.append(boundary);host.append(details)}const evidence=ledger?.summary?.world_model_advisor,node=document.querySelector('#manager-advisor-evidence-summary');if(evidence?.advised_decisions)node.textContent+=` \u00b7 \u6218\u672f\u8fd0\u884c\u65f6\u7ed1\u5b9a ${evidence.verified_tactical_bindings||0}/${evidence.executed_advised_decisions||0} \u00b7 \u5efa\u8bae\u76f4\u63a5\u6267\u884c ${evidence.direct_recommendation_executions||0} \u6b21`}
function renderManagerDecisionLedgerFutureReviews(season){const ledger=season?.manager_decision_ledger,entries=visibleManagerLedgerEntries(season),summary=ledger?.summary?.world_model_future_reviews;if(summary?.reviewed_future_sets){const host=document.querySelector('#manager-decision-ledger-summary');host.textContent+=' \u00b7 \u672a\u6765\u8bc1\u636e\u590d\u6838 '+summary.reviewed_future_sets+' \u6b21\uff08\u4fdd\u7559 '+summary.kept_after_review+' / \u4fee\u6539 '+summary.revised_after_review+'\uff09'}for(const [index,row] of entries.entries()){const reviews=row.world_model_future_reviews||[],host=managerDecisionLedgerList.children[index];if(!host||!reviews.length)continue;const details=document.createElement('details'),heading=document.createElement('summary');heading.textContent='\u4e16\u754c\u6a21\u578b\u672a\u6765\u8bc1\u636e\u590d\u6838 \u00b7 '+reviews.length+' \u6b21';details.append(heading);for(const review of reviews){const evidence=review.evidence_summary||{},line=document.createElement('p');line.className='status';line.textContent=(review.intent==='keep_after_review'?'\u590d\u6838\u540e\u4fdd\u7559':'\u590d\u6838\u540e\u4fee\u6539')+' \u00b7 \u4efb\u52a1 '+String(review.task_id||'').slice(0,12)+' \u00b7 \u52a8\u4f5c\u5206\u53c9\u65f6\u70b9 '+Number(evidence.action_divergence_scenarios||0)+' \u00b7 \u65f6\u95f4\u654f\u611f\u6027 '+(evidence.timing_sensitivity_observed?'\u5df2\u89c2\u5bdf\u5230':'\u672a\u89c2\u5bdf\u5230')+' \u00b7 \u4e0d\u6388\u4e88\u51b3\u7b56\u8d28\u91cf\u6216\u8d5b\u679c\u56e0\u679c';details.append(line)}host.append(details)}}
function appendFutureMechanismExamplesBase(context){const {host,scenarios,title}=context,rows=(scenarios||[]).flatMap(scenario=>(scenario.mechanism_examples||[]).map(example=>({scenario,example})));if(!host||!rows.length)return;const details=document.createElement('details'),heading=document.createElement('summary');heading.textContent=title+' · '+rows.length+' 条有界样例';details.append(heading);for(const {scenario,example} of rows){const action=document.createElement('p');action.className='status';action.textContent=Number(scenario.branch_minute||0).toFixed(1)+' 分钟分叉 · '+example.clock+' · '+example.team+' · '+example.baseline_action+' → '+example.treatment_action+(example.recommended_action?' · 模型建议 '+example.recommended_action:'')+' · '+(example.directly_observed?'运行时已绑定':'运行时未绑定')+' · '+(example.local_policy_attribution_eligible?'局部归因合格':'仅动作分歧');details.append(action);for(const window of example.downstream_windows||[]){const delta=window.delta||{},line=document.createElement('p');line.className='status';line.textContent='+'+Number(window.window_sec||0)+' 秒描述窗口：传球 '+Number(delta.passes||0)+' · 射门 '+Number(delta.shots||0)+' · 进球 '+Number(delta.goals||0)+' · 球权丢失 '+Number(delta.turnovers||0)+' · 其他策略改变 '+Number(window.additional_policy_changes||0)+'；不授予下游因果';details.append(line)}}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='动作改变可在身份完整时获得模拟器局部归因；后续共享时钟窗口始终只是描述。';details.append(boundary);host.append(details);context.details=details}
function appendFutureMechanismExamplesCrossMetrics(context){const {scenarios,details}=context;if(!details?.matches('details'))return;const windows=(scenarios||[]).flatMap(scenario=>(scenario.mechanism_examples||[]).flatMap(example=>example.downstream_windows||[])),crossWindows=windows.filter(window=>Number(window?.delta?.crosses||0)!==0);if(!crossWindows.length)return;const note=document.createElement('p');note.className='status';note.textContent='\u4f20\u4e2d\u5df2\u4ece\u4f20\u7403\u4e2d\u62c6\u5206\u4e3a\u72ec\u7acb\u63cf\u8ff0\u6307\u6807 \u00b7 '+crossWindows.length+' \u4e2a\u65f6\u95f4\u7a97\u53e3\u5b58\u5728\u975e\u96f6\u4f20\u4e2d\u53d8\u5316 \u00b7 \u4e0d\u6388\u4e88\u4e0b\u6e38\u56e0\u679c';details.insertBefore(note,details.lastElementChild)}
function appendFutureMechanismExamplesPolicySemantics(context){const {scenarios,details}=context;if(!details?.matches('details'))return;const examples=(scenarios||[]).flatMap(scenario=>scenario.mechanism_examples||[]),labels={direct_preference:'\u76f4\u63a5\u5df2\u9a8c\u8bc1\u504f\u597d',suppression_only:'\u4ec5\u6291\u5236\uff0c\u65e0\u76f4\u63a5\u63a8\u8350',none:'\u65e0\u6388\u6743\u4fe1\u53f7',legacy_unclassified:'\u65e7\u8bc1\u636e\u672a\u5206\u7c7b'};for(const example of examples){if(!example.signal_mode||example.signal_mode==='legacy_unclassified')continue;const line=document.createElement('p'),hold=example.hold_reference_redistributed?'\u6301\u7403\u95f4\u63a5\u518d\u5206\u914d '+(Number(example.hold_reference_probability_delta||0)*100).toFixed(2)+' pp':'\u6301\u7403\u65e0\u95f4\u63a5\u518d\u5206\u914d';line.className='status';line.textContent=example.clock+' \u00b7 '+(labels[example.signal_mode]||example.signal_mode)+' \u00b7 \u4e3b\u8981\u4fe1\u53f7 '+example.primary_signal_action+' \u00b7 '+hold+' \u00b7 \u6301\u7403\u65e0\u76f4\u63a5\u6a21\u578b\u6743\u9650';details.insertBefore(line,details.lastElementChild)}}
const FUTURE_MECHANISM_EXAMPLE_RENDER_STAGES=Object.freeze([appendFutureMechanismExamplesBase,appendFutureMechanismExamplesCrossMetrics,appendFutureMechanismExamplesPolicySemantics]);
function appendFutureMechanismExamples(host,scenarios,title){const context={host,scenarios:scenarios||[],title,details:null};for(const appendStage of FUTURE_MECHANISM_EXAMPLE_RENDER_STAGES)appendStage(context)}
function renderManagerDecisionLedgerFutureScenarioEvidence(season){const entries=visibleManagerLedgerEntries(season),labels={descriptive_only_ineligible:'资格未通过，仅描述',no_realized_action_divergence:'策略介入但动作未分叉',action_divergence_without_local_attribution:'动作分叉，局部归因不足',local_action_divergence_with_descriptive_future_difference:'局部归因成立，未来出现描述差异',local_action_divergence_without_measured_future_difference:'局部归因成立，已测未来未变化'};for(const [index,row] of entries.entries()){const scenarios=(row.world_model_future_reviews||[]).flatMap(review=>review.scenario_evidence||[]),host=managerDecisionLedgerList.children[index];if(!host||!scenarios.length)continue;const details=document.createElement('details'),heading=document.createElement('summary');heading.textContent='已封存的分叉机制链 · '+scenarios.length+' 个时点';details.append(heading);for(const scenario of scenarios){const line=document.createElement('p');line.className='status';line.textContent=Number(scenario.branch_minute||0).toFixed(1)+' 分钟 · '+(labels[scenario.future_status]||scenario.future_status)+' · 动作 '+Number(scenario.changed_actions||0)+' · 局部归因 '+Number(scenario.locally_attributable_changes||0)+' · 后续差异 '+Number(scenario.descriptive_future_difference_count||0);details.append(line)}host.append(details)}}
function renderManagerDecisionLedgerMechanismExamples(season){const ledger=season?.manager_decision_ledger,entries=visibleManagerLedgerEntries(season),evidence=ledger?.summary?.world_model_future_reviews;if(evidence?.retained_mechanism_examples){const summary=document.querySelector('#manager-decision-ledger-summary');summary.textContent+=' · 具体动作链 '+evidence.retained_mechanism_examples+' 条（局部归因 '+evidence.locally_attributable_mechanism_examples+'）'}for(const [index,row] of entries.entries()){const scenarios=(row.world_model_future_reviews||[]).flatMap(review=>review.scenario_evidence||[]);appendFutureMechanismExamples(managerDecisionLedgerList.children[index],scenarios,'已封存的具体动作采用链')}}
function renderManagerDecisionLedgerMechanismSemantics(season){const evidence=season?.manager_decision_ledger?.summary?.world_model_future_reviews,summary=document.querySelector('#manager-decision-ledger-summary');if(!summary||!evidence?.retained_mechanism_examples)return;summary.textContent+=' · 传中动作 '+Number(evidence.cross_action_mechanism_examples||0)+' · 直接偏好 '+Number(evidence.direct_preference_mechanism_examples||0)+' · 仅抑制 '+Number(evidence.suppression_only_mechanism_examples||0)+' · 持球基准再分配 '+Number(evidence.hold_reference_redistribution_examples||0)+' · 非零传中窗口 '+Number(evidence.nonzero_cross_descriptive_windows||0)}
function renderManagerDecisionLedgerFutureReviewExecution(season){const ledger=season?.manager_decision_ledger,entries=visibleManagerLedgerEntries(season),evidence=ledger?.summary?.world_model_future_review_execution;if(evidence?.fixtures_with_review_execution_trace){managerDecisionLedgerSummary.textContent+=' · 复核进入最终选择 '+(Number(evidence.fixtures_with_review_execution_trace||0)-Number(evidence.terminal_review_superseded_before_execution||0))+' 场 · 运行时验证 '+Number(evidence.reviewed_selection_runtime_verified||0)+' 场'}const stateLabels={terminal_review_superseded_before_execution:'最终复核已被未复核编辑覆盖',reviewed_selection_awaiting_execution:'复核选择已冻结，等待比赛',reviewed_selection_runtime_verified:'复核选择已由比赛引擎战术绑定验证',reviewed_selection_execution_evidence_unavailable:'复核选择存在，但直接执行证据不可用'},relationLabels={followed_by_later_review:'后续又复核',selected_for_fixture:'进入最终选择',superseded_by_unreviewed_edit:'被未复核编辑覆盖'};for(const [index,row] of entries.entries()){const trace=row.future_review_execution_trace,host=managerDecisionLedgerList.children[index];if(!host||!trace?.available)continue;const details=document.createElement('details'),heading=document.createElement('summary'),chain=document.createElement('p'),runtime=document.createElement('p'),boundary=document.createElement('p');heading.textContent='世界模型复核 → 正式执行';chain.className=runtime.className=boundary.className='status';chain.textContent=(trace.review_chain||[]).map((item,reviewIndex)=>'复核 '+(reviewIndex+1)+'：'+(relationLabels[item.relation_to_final_selection]||item.relation_to_final_selection)).join(' → ');runtime.textContent=(stateLabels[trace.end_to_end_state]||trace.end_to_end_state)+(trace.runtime_binding?.status==='verified'?' · 引擎采用 '+trace.runtime_binding.applied_tactic+' · 比赛 '+trace.runtime_binding.match_id:'');boundary.textContent='这里只验证复核、最终决策身份与引擎战术绑定；不把赛前模拟路径匹配到比分，也不估计赛果因果。';details.append(heading,chain,runtime,boundary);host.append(details)}}
function appendOfficialActionExecutionBase(context){const {host,evidence,title}=context;if(!host||!evidence?.available)return;const counts=evidence.retained_record_evidence||{},details=document.createElement('details'),heading=document.createElement('summary'),summary=document.createElement('p'),boundary=document.createElement('p');heading.textContent=title;summary.className=boundary.className='status';summary.textContent='经理球队保留动作 '+Number(counts.records||0)+' · 非零概率影响 '+Number(counts.influenced_decisions||0)+' · 局部动作改变 '+Number(counts.locally_attributable_action_changes||0)+' · 球事件身份匹配 '+Number(counts.direct_ball_event_links||0)+' · 未解析 '+Number(counts.unresolved_or_missing_ball_event_links||0);details.append(heading,summary);for(const example of evidence.examples||[]){const line=document.createElement('p'),local=example.simulator_local_action_attribution?'共享随机数下局部动作归因':'未形成局部动作改变',linked=example.runtime_link?.direct_ball_event_identity?'球事件身份已验证':example.runtime_link?.status||'无球事件链接';line.className='status';line.textContent=example.clock+' · '+example.counterfactual_baseline_action+' → '+example.actual_action+' · 模型建议 '+example.recommended_action+' · '+local+' · '+linked;details.append(line)}boundary.textContent='这是正式比赛中的动作选择与身份链接证据；不比较比分、不证明战术质量，也不授权赛果或现实足球因果。';details.append(boundary);host.append(details);context.details=details}
function appendOfficialActionExecutionPolicySemantics(context){const {evidence,details}=context;if(!details?.matches('details'))return;const modeLabels={direct_preference:'\u76f4\u63a5\u5df2\u9a8c\u8bc1\u504f\u597d',suppression_only:'\u4ec5\u6291\u5236\uff0c\u65e0\u76f4\u63a5\u63a8\u8350',none:'\u65e0\u6388\u6743\u4fe1\u53f7',legacy_unclassified:'\u65e7\u8bc1\u636e\u672a\u5206\u7c7b'};for(const example of evidence.examples||[]){const signal=example.policy_signal;if(!signal||signal.mode==='legacy_unclassified')continue;const reference=example.reference_action_effect||{},line=document.createElement('p'),primary=signal.primary_action==='none'?'\u65e0':signal.primary_action,referenceText=reference.received_redistributed_probability?'\u6301\u7403\u4ec5\u95f4\u63a5\u83b7\u5f97 '+(Number(reference.probability_delta||0)*100).toFixed(2)+' pp':'\u6301\u7403\u672a\u83b7\u5f97\u95f4\u63a5\u518d\u5206\u914d';line.className='status';line.textContent=example.clock+' \u00b7 \u4fe1\u53f7\u8bed\u4e49 '+(modeLabels[signal.mode]||signal.mode)+' \u00b7 \u4e3b\u8981\u52a8\u4f5c '+primary+' \u00b7 '+referenceText+' \u00b7 \u6301\u7403\u65e0\u76f4\u63a5\u6a21\u578b\u6743\u9650';details.insertBefore(line,details.lastElementChild)}}
function renderManagerDecisionLedgerOfficialActionExecution(season){const ledger=season?.manager_decision_ledger,entries=visibleManagerLedgerEntries(season),summary=ledger?.summary?.world_model_official_action_execution;if(summary?.fixtures_with_official_action_evidence){managerDecisionLedgerSummary.textContent+=' · 正式动作证据 '+Number(summary.fixtures_with_official_action_evidence||0)+' 场（局部改变 '+Number(summary.locally_attributable_action_changes||0)+'）'}for(const [index,row] of entries.entries()){appendOfficialActionExecution(managerDecisionLedgerList.children[index],row.execution?.world_model_action_execution,'正式比赛中的世界模型动作采用')}}
function appendManagerWorldEvolutionThreadBase(host,thread){if(!host||!thread?.available)return;const details=document.createElement('details'),heading=document.createElement('summary'),flow=document.createElement('ol'),boundary=document.createElement('p'),stageLabels={prematch_future_review:'赛前世界模型分叉复核',frozen_manager_decision:'最终冻结决策',official_tactical_runtime:'引擎战术绑定',official_world_model_actions:'正式比赛动作采用',observed_match_result:'观察比赛事实',persistent_world_state:'持久世界状态'},statusLabels={not_reviewed:'未复核',selected_for_fixture:'复核结果进入最终选择',superseded_before_execution:'复核后被未复核编辑覆盖',frozen:'已冻结',awaiting_execution:'等待正式比赛',runtime_verified:'运行时已验证',execution_evidence_unavailable:'执行证据不可用',locally_attributable_action_changes_observed:'观察到局部动作改变',probability_influence_without_realized_action_change:'概率受影响但动作未改变',no_nonzero_action_influence_observed:'未观察到非零影响',not_applicable_stable_mode:'稳定模式不启用',legacy_evidence_unavailable:'旧报告无此证据',action_evidence_unavailable:'动作证据不可用',observed_descriptive:'比赛事实已记录，仅描述',result_evidence_unavailable:'赛果证据不可用',after_recovery_recorded:'比赛与恢复后世界状态已记录',after_match_recorded_recovery_pending:'赛后状态已记录，恢复待结算',persistent_state_evidence_unavailable:'持久状态证据不可用'};heading.textContent='统一世界演化链 · '+(thread.world_model_runtime_chain_complete?'世界模型运行链完整':thread.official_runtime_chain_complete?'正式比赛链完整':'证据链待闭合');flow.className='journal-list';flow.setAttribute('aria-label','赛前复核到持久世界状态的证据时间线');for(const stage of thread.stages||[]){const item=document.createElement('li'),title=document.createElement('strong'),meta=document.createElement('p');title.textContent=stageLabels[stage.stage_id]||stage.stage_id;meta.className='status';meta.textContent=statusLabels[stage.status]||stage.status;if(stage.stage_id==='official_world_model_actions'&&Number(stage.retained_records||0)){meta.textContent+=' · 保留动作 '+Number(stage.retained_records||0)+' · 局部改变 '+Number(stage.locally_attributable_action_changes||0)+' · 球事件链接 '+Number(stage.direct_ball_event_links||0)}if(stage.stage_id==='persistent_world_state'&&stage.metrics_delta){meta.textContent+=' · 疲劳 '+Number(stage.metrics_delta.team_fatigue_ema||0).toFixed(3)+' · 士气 '+Number(stage.metrics_delta.squad_morale_ema||0).toFixed(3)}item.append(title,meta);flow.append(item)}if((thread.continuity_gaps||[]).length){const gaps=document.createElement('p');gaps.className='status';gaps.textContent='证据断点：'+thread.continuity_gaps.join(' · ');details.append(heading,flow,gaps)}else{details.append(heading,flow)}boundary.className='status';boundary.textContent='时间线只连接已验证的身份、正式运行时、局部动作与状态转移；同场出现不等于因果，完整链也不等于赛果改善。';details.append(boundary);host.append(details)}
function appendMetaLearningGovernance(details,society){const updates=society?.meta_learning_updates||[];if(!details||!updates.length)return;const panel=document.createElement('section'),heading=document.createElement('h4'),list=document.createElement('ul'),boundary=document.createElement('p'),statusLabels={shadow:'\u5f71\u5b50\u63d0\u6848',observed_pending_evaluation:'\u5df2\u89c2\u5bdf\uff0c\u5f85\u914d\u5bf9\u8bc4\u4f30',committed:'\u5df2\u6388\u6743\u751f\u6548',rejected:'\u8bc1\u636e\u62d2\u7edd',rolled_back:'\u5df2\u56de\u6eda',expired:'\u5df2\u8fc7\u671f'},updateLabels={created:'\u65b0\u5efa\u63d0\u6848',status_changed:'\u72b6\u6001\u63a8\u8fdb',observation_recorded:'\u65b0\u589e\u89c2\u5bdf',evidence_updated:'\u8bc1\u636e\u66f4\u65b0',retention_removed:'\u8d85\u51fa\u4fdd\u7559\u7a97\u53e3'},parameterLabels={risk_budget:'\u98ce\u9669\u9884\u7b97',pressing_intensity:'\u903c\u62a2\u5f3a\u5ea6',line_height:'\u9632\u7ebf\u9ad8\u5ea6',rotation_aggressiveness:'\u8f6e\u6362\u6fc0\u8fdb\u5ea6',icon_patience:'\u6838\u5fc3\u8010\u5fc3',w_h_delta:'\u5386\u53f2\u6743\u91cd',w_x_delta:'\u5916\u90e8\u4fe1\u53f7\u6743\u91cd'},scaleLabels={fast:'\u5feb',medium:'\u4e2d',slow:'\u6162'},nextLabels={identity_bound_matched_evaluation:'\u9700\u8981\u8eab\u4efd\u7ed1\u5b9a\u7684\u914d\u5bf9\u8bc4\u4f30',monitor_and_revalidate_before_reuse:'\u7ee7\u7eed\u76d1\u6d4b\uff0c\u590d\u7528\u524d\u91cd\u65b0\u9a8c\u8bc1',new_identity_bound_proposal:'\u9700\u8981\u65b0\u7684\u8eab\u4efd\u7ed1\u5b9a\u63d0\u6848',record_not_retained:'\u8bb0\u5f55\u5df2\u79fb\u51fa\u6709\u754c\u7a97\u53e3'},authorityLabels={withheld:'\u6743\u9650\u4fdd\u7559',granted:'\u53c2\u6570\u6743\u9650\u5df2\u6388\u4e88',denied:'\u53c2\u6570\u6743\u9650\u5df2\u62d2\u7edd',rolled_back:'\u6388\u6743\u5df2\u56de\u6eda',expired:'\u6388\u6743\u5df2\u8fc7\u671f',not_retained:'\u8bb0\u5f55\u5df2\u79fb\u51fa'};panel.className='decision-preview';heading.textContent='\u957f\u671f\u9002\u5e94\u6cbb\u7406';list.className='journal-list';list.setAttribute('role','list');list.setAttribute('aria-label','\u672c\u573a\u6bd4\u8d5b\u540e\u7684\u957f\u671f\u9002\u5e94\u6cbb\u7406\u53d8\u5316');const signed=value=>`${Number(value)>=0?'+':''}${Number(value).toFixed(3)}`;for(const update of updates){const item=document.createElement('li'),title=document.createElement('strong'),state=document.createElement('p'),changes=document.createElement('p'),evidence=document.createElement('p'),next=document.createElement('p'),evaluation=update.evaluation_after||{};title.textContent=(updateLabels[update.update_kind]||update.update_kind)+' \u00b7 '+String(update.proposal_id||'').slice(0,10);state.className=changes.className=evidence.className=next.className='status';state.textContent='\u72b6\u6001 '+(statusLabels[update.before_status]||update.before_status||'\u65e0')+' \u2192 '+(statusLabels[update.after_status]||update.after_status||'\u5df2\u79fb\u51fa')+' \u00b7 \u89c2\u5bdf '+Number(update.observation_count_after||0)+'/'+Number(update.observation_window||0)+' \u00b7 \u672c\u6b21 '+signed(update.observation_count_delta||0);changes.textContent='\u6709\u754c\u53c2\u6570\uff1a'+(update.changes||[]).map(change=>(parameterLabels[change.parameter]||change.parameter)+' '+signed(change.proposed_delta)+'\uff08'+(scaleLabels[change.time_scale]||change.time_scale)+'\u901f\uff09').join(' \u00b7 ');evidence.textContent=evaluation.available?'\u914d\u5bf9\u8bc1\u636e '+Number(evaluation.matched_units||0)+' \u7ec4 \u00b7 ATE '+signed(evaluation.average_treatment_effect)+' \u00b7 95% ['+signed(evaluation.ci_low)+', '+signed(evaluation.ci_high)+'] \u00b7 \u95e8\u69db '+signed(evaluation.minimum_effect)+' \u00b7 '+(evaluation.lower_bound_clears_threshold?'\u4e0b\u754c\u901a\u8fc7':'\u4e0b\u754c\u672a\u901a\u8fc7'):'\u914d\u5bf9\u6548\u5e94\u8bc1\u636e\u5c1a\u672a\u8bb0\u5f55';next.textContent=(authorityLabels[update.authority_state]||update.authority_state)+' \u00b7 \u4e0b\u4e00\u8bc1\u636e\uff1a'+(nextLabels[update.next_required_evidence]||update.next_required_evidence);item.append(title,state,changes,evidence,next);list.append(item)}boundary.className='status';boundary.textContent='\u53ea\u663e\u793a\u5185\u5bb9\u65e0\u5173\u7684\u63d0\u6848\u8eab\u4efd\u3001\u6709\u754c\u53c2\u6570\u3001\u89c2\u5bdf\u8fdb\u5ea6\u548c\u8bc4\u4f30\u6458\u8981\uff1b\u53cd\u601d\u6587\u672c\u3001\u8bb0\u5fc6\u8bc1\u636e\u548c\u914d\u5bf9\u884c\u4e0d\u8fdb\u5165\u4ea7\u54c1\uff0c\u672c\u89c6\u56fe\u4e5f\u4e0d\u63d0\u4f9b\u53c2\u6570\u6388\u6743\u3002';panel.append(heading,list,boundary);details.insertBefore(panel,details.lastElementChild)}
function appendManagerWorldEvolutionThreadSocietyContinuity(host,thread){const stage=(thread?.stages||[]).find(row=>row.stage_id==='persistent_world_state'),society=stage?.society_transition,meta=society?.meta_learning_after||{};if(!host||!society?.available)return;const details=host.lastElementChild;if(!details||details.tagName!=='DETAILS')return;const line=document.createElement('p');line.className='status';line.textContent='\u793e\u4f1a\u8ba4\u77e5\u8fde\u7eed\u6027 \u00b7 \u8bb0\u5fc6 '+Number(society.memory_record_delta||0)+' \u00b7 \u8ba4\u77e5\u8bb0\u5fc6 '+Number(society.cognitive_memory_delta||0)+' \u00b7 \u5f85\u8bc4\u4f30 '+(Number(meta.observed_pending_evaluation||0)+Number(meta.shadow||0))+' \u00b7 \u5df2\u6388\u6743 '+Number(meta.committed||0)+' \u00b7 \u72b6\u6001\u53d8\u5316 '+Number((society.changed_state_fields||[]).length)+' \u00b7 \u53ea\u8868\u793a\u6301\u4e45\u5316\u4e8b\u5b9e';details.insertBefore(line,details.lastElementChild);appendMetaLearningGovernance(details,society)}
function renderManagerDecisionLedgerWorldEvolutionThread(season){const ledger=season?.manager_decision_ledger,entries=visibleManagerLedgerEntries(season),summary=ledger?.summary?.manager_world_evolution;if(summary?.fixtures_with_world_evolution_thread){managerDecisionLedgerSummary.textContent+=' · 统一世界链 '+Number(summary.fixtures_with_world_evolution_thread||0)+' 场（完整世界模型链 '+Number(summary.world_model_runtime_chain_complete||0)+'）'}for(const [index,row] of entries.entries()){appendManagerWorldEvolutionThread(managerDecisionLedgerList.children[index],row.world_evolution_thread)}}
function renderManagerDecisionLedgerMetaLearningSummary(season){const summary=season?.manager_decision_ledger?.summary?.manager_world_evolution;if(summary?.meta_learning_pending||summary?.meta_learning_authorized){managerDecisionLedgerSummary.textContent+=' \u00b7 Meta \u5f85\u8bc4\u4f30 '+Number(summary.meta_learning_pending||0)+' / \u5df2\u6388\u6743 '+Number(summary.meta_learning_authorized||0)}}
function renderManagerIntelligenceTacticalBinding(command,configured){const debrief=command?.postmatch_debrief,binding=debrief?.tactical_binding;if(!configured||!debrief?.available||!binding)return;if(!binding.available){const unavailable=document.createElement('p');unavailable.className='status';unavailable.textContent=`\u6218\u672f\u8fd0\u884c\u65f6\u7ed1\u5b9a\u4e0d\u53ef\u7528\uff1a${binding.reason||'unknown'}`;matchdayAttribution.append(unavailable);return}const vector=binding.initial_vector||{},value=`${binding.applied_tactic} \u00b7 ${binding.binding_kind==='native_team_vector'?'\u539f\u751f\u7403\u961f\u5411\u91cf':'\u9501\u5b9a\u6218\u672f\u9884\u8bbe'} \u00b7 \u903c\u62a2 ${Math.round(Number(vector.pressing_intensity||0)*100)}% \u00b7 \u9632\u7ebf ${Math.round(Number(vector.line_height||0)*100)}% \u00b7 \u7eb5\u5411 ${Math.round(Number(vector.verticality||0)*100)}% \u00b7 \u53d8\u5316 ${binding.changed_controls?.length||0}/22 \u7ef4`,node=card('\u5f15\u64ce\u6218\u672f\u7ed1\u5b9a',value);node.setAttribute('role','listitem');matchdayAttribution.append(node)}
function renderManagerIntelligenceOfficialActionExecution(command,configured){const debrief=command?.postmatch_debrief,evidence=debrief?.world_model_action_execution;if(!configured||!debrief?.available||!evidence)return;if(evidence.available){appendOfficialActionExecution(matchdayAttribution,evidence,'正式比赛中的世界模型动作采用')}else{const unavailable=document.createElement('p');unavailable.className='status';unavailable.textContent='世界模型动作执行证据：'+(evidence.reason||'不可用');matchdayAttribution.append(unavailable)}}
function renderManagerIntelligenceWorldEvolutionThread(command,configured){const matchId=command?.postmatch_debrief?.match_id,entry=(command?.decision_ledger?.entries||[]).find(row=>row.execution?.match_id===matchId);if(configured&&entry?.world_evolution_thread)appendManagerWorldEvolutionThread(matchdayAttribution,entry.world_evolution_thread)}
const MANAGER_INTELLIGENCE_RENDER_STAGES=Object.freeze([renderManagerIntelligenceBase,renderManagerIntelligenceClubSupport,renderManagerIntelligenceTacticalBinding,renderManagerIntelligenceOfficialActionExecution,renderManagerIntelligenceWorldEvolutionThread]);
function renderManagerIntelligence(command,configured){for(const renderStage of MANAGER_INTELLIGENCE_RENDER_STAGES)renderStage(command,configured)}
function appendManagerWorldEvolutionThreadReviewedFutures(host,thread){const stage=(thread?.stages||[]).find(row=>row.stage_id==='prematch_future_review'),details=host?.lastElementChild,item=details?.querySelector('ol li');if(!stage||!item)return;const note=document.createElement('p'),intentLabels={keep_after_review:'复核后保留方案',revise_after_review:'复核后修改方案'};note.className='status';note.textContent=stage.status==='not_reviewed'?'未生成或复核赛前多时点未来。':'分叉证据：'+(intentLabels[stage.review_intent]||'已复核')+' · '+Number(stage.eligible_scenarios||0)+' / '+Number(stage.fixed_scenario_budget||0)+' 个时点可用 · 动作分叉 '+Number(stage.action_divergence_scenarios||0)+' · 局部归因 '+Number(stage.local_attribution_scenarios||0)+' · 描述差异 '+Number(stage.descriptive_future_difference_scenarios||0)+' · 时间敏感性 '+(stage.timing_sensitivity_observed?'已观察到':'未观察到')+' · '+(stage.status==='selected_for_fixture'?'已进入最终选择':'未进入最终选择');item.append(note)}
function appendManagerWorldEvolutionThreadMechanismSemantics(host,thread){const stage=(thread?.stages||[]).find(row=>row.stage_id==='prematch_future_review'),details=host?.lastElementChild,item=details?.querySelector('ol li');if(!stage||!item||stage.status==='not_reviewed')return;const note=document.createElement('p');note.className='status';note.textContent='动作语义：传中 '+Number(stage.cross_action_mechanism_examples||0)+' · 直接偏好 '+Number(stage.direct_preference_mechanism_examples||0)+' · 仅抑制 '+Number(stage.suppression_only_mechanism_examples||0)+' · 持球基准再分配 '+Number(stage.hold_reference_redistribution_examples||0)+' · 非零传中窗口 '+Number(stage.nonzero_cross_descriptive_windows||0);item.append(note)}
function appendManagerWorldEvolutionThreadOfficialActionSemantics(host,thread){const stages=thread?.stages||[],index=stages.findIndex(row=>row.stage_id==='official_world_model_actions'),stage=stages[index],items=host?.lastElementChild?.querySelectorAll('ol li'),item=index>=0?items?.[index]:null;if(!stage||!item||!Number(stage.retained_semantic_examples||0))return;const note=document.createElement('p');note.className='status';note.textContent='正式动作有界样例：'+Number(stage.retained_semantic_examples||0)+' / '+Number(stage.retained_records||0)+' 条 · '+(stage.semantic_example_coverage_complete?'覆盖完整':stage.semantic_examples_truncated?'样例已截断':'覆盖不完整')+' · 传中 '+Number(stage.semantic_cross_action_examples||0)+' · 直接偏好 '+Number(stage.semantic_direct_preference_examples||0)+' · 仅抑制 '+Number(stage.semantic_suppression_only_examples||0)+' · 持球再分配 '+Number(stage.semantic_hold_reference_redistribution_examples||0)+' · 直接传中事件 '+Number(stage.semantic_direct_cross_ball_event_examples||0)+' · 不代表赛果因果';item.append(note)}
function appendReviewedScenarioArchive(host,scenarios,title){if(!host||!Array.isArray(scenarios)||!scenarios.length)return;const labels={descriptive_only_ineligible:'资格未通过，仅描述',no_realized_action_divergence:'动作未分叉',action_divergence_without_local_attribution:'动作分叉，局部归因不足',local_action_divergence_with_descriptive_future_difference:'局部归因，未来有描述差异',local_action_divergence_without_measured_future_difference:'局部归因，未测到未来差异'},details=document.createElement('details'),heading=document.createElement('summary'),list=document.createElement('ol'),boundary=document.createElement('p');heading.textContent=title+' · '+scenarios.length+' 个时点';list.className='journal-list';list.setAttribute('aria-label','身份封存的赛前分叉档案');for(const scenario of scenarios){const item=document.createElement('li'),name=document.createElement('strong'),facts=document.createElement('p'),identity=document.createElement('p');name.textContent=Number(scenario.branch_minute||0).toFixed(1)+' 分钟 · '+(labels[scenario.future_status]||scenario.future_status);facts.className=identity.className='status';facts.textContent='动作改变 '+Number(scenario.changed_actions||0)+' · 局部归因 '+Number(scenario.locally_attributable_changes||0)+' · 描述差异 '+Number(scenario.descriptive_future_difference_count||0)+' · 锚点 '+(scenario.anchor_verified?'已验证':'未验证');identity.textContent='分叉身份 '+String(scenario.source_scenario_identity||'').slice(0,12)+' · 档案身份 '+String(scenario.archive_identity||'').slice(0,12);item.append(name,facts,identity);list.append(item)}boundary.className='status';boundary.textContent='这些是赛前模拟中未被正式世界同时经历的分叉；不排名、不与观察比分匹配，也不构成赛果预测、效果估计或现实足球因果。';details.append(heading,list,boundary);host.append(details)}
function appendManagerWorldEvolutionThreadScenarioArchive(host,thread){const stage=(thread?.stages||[]).find(row=>row.stage_id==='prematch_future_review'),item=host?.lastElementChild?.querySelector('ol li');appendReviewedScenarioArchive(item,stage?.reviewed_scenarios,'已封存的赛前分叉')}
function appendReviewWorldContinuity(host,certificate){if(!host||!certificate)return;const labels={complete:'复核决策已贯通正式世界',awaiting_official_match:'等待正式比赛',review_superseded:'复核后决策已被覆盖',tactical_runtime_binding_unavailable:'正式战术绑定缺失',official_action_evidence_unavailable:'正式动作证据缺失',not_reviewed:'本场未复核赛前分叉'},line=document.createElement('p'),boundary=document.createElement('p');line.className=boundary.className='status';line.textContent=(labels[certificate.status]||certificate.status)+' · 决策身份 '+(certificate.final_decision_identity_bound?'已绑定':'未绑定')+' · 战术运行时 '+(certificate.runtime_tactic_verified?'已验证':'未验证')+' · 同场动作证据 '+(certificate.official_action_same_match?'已连接':'未连接')+' · 分叉档案 '+Number(certificate.scenario_archive_count||0);boundary.textContent='连续性证书只验证复核、最终决策、正式战术和同场动作证据；不会把赛前模拟分叉中的机会逐条匹配为正式比赛动作，也不证明赛果改善。';host.append(line,boundary)}
function appendManagerWorldEvolutionThreadReviewWorldCertificate(host,thread){const details=host?.lastElementChild;if(details?.tagName==='DETAILS')appendReviewWorldContinuity(details,thread?.review_to_official_world)}
const renderManagerWorldModelAdviceWithoutStatusReset=renderManagerWorldModelAdvice;
renderManagerWorldModelAdvice=(...args)=>{managerWorldModelAdviceSummary.className='status';return renderManagerWorldModelAdviceWithoutStatusReset(...args)};
function renderManagerInterventionWorkspace(season){managerInterventionWorkspace.replaceChildren();const workspace=season?.manager_intervention_workspace;if(!workspace?.available)return;const stateLabels={decision_required:'等待冻结经理决策',ready_to_explore:'已冻结，可生成有界未来',future_generation_in_progress:'正在生成预注册未来',future_generation_interrupted:'未来生成中断，可安全恢复',future_generation_failed:'未来生成失败，可检查后重试',evidence_ready_for_review:'机制证据已就绪，等待经理复核',review_recorded_decision_refrozen:'复核已记录，最终决策已重新冻结'},stageLabels={freeze_intervention:'1 · 冻结干预方案',generate_bounded_futures:'2 · 生成有界未来',inspect_local_mechanism:'3 · 检查局部动作机制',record_manager_review:'4 · 记录经理复核',advance_official_world:'5 · 推进正式足球世界'},statusLabels={decision_frozen:'已完成',available:'可开始',running:'进行中',interrupted:'已中断',failed:'已失败',evidence_complete:'证据完成',scenario_evidence_available:'逐时点机制可检查',aggregate_only:'仅有聚合证据',awaiting_future_set:'等待未来集',review_available:'可选择保留或修改',review_recorded:'已记录',optional_not_started:'尚未复核（可选）',ready_for_official_match:'正式比赛可推进'};const heading=document.createElement('p'),flow=document.createElement('ol'),facts=document.createElement('p'),next=document.createElement('p'),boundary=document.createElement('p');heading.className='status';heading.textContent=(stateLabels[workspace.workflow_state]||workspace.workflow_state)+' · 会话 '+String(workspace.workspace_identity||'').slice(0,12);flow.className='journal-list';flow.setAttribute('aria-label','经理反事实干预五步流程');for(const stage of workspace.stages||[]){const item=document.createElement('li'),title=document.createElement('strong'),status=document.createElement('p');title.textContent=stageLabels[stage.stage_id]||stage.stage_id;status.className='status';status.textContent=statusLabels[stage.status]||stage.status;item.append(title,status);flow.append(item)}const evidence=workspace.evidence_summary||{};facts.className='status';facts.textContent='未来集 '+Number(evidence.future_sets||0)+' · 注册时点 '+Number(evidence.registered_scenarios||0)+' · 动作分叉 '+Number(evidence.action_divergence_scenarios||0)+' · 局部归因 '+Number(evidence.local_attribution_scenarios||0)+' · 已复核 '+Number(evidence.reviewed_future_sets||0);const allowed=workspace.allowed_actions||{};next.className='status';next.textContent='下一步：'+(allowed.review_future_set?'检查逐时点证据并明确保留或修改当前方案':workspace.workflow_state==='future_generation_in_progress'?'等待固定场景预算完成':workspace.workflow_state==='future_generation_interrupted'?'从已验证进度安全恢复':workspace.workflow_state==='future_generation_failed'?'检查失败原因后重新提交':allowed.request_future_set?'生成新的身份绑定多时点未来':'推进正式比赛');if((workspace.continuity_gaps||[]).length){const gaps=document.createElement('p');gaps.className='status';gaps.textContent='证据断点：'+workspace.continuity_gaps.join(' · ');managerInterventionWorkspace.append(heading,flow,facts,next,gaps)}else managerInterventionWorkspace.append(heading,flow,facts,next);boundary.className='status';boundary.textContent='工作台只组织同一冻结决策的模拟器分叉、局部动作证据和显式复核；不排名时点、不预测比分，也不授权赛果或现实足球因果。';managerInterventionWorkspace.append(boundary)}
function renderManagerFutureSets(season,managed){const available=currentStudioMode==='research'&&Boolean(managed?.manager_decision);managerFutureSet.hidden=!available;managerFutureSetList.replaceChildren();if(!available){managerFutureSetSummary.textContent='';return}const sets=season?.manager_future_sets||[];managerFutureSetSummary.textContent=sets.length?`${sets.length} \u4e2a\u8d5b\u524d\u672a\u6765\u5b9e\u9a8c\u5df2\u7ed1\u5b9a\u672c\u8d5b\u5b63\uff1b\u53ea\u6709\u5f53\u524d revision \u7684\u7ed3\u679c\u53ef\u7ee7\u7eed\u7528\u4e8e\u672c\u8f6e\u590d\u76d8\u3002`:'\u51b3\u7b56\u5df2\u51bb\u7ed3\u3002\u53ef\u590d\u7528\u6b63\u5f0f\u5bf9\u9635\u3001seed \u548c\u53cc\u65b9\u5b9e\u9645\u6218\u672f\uff0c\u751f\u6210\u56fa\u5b9a\u591a\u65f6\u70b9\u53cc\u4e16\u754c\u8bc1\u636e\u3002';for(const item of sets){const context=item.manager_context||{},node=card(`\u7b2c ${context.matchday??'?'} \u8f6e \u00b7 ${item.home||'?'} vs ${item.away||'?'}`,item.state),meta=document.createElement('p'),boundary=document.createElement('p'),links=document.createElement('p');node.setAttribute('role','listitem');meta.className='status';meta.textContent=`\u573a\u666f ${item.scenario_id||item.task_id||'?'} \u00b7 \u4e0a\u4e0b\u6587 ${String(context.context_identity||'').slice(0,12)} \u00b7 ${item.binding_current?'\u5f53\u524d\u7ed1\u5b9a':'\u5386\u53f2\u8bc1\u636e'}`;boundary.className='status';boundary.textContent='\u53ea\u63cf\u8ff0\u6a21\u62df\u5668\u5185\u52a8\u4f5c\u4e0e\u672a\u6765\u5dee\u5f02\uff1b\u4e0d\u9884\u6d4b\u6bd4\u5206\u3001\u4e0d\u9009\u62e9\u6700\u4f73\u65f6\u70b9\u3001\u4e0d\u6388\u6743\u8d5b\u679c\u56e0\u679c\u3002';const link=artifactLink('\u6253\u5f00\u8d5b\u524d\u672a\u6765\u5b9e\u9a8c',item.fork_set_url);if(link)links.append(link);node.append(meta,boundary,links);managerFutureSetList.append(node)}}
async function requestManagerFutureExperiment(){const managed=currentSeason?.next_manager_fixture,values=String(managerFutureSetMinutes.value||'').split(',').map(value=>Number(value.trim())),valid=values.length>=2&&values.length<=4&&values.every((value,index)=>Number.isFinite(value)&&value>=0&&value<=90&&(index===0||value>values[index-1]));if(!managed?.manager_decision){announce(managerFutureSetSummary,'\u8bf7\u5148\u51bb\u7ed3\u672c\u573a\u7ecf\u7406\u51b3\u7b56\u3002','error',true);return}if(!valid){announce(managerFutureSetSummary,'\u8bf7\u8f93\u5165 2\u20134 \u4e2a\u4e25\u683c\u9012\u589e\u4e14\u4f4d\u4e8e 0\u201390 \u7684\u5206\u949f\u3002','error',true);managerFutureSetMinutes.focus();return}requestManagerFutureSet.disabled=true;requestManagerFutureSet.setAttribute('aria-busy','true');announce(managerFutureSetSummary,'\u6b63\u5728\u63d0\u4ea4\u8eab\u4efd\u7ed1\u5b9a\u7684\u8d5b\u524d\u672a\u6765\u5b9e\u9a8c\u2026\u2026');try{const key=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random(),data=await api('/api/v1/seasons/world-model-future-set',{method:'POST',headers:{'Idempotency-Key':key},body:JSON.stringify({fixture_id:managed.fixture_id,branch_times_sec:values.map(value=>value*60)})});activePoll=data.task.task_id;await pollTask(data.task.task_id)}catch(error){announce(managerFutureSetSummary,error.message,'error',true)}finally{requestManagerFutureSet.disabled=false;requestManagerFutureSet.setAttribute('aria-busy','false')}}
async function reviewManagerFutureEvidence(item,intent,button){button.disabled=true;button.setAttribute('aria-busy','true');try{const managed=currentSeason?.next_manager_fixture,payload={task_id:item.task_id,fixture_id:managed?.fixture_id,expected_revision:currentSeason?.revision,intent};if(intent==='revise_after_review'){const decisionPayload=managerDecisionPayload(true);if(decisionPayload.expected_revision!==currentSeason?.revision)throw new Error('\u5f53\u524d\u65b9\u6848\u9884\u89c8\u5df2\u8fc7\u671f\uff0c\u8bf7\u7b49\u5f85\u5237\u65b0\u3002');payload.decision=decisionPayload.decision}announce(managerFutureSetSummary,'\u6b63\u5728\u8bb0\u5f55\u8eab\u4efd\u7ed1\u5b9a\u7684\u590d\u6838\u6536\u636e\u2026\u2026');await api('/api/v1/seasons/world-model-future-review',{method:'POST',body:JSON.stringify(payload)});await refresh()}catch(error){announce(managerFutureSetSummary,error.message,'error',true)}finally{button.disabled=false;button.setAttribute('aria-busy','false')}}
const renderManagerFutureSetsBase=renderManagerFutureSets;
function renderManagerFutureSetsReviewActions(season,managed){const sets=season?.manager_future_sets||[],nodes=[...managerFutureSetList.children];for(const [index,item] of sets.entries()){const node=nodes[index],receipt=item.review_receipt;if(!node)continue;const evidence=document.createElement('p');evidence.className='status';evidence.textContent='\u673a\u5236\u8bc1\u636e\uff1a\u52a8\u4f5c\u5206\u53c9 '+Number(item.action_divergence_scenarios||0)+' \u4e2a\u65f6\u70b9 \u00b7 \u5c40\u90e8\u5f52\u56e0 '+Number(item.local_attribution_scenarios||0)+' \u00b7 \u540e\u7eed\u63cf\u8ff0\u5dee\u5f02 '+Number(item.descriptive_future_difference_scenarios||0);node.append(evidence);if(receipt){const reviewed=document.createElement('p');reviewed.className='status';reviewed.textContent=(receipt.intent==='keep_after_review'?'\u7ecf\u7406\u590d\u6838\u540e\u4fdd\u7559\u51b3\u7b56':'\u7ecf\u7406\u590d\u6838\u540e\u4fee\u6539\u51b3\u7b56')+' \u00b7 \u6536\u636e '+String(receipt.review_identity||'').slice(0,12)+' \u00b7 \u4e0d\u4ee3\u8868\u51b3\u7b56\u8d28\u91cf\u6216\u8d5b\u679c\u56e0\u679c';node.append(reviewed);continue}if(item.state!=='completed'||!item.binding_current)continue;const actions=document.createElement('p'),keep=document.createElement('button'),revise=document.createElement('button');keep.type=revise.type='button';keep.textContent='\u590d\u6838\u540e\u4fdd\u7559\u5f53\u524d\u51b3\u7b56';revise.textContent='\u6309\u5f53\u524d\u8868\u5355\u4fee\u6539\u5e76\u8bb0\u5f55\u590d\u6838';keep.addEventListener('click',()=>void reviewManagerFutureEvidence(item,'keep_after_review',keep));revise.addEventListener('click',()=>void reviewManagerFutureEvidence(item,'revise_after_review',revise));actions.append(keep,document.createTextNode(' '),revise);node.append(actions)}}
function renderManagerFutureSetsScenarioEvidence(season,managed){const sets=season?.manager_future_sets||[],nodes=[...managerFutureSetList.children],labels={descriptive_only_ineligible:'资格未通过，仅描述',no_realized_action_divergence:'策略介入但动作未分叉',action_divergence_without_local_attribution:'动作分叉，局部归因不足',local_action_divergence_with_descriptive_future_difference:'局部归因成立，未来出现描述差异',local_action_divergence_without_measured_future_difference:'局部归因成立，已测未来未变化'};for(const [index,item] of sets.entries()){const node=nodes[index],scenarios=item.scenario_evidence||[];if(!node||!scenarios.length)continue;const details=document.createElement('details'),heading=document.createElement('summary');heading.textContent='展开分叉机制链 · '+scenarios.length+' 个预注册时点';details.append(heading);for(const scenario of scenarios){const line=document.createElement('p'),identity=String(scenario.scenario_identity||'').slice(0,12);line.className='status';line.textContent=Number(scenario.branch_minute||0).toFixed(1)+' 分钟 · 前缀'+(scenario.anchor_verified?'已验证':'未验证')+' · '+(labels[scenario.future_status]||scenario.future_status)+' · 动作 '+Number(scenario.changed_actions||0)+' · 局部归因 '+Number(scenario.locally_attributable_changes||0)+' · 后续差异 '+Number(scenario.descriptive_future_difference_count||0)+' · '+identity;details.append(line)}const boundary=document.createElement('p');boundary.className='status';boundary.textContent='每行身份随复核收据封存；时点之间不排名，后续差异不授予赛果因果。';details.append(boundary);node.append(details)}}
const MANAGER_FUTURE_SET_RENDER_STAGES=Object.freeze([renderManagerFutureSetsBase,renderManagerFutureSetsReviewActions,renderManagerFutureSetsScenarioEvidence,renderManagerFutureSetsMechanismExamples]);
renderManagerFutureSets=(season,managed)=>{for(const renderStage of MANAGER_FUTURE_SET_RENDER_STAGES)renderStage(season,managed)};
function renderManagerFutureSetsMechanismExamples(season,managed){const sets=season?.manager_future_sets||[],nodes=[...managerFutureSetList.children];for(const [index,item] of sets.entries()){const node=nodes[index];appendFutureMechanismExamples(node,item.scenario_evidence||[],'查看具体动作采用链');if(node&&item.state==='interrupted'&&item.binding_current){const resume=document.createElement('button');resume.type='button';resume.textContent='从已验证进度恢复未来生成';resume.addEventListener('click',()=>void resumeInterruptedTask(item.task_id,resume));node.append(resume)}}}
function managerNavigationTarget(id){if(!id)return null;const host=document.getElementById(id);if(!host)return null;let target=host;if(id==='manager-future-set-list')target=host.querySelector('button:not(:disabled)');else if(!host.matches('button,input,select,textarea,a[href],[tabindex]'))target=host.querySelector('button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),a[href],[tabindex]');return target&&!target.disabled&&!target.closest('[hidden]')?target:null}
function activateManagerNavigationTarget(id){const target=managerNavigationTarget(id);if(!target){announce(managerWorldNavigatorSummary,'当前导航目标不可用，请刷新赛季状态。','error',true);return false}target.scrollIntoView({block:'center',behavior:'smooth'});target.focus();return true}
function syncManagerWorldChapterSelection(fixtureId){for(const item of managerDecisionLedgerList.children){if(item.dataset.fixtureId===fixtureId)item.setAttribute('aria-current','location');else item.removeAttribute('aria-current')}for(const button of managerWorldNavigatorHistory.querySelectorAll('button[data-fixture-id]'))button.setAttribute('aria-pressed',String(button.dataset.fixtureId===fixtureId))}
function openManagerWorldChapter(fixtureId,{updateLocation=true,reportMissing=true}={}){const seasonId=String(currentSeason?.season_id||''),target=[...managerDecisionLedgerList.children].find(item=>item.dataset.fixtureId===fixtureId&&item.dataset.chapterComplete==='true');if(!seasonId||!target){syncManagerWorldChapterSelection(null);if(reportMissing)announce(managerWorldNavigatorSummary,'请求的完整世界章节当前不可用，请刷新赛季状态。','error',true);return false}for(const detail of target.querySelectorAll('details'))detail.open=true;const selectedRef=managerWorldChapterFromLocation();if(updateLocation&&(selectedRef?.seasonId!==seasonId||selectedRef?.fixtureId!==fixtureId)){const url=new URL(location.href);url.hash=managerWorldChapterHashPrefix.slice(1)+encodeURIComponent(seasonId)+'/'+encodeURIComponent(fixtureId);history.pushState({seasonId,worldChapter:fixtureId},'',url)}syncManagerWorldChapterSelection(fixtureId);target.scrollIntoView({block:'center',behavior:'smooth'});target.focus();return true}
function navigateManagerWorldChapter(fixtureId,expectedChapterIdentity=''){const seasonId=String(currentSeason?.season_id||''),entries=currentSeason?.manager_decision_ledger?.entries||[],completed=entries.some(row=>row.fixture_id===fixtureId&&row.lifecycle_state!=='frozen_awaiting_execution'),gapRows=currentSeason?.manager_world_navigator?.summary?.continuity_gap_counts||[],stateRows=currentSeason?.manager_world_navigator?.summary?.world_model_action_adoption_ledger?.state_counts||[],trajectoryRows=currentSeason?.manager_world_navigator?.world_trajectory?.points||[],diagnosticRows=gapRows.concat(stateRows,trajectoryRows),bindingCurrent=!expectedChapterIdentity||diagnosticRows.some(row=>(row.latest_fixture_id===fixtureId&&row.latest_chapter_identity===expectedChapterIdentity)||(row.fixture_id===fixtureId&&row.chapter_identity===expectedChapterIdentity));if(!seasonId||!completed||!bindingCurrent){announce(managerWorldNavigatorSummary,'诊断绑定的世界章节已过期或不可用。','error',true);return false}const selectedRef=managerWorldChapterFromLocation();if(selectedRef?.seasonId!==seasonId||selectedRef?.fixtureId!==fixtureId){const url=new URL(location.href);url.hash=managerWorldChapterHashPrefix.slice(1)+encodeURIComponent(seasonId)+'/'+encodeURIComponent(fixtureId);history.pushState({seasonId,worldChapter:fixtureId},'',url)}renderManagerDecisionLedger(currentSeason);renderManagerWorldNavigator(currentSeason);return true}
function restoreManagerWorldChapterLocation(){const chapterHashPresent=String(location.hash||'').startsWith(managerWorldChapterHashPrefix),selectedRef=managerWorldChapterFromLocation();if(!selectedRef){syncManagerWorldChapterSelection(null);if(chapterHashPresent)announce(managerWorldNavigatorSummary,'世界章节链接格式无效，未打开任何章节。','error',true);return false}if(selectedRef.seasonId!==currentSeason?.season_id){syncManagerWorldChapterSelection(null);announce(managerWorldNavigatorSummary,'该世界章节链接属于另一个赛季，未打开任何章节。','error',true);return false}return openManagerWorldChapter(selectedRef.fixtureId,{updateLocation:false,reportMissing:true})}
function renderManagerWorldNavigatorBase(season){managerWorldNavigatorCurrent.replaceChildren();managerWorldInfluencePath.replaceChildren();managerWorldGapList.replaceChildren();managerWorldGapDiagnostics.hidden=true;managerWorldNavigatorSecondary.replaceChildren();managerWorldNavigatorHistory.replaceChildren();managerWorldNavigatorGaps.textContent='';managerWorldNavigatorAction.hidden=true;managerWorldNavigatorAction.disabled=false;managerWorldNavigatorAction.removeAttribute('data-target');const navigator=season?.manager_world_navigator;if(!navigator?.available){managerWorldNavigator.hidden=true;managerWorldNavigatorSummary.textContent='';return}managerWorldNavigator.hidden=false;const summary=navigator.summary||{},current=navigator.current_chapter||{},fixture=current.fixture||{},stateLabels={decision_required:'等待经理决策',ready_to_explore:'决策已冻结，可探索未来',future_generation_in_progress:'有界未来生成中',future_generation_interrupted:'未来生成已中断',future_generation_failed:'未来生成失败',evidence_ready_for_review:'未来证据等待复核',review_recorded_decision_refrozen:'复核完成，等待正式比赛',season_complete:'赛季世界已封存'},stageStatusLabels={decision_required:'等待决策',blocked_by_decision:'等待冻结决策',decision_frozen:'已冻结',available:'可开始',running:'生成中',interrupted:'已中断',failed:'已失败',evidence_complete:'证据完成',scenario_evidence_available:'机制证据可检查',aggregate_only:'仅有聚合证据',awaiting_future_set:'等待未来集',review_available:'等待经理复核',review_recorded:'已复核',optional_not_started:'可选，尚未开始',ready_for_official_match:'正式比赛可推进'};managerWorldNavigatorSummary.textContent=(stateLabels[current.workflow_state]||current.workflow_state)+' · 已完成世界 '+Number(summary.completed_world_chapters||0)+' · 完整正式链 '+Number(summary.official_runtime_chapters||0)+' · 完整世界模型链 '+Number(summary.world_model_runtime_chapters||0)+' · 局部动作改变 '+Number(summary.local_action_changes||0)+' · 证据断点章节 '+Number(summary.chapters_with_continuity_gaps||0);const influence=summary.world_model_influence_path||{},influenceLabels=[['review_selected','复核进入最终选择'],['reviewed_future_scenario_evidence','复核保留逐时点分叉证据'],['reviewed_future_action_divergence','复核未来中观察到动作分叉'],['reviewed_future_local_attribution','复核未来中具有局部归因'],['reviewed_future_timing_sensitivity','复核未来中观察到时间敏感性'],['runtime_tactic_verified','战术运行时验证'],['official_action_evidence','正式动作证据可用'],['chapters_with_local_action_changes','至少一处局部动作改变'],['persistent_world_transitions','持久世界状态可用'],['complete_world_model_runtime_chains','完整世界模型运行链']];for(const [key,label] of influenceLabels){const item=document.createElement('li'),title=document.createElement('strong'),coverage=document.createElement('span');title.textContent=label;coverage.className='status';coverage.textContent=' · '+Number(influence[key]||0)+' / '+Number(summary.completed_world_chapters||0)+' 场';item.append(title,coverage);managerWorldInfluencePath.append(item)}const gapRows=summary.continuity_gap_counts||[],gapLabels={review_to_final_decision_continuity_broken:'复核未进入最终决策',tactical_runtime_binding_unavailable:'正式战术绑定证据缺失',world_model_action_evidence_unavailable:'世界模型动作证据缺失',observed_result_unavailable:'观察赛果证据缺失',persistent_world_state_unavailable:'持久世界状态缺失'},gapGuidance={review_to_final_decision_continuity_broken:'未来比赛应在最后一次编辑后重新复核并冻结',tactical_runtime_binding_unavailable:'检查比赛报告与战术运行时绑定',world_model_action_evidence_unavailable:'检查比赛是否启用世界模型并保留动作证据身份',observed_result_unavailable:'检查比赛报告是否完整持久化',persistent_world_state_unavailable:'检查赛后与恢复状态快照'};managerWorldGapDiagnostics.hidden=!Number(summary.completed_world_chapters||0);managerWorldGapSummary.textContent=gapRows.length?'赛季证据断点诊断 · '+Number(summary.chapters_with_continuity_gaps||0)+' 场 / '+gapRows.length+' 类':'赛季证据断点诊断 · '+Number(summary.chapters_without_continuity_gaps||0)+' 场无已声明断点';if(gapRows.length){for(const row of gapRows){const code=String(row.gap||''),label=gapLabels[code]||'未知断点 · '+code,node=card(label,Number(row.chapters||0)+' 场 · '+(gapGuidance[code]||'检查对应章节的完整证据时间线')+' · '+code),open=document.createElement('button'),fixtureId=String(row.latest_fixture_id||''),chapterIdentity=String(row.latest_chapter_identity||''),matchday=Number(row.latest_matchday);node.setAttribute('role','listitem');open.type='button';open.className='secondary';open.textContent='打开最近受影响章节 · 第 '+matchday+' 轮';open.setAttribute('aria-label',label+'：打开最近受影响章节，第 '+matchday+' 轮');open.dataset.fixtureId=fixtureId;open.dataset.chapterIdentity=chapterIdentity;open.disabled=!fixtureId||matchday<1||!Number.isInteger(matchday)||!/^[0-9a-f]{64}$/.test(chapterIdentity);open.addEventListener('click',()=>navigateManagerWorldChapter(open.dataset.fixtureId,open.dataset.chapterIdentity));node.append(open);managerWorldGapList.append(node)}}else if(Number(summary.completed_world_chapters||0)){const none=document.createElement('p');none.className='status';none.setAttribute('role','listitem');none.textContent='所有已完成章节均无已声明连续性断点。';managerWorldGapList.append(none)}const heading=document.createElement('p'),flow=document.createElement('ol'),boundary=document.createElement('p');heading.className='status';heading.textContent=fixture.fixture_id?'当前位置：第 '+Number(fixture.matchday||0)+' 轮 · '+fixture.home+' vs '+fixture.away+' · 会话 '+String(current.workspace_identity||'').slice(0,12):'当前位置：赛季已完成';flow.className='journal-list';flow.setAttribute('aria-label','当前足球世界阶段');const stageLabels={freeze_intervention:'冻结干预',generate_bounded_futures:'生成未来',inspect_local_mechanism:'检查机制',record_manager_review:'经理复核',advance_official_world:'正式世界'};for(const stage of current.stages||[]){const item=document.createElement('li'),title=document.createElement('strong'),status=document.createElement('span');title.textContent=stageLabels[stage.stage_id]||stage.stage_id;status.className='status';status.textContent=' · '+(stageStatusLabels[stage.status]||stage.status);item.append(title,status);flow.append(item)}boundary.className='status';boundary.textContent='导航只连接当前干预会话与已完成的可回放世界；章节完整不代表赛果改善或现实足球因果。';managerWorldNavigatorCurrent.append(heading,flow,boundary);const currentGaps=current.continuity_gaps||[];managerWorldNavigatorGaps.className='status'+(currentGaps.length?' error':'');managerWorldNavigatorGaps.textContent=currentGaps.length?'当前会话连续性断点：'+currentGaps.join(' · '):'当前会话无已声明连续性断点。';const primary=navigator.primary_action;if(primary){managerWorldNavigatorAction.hidden=false;managerWorldNavigatorAction.textContent=primary.label;managerWorldNavigatorAction.disabled=!primary.target_element_id;if(primary.target_element_id)managerWorldNavigatorAction.dataset.target=primary.target_element_id}for(const action of navigator.secondary_actions||[]){const button=document.createElement('button');button.type='button';button.className='secondary';button.textContent=action.label;button.dataset.actionId=String(action.action_id||'');button.disabled=!action.target_element_id;if(action.target_element_id)button.dataset.target=action.target_element_id;button.addEventListener('click',()=>activateManagerNavigationTarget(button.dataset.target));managerWorldNavigatorSecondary.append(button)}for(const chapter of (navigator.history_chapters||[]).slice(0,6)){const score=chapter.score||{},node=card('第 '+Number(chapter.matchday||0)+' 轮 · '+chapter.fixture_id,(score.home===undefined?'结果不可用':score.home+'–'+score.away)),facts=document.createElement('p'),gaps=document.createElement('p');node.setAttribute('role','listitem');facts.className=gaps.className='status';facts.textContent=(chapter.world_model_runtime_chain_complete?'世界模型运行链完整':chapter.official_runtime_chain_complete?'正式运行链完整':'证据链不完整')+' · 局部动作改变 '+Number(chapter.locally_attributable_action_changes||0)+' · 世界状态 '+(chapter.persistent_transition_identity?'已持久化':'不可用')+' · 章节 '+String(chapter.chapter_identity||'').slice(0,12);gaps.textContent=(chapter.continuity_gaps||[]).length?'断点：'+chapter.continuity_gaps.join(' · '):'无已声明证据断点';const open=document.createElement('button');open.type='button';open.className='secondary';open.textContent='打开完整世界章节';open.dataset.fixtureId=String(chapter.fixture_id||'');open.setAttribute('aria-pressed','false');open.addEventListener('click',()=>openManagerWorldChapter(open.dataset.fixtureId));node.append(facts,gaps,open);managerWorldNavigatorHistory.append(node)}restoreManagerWorldChapterLocation()}
function renderManagerWorldActionAdoptionLedgerBase(season){managerWorldActionAdoptionMetrics.replaceChildren();managerWorldActionAdoptionStates.replaceChildren();const summary=season?.manager_world_navigator?.summary||{},ledger=summary.world_model_action_adoption_ledger,completed=Number(summary.completed_world_chapters||0);if(!ledger||!completed){managerWorldActionAdoptionSummary.textContent='尚无已完成世界可形成动作采纳账本。';managerWorldActionAdoptionDiagnostics.hidden=true;return}const rate=value=>value===null||value===undefined?'无分母':(Number(value)*100).toFixed(1)+'%',metrics=[['保留动作决策',Number(ledger.retained_records||0)],['非零概率影响',Number(ledger.influenced_decisions||0)],['局部动作改变',Number(ledger.locally_attributable_action_changes||0)],['球事件身份匹配',Number(ledger.direct_ball_event_links||0)]];for(const [label,value] of metrics){const node=card(label,String(value));node.setAttribute('role','listitem');managerWorldActionAdoptionMetrics.append(node)}managerWorldActionAdoptionSummary.textContent='动作证据 '+Number(ledger.fixtures_with_action_evidence||0)+' / '+completed+' 场 · 完整记录覆盖 '+Number(ledger.fixtures_with_complete_record_coverage||0)+' 场 · 记录截断 '+Number(ledger.fixtures_with_incomplete_record_coverage||0)+' 场 · 影响率 '+rate(ledger.influence_rate)+' · 受影响决策中的动作改变率 '+rate(ledger.realized_change_rate_among_influenced)+' · 可解析球事件链接覆盖 '+rate(ledger.direct_ball_event_link_coverage)+' · 未解析 '+Number(ledger.unresolved_ball_event_links||0);const rows=ledger.state_counts||[],labels={realized_action_change:'发生局部动作改变',probability_influence_only:'概率受影响但采样动作未改变',no_nonzero_influence:'未观察到非零动作影响',not_applicable_stable_mode:'稳定模式不适用世界模型动作层',evidence_unavailable:'正式动作证据不可用'};managerWorldActionAdoptionDiagnostics.hidden=!rows.length;for(const row of rows){const state=String(row.state||''),label=labels[state]||'未知采纳状态 · '+state,fixtureId=String(row.latest_fixture_id||''),chapterIdentity=String(row.latest_chapter_identity||''),matchday=Number(row.latest_matchday),node=card(label,Number(row.chapters||0)+' 场 · 最近第 '+matchday+' 轮'),open=document.createElement('button');node.setAttribute('role','listitem');open.type='button';open.className='secondary';open.textContent='打开最近该状态章节 · 第 '+matchday+' 轮';open.setAttribute('aria-label',label+'：打开最近章节，第 '+matchday+' 轮');open.dataset.fixtureId=fixtureId;open.dataset.chapterIdentity=chapterIdentity;open.disabled=!fixtureId||matchday<1||!Number.isInteger(matchday)||!/^[0-9a-f]{64}$/.test(chapterIdentity);open.addEventListener('click',()=>navigateManagerWorldChapter(open.dataset.fixtureId,open.dataset.chapterIdentity));node.append(open);managerWorldActionAdoptionStates.append(node)}}
function managerWorldPropagationText(world){const outcomes=world?.outcomes||{},metrics=world?.match_metrics_delta_totals||{},changes=world?.transition_summary_totals||{};return '同章后续事实：赛果 '+Number(world?.results_available||0)+' 场（'+Number(outcomes.win||0)+'胜 '+Number(outcomes.draw||0)+'平 '+Number(outcomes.loss||0)+'负，'+Number(world?.points_earned||0)+' 分） · 持久状态 '+Number(world?.persistent_state_chapters||0)+' 场 · 比赛后疲劳总变化 '+Number(metrics.team_fatigue_ema||0).toFixed(3)+' · 士气总变化 '+Number(metrics.squad_morale_ema||0).toFixed(3)+' · 新伤 '+Number(changes.new_injuries||0)+' · 新停赛 '+Number(changes.new_suspensions||0)}
function renderManagerWorldActionAdoptionLedgerWorldPropagation(season){const navigator=season?.manager_world_navigator,ledger=navigator?.summary?.world_model_action_adoption_ledger;if(!ledger)return;const overall=ledger.descriptive_world_after;if(overall)managerWorldActionAdoptionSummary.textContent+=' · 后续赛果 '+Number(overall.results_available||0)+' 场 · 持久状态 '+Number(overall.persistent_state_chapters||0)+' 场';const rows=ledger.state_counts||[],nodes=[...managerWorldActionAdoptionStates.children];for(const [index,row] of rows.entries()){const node=nodes[index],world=row.descriptive_world_after;if(!node||!world)continue;const propagation=document.createElement('p');propagation.className='status';propagation.textContent=managerWorldPropagationText(world);node.insertBefore(propagation,node.lastChild)}const chapters=(navigator.history_chapters||[]).slice(0,6),history=[...managerWorldNavigatorHistory.children],outcomeLabels={win:'胜',draw:'平',loss:'负'};for(const [index,chapter] of chapters.entries()){const node=history[index],world=chapter.descriptive_world_after;if(!node||!world)continue;const line=document.createElement('p'),metrics=world.metrics_delta||{},changes=world.transition_summary||{};line.className='status';line.textContent='同章后续：'+(world.result_available?(outcomeLabels[world.outcome]||world.outcome)+' · '+Number(world.points_earned||0)+' 分':'赛果不可用')+' · '+(world.persistent_state_available?'疲劳 '+Number(metrics.team_fatigue_ema||0).toFixed(3)+' · 士气 '+Number(metrics.squad_morale_ema||0).toFixed(3)+' · 新伤 '+Number(changes.new_injuries||0):'持久状态不可用');node.insertBefore(line,node.lastChild)}}
function renderManagerWorldActionAdoptionLedgerBoundedSemantics(season){const sample=season?.manager_world_navigator?.summary?.world_model_action_adoption_ledger?.bounded_semantic_examples;if(!sample)return;managerWorldActionAdoptionSummary.textContent+=' · 有界语义样例 '+Number(sample.retained_semantic_examples||0)+' · 覆盖完整 '+Number(sample.fixtures_with_complete_coverage||0)+' 场 · 截断 '+Number(sample.fixtures_with_truncated_examples||0)+' 场 · 传中 '+Number(sample.semantic_cross_action_examples||0)+' · 直接偏好 '+Number(sample.semantic_direct_preference_examples||0)+' · 仅抑制 '+Number(sample.semantic_suppression_only_examples||0)+' · 持球再分配 '+Number(sample.semantic_hold_reference_redistribution_examples||0)+' · 全量分布未授权'}
const MANAGER_WORLD_ACTION_ADOPTION_LEDGER_RENDER_STAGES=Object.freeze([renderManagerWorldActionAdoptionLedgerBase,renderManagerWorldActionAdoptionLedgerWorldPropagation,renderManagerWorldActionAdoptionLedgerBoundedSemantics]);
function renderManagerWorldActionAdoptionLedger(season){for(const renderStage of MANAGER_WORLD_ACTION_ADOPTION_LEDGER_RENDER_STAGES)renderStage(season)}
function renderManagerWorldStory(season){const story=season?.manager_world_story;managerWorldStoryStages.replaceChildren();managerWorldStoryOpen.hidden=true;managerWorldStoryOpen.disabled=true;managerWorldStoryOpen.dataset.fixtureId='';managerWorldStoryOpen.dataset.chapterIdentity='';managerWorldStory.hidden=!story?.available;if(!story?.available){managerWorldStorySummary.textContent='';return}const stageLabels={manager_intervention:'\u7ecf\u7406\u5e72\u9884',counterfactual_review:'\u672a\u6765\u590d\u6838',official_action_adoption:'\u6b63\u5f0f\u52a8\u4f5c',persistent_world_transition:'\u4e16\u754c\u5ef6\u7eed',outcome_evidence:'\u8d5b\u679c\u8fb9\u754c'},statusLabels={complete:'\u5df2\u5b8c\u6210',selected:'\u5df2\u9009\u62e9',reviewed_not_selected:'\u5df2\u590d\u6838\u4f46\u672a\u91c7\u7528',not_used:'\u672c\u8f6e\u672a\u4f7f\u7528',changed:'\u52a8\u4f5c\u5df2\u6539\u53d8',influenced_only:'\u4ec5\u6982\u7387\u5f71\u54cd',no_influence:'\u65e0\u975e\u96f6\u5f71\u54cd',not_applicable:'\u7a33\u5b9a\u6a21\u5f0f\u4e0d\u9002\u7528',evidence_unavailable:'\u8bc1\u636e\u4e0d\u53ef\u7528',world_persisted:'\u5df2\u6301\u4e45\u5316',descriptive_result_only:'\u4ec5\u63cf\u8ff0\u8d5b\u679c',pending:'\u5f85\u6267\u884c',action_required:'\u9700\u8981\u51b3\u7b56',available:'\u53ef\u590d\u6838'};for(const stage of story.stages||[]){const item=document.createElement('li');item.className='world-story-step';item.dataset.status=String(stage.status||'pending');if(['action_required','available'].includes(stage.status))item.setAttribute('aria-current','step');item.textContent=(stageLabels[stage.stage_id]||stage.stage_id)+' \u00b7 '+(statusLabels[stage.status]||stage.status);managerWorldStoryStages.append(item)}const source=story.source||{},metrics=story.metrics||{},previous=story.previous_completed,stateLabels={local_action_change_observed:'\u672c\u8f6e\u52a8\u4f5c\u6539\u53d8',probability_influence_only:'\u4e16\u754c\u6a21\u578b\u5f71\u54cd\u4e86\u52a8\u4f5c\u6982\u7387\u4f46\u672a\u8de8\u8fc7\u91c7\u6837\u8fb9\u754c',no_nonzero_influence:'\u672a\u89c2\u5bdf\u5230\u975e\u96f6\u52a8\u4f5c\u5f71\u54cd',not_applicable:'\u8be5\u6a21\u5f0f\u4e0d\u9002\u7528\u52a8\u4f5c\u91c7\u7528',evidence_gap:'\u8bc1\u636e\u94fe\u4e0d\u5b8c\u6574',awaiting_official_world:'\u7b49\u5f85\u6b63\u5f0f\u4e16\u754c'},active=story.view_mode==='active_chapter',prefix=active?'\u5f53\u524d\u4e16\u754c':'\u6700\u8fd1\u5b8c\u6210\u4e16\u754c';managerWorldStorySummary.textContent=prefix+' \u00b7 \u7b2c '+Number(source.matchday||0)+' \u8f6e \u00b7 '+String(source.fixture_id||'')+' \u00b7 '+(stateLabels[story.story_state]||story.story_state)+' \u00b7 \u5c40\u90e8\u52a8\u4f5c\u6539\u53d8 '+Number(metrics.locally_attributable_action_changes||0)+' \u00b7 \u8d5b\u679c\u6539\u5584\u4ecd\u672a\u5efa\u7acb';if(active&&previous?.source){managerWorldStorySummary.textContent+=' \u00b7 \u4e0a\u4e00\u5b8c\u6210\u4e16\u754c \u7b2c '+Number(previous.source.matchday||0)+' \u8f6e\uff1a'+(stateLabels[previous.story_state]||previous.story_state)+'\uff0c\u5c40\u90e8\u52a8\u4f5c\u6539\u53d8 '+Number(previous.metrics?.locally_attributable_action_changes||0)}const openSource=source.navigable?source:previous?.source;if(openSource?.navigable&&openSource.chapter_identity&&openSource.fixture_id){managerWorldStoryOpen.hidden=false;managerWorldStoryOpen.disabled=false;managerWorldStoryOpen.textContent=active?'\u6253\u5f00\u4e0a\u4e00\u8f6e\u5b8c\u6574\u4e16\u754c\u8bc1\u636e':'\u6253\u5f00\u672c\u8f6e\u5b8c\u6574\u4e16\u754c\u8bc1\u636e';managerWorldStoryOpen.dataset.fixtureId=String(openSource.fixture_id);managerWorldStoryOpen.dataset.chapterIdentity=String(openSource.chapter_identity);managerWorldStoryOpen.onclick=()=>navigateManagerWorldChapter(managerWorldStoryOpen.dataset.fixtureId,managerWorldStoryOpen.dataset.chapterIdentity)}}
function renderManagerWorldTrajectory(season){managerWorldTrajectoryList.replaceChildren();const trajectory=season?.manager_world_navigator?.world_trajectory,points=trajectory?.points||[];managerWorldTrajectory.hidden=!points.length;if(!points.length)return;managerWorldTrajectorySummary.textContent='逐轮世界轨迹 · '+Number(trajectory.visible_points||0)+' / '+Number(trajectory.total_points||0)+' 个章节'+(trajectory.points_truncated?' · 仅显示最近窗口':'')+' · 赛果证据 '+Number(trajectory.results_available||0)+' · 持久状态 '+Number(trajectory.persistent_state_chapters||0);const outcomes={win:'胜',draw:'平',loss:'负'},states={realized_action_change:'动作已改变',probability_influence_only:'仅概率影响',no_nonzero_influence:'无非零影响',not_applicable_stable_mode:'稳定模式不适用',evidence_unavailable:'动作证据不可用'},markerLabels={local_action_change:'局部动作改变',result_evidence_unavailable:'赛果缺失',persistent_state_evidence_unavailable:'持久状态缺失',recovery_pending:'恢复待完成',new_injury:'新增伤病',new_suspension:'新增停赛',continuity_gap:'证据链断点'};for(const point of points){const cumulative=point.cumulative||{},metrics=point.match_metrics_delta||{},changes=point.match_transition_summary||{},node=card('第 '+Number(point.matchday||0)+' 轮 · '+String(point.fixture_id||''),(point.result_available?(outcomes[point.outcome]||point.outcome)+' · '+Number(point.points_earned||0)+' 分':'赛果不可用')),facts=document.createElement('p'),running=document.createElement('p'),markers=document.createElement('p'),open=document.createElement('button');node.setAttribute('role','listitem');facts.className=running.className=markers.className='status';facts.textContent=(states[point.action_adoption_state]||point.action_adoption_state)+' · 局部动作改变 '+Number(point.locally_attributable_action_changes||0)+' · '+(point.persistent_state_available?'疲劳 '+Number(metrics.team_fatigue_ema||0).toFixed(3)+' · 士气 '+Number(metrics.squad_morale_ema||0).toFixed(3)+' · 新伤 '+Number(changes.new_injuries||0):'持久状态不可用');running.textContent='赛季累计：'+Number(cumulative.points_earned||0)+' 分 · 赛果证据 '+Number(cumulative.results_available||0)+' 场 · 持久状态 '+Number(cumulative.persistent_state_chapters||0)+' 场 · 动作改变 '+Number(cumulative.locally_attributable_action_changes||0);const labels=(point.world_change_markers||[]).map(value=>markerLabels[value]||value);markers.textContent=labels.length?'变化标记：'+labels.join(' · '):'本章无已声明变化标记';open.type='button';open.className='secondary';open.textContent='打开该轮完整世界章节';open.dataset.fixtureId=String(point.fixture_id||'');open.dataset.chapterIdentity=String(point.chapter_identity||'');open.setAttribute('aria-pressed','false');open.disabled=!open.dataset.fixtureId||!/^[0-9a-f]{64}$/.test(open.dataset.chapterIdentity);open.addEventListener('click',()=>navigateManagerWorldChapter(open.dataset.fixtureId,open.dataset.chapterIdentity));node.append(facts,running,markers,open);managerWorldTrajectoryList.append(node)}}
function renderManagerWorldReviewedFutureContinuityBase(season){const trajectory=season?.manager_world_navigator?.world_trajectory,points=trajectory?.points||[],nodes=[...managerWorldTrajectoryList.children],intentLabels={keep_after_review:'保留原方案',revise_after_review:'修改后冻结'};if(points.length)managerWorldTrajectorySummary.textContent+=' · 赛前未来复核 '+Number(trajectory.reviewed_future_chapters||0)+' 场 · 进入最终选择 '+Number(trajectory.reviewed_future_selected_chapters||0)+' 场';for(const [index,point] of points.entries()){const node=nodes[index],future=point.reviewed_future_context;if(!node||!future)continue;const line=document.createElement('p');line.className='status';line.textContent=future.available?'赛前分叉：'+(intentLabels[future.intent]||'已复核')+' · '+Number(future.eligible_scenarios||0)+' / '+Number(future.fixed_scenario_budget||0)+' 个时点可用 · 动作分叉 '+Number(future.action_divergence_scenarios||0)+' · 局部归因 '+Number(future.local_attribution_scenarios||0)+' · 时间敏感性 '+(future.timing_sensitivity_observed?'已观察到':'未观察到')+' · '+(future.selected_for_fixture?'已进入正式选择':'复核后被覆盖'):'赛前分叉：未复核';node.insertBefore(line,node.lastChild)}}
function renderManagerWorldReviewedFutureMechanismSemantics(season){const trajectory=season?.manager_world_navigator?.world_trajectory,points=trajectory?.points||[],nodes=[...managerWorldTrajectoryList.children];if(points.length)managerWorldTrajectorySummary.textContent+=' · 传中动作 '+Number(trajectory.cross_action_mechanism_examples||0)+' · 直接偏好 '+Number(trajectory.direct_preference_mechanism_examples||0)+' · 仅抑制 '+Number(trajectory.suppression_only_mechanism_examples||0)+' · 持球基准再分配 '+Number(trajectory.hold_reference_redistribution_examples||0)+' · 非零传中窗口 '+Number(trajectory.nonzero_cross_descriptive_windows||0);for(const [index,point] of points.entries()){const node=nodes[index],future=point.reviewed_future_context;if(!node||!future?.available)continue;const line=document.createElement('p');line.className='status';line.textContent='分叉动作语义：传中 '+Number(future.cross_action_mechanism_examples||0)+' · 直接偏好 '+Number(future.direct_preference_mechanism_examples||0)+' · 仅抑制 '+Number(future.suppression_only_mechanism_examples||0)+' · 持球基准再分配 '+Number(future.hold_reference_redistribution_examples||0)+' · 非零传中窗口 '+Number(future.nonzero_cross_descriptive_windows||0);node.insertBefore(line,node.lastChild)}}
function renderManagerWorldReviewedFutureOfficialActionSemantics(season){const points=season?.manager_world_navigator?.world_trajectory?.points||[],nodes=[...managerWorldTrajectoryList.children];for(const [index,point] of points.entries()){const node=nodes[index],sample=point.bounded_official_action_semantics;if(!node||!sample)continue;const line=document.createElement('p');line.className='status';line.textContent='正式动作有界样例：'+Number(sample.retained_semantic_examples||0)+' 条 · '+(sample.semantic_example_coverage_complete?'覆盖本场全部保留记录':sample.semantic_examples_truncated?'样例已截断':'语义样例不可完整覆盖')+' · 传中 '+Number(sample.semantic_cross_action_examples||0)+' · 直接偏好 '+Number(sample.semantic_direct_preference_examples||0)+' · 仅抑制 '+Number(sample.semantic_suppression_only_examples||0)+' · 持球再分配 '+Number(sample.semantic_hold_reference_redistribution_examples||0)+' · 不代表全量分布';node.insertBefore(line,node.lastChild)}}
function renderManagerWorldScenarioArchive(season){const trajectory=season?.manager_world_navigator?.world_trajectory,points=trajectory?.points||[],nodes=[...managerWorldTrajectoryList.children];if(points.length)managerWorldTrajectorySummary.textContent+=' · 已封存分叉 '+Number(trajectory.reviewed_scenario_archives||0)+' 个';for(const [index,point] of points.entries())appendReviewedScenarioArchive(nodes[index],point.reviewed_future_context?.scenarios,'查看该轮赛前分叉档案')}
function renderManagerWorldReviewCertificate(season){const navigator=season?.manager_world_navigator,summary=navigator?.summary||{},trajectory=navigator?.world_trajectory,points=trajectory?.points||[],nodes=[...managerWorldTrajectoryList.children];if(navigator?.available)managerWorldNavigatorSummary.textContent+=' · 复核贯通正式世界 '+Number(summary.reviewed_world_model_runtime_chapters||0)+' 场';if(points.length)managerWorldTrajectorySummary.textContent+=' · 复核贯通 '+Number(trajectory.reviewed_world_model_chain_complete||0)+' 场';for(const [index,point] of points.entries())appendReviewWorldContinuity(nodes[index],point.review_to_official_world)}
function renderManagerWorldReviewInfluence(season){const navigator=season?.manager_world_navigator,summary=navigator?.summary||{},influence=summary.world_model_influence_path||{};if(!navigator?.available)return;const item=document.createElement('li'),title=document.createElement('strong'),coverage=document.createElement('span');title.textContent='复核决策贯通同场正式动作';coverage.className='status';coverage.textContent=' · '+Number(influence.complete_reviewed_world_model_chains||0)+' / '+Number(summary.completed_world_chapters||0)+' 场';item.append(title,coverage);managerWorldInfluencePath.append(item)}
const syncManagerWorldChapterSelectionWithoutTrajectory=syncManagerWorldChapterSelection;
syncManagerWorldChapterSelection=fixtureId=>{syncManagerWorldChapterSelectionWithoutTrajectory(fixtureId);for(const button of managerWorldTrajectoryList.querySelectorAll('button[data-fixture-id]'))button.setAttribute('aria-pressed',String(button.dataset.fixtureId===fixtureId))};
function retainedActionSemanticTextBase(context){const {semantic}=context;if(!semantic)return;const actual=semantic.actual_action_counts||{},primary=semantic.primary_signal_action_counts||{},modes=semantic.signal_mode_counts||{},coverage=semantic.full_source_distribution_authorized?'覆盖全部源动作记录':'仅覆盖当前保留记录（源记录存在截断）';context.text='全量保留记录语义 '+Number(semantic.records||0)+' 条 · 实际动作：传球 '+Number(actual.pass||0)+' / 传中 '+Number(actual.cross||0)+' / 射门 '+Number(actual.shot||0)+' / 持球 '+Number(actual.hold||0)+' / 无动作 '+Number(actual.none||0)+' · 主信号：传球 '+Number(primary.pass||0)+' / 传中 '+Number(primary.cross||0)+' / 射门 '+Number(primary.shot||0)+' / 持球 '+Number(primary.hold||0)+' / 无 '+Number(primary.none||0)+' · 信号模式：直接偏好 '+Number(modes.direct_preference||0)+' / 仅抑制 '+Number(modes.suppression_only||0)+' / 无信号 '+Number(modes.none||0)+' / 旧版未分类 '+Number(modes.legacy_unclassified||0)+' · 持球基准再分配 '+Number(semantic.hold_reference_redistribution_records||0)+' · 直接传中事件 '+Number(semantic.direct_cross_ball_event_links||0)+' · 局部传中改变 '+Number(semantic.locally_attributable_cross_changes||0)+' · '+coverage+' · 不授权赛果归因'}
function retainedActionSemanticTextTransitions(context){const {semantic}=context,matrix=semantic?.locally_attributable_action_transition_counts;if(!matrix)return;const labels={hold:'持球',pass:'传球',cross:'传中',shot:'射门',none:'无动作'},transitions=[];for(const [before,row] of Object.entries(matrix))for(const [after,count] of Object.entries(row||{}))if(Number(count)>0)transitions.push((labels[before]||before)+'→'+(labels[after]||after)+' '+Number(count));context.text+=' · 世界模型局部动作转移：'+(transitions.join(' / ')||'无')}
function appendOfficialActionExecutionRetainedRecordSemantics(context){const {evidence,details}=context,semantic=evidence?.retained_record_semantics;if(!details?.matches('details')||!semantic)return;const line=document.createElement('p');line.className='status';line.textContent=retainedActionSemanticText(semantic);details.insertBefore(line,details.lastElementChild)}
const OFFICIAL_ACTION_EXECUTION_RENDER_STAGES=Object.freeze([appendOfficialActionExecutionBase,appendOfficialActionExecutionPolicySemantics,appendOfficialActionExecutionRetainedRecordSemantics]);
function appendOfficialActionExecution(host,evidence,title){const context={host,evidence,title,details:null};for(const appendStage of OFFICIAL_ACTION_EXECUTION_RENDER_STAGES)appendStage(context)}
function renderManagerDecisionLedgerRetainedRecordSemantics(season){const semantic=season?.manager_decision_ledger?.summary?.world_model_official_action_execution?.retained_record_semantics;if(!semantic||!semantic.fixtures_with_v2_semantics)return;managerDecisionLedgerSummary.textContent+=' · 全量动作语义 '+Number(semantic.retained_records_with_v2_semantics||0)+' 条 / '+Number(semantic.fixtures_with_v2_semantics||0)+' 场 · 动作转移矩阵 '+Number(semantic.fixtures_with_v3_transition_semantics||0)+' 场 · '+(semantic.full_source_transition_distribution_authorized?'全部源转移覆盖':'存在旧版或源记录截断')+' · 不授权赛果归因'}
function appendManagerWorldEvolutionThreadRetainedRecordSemantics(host,thread){const stages=thread?.stages||[],index=stages.findIndex(row=>row.stage_id==='official_world_model_actions'),stage=stages[index],items=host?.lastElementChild?.querySelectorAll('ol li'),item=index>=0?items?.[index]:null;if(!item||!stage?.retained_record_semantics)return;const line=document.createElement('p');line.className='status';line.textContent=retainedActionSemanticText(stage.retained_record_semantics);item.append(line)}
const MANAGER_WORLD_EVOLUTION_THREAD_STAGES=Object.freeze([appendManagerWorldEvolutionThreadBase,appendManagerWorldEvolutionThreadSocietyContinuity,appendManagerWorldEvolutionThreadReviewedFutures,appendManagerWorldEvolutionThreadMechanismSemantics,appendManagerWorldEvolutionThreadOfficialActionSemantics,appendManagerWorldEvolutionThreadScenarioArchive,appendManagerWorldEvolutionThreadReviewWorldCertificate,appendManagerWorldEvolutionThreadRetainedRecordSemantics]);
function appendManagerWorldEvolutionThread(host,thread){for(const appendStage of MANAGER_WORLD_EVOLUTION_THREAD_STAGES)appendStage(host,thread)}
function renderManagerWorldRetainedRecordSemantics(season){const navigator=season?.manager_world_navigator,ledger=navigator?.summary?.world_model_action_adoption_ledger?.retained_record_semantics,points=navigator?.world_trajectory?.points||[],trajectoryNodes=[...managerWorldTrajectoryList.children],chapters=(navigator?.history_chapters||[]).slice(0,6),historyNodes=[...managerWorldNavigatorHistory.children];if(ledger){managerWorldActionAdoptionSummary.textContent+=' · V3 全量语义 '+Number(ledger.records||0)+' 条 / '+Number(ledger.fixtures_with_v2_semantics||0)+' 场 · 动作转移矩阵 '+Number(ledger.fixtures_with_v3_transition_semantics||0)+' 场 · 旧版转移缺失 '+Number(ledger.fixtures_without_v3_transition_semantics||0)+' 场 · '+(ledger.full_source_transition_distribution_authorized?'全部源转移分布已授权':'全源转移分布未授权')+' · 不授权赛果归因'}for(const [index,point] of points.entries()){const node=trajectoryNodes[index],semantic=point.official_retained_record_semantics;if(!node||!semantic)continue;const line=document.createElement('p');line.className='status';line.textContent=retainedActionSemanticText(semantic);node.insertBefore(line,node.lastElementChild)}for(const [index,chapter] of chapters.entries()){const node=historyNodes[index],semantic=chapter.action_adoption?.retained_record_semantics;if(!node||!semantic)continue;const line=document.createElement('p');line.className='status';line.textContent=retainedActionSemanticText(semantic);node.insertBefore(line,node.lastElementChild)}}
function renderManagerWorldTransitionPropagation(season){const propagation=season?.manager_world_navigator?.summary?.world_model_action_adoption_ledger?.local_transition_descriptive_propagation;if(!propagation)return;const rows=propagation.strata||[],labels={hold:'持球',pass:'传球',cross:'传中',shot:'射门',none:'无动作'};managerWorldActionAdoptionSummary.textContent+=' · 动作转移后续事实 '+Number(rows.length)+' 组 · V3覆盖 '+Number(propagation.fixtures_with_v3_transition_semantics||0)+' 场 · 分层可重叠 · 禁止跨层效果比较';for(const row of rows){const before=labels[row.counterfactual_baseline_action]||row.counterfactual_baseline_action,after=labels[row.actual_action]||row.actual_action,node=card(before+'→'+after,Number(row.transition_occurrences||0)+' 次局部转移 · '+Number(row.chapters||0)+' 个世界章节'),world=document.createElement('p'),boundary=document.createElement('p'),open=document.createElement('button');node.setAttribute('role','listitem');world.className=boundary.className='status';world.textContent=managerWorldPropagationText(row.descriptive_world_after);boundary.textContent='同章描述性共现 · '+Number(row.chapters_with_other_local_transitions||0)+' 个章节还含其他转移 · '+(row.full_source_transition_distribution_authorized?'源转移记录覆盖完整':'源转移记录可能截断')+' · 不排名、不比较效果、不归因赛果';open.type='button';open.className='secondary';open.textContent='打开最近该转移章节 · 第 '+Number(row.latest_matchday||0)+' 轮';open.dataset.fixtureId=String(row.latest_fixture_id||'');open.dataset.chapterIdentity=String(row.latest_chapter_identity||'');open.disabled=!open.dataset.fixtureId||!/^[0-9a-f]{64}$/.test(open.dataset.chapterIdentity);open.addEventListener('click',()=>navigateManagerWorldChapter(open.dataset.fixtureId,open.dataset.chapterIdentity));node.append(world,boundary,open);managerWorldActionAdoptionStates.append(node)}}
function renderManagerWorldActionTransitionMap(season){managerWorldActionTransitionMap.hidden=true;managerWorldActionTransitionMapSummary.textContent='';managerWorldActionTransitionTableBody.replaceChildren();managerWorldActionTransitionTable.hidden=false;const navigator=season?.manager_world_navigator,summary=navigator?.summary||{},completed=Number(summary.completed_world_chapters||0);if(!navigator?.available||!completed)return;managerWorldActionTransitionMap.hidden=false;const semantic=summary.world_model_action_adoption_ledger?.retained_record_semantics,v3=Number(semantic?.fixtures_with_v3_transition_semantics||0),missing=Number(semantic?.fixtures_without_v3_transition_semantics||0),matrix=semantic?.locally_attributable_action_transition_counts;if(!matrix||!v3){managerWorldActionTransitionTable.hidden=true;managerWorldActionTransitionMapSummary.textContent='尚无 V3 动作转移证据 · 已完成 '+completed+' 场 · 旧版或缺失矩阵 '+Math.max(missing,completed)+' 场。这里显示的是证据缺口，不是“世界模型没有改变动作”。';return}const actions=['hold','pass','cross','shot','none'],labels={hold:'持球',pass:'传球',cross:'传中',shot:'射门',none:'无动作'};let total=0;for(const before of actions){const row=document.createElement('tr'),heading=document.createElement('th');heading.scope='row';heading.textContent=labels[before];row.append(heading);for(const after of actions){const cell=document.createElement('td'),count=Number(matrix?.[before]?.[after]||0);total+=count;cell.textContent=String(count);cell.dataset.active=String(count>0);cell.dataset.diagonal=String(before===after);cell.setAttribute('aria-label',labels[before]+'转为'+labels[after]+'：'+count+' 次');row.append(cell)}managerWorldActionTransitionTableBody.append(row)}const sourceCoverage=semantic.full_source_transition_distribution_authorized?'全部源转移记录覆盖完整':'全源分布未授权（含旧版证据或源记录截断）';managerWorldActionTransitionMapSummary.textContent='V3 转移矩阵 '+v3+' / '+completed+' 场 · 旧版或缺失 '+missing+' 场 · 局部动作改变 '+total+' 次 · '+sourceCoverage+'。矩阵格是计数，不是效果值。'}
function retainedActionSemanticTextExactExpectation(context){const {semantic}=context;if(!semantic)return;if(semantic.schema_version!==3){context.text+=' · V4共享采样期望不可用（旧版证据）';return}context.text+=' · 共享采样期望动作改变 '+Number(semantic.expected_counterfactual_action_changes||0).toFixed(3)+' · 精确记录 '+Number(semantic.records_with_exact_change_probability||0)+' / '+Number(semantic.records||0)+' · '+(semantic.full_source_expectation_authorized?'全源精确期望已授权':'精确期望仅覆盖保留记录')+' · 概率质量不是额外观察事件'}
const RETAINED_ACTION_SEMANTIC_TEXT_STAGES=Object.freeze([retainedActionSemanticTextBase,retainedActionSemanticTextTransitions,retainedActionSemanticTextExactExpectation]);
function retainedActionSemanticText(semantic){const context={semantic,text:''};for(const textStage of RETAINED_ACTION_SEMANTIC_TEXT_STAGES)textStage(context);return context.text}
function renderManagerDecisionLedgerExactActionExpectation(season){const semantic=season?.manager_decision_ledger?.summary?.world_model_official_action_execution?.retained_record_semantics;if(!semantic?.fixtures_with_v4_expectation_semantics)return;managerDecisionLedgerSummary.textContent+=' · 共享采样期望改变 '+Number(semantic.expected_counterfactual_action_changes||0).toFixed(3)+' / 精确证据 '+Number(semantic.fixtures_with_v4_expectation_semantics||0)+' 场 · 旧版缺失 '+Number(semantic.fixtures_without_v4_expectation_semantics||0)+' 场'}
const MANAGER_DECISION_LEDGER_RENDER_STAGES=Object.freeze([renderManagerDecisionLedgerBase,renderManagerDecisionLedgerAdvisorEvidence,renderManagerDecisionLedgerExecutionTrace,renderManagerDecisionLedgerFutureReviews,renderManagerDecisionLedgerFutureScenarioEvidence,renderManagerDecisionLedgerMechanismExamples,renderManagerDecisionLedgerMechanismSemantics,renderManagerDecisionLedgerFutureReviewExecution,renderManagerDecisionLedgerOfficialActionExecution,renderManagerDecisionLedgerWorldEvolutionThread,renderManagerDecisionLedgerMetaLearningSummary,renderManagerDecisionLedgerRetainedRecordSemantics,renderManagerDecisionLedgerExactActionExpectation]);
function renderManagerDecisionLedger(season){for(const renderStage of MANAGER_DECISION_LEDGER_RENDER_STAGES)renderStage(season)}
function renderManagerWorldExactActionExpectation(season){const navigator=season?.manager_world_navigator,completed=Number(navigator?.summary?.completed_world_chapters||0),semantic=navigator?.summary?.world_model_action_adoption_ledger?.retained_record_semantics;if(!navigator?.available||!completed||!semantic)return;const exactFixtures=Number(semantic.fixtures_with_v4_expectation_semantics||0),missing=Number(semantic.fixtures_without_v4_expectation_semantics||0);if(exactFixtures){const node=card('共享采样期望改变',Number(semantic.expected_counterfactual_action_changes||0).toFixed(3)+' · '+exactFixtures+'/'+completed+' 场精确');node.setAttribute('role','listitem');managerWorldActionAdoptionMetrics.append(node);managerWorldActionAdoptionSummary.textContent+=' · V4精确期望 '+exactFixtures+' 场 · 旧版缺失 '+missing+' 场 · '+(semantic.full_source_expectation_authorized?'全源精确期望已授权':'只授权已保留精确记录')}else{managerWorldActionAdoptionSummary.textContent+=' · V4共享采样期望不可用：'+missing+' 场为旧版或缺失证据，不能解释为零影响'}}
function renderManagerWorldActionTransitionDrilldown(season){managerWorldActionTransitionDetail.hidden=true;managerWorldActionTransitionSelection.textContent='';managerWorldActionTransitionWorld.textContent='';managerWorldActionTransitionBoundary.textContent='';managerWorldActionTransitionOpen.disabled=true;managerWorldActionTransitionOpen.dataset.fixtureId='';managerWorldActionTransitionOpen.dataset.chapterIdentity='';const propagation=season?.manager_world_navigator?.summary?.world_model_action_adoption_ledger?.local_transition_descriptive_propagation,rows=propagation?.strata||[],byTransition=new Map(rows.map(row=>[row.transition_id,row])),actions=['hold','pass','cross','shot','none'],labels={hold:'持球',pass:'传球',cross:'传中',shot:'射门',none:'无动作'},cells=[...managerWorldActionTransitionTableBody.querySelectorAll('td')];let index=0;for(const before of actions)for(const after of actions){const cell=cells[index++];if(!cell)continue;const count=Number(cell.textContent||0),transitionId=before+'_to_'+after,evidence=byTransition.get(transitionId),button=document.createElement('button');button.type='button';button.className='transition-cell-button';button.textContent=String(count);button.dataset.transitionId=transitionId;button.disabled=count<=0||!evidence;button.setAttribute('aria-controls','manager-world-action-transition-detail');button.setAttribute('aria-pressed','false');button.setAttribute('aria-label',labels[before]+'转为'+labels[after]+'：'+count+' 次'+(count>0&&!evidence?'，后续传播证据缺失':'，查看后续世界证据'));button.addEventListener('click',()=>{for(const candidate of managerWorldActionTransitionTableBody.querySelectorAll('.transition-cell-button'))candidate.setAttribute('aria-pressed','false');button.setAttribute('aria-pressed','true');managerWorldActionTransitionSelection.textContent=labels[before]+' → '+labels[after]+' · '+Number(evidence.transition_occurrences||0)+' 次局部转移 · '+Number(evidence.chapters||0)+' 个世界章节';managerWorldActionTransitionWorld.textContent=managerWorldPropagationText(evidence.descriptive_world_after);managerWorldActionTransitionBoundary.textContent='同章描述性共现 · '+Number(evidence.chapters_with_other_local_transitions||0)+' 个章节还含其他转移 · '+(evidence.full_source_transition_distribution_authorized?'源转移记录覆盖完整':'源转移记录可能截断')+' · 不排名、不比较跨格效果、不归因赛果';managerWorldActionTransitionOpen.disabled=false;managerWorldActionTransitionOpen.dataset.fixtureId=String(evidence.latest_fixture_id||'');managerWorldActionTransitionOpen.dataset.chapterIdentity=String(evidence.latest_chapter_identity||'');managerWorldActionTransitionOpen.disabled=!managerWorldActionTransitionOpen.dataset.fixtureId||!/^[0-9a-f]{64}$/.test(managerWorldActionTransitionOpen.dataset.chapterIdentity);managerWorldActionTransitionOpen.textContent='打开最近相关世界章节 · 第 '+Number(evidence.latest_matchday||0)+' 轮';managerWorldActionTransitionDetail.hidden=false;managerWorldActionTransitionDetail.focus()});cell.replaceChildren(button)}if(rows.length)managerWorldActionTransitionMapSummary.textContent+=' 选择任一非零格可检查同章后续世界事实。'}
function renderManagerWorldReviewedFutureContinuity(season){renderManagerWorldReviewedFutureContinuityBase(season);renderManagerWorldReviewedFutureMechanismSemantics(season);renderManagerWorldReviewedFutureOfficialActionSemantics(season)}
const MANAGER_WORLD_RENDER_STAGES=Object.freeze([renderManagerWorldNavigatorBase,renderManagerWorldStory,renderManagerWorldActionAdoptionLedger,renderManagerWorldTrajectory,renderManagerWorldReviewedFutureContinuity,renderManagerWorldScenarioArchive,renderManagerWorldReviewCertificate,renderManagerWorldReviewInfluence,renderManagerWorldRetainedRecordSemantics,renderManagerWorldTransitionPropagation,renderManagerWorldActionTransitionMap,renderManagerWorldExactActionExpectation,renderManagerWorldActionTransitionDrilldown]);
function renderManagerWorldNavigator(season){for(const renderStage of MANAGER_WORLD_RENDER_STAGES)renderStage(season)}
managerWorldActionTransitionOpen.addEventListener('click',()=>navigateManagerWorldChapter(managerWorldActionTransitionOpen.dataset.fixtureId,managerWorldActionTransitionOpen.dataset.chapterIdentity));
managerWorldNavigatorAction.addEventListener('click',()=>activateManagerNavigationTarget(managerWorldNavigatorAction.dataset.target));
window.addEventListener('popstate',()=>{if(currentSeason){renderManagerDecisionLedger(currentSeason);renderManagerWorldNavigator(currentSeason)}else restoreManagerWorldChapterLocation()});
const SEASON_RENDER_STAGES=Object.freeze([renderSeasonBase,renderSeasonCommandCenter,renderSeasonClubTimeline,renderSeasonManagerFutureSets]);
renderSeason=(season,configured,history=[],historySummary={})=>{for(const renderStage of SEASON_RENDER_STAGES)renderStage(season,configured,history,historySummary)};
function renderSeasonManagerFutureSets(season,configured,history=[],historySummary={}){renderManagerFutureSets(season,season?.next_manager_fixture);renderManagerInterventionWorkspace(season);renderManagerWorldNavigator(season)}
managerManual.addEventListener('change',()=>renderManagerSquad(currentSeason,currentSeason?.next_manager_fixture));
managerRotation.addEventListener('change',()=>renderManagerSquad(currentSeason,currentSeason?.next_manager_fixture));
requestManagerAdvice.addEventListener('click',()=>void requestManagerWorldModelAdvice());
requestManagerFutureSet.addEventListener('click',()=>void requestManagerFutureExperiment());
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
forkForm.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(forkForm),minutes=String(f.get('branch_minutes')||'').split(',').map(value=>Number(value.trim())),plan={seed:Number(f.get('seed')),home_tactic:f.get('home_tactic'),away_tactic:f.get('away_tactic'),branch_times_sec:minutes.map(value=>value*60)},key=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random();submit(forkForm,'/api/v1/world-model-fork-sets',{home:f.get('home'),away:f.get('away'),fast:f.get('fast')==='on',plan},{'Idempotency-Key':key})});
studyForm.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(studyForm),focus=f.get('focus_side'),baseline=f.get('baseline_tactic'),treatment=f.get('treatment_tactic'),opponent=f.get('opponent_tactic'),budget=Number(f.get('pair_budget')),start=Number(f.get('seed_start')),seeds=Array.from({length:budget},(_,index)=>start+index),plan={study_id:f.get('study_id'),fixture:{home:f.get('home'),away:f.get('away')},baseline:{home_tactic:focus==='home'?baseline:opponent,away_tactic:focus==='away'?baseline:opponent},treatment:{home_tactic:focus==='home'?treatment:opponent,away_tactic:focus==='away'?treatment:opponent},seeds,fast:f.get('fast')==='on',analysis_plan:{smallest_effect_size:0.05}},key=globalThis.crypto?.randomUUID?.()||String(Date.now())+'-'+Math.random();submit(studyForm,'/api/v1/tactical-studies',{plan},{'Idempotency-Key':key})});
createBackupButton.addEventListener('click',async()=>{createBackupButton.disabled=true;createBackupButton.setAttribute('aria-busy','true');setRecoveryMessage('正在创建并校验备份……');try{await api('/api/v1/backups',{method:'POST'});setRecoveryMessage('备份已创建并通过校验。');await refreshRecovery()}catch(e){setRecoveryMessage(e.message,true,true)}finally{createBackupButton.disabled=false;createBackupButton.setAttribute('aria-busy','false')}});
restoreForm.addEventListener('submit',async e=>{e.preventDefault();const id=restoreForm.dataset.backupId,f=new FormData(restoreForm),button=restoreForm.querySelector('button[type="submit"]');button.disabled=true;setBusy(restoreForm,true);setRecoveryMessage('正在执行事务恢复……');try{await api('/api/v1/backups/'+encodeURIComponent(id)+'/restore',{method:'POST',body:JSON.stringify({confirmation:f.get('confirmation'),replace:f.get('replace')==='on'})});closeRestore(false);setRecoveryMessage('恢复完成，任务历史已安全重置。',false,true);await Promise.all([refresh(),refreshRecovery()])}catch(error){setRecoveryMessage(error.message,true,true)}finally{button.disabled=false;setBusy(restoreForm,false)}});
cancelRestore.addEventListener('click',()=>closeRestore(true));
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!restoreForm.hidden){event.preventDefault();closeRestore(true)}});
logoutButton.addEventListener('click',async()=>{logoutButton.disabled=true;logoutButton.setAttribute('aria-busy','true');try{await api('/api/v1/logout',{method:'POST',body:'{}'});location.replace('/login')}catch(e){announce(message,e.message,'error',true);logoutButton.disabled=false;logoutButton.setAttribute('aria-busy','false')}});
void Promise.all([refresh(),refreshRecovery(),refreshProductValueStudy()]);
</script>
</body></html>"""


_LOGIN_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="gfs-csrf" content="__CSRF__"><title>GFS Studio 登录</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#090d0c;color:#f7f6f0;font:16px/1.5 system-ui,sans-serif}.panel{width:min(440px,calc(100% - 2rem));padding:2rem;border:1px solid #31413a;border-radius:18px;background:#151b19}h1{margin:.2rem 0}.muted{color:#aab2ad}form{display:grid;gap:.8rem;margin-top:1.4rem}label{display:grid;gap:.35rem}input,button{min-block-size:44px;font:inherit;padding:.75rem;border-radius:9px}input{color:inherit;background:#0b100f;border:1px solid #455b52}button{font-weight:750;background:#7ee2a8;border:0;color:#07100c}input:focus-visible,button:focus-visible,[tabindex="-1"]:focus-visible{outline:3px solid #ffd166;outline-offset:3px}.error{color:#ff7b72;min-height:1.5rem}@media(forced-colors:active){input,button,.panel{border:1px solid CanvasText}}
</style></head><body><main class="panel"><p class="muted">Authenticated deployment</p><h1>进入 GFS Studio</h1><p class="muted">请输入独立的产品访问令牌。它不是模型提供商 API Key。</p><form id="login" aria-busy="false"><label>产品访问令牌<input name="token" type="password" autocomplete="current-password" required></label><button type="submit">安全登录</button></form><p id="message" class="error" role="alert" aria-live="assertive" aria-atomic="true" tabindex="-1"></p></main><script nonce="__NONCE__">
const form=document.querySelector('#login'),message=document.querySelector('#message'),csrf=document.querySelector('meta[name="gfs-csrf"]').content;form.addEventListener('submit',async e=>{e.preventDefault();const button=form.querySelector('button'),token=new FormData(form).get('token');button.disabled=true;form.setAttribute('aria-busy','true');message.textContent='';try{const response=await fetch('/api/v1/login',{method:'POST',headers:{'Content-Type':'application/json','X-GFS-CSRF':csrf},body:JSON.stringify({access_token:token})});const data=await response.json();if(!response.ok)throw new Error(data.error?.message||'登录失败');location.replace('/')}catch(error){message.textContent=error.message;message.focus()}finally{button.disabled=false;form.setAttribute('aria-busy','false');form.reset()}});
</script></body></html>"""
