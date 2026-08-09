import io
import json
import threading
from urllib.request import urlopen
from wsgiref.util import setup_testing_defaults

import pytest

from src.cli import build_parser, cmd_studio_web
from src.product.web import ProductWebApp, _is_loopback_host, create_product_web_server
from src.product.web_security import WebAccessPolicy
from src.product.tasks import BackgroundMatchWorker


def _request(
    app, method="GET", path="/", payload=None, *, csrf=None, host=None,
    idempotency_key=None, forwarded_proto=None, cookie=None,
):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    environ = {}
    setup_testing_defaults(environ)
    environ.update({
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    })
    if payload is not None:
        environ["CONTENT_TYPE"] = "application/json"
    if csrf is not None:
        environ["HTTP_X_GFS_CSRF"] = csrf
    if host is not None:
        environ["HTTP_HOST"] = host
    if idempotency_key is not None:
        environ["HTTP_IDEMPOTENCY_KEY"] = idempotency_key
    if forwarded_proto is not None:
        environ["HTTP_X_FORWARDED_PROTO"] = forwarded_proto
    if cookie is not None:
        environ["HTTP_COOKIE"] = cookie
    observed = {}

    def start_response(status, headers):
        observed["status"] = status
        observed["headers"] = dict(headers)

    response = b"".join(app(environ, start_response))
    observed["body"] = response
    if observed["headers"]["Content-Type"].startswith("application/json"):
        observed["json"] = json.loads(response)
    return observed


def test_root_is_accessible_and_hardened(tmp_path):
    response = _request(ProductWebApp(tmp_path))
    document = response["body"].decode("utf-8")
    assert response["status"].startswith("200")
    assert "Content-Security-Policy" in response["headers"]
    assert response["headers"]["X-Frame-Options"] == "DENY"
    assert '<a class="skip" href="#main">' in document
    assert 'id="main"' in document
    assert 'aria-live="polite"' in document
    assert "prefers-reduced-motion" in document
    assert 'id="logout-button" type="button" hidden' in document
    assert 'aria-labelledby="recovery-title"' in document
    assert 'id="backup-list"' in document and 'role="list"' in document
    assert 'id="restore-form" hidden' in document
    assert 'aria-labelledby="release-title"' in document
    assert 'id="release-summary"' in document
    assert 'id="release-gates"' in document and 'role="list"' in document
    assert "renderRelease(data)" in document
    assert "recommended_gate_id" in document
    assert "dataset.actionState" in document
    assert 'name="confirmation"' in document and 'name="replace"' in document
    assert 'type="file"' not in document
    assert "innerHTML" not in document


def test_health_is_liveness_only_and_never_calls_provider(tmp_path):
    response = _request(ProductWebApp(tmp_path), path="/healthz")
    assert response["json"] == {
        "schema_version": 1,
        "status": "ok",
        "service": "gfs-product-web",
        "external_calls_made": False,
        "background_worker_alive": None,
    }


def test_operations_are_aggregated_and_paths_are_privacy_normalized(tmp_path):
    app = ProductWebApp(tmp_path)
    _request(app, path="/api/v1/tasks/private-user-component")
    operations = _request(app, path="/api/v1/operations")
    assert operations["status"].startswith("200")
    metrics = operations["json"]
    assert metrics["requests"]["routes"]["/api/v1/tasks/{task_id}"] == 1
    assert metrics["privacy"]["raw_events_exposed"] is False
    assert "private-user-component" not in json.dumps(metrics)
    assert "events" not in metrics


def test_telemetry_failure_does_not_break_liveness(tmp_path, monkeypatch):
    app = ProductWebApp(tmp_path)
    monkeypatch.setattr(app.telemetry, "record", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    response = _request(app, path="/healthz")
    assert response["status"].startswith("200")
    assert app.telemetry.write_failures == 1


def test_non_loopback_host_header_is_rejected_against_dns_rebinding(tmp_path):
    response = _request(ProductWebApp(tmp_path), path="/api/v1/studio", host="attacker.example")
    assert response["status"].startswith("400")
    assert response["json"]["error"]["code"] == "invalid_host"


def test_web_workflow_requires_csrf_then_creates_and_reads_studio(tmp_path):
    app = ProductWebApp(tmp_path)
    payload = {"name": "Beta Lab", "mode": "stable", "seed": 17}
    rejected = _request(app, "POST", "/api/v1/studio", payload)
    assert rejected["status"].startswith("403")
    assert rejected["json"]["error"]["code"] == "csrf_rejected"

    created = _request(
        app, "POST", "/api/v1/studio", payload, csrf=app.csrf_token,
    )
    assert created["status"].startswith("201")
    assert created["json"]["studio"]["name"] == "Beta Lab"
    assert created["json"]["studio"]["workflow"]["state"] == "blocked"

    status = _request(app, path="/api/v1/studio")
    assert status["json"]["configured"]
    assert status["json"]["studio"]["seed"] == 17
    assert status["headers"]["X-GFS-CSRF-Token"] == app.csrf_token

    duplicate = _request(
        app, "POST", "/api/v1/studio", payload, csrf=app.csrf_token,
    )
    assert duplicate["status"].startswith("409")
    assert duplicate["json"]["error"]["code"] == "studio_exists"


def test_match_route_validates_input_before_domain_execution(tmp_path):
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/matches", {
        "home": "Brazil", "away": "brazil", "fast": True,
    }, csrf=app.csrf_token)
    assert response["status"].startswith("422")
    assert response["json"]["error"]["code"] == "same_team"


def test_match_route_queues_idempotently_then_returns_report_url(tmp_path, monkeypatch):
    dashboard = tmp_path / "outputs/studio/demo/matches/0001.html"
    dashboard.parent.mkdir(parents=True)
    dashboard.write_text("<html><body>report</body></html>", encoding="utf-8")

    class FakeWorkspace:
        def readiness(self):
            return {"ready": True, "blockers": []}

        def run_match(self, home, away, *, fast):
            return {
                "match_id": "0001-brazil-vs-argentina",
                "fixture": {"home": home, "away": away, "fast": fast},
                "result": {"score": {"home": 1, "away": 0}},
                "integrity": {"accepted": True, "state": "accepted", "blockers": []},
                "report_path": str(dashboard.with_suffix(".json")),
                "dashboard_path": str(dashboard),
                "raw_summary": {"large": "must-not-cross-api-boundary"},
            }

    monkeypatch.setattr(
        "src.product.web.ProductWorkspace.load", lambda _root: FakeWorkspace(),
    )
    app = ProductWebApp(tmp_path)
    response = _request(app, "POST", "/api/v1/matches", {
        "home": "Brazil", "away": "Argentina", "fast": True,
    }, csrf=app.csrf_token, idempotency_key="browser-request-1")
    assert response["status"].startswith("202")
    task_id = response["json"]["task"]["task_id"]
    duplicate = _request(app, "POST", "/api/v1/matches", {
        "home": "Brazil", "away": "Argentina", "fast": True,
    }, csrf=app.csrf_token, idempotency_key="browser-request-1")
    assert duplicate["status"].startswith("200")
    assert duplicate["json"]["created"] is False
    assert duplicate["json"]["task"]["task_id"] == task_id
    conflict = _request(app, "POST", "/api/v1/matches", {
        "home": "France", "away": "Spain", "fast": True,
    }, csrf=app.csrf_token, idempotency_key="browser-request-1")
    assert conflict["status"].startswith("409")
    assert conflict["json"]["error"]["code"] == "idempotency_conflict"

    assert BackgroundMatchWorker(app.task_queue).run_once()
    observed = _request(app, path=f"/api/v1/tasks/{task_id}")
    assert observed["json"]["task"]["state"] == "completed"
    dashboard_url = observed["json"]["task"]["result"]["dashboard_url"]
    assert dashboard_url.endswith("/0001.html")
    assert "raw_summary" not in observed["json"]

    artifact = _request(app, path=dashboard_url)
    assert artifact["status"].startswith("200")
    assert artifact["body"].startswith(b"<html>")


def test_artifacts_are_html_only_and_cannot_escape_workspace(tmp_path):
    secret = tmp_path / "data/private.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("private", encoding="utf-8")
    app = ProductWebApp(tmp_path)
    traversal = _request(app, path="/artifacts/outputs/studio/../../../data/private.txt")
    assert traversal["status"].startswith("404")
    assert b"private" not in traversal["body"]


def test_mutations_fail_fast_when_another_operation_is_running(tmp_path):
    app = ProductWebApp(tmp_path)
    app._mutation_lock.acquire()
    try:
        response = _request(app, "POST", "/api/v1/studio", {
            "name": "Busy", "mode": "stable", "seed": 42,
        }, csrf=app.csrf_token)
    finally:
        app._mutation_lock.release()
    assert response["status"].startswith("409")
    assert response["json"]["error"]["code"] == "operation_in_progress"


def test_web_recovery_center_creates_verifies_and_explicitly_restores(tmp_path):
    app = ProductWebApp(tmp_path)
    created_studio = _request(
        app, "POST", "/api/v1/studio",
        {"name": "Recovery Center", "mode": "stable", "seed": 42},
        csrf=app.csrf_token,
    )
    assert created_studio["status"].startswith("201")
    empty = _request(app, path="/api/v1/recovery")
    assert empty["json"]["backups"] == []

    no_csrf = _request(app, "POST", "/api/v1/backups", {})
    assert no_csrf["status"].startswith("403")
    created = _request(
        app, "POST", "/api/v1/backups", {}, csrf=app.csrf_token,
    )
    assert created["status"].startswith("201")
    backup_id = created["json"]["backup_id"]
    assert created["json"]["valid"] is True
    assert str(tmp_path) not in json.dumps(created["json"])
    catalog = _request(app, path="/api/v1/recovery")["json"]
    assert catalog["count"] == 1
    assert catalog["backups"][0]["backup_id"] == backup_id

    verified = _request(
        app, "POST", f"/api/v1/backups/{backup_id}/verify", {},
        csrf=app.csrf_token,
    )
    assert verified["status"].startswith("200") and verified["json"]["valid"]
    session = tmp_path / "data/persistence/product_session.json"
    session.write_text(session.read_text(encoding="utf-8").replace(
        "Recovery Center", "Changed After Backup",
    ), encoding="utf-8")

    wrong_confirmation = _request(
        app, "POST", f"/api/v1/backups/{backup_id}/restore",
        {"confirmation": "错误确认", "replace": True}, csrf=app.csrf_token,
    )
    assert wrong_confirmation["status"].startswith("422")
    assert "Changed After Backup" in session.read_text(encoding="utf-8")

    app._mutation_lock.acquire()
    try:
        admission_blocked = _request(
            app, "POST", "/api/v1/matches",
            {"home": "Brazil", "away": "Argentina", "fast": True},
            csrf=app.csrf_token,
        )
    finally:
        app._mutation_lock.release()
    assert admission_blocked["status"].startswith("409")
    assert admission_blocked["json"]["error"]["code"] == "operation_in_progress"

    task, _ = app.task_queue.submit_match("Brazil", "Argentina", fast=True)
    blocked = _request(
        app, "POST", f"/api/v1/backups/{backup_id}/restore",
        {"confirmation": backup_id, "replace": True}, csrf=app.csrf_token,
    )
    assert blocked["status"].startswith("409")
    assert blocked["json"]["error"]["code"] == "restore_blocked"
    claimed = app.task_queue.claim_next("test-worker")
    app.task_queue.complete(claimed["task_id"], "test-worker", {"ok": True})

    restored = _request(
        app, "POST", f"/api/v1/backups/{backup_id}/restore",
        {"confirmation": backup_id, "replace": True}, csrf=app.csrf_token,
    )
    assert restored["status"].startswith("200")
    assert restored["json"]["restored"] is True
    assert restored["json"]["task_history_reset"] is True
    assert "Recovery Center" in session.read_text(encoding="utf-8")
    assert app.task_queue.list_tasks() == []
    assert task["task_id"] not in json.dumps(app.task_queue.list_tasks())

    invalid_path = _request(
        app, "POST", "/api/v1/backups/../escape/verify", {}, csrf=app.csrf_token,
    )
    assert invalid_path["status"].startswith("404")
    wrong_method = _request(app, "GET", "/api/v1/backups")
    assert wrong_method["status"].startswith("405")


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_are_allowed(host):
    assert _is_loopback_host(host)


def test_remote_binding_is_rejected_before_socket_creation(tmp_path):
    assert not _is_loopback_host("0.0.0.0")
    with pytest.raises(ValueError, match="non-loopback Web binding"):
        create_product_web_server(tmp_path, host="0.0.0.0")


def test_server_serves_real_http_on_ephemeral_loopback_port(tmp_path):
    server = create_product_web_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/healthz", timeout=3,
        ) as response:
            payload = json.loads(response.read())
            assert response.status == 200
            assert payload["status"] == "ok"
            assert payload["external_calls_made"] is False
            assert response.headers["X-Content-Type-Options"] == "nosniff"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_server_lease_rejects_a_second_web_process_for_same_workspace(tmp_path):
    first = create_product_web_server(tmp_path, port=0)
    try:
        with pytest.raises(RuntimeError, match="lease is already owned"):
            create_product_web_server(tmp_path, port=0)
    finally:
        first.server_close()


def test_remote_login_requires_allowed_host_https_and_secure_session(tmp_path):
    access_token = "deployment-test-token-" + "x" * 32
    policy = WebAccessPolicy(
        remote=True, access_token=access_token,
        allowed_hosts=("studio.example",),
    )
    app = ProductWebApp(tmp_path, access_policy=policy)

    wrong_host = _request(
        app, path="/login", host="attacker.example", forwarded_proto="https",
    )
    assert wrong_host["status"].startswith("400")
    insecure = _request(
        app, path="/login", host="studio.example", forwarded_proto="http",
    )
    assert insecure["status"].startswith("426")

    redirected = _request(
        app, path="/", host="studio.example", forwarded_proto="https",
    )
    assert redirected["status"].startswith("302")
    assert redirected["headers"]["Location"] == "/login"
    login_page = _request(
        app, path="/login", host="studio.example", forwarded_proto="https",
    )
    assert login_page["status"].startswith("200")
    assert b"access_token" in login_page["body"]

    rejected = _request(
        app, "POST", "/api/v1/login", {"access_token": "wrong"},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    assert rejected["status"].startswith("401")
    accepted = _request(
        app, "POST", "/api/v1/login", {"access_token": access_token},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    assert accepted["status"].startswith("200")
    assert access_token.encode() not in accepted["body"]
    cookie_header = accepted["headers"]["Set-Cookie"]
    assert all(value in cookie_header for value in (
        "HttpOnly", "Secure", "SameSite=Strict",
    ))
    cookie = cookie_header.split(";", 1)[0]

    unauthorized = _request(
        app, path="/api/v1/studio",
        host="studio.example", forwarded_proto="https",
    )
    assert unauthorized["status"].startswith("401")
    authenticated = _request(
        app, path="/api/v1/studio", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert authenticated["status"].startswith("200")
    assert authenticated["json"]["access"]["mode"] == "remote_authenticated"
    assert access_token not in json.dumps(authenticated["json"])
    remote_document = _request(
        app, path="/", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )["body"].decode("utf-8")
    assert 'id="logout-button" type="button">' in remote_document

    malformed = _request(
        app, "POST", "/api/v1/login", {"access_token": 42},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    assert malformed["status"].startswith("401")
    for _ in range(4):
        attempt = _request(
            app, "POST", "/api/v1/login", {"access_token": "wrong"},
            csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
        )
        assert attempt["status"].startswith("401")
    limited = _request(
        app, "POST", "/api/v1/login", {"access_token": access_token},
        csrf=app.csrf_token, host="studio.example", forwarded_proto="https",
    )
    assert limited["status"].startswith("429")
    assert int(limited["headers"]["Retry-After"]) >= 1
    operations = _request(
        app, path="/api/v1/operations", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert operations["json"]["authentication"] == {
        "accepted": 1, "rejected": 6, "rate_limited": 1, "logged_out": 0,
    }
    assert access_token not in app.telemetry.path.read_text(encoding="utf-8")

    missing_csrf = _request(
        app, "POST", "/api/v1/logout", {}, cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert missing_csrf["status"].startswith("403")
    logout = _request(
        app, "POST", "/api/v1/logout", {}, csrf=app.csrf_token, cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert logout["status"].startswith("200")
    assert logout["json"]["revoked"] is True
    assert "Max-Age=0" in logout["headers"]["Set-Cookie"]
    expired = _request(
        app, path="/api/v1/studio", cookie=cookie,
        host="studio.example", forwarded_proto="https",
    )
    assert expired["status"].startswith("401")
    assert app.telemetry.snapshot()["authentication"]["logged_out"] == 1


def test_web_cli_parses_explicit_remote_security_boundary():
    args = build_parser().parse_args([
        "studio", "web", "--host", "0.0.0.0", "--allow-remote",
        "--allowed-host", "studio.example", "--allowed-host", "backup.example",
    ])
    assert args.allow_remote
    assert args.host == "0.0.0.0"
    assert args.allowed_host == ["studio.example", "backup.example"]


def test_local_web_cli_ignores_unrelated_remote_secret_file(monkeypatch, tmp_path):
    captured = {}

    class FakeServer:
        server_port = 8765

        @staticmethod
        def serve_forever():
            raise KeyboardInterrupt

        @staticmethod
        def server_close():
            return None

    def fake_create_server(*args, **kwargs):
        captured.update(kwargs)
        return FakeServer()

    monkeypatch.setenv("GFS_WEB_ACCESS_TOKEN_FILE", str(tmp_path / "missing"))
    monkeypatch.setattr("src.product.create_product_web_server", fake_create_server)
    args = build_parser().parse_args(["--base-dir", str(tmp_path), "studio", "web"])
    assert cmd_studio_web(args) == 0
    assert captured["allow_remote"] is False
    assert captured["access_token"] == ""
