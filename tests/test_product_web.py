import io
import json
import threading
from urllib.request import urlopen
from wsgiref.util import setup_testing_defaults

import pytest

from src.product.web import ProductWebApp, _is_loopback_host, create_product_web_server
from src.product.tasks import BackgroundMatchWorker


def _request(
    app, method="GET", path="/", payload=None, *, csrf=None, host=None,
    idempotency_key=None,
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


def test_health_is_liveness_only_and_never_calls_provider(tmp_path):
    response = _request(ProductWebApp(tmp_path), path="/healthz")
    assert response["json"] == {
        "schema_version": 1,
        "status": "ok",
        "service": "gfs-product-web",
        "external_calls_made": False,
        "background_worker_alive": None,
    }


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


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_hosts_are_allowed(host):
    assert _is_loopback_host(host)


def test_remote_binding_is_rejected_before_socket_creation(tmp_path):
    assert not _is_loopback_host("0.0.0.0")
    with pytest.raises(ValueError, match="loopback-only"):
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
