import http.client
import json
from pathlib import Path
from threading import Thread

from gfs_studio.api import web_gateway

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_routes_security_and_error_contracts():
    gateway = web_gateway(PROJECT_ROOT)
    health = gateway.handle("GET", "/health", host="127.0.0.1:8765")
    assert health.status == 200
    assert json.loads(health.body)["status"] == "ok"
    overview = gateway.handle("GET", "/api/studio", host="127.0.0.1:8765")
    payload = json.loads(overview.body)
    assert payload["m2"]["active"]["runs"] == 247
    assert payload["release"]["release_ready"] is False
    assert payload["deployment"] == "loopback_only"
    page = gateway.handle("GET", "/", host="127.0.0.1:8765")
    assert page.status == 200
    assert b"Run a micro match" in page.body
    assert gateway.handle("GET", "/health", host="example.com").status == 403
    assert gateway.handle("POST", "/health", host="127.0.0.1").status == 405
    assert gateway.handle("GET", "/missing", host="127.0.0.1").status == 404
    assert gateway.handle("POST", "/api/micro-match", host="127.0.0.1", body=b"not-json").status == 400


def test_real_loopback_http_and_micro_match_submission():
    gateway = web_gateway(PROJECT_ROOT)
    server = gateway.serve(port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        connection = http.client.HTTPConnection(host, port, timeout=30)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["status"] == "ok"

        connection.request(
            "POST",
            "/api/micro-match",
            body=json.dumps({"home": "Brazil", "away": "Argentina", "seed": 42}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["home"] == "Brazil"
        assert payload["away"] == "Argentina"
        assert payload["score"] == [0, 3]
        assert payload["ticks"] == 180
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        gateway.close()
