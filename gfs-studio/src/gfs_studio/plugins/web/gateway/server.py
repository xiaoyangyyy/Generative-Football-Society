from __future__ import annotations

import json
from wsgiref.simple_server import make_server

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_studio.contracts.web import HttpError, WebResponse

MAX_BODY_BYTES = 64 * 1024


class WebGateway:
    def __init__(self, workflows=None) -> None:
        self.workflows = workflows
        self.api = None
        self.views = None
        self.authentication = None

    def handle(self, method: str, path: str, *, host: str, body: bytes = b"", headers: dict | None = None) -> WebResponse:
        try:
            if len(body) > MAX_BODY_BYTES:
                raise HttpError(413, "request body is too large")
            if self.authentication is not None:
                self.authentication.authorize(host, headers or {})
            for router in (self.api, self.views):
                if router is not None:
                    response = router.handle(method, path, body)
                    if response is not None:
                        return response
            raise HttpError(404, "route not found")
        except HttpError as error:
            return self.json(error.status, {"error": error.message})
        except (ValueError, TypeError, KeyError) as error:
            return self.json(400, {"error": str(error)})
        except PermissionError:
            return self.json(403, {"error": "request is not authorized"})
        except FileNotFoundError:
            return self.json(404, {"error": "resource not found"})
        except TimeoutError:
            return self.json(408, {"error": "request deadline expired"})
        except Exception:
            return self.json(500, {"error": "internal server error"})

    def __call__(self, environ, start_response):
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            return _finish(start_response, 400, {"error": "invalid content length"})
        if length < 0:
            return _finish(start_response, 400, {"error": "invalid content length"})
        if length > MAX_BODY_BYTES:
            return _finish(start_response, 413, {"error": "request body is too large"})
        body = environ["wsgi.input"].read(length) if length else b""
        headers = {
            name: environ[name]
            for name in ("HTTP_X_FORWARDED_FOR", "HTTP_X_FORWARDED_HOST", "HTTP_FORWARDED")
            if name in environ
        }
        response = self.handle(
            environ["REQUEST_METHOD"],
            environ["PATH_INFO"],
            host=environ.get("HTTP_HOST", ""),
            body=body,
            headers=headers,
        )
        start_response(
            f"{response.status} {_REASONS.get(response.status, 'Unknown')}",
            [("Content-Type", response.content_type), *response.headers],
        )
        return [response.body]

    def close(self) -> None:
        if self.workflows is not None:
            self.workflows.close()

    def serve(self, host="127.0.0.1", port=8765):
        if host != "127.0.0.1":
            raise ValueError("only loopback deployment is supported")
        return make_server(host, port, self)

    @staticmethod
    def payload(body: bytes) -> dict:
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HttpError(400, "invalid JSON body") from error
        if not isinstance(payload, dict):
            raise HttpError(400, "JSON body must be an object")
        return payload

    @staticmethod
    def team(payload: dict, key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 80:
            raise HttpError(400, f"{key} must be a non-empty team name")
        return value.strip()

    @staticmethod
    def seed(payload: dict) -> int:
        value = payload.get("seed", 42)
        if isinstance(value, bool) or not isinstance(value, int):
            raise HttpError(400, "seed must be an integer")
        return value

    @staticmethod
    def backup_id(payload: dict) -> str:
        value = payload.get("backup_id")
        if not isinstance(value, str):
            raise HttpError(400, "backup_id is required")
        return value

    @staticmethod
    def json(status: int, payload) -> WebResponse:
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        return WebGateway.response(status, body, "application/json; charset=utf-8")

    @staticmethod
    def response(status: int, body: bytes, content_type: str) -> WebResponse:
        return WebResponse(
            status,
            body,
            content_type,
            (
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
                ("X-Content-Type-Options", "nosniff"),
                ("X-Frame-Options", "DENY"),
                ("Content-Security-Policy", "default-src 'self'"),
            ),
        )


class WebPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.web.gateway",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.web",
        provides=("http-gateway",),
        requires=(
            "gfs.studio.product.studio",
            "gfs.studio.product.manager",
            "gfs.studio.product.recovery",
            "gfs.studio.product.tasks",
        ),
    )

    def __init__(self, workflows=None, manager=None, recovery=None, tasks=None) -> None:
        del manager, recovery, tasks
        self._workflows = workflows
        self.gateway = None

    def activate(self, context: PluginContext) -> None:
        workflows = self._workflows or context.require("studio-workflows")
        self.gateway = WebGateway(workflows)
        context.provide("http-gateway", self.gateway)

    def deactivate(self) -> None:
        if self.gateway is not None:
            self.gateway.close()
            self.gateway = None


def _finish(start_response, status, payload):
    response = WebGateway.json(status, payload)
    start_response(
        f"{status} {_REASONS.get(status, 'Unknown')}",
        [("Content-Type", response.content_type), *response.headers],
    )
    return [response.body]


_REASONS = {
    200: "OK",
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    413: "Payload Too Large",
    500: "Internal Server Error",
    503: "Service Unavailable",
}
