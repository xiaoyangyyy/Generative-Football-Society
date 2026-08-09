"""Dependency-free local Web Beta over the canonical product workspace."""

from __future__ import annotations

import html
import ipaddress
import json
import logging
import secrets
import threading
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlsplit
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from src.product.control_plane import ProductControlPlane
from src.product.workspace import ProductWorkspace, StudioConfig
from src.simulation.llm_gateway import provider_preflight


LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 64 * 1024
JSON_HEADERS = [("Content-Type", "application/json; charset=utf-8")]
STATUS_TEXT = {
    200: "OK", 201: "Created", 400: "Bad Request", 403: "Forbidden",
    404: "Not Found", 405: "Method Not Allowed", 409: "Conflict",
    413: "Payload Too Large", 415: "Unsupported Media Type",
    422: "Unprocessable Entity", 500: "Internal Server Error",
    503: "Service Unavailable",
}


class WebRequestError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ProductWebApp:
    """Small WSGI adapter; all domain decisions stay in ProductWorkspace."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.csrf_token = secrets.token_urlsafe(32)
        self._mutation_lock = threading.Lock()

    def __call__(
        self, environ: dict[str, Any],
        start_response: Callable[[str, list[tuple[str, str]]], Any],
    ) -> Iterable[bytes]:
        request_id = secrets.token_hex(8)
        try:
            status, headers, body = self._dispatch(environ)
        except WebRequestError as exc:
            status = exc.status
            headers = list(JSON_HEADERS)
            body = _json_bytes({
                "error": {"code": exc.code, "message": exc.message},
                "request_id": request_id,
            })
        except Exception:
            LOGGER.exception("Unhandled Web Beta request error request_id=%s", request_id)
            status = 500
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
        return [body]

    def _dispatch(
        self, environ: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        self._require_loopback_host(environ)
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        if method == "GET" and path == "/":
            return self._html_response()
        if method == "GET" and path == "/healthz":
            return self._json_response(200, {
                "schema_version": 1,
                "status": "ok",
                "service": "gfs-product-web",
                "external_calls_made": False,
            })
        if method == "GET" and path == "/readyz":
            payload = self._studio_status()
            ready = bool(payload.get("configured") and (
                payload.get("studio", {}).get("readiness", {}).get("ready")
            ))
            return self._json_response(200 if ready else 503, {
                "schema_version": 1, "ready": ready,
                "configured": payload["configured"],
                "blockers": (
                    payload.get("studio", {}).get("readiness", {}).get("blockers", [])
                ),
            })
        if method == "GET" and path == "/api/v1/studio":
            return self._json_response(200, self._studio_status(), csrf=True)
        if method == "POST" and path == "/api/v1/studio":
            self._require_csrf(environ)
            return self._create_studio(self._read_json(environ))
        if method == "POST" and path == "/api/v1/matches":
            self._require_csrf(environ)
            return self._run_match(self._read_json(environ))
        if method == "GET" and path.startswith("/artifacts/"):
            return self._artifact_response(path.removeprefix("/artifacts/"))
        if path in {"/api/v1/studio", "/api/v1/matches"}:
            raise WebRequestError(405, "method_not_allowed", "Method not allowed")
        raise WebRequestError(404, "not_found", "Resource not found")

    def _json_response(
        self, status: int, payload: Any, *, csrf: bool = False,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        headers = list(JSON_HEADERS)
        if csrf:
            headers.append(("X-GFS-CSRF-Token", self.csrf_token))
        return status, headers, _json_bytes(payload)

    def _html_response(self) -> tuple[int, list[tuple[str, str]], bytes]:
        nonce = secrets.token_urlsafe(18)
        document = _INDEX_HTML.replace("__NONCE__", html.escape(nonce, quote=True))
        document = document.replace(
            "__CSRF__", html.escape(self.csrf_token, quote=True),
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
    def _require_loopback_host(environ: dict[str, Any]) -> None:
        authority = str(
            environ.get("HTTP_HOST") or environ.get("SERVER_NAME") or "",
        ).strip()
        try:
            hostname = urlsplit("//" + authority).hostname or ""
        except ValueError as exc:
            raise WebRequestError(400, "invalid_host", "Invalid Host header") from exc
        if not _is_loopback_host(hostname):
            raise WebRequestError(400, "invalid_host", "Host must resolve to loopback")

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

    def _run_match(
        self, payload: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        home = self._text_field(payload, "home", maximum=80)
        away = self._text_field(payload, "away", maximum=80)
        if home.casefold() == away.casefold():
            raise WebRequestError(422, "same_team", "Home and away teams must differ")
        fast = payload.get("fast", False)
        if not isinstance(fast, bool):
            raise WebRequestError(422, "invalid_fast", "fast must be boolean")
        if not self._mutation_lock.acquire(blocking=False):
            raise WebRequestError(409, "operation_in_progress", "Another mutation is running")
        try:
            try:
                report = ProductWorkspace.load(self.root).run_match(home, away, fast=fast)
            except FileNotFoundError as exc:
                raise WebRequestError(404, "studio_missing", "Create a Studio first") from exc
            except TimeoutError as exc:
                raise WebRequestError(
                    409, "studio_busy", "The Studio is busy; retry after the active operation",
                ) from exc
            except RuntimeError as exc:
                raise WebRequestError(
                    409, "match_blocked", "Match blocked by readiness or transaction gates",
                ) from exc
            dashboard = Path(str(report["dashboard_path"])).resolve()
            try:
                dashboard_relative = dashboard.relative_to(self.root).as_posix()
            except ValueError as exc:
                raise WebRequestError(500, "artifact_outside_workspace", "Invalid report path") from exc
            response = {
                "schema_version": 1,
                "match_id": report["match_id"],
                "fixture": report["fixture"],
                "result": report["result"],
                "integrity": report["integrity"],
                "dashboard_url": "/artifacts/" + dashboard_relative,
            }
            return self._json_response(201, response)
        finally:
            self._mutation_lock.release()

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


def create_product_web_server(
    root: str | Path, *, host: str = "127.0.0.1", port: int = 8765,
):
    """Create a loopback-only server; remote deployment needs an auth layer."""
    if not _is_loopback_host(host):
        raise ValueError("Web Beta is loopback-only; put an authenticated gateway in front")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be 0 (automatic) or between 1 and 65535")
    return make_server(
        host, port, ProductWebApp(root),
        server_class=ThreadingWSGIServer,
        handler_class=WSGIRequestHandler,
    )


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
    input,select,button { font:inherit; border-radius:9px; border:1px solid #455b52; padding:.72rem .8rem }
    input,select { color:var(--ink); background:#0b100f } button { color:#07100c; background:var(--accent);
      border-color:transparent; font-weight:750; cursor:pointer } button:disabled { opacity:.48; cursor:wait }
    input:focus-visible,select:focus-visible,button:focus-visible,a:focus-visible { outline:3px solid var(--warn); outline-offset:3px }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:.7rem } .check { display:flex; align-items:center; gap:.6rem }
    .check input { width:1.1rem; height:1.1rem } .status { min-height:1.6rem; color:var(--muted) }
    .error { color:var(--danger) } .ok { color:var(--accent) } a { color:var(--accent) }
    pre { max-height:340px; overflow:auto; padding:1rem; border-radius:10px; background:#070a09;
      color:#cbd5d0; white-space:pre-wrap; overflow-wrap:anywhere }
    [hidden] { display:none!important }
    @media(prefers-reduced-motion:no-preference){ section { animation:rise .45s ease both }
      @keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}} }
  </style>
</head>
<body>
<a class="skip" href="#main">跳到主要内容</a>
<header><div class="eyebrow">Generative Football Society</div><h1>一个入口，观察完整足球社会。</h1>
<p class="lede">配置工作区、核验模型与证据、运行比赛，并在同一条可审计工作流中查看结果。</p></header>
<main id="main" class="grid">
  <section class="overview" aria-labelledby="overview-title"><h2 id="overview-title">系统状态</h2>
    <div id="cards" class="cards" aria-live="polite"></div><p id="workflow" class="status"></p></section>
  <section class="action" aria-labelledby="action-title"><h2 id="action-title">下一步操作</h2>
    <form id="setup-form"><label>工作区名称<input name="name" maxlength="80" required value="My GFS Studio"></label>
      <div class="row"><label>模式<select name="mode"><option value="stable">稳定</option><option value="research">研究</option><option value="cognitive">认知</option></select></label>
      <label>随机种子<input name="seed" type="number" min="0" max="2147483647" value="42" required></label></div>
      <button type="submit">创建工作区</button></form>
    <form id="match-form" hidden><div class="row"><label>主队<input name="home" maxlength="80" value="Brazil" required></label>
      <label>客队<input name="away" maxlength="80" value="Argentina" required></label></div>
      <label class="check"><input name="fast" type="checkbox" checked>快速模式</label><button type="submit">运行比赛</button></form>
    <p id="message" class="status" role="status" aria-live="polite"></p><p id="report-link" hidden></p></section>
  <section class="wide" aria-labelledby="evidence-title"><h2 id="evidence-title">证据与运行详情</h2><pre id="details">正在读取……</pre></section>
</main>
<script nonce="__NONCE__">
const csrf=document.querySelector('meta[name="gfs-csrf"]').content;
const cards=document.querySelector('#cards'),setup=document.querySelector('#setup-form'),match=document.querySelector('#match-form');
const message=document.querySelector('#message'),details=document.querySelector('#details'),workflow=document.querySelector('#workflow'),report=document.querySelector('#report-link');
const esc=v=>String(v??'—');
function card(label,value){const el=document.createElement('div');el.className='card';const a=document.createElement('span');a.className='label';a.textContent=label;const b=document.createElement('strong');b.className='value';b.textContent=esc(value);el.append(a,b);return el}
function render(data){cards.replaceChildren();const s=data.studio;if(!data.configured){cards.append(card('工作区','未配置'),card('API 调用','0'));setup.hidden=false;match.hidden=true;workflow.textContent='创建工作区后，系统会先执行证据就绪检查。'}else{const r=s.readiness||{},w=s.workflow||{};cards.append(card('模式',s.mode),card('就绪',r.ready?'是':'否'),card('已完成比赛',s.matches_played),card('工作流',w.state));setup.hidden=true;match.hidden=false;workflow.textContent=r.ready?'证据门禁通过，可以运行比赛。':'阻塞项：'+(r.blockers||[]).join(', ')}details.textContent=JSON.stringify(data,null,2)}
async function api(path,options={}){const response=await fetch(path,{...options,headers:{'Content-Type':'application/json','X-GFS-CSRF':csrf,...options.headers}});const data=await response.json();if(!response.ok)throw new Error(data.error?.message||'请求失败');return data}
async function refresh(){try{render(await api('/api/v1/studio'))}catch(e){message.className='status error';message.textContent=e.message}}
async function submit(form,path,payload){const button=form.querySelector('button');button.disabled=true;message.className='status';message.textContent='正在处理，请勿关闭页面……';report.hidden=true;try{const data=await api(path,{method:'POST',body:JSON.stringify(payload)});message.className='status ok';message.textContent='操作成功。';if(data.dashboard_url){report.replaceChildren();const a=document.createElement('a');a.href=data.dashboard_url;a.target='_blank';a.rel='noopener';a.textContent='打开比赛仪表板';report.append(a);report.hidden=false;details.textContent=JSON.stringify(data,null,2)}else await refresh()}catch(e){message.className='status error';message.textContent=e.message}finally{button.disabled=false}}
setup.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(setup);submit(setup,'/api/v1/studio',{name:f.get('name'),mode:f.get('mode'),seed:Number(f.get('seed'))})});
match.addEventListener('submit',e=>{e.preventDefault();const f=new FormData(match);submit(match,'/api/v1/matches',{home:f.get('home'),away:f.get('away'),fast:f.get('fast')==='on'})});
refresh();
</script>
</body></html>"""
