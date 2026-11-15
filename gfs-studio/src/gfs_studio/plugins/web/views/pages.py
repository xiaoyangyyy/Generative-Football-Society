import html

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_studio.contracts.web import HttpError


class WebViews:
    def __init__(self, gateway, api) -> None:
        self.gateway = gateway
        self.api = api

    def handle(self, method: str, path: str, body: bytes):
        del body
        if path not in {"/", "/tasks"}:
            return None
        if method != "GET":
            raise HttpError(405, "method not allowed")
        if path == "/":
            content = _HTML
        else:
            content = _tasks_html(self.api.product["tasks"].list())
        return self.gateway.response(200, content, "text/html; charset=utf-8")


class ViewsPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.web.views",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.web",
        provides=("web-views",),
        requires=("gfs.studio.web.api",),
    )

    def activate(self, context: PluginContext) -> None:
        gateway = context.require("http-gateway")
        views = WebViews(gateway, context.require("http-api"))
        gateway.views = views
        context.provide("web-views", views)

    def deactivate(self) -> None:
        pass


_HTML = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GFS Studio</title><style>body{font-family:system-ui;max-width:960px;margin:2rem auto;padding:0 1rem;background:#f6f8fb;color:#18202b}section{background:white;padding:1rem;margin:1rem 0;border-radius:8px}label{display:block;margin:.5rem 0}input{display:block;padding:.5rem;min-width:16rem}button{padding:.6rem 1rem}pre{white-space:pre-wrap}</style></head>
<body><h1>Generative Football Society Studio</h1>
<section><h2>Run a micro match</h2><form id="match"><label>Home <input id="home" value="Brazil" required></label><label>Away <input id="away" value="Argentina" required></label><label>Seed <input id="seed" type="number" value="42"></label><button type="submit">Run fast match</button></form><pre id="result" role="status">Ready.</pre></section>
<section><h2>API</h2><ul><li><a href="/tasks">Task dashboard</a></li><li><a href="/api/studio">Studio overview</a></li><li><a href="/api/research/m2">M2 evidence</a></li><li><a href="/api/world-model">World model status</a></li></ul></section>
<script>document.getElementById('match').addEventListener('submit',async event=>{event.preventDefault();const result=document.getElementById('result');result.textContent='Running...';const response=await fetch('/api/micro-match',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({home:document.getElementById('home').value,away:document.getElementById('away').value,seed:Number(document.getElementById('seed').value)})});const data=await response.json();result.textContent=JSON.stringify(data,null,2);});</script></body></html>"""


def _tasks_html(tasks) -> bytes:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(task['idempotency_key'])}</td>"
        f"<td>{html.escape(task['state'])}</td>"
        f"<td><a href=\"/api/tasks/{html.escape(task['task_id'])}\">details</a></td>"
        "</tr>"
        for task in tasks
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>GFS Tasks</title></head><body><h1>Task dashboard</h1>"
        "<p><a href=\"/\">Back to Studio</a></p>"
        "<table><thead><tr><th>Task</th><th>State</th><th>Details</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    ).encode("utf-8")
