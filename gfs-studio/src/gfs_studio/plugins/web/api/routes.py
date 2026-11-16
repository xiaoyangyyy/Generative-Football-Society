from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.identity import content_identity
from gfs_studio.contracts.web import HttpError


class ApiRoutes:
    def __init__(self, gateway, product, latent_models) -> None:
        self.gateway = gateway
        self.product = product
        self.latent_models = latent_models

    def handle(self, method: str, path: str, body: bytes):
        json = self.gateway.json
        if method == "GET" and path == "/health":
            return json(200, {"status": "ok", "transport": "loopback"})
        if method == "GET" and path == "/api/studio":
            return json(200, self.product["workflows"].overview())
        if method == "GET" and path == "/api/research/m2":
            return json(200, self.product["workflows"].overview()["m2"])
        if method == "GET" and path == "/api/world-model":
            checkpoint = (
                self.product["workflows"].project_root
                / "data" / "world-model" / "latent_wm_rollout_calibrated_candidate.pt"
            )
            return json(200, self.latent_models.load(checkpoint).status())
        if method == "GET" and path == "/api/tasks":
            return json(200, {"tasks": self.product["tasks"].list()})
        if method == "GET" and path.startswith("/api/tasks/") and path not in _ROUTES:
            task = self.product["tasks"].get(path.rsplit("/", 1)[-1])
            if task is None:
                raise HttpError(404, "task not found")
            return json(200, task)
        if method == "POST":
            payload = self.gateway.payload(body)
            if path == "/api/micro-match":
                task_payload = {
                    "home": self.gateway.team(payload, "home"),
                    "away": self.gateway.team(payload, "away"),
                    "seed": self.gateway.seed(payload),
                }
                key = payload.get("idempotency_key")
                if key is None:
                    key = "micro:" + content_identity(task_payload)
                return self._submit_task(key, task_payload)
            if path == "/api/manager/briefing":
                return json(200, self.product["manager"].briefing(
                    self.gateway.team(payload, "home"),
                    self.gateway.team(payload, "away"),
                    self.gateway.seed(payload),
                ))
            if path == "/api/tasks":
                key = payload.get("idempotency_key")
                if not isinstance(key, str):
                    raise HttpError(400, "idempotency_key is required")
                return self._submit_task(key, {
                    "home": self.gateway.team(payload, "home"),
                    "away": self.gateway.team(payload, "away"),
                    "seed": self.gateway.seed(payload),
                })
            if path == "/api/tasks/recover":
                return json(200, {"recovered": self.product["tasks"].recover()})
            if path == "/api/tasks/resume":
                return json(200, {"resumed": self.product["tasks"].resume(
                    payload.get("task_id"),
                )})
            if path == "/api/tasks/cancel":
                task_id = payload.get("task_id")
                if not isinstance(task_id, str):
                    raise HttpError(400, "task_id is required")
                return json(202, self.product["tasks"].cancel(task_id))
            if path == "/api/tasks/wait":
                task_id = payload.get("task_id")
                timeout = payload.get("timeout", 30.0)
                if not isinstance(task_id, str):
                    raise HttpError(400, "task_id is required")
                if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                    raise HttpError(400, "timeout must be numeric")
                return json(200, self.product["tasks"].wait(task_id, timeout))
            if path == "/api/backups":
                return json(200, self.product["recovery"].create())
            if path == "/api/backups/verify":
                return json(200, self.product["recovery"].verify(
                    self.gateway.backup_id(payload),
                ))
            if path == "/api/backups/restore":
                destination = payload.get("destination")
                replace = payload.get("replace", False)
                if not isinstance(destination, str) or not destination.strip():
                    raise HttpError(400, "destination is required")
                if not isinstance(replace, bool):
                    raise HttpError(400, "replace must be boolean")
                return json(200, self.product["recovery"].restore(
                    self.gateway.backup_id(payload), destination, replace=replace,
                ))
            if path == "/api/comparison":
                baseline_seed = payload.get("baseline_seed", 42)
                treatment_seed = payload.get("treatment_seed", 43)
                if any(isinstance(value, bool) or not isinstance(value, int) for value in (baseline_seed, treatment_seed)):
                    raise HttpError(400, "comparison seeds must be integers")
                return json(200, self.product["comparison"].compare_seeds(
                    self.gateway.team(payload, "home"),
                    self.gateway.team(payload, "away"),
                    baseline_seed,
                    treatment_seed,
                ))
        if path in _ROUTES:
            raise HttpError(405, "method not allowed")
        return None

    def _submit_task(self, key: str, payload: dict):
        task = self.product["tasks"].submit(key, payload)
        return self.gateway.json(202, {
            "task_id": task["task_id"],
            "state": task["state"],
            "status_url": f"/api/tasks/{task['task_id']}",
        })


class ApiPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.web.api",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.web",
        provides=("http-api",),
        requires=("gfs.studio.web.gateway", "gfs.studio.product", "gfs.studio.world-model.latent"),
    )

    def activate(self, context: PluginContext) -> None:
        gateway = context.require("http-gateway")
        routes = ApiRoutes(
            gateway,
            context.require("product-bundle"),
            context.require("latent-models"),
        )
        gateway.api = routes
        context.provide("http-api", routes)

    def deactivate(self) -> None:
        pass


_ROUTES = {
    "/health", "/api/studio", "/api/research/m2", "/api/world-model",
    "/api/micro-match", "/api/manager/briefing", "/api/backups",
    "/api/backups/verify", "/api/backups/restore", "/api/tasks",
    "/api/tasks/recover", "/api/tasks/resume", "/api/tasks/cancel",
    "/api/tasks/wait", "/api/comparison",
}
