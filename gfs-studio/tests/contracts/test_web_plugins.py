import json
from pathlib import Path

from gfs_core.kernel import PluginKernel
from gfs_studio.bootstrap.builtin_plugins import build_planning_registry, build_web_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_web_bundle_separates_routes_views_and_loopback_authentication():
    kernel = PluginKernel(build_web_registry(PROJECT_ROOT))
    try:
        kernel.start(["gfs.studio.web"])
        web = kernel.context.require("web-bundle")
        assert set(web) == {"gateway", "api", "views", "authentication"}
        assert web["authentication"].status()["production_authentication"] is False
        assert web["gateway"].handle(
            "GET", "/", host="127.0.0.1",
        ).status == 200
        assert web["gateway"].handle(
            "GET", "/health", host="example.com",
        ).status == 403
        forwarded = web["gateway"].handle(
            "GET", "/health", host="127.0.0.1",
            headers={"HTTP_X_FORWARDED_FOR": "192.0.2.1"},
        )
        assert forwarded.status == 403
        missing = web["gateway"].handle("GET", "/missing", host="127.0.0.1")
        assert json.loads(missing.body)["error"] == "route not found"
    finally:
        kernel.stop()


def test_world_model_bundle_exposes_real_leaf_capabilities():
    kernel = PluginKernel(build_planning_registry(PROJECT_ROOT))
    try:
        kernel.start(["gfs.studio.world-model"])
        bundle = kernel.context.require("world-model-bundle")
        assert set(bundle) == {"latent", "frame", "planning", "policies"}
        assert bundle["frame"].manifest()["artifact_verified"] is False
        assert bundle["latent"].load(
            PROJECT_ROOT / "data" / "world-model" / "latent_wm_rollout_calibrated_candidate.pt"
        ).status()["deployed"] is False
    finally:
        kernel.stop()
