import json
from pathlib import Path
import shutil

from gfs_studio.bootstrap.application import StudioApplication


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = PROJECT_ROOT.parent / "gfs-core"


def test_merged_container_workspace_resolves_embedded_core_data(tmp_path):
    shutil.copytree(
        CORE_ROOT / "data", tmp_path / "data",
        ignore=shutil.ignore_patterns("persistence"),
    )
    shutil.copytree(PROJECT_ROOT / "evidence", tmp_path / "evidence")
    application = StudioApplication(tmp_path)
    gateway = application.web_gateway()
    try:
        response = gateway.handle("GET", "/health", host="127.0.0.1")
        assert response.status == 200
        assert json.loads(response.body)["status"] == "ok"
    finally:
        application.close()
