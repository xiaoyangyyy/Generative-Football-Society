import json
from pathlib import Path

from gfs_studio.api import m2_status, release_status
from gfs_studio.bootstrap.application import StudioApplication
from gfs_studio.plugins.evidence.release.current import current_verification

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_m2_status_distinguishes_active_v6_and_frozen_v1():
    status = m2_status(PROJECT_ROOT)
    active = status["active"]
    previous = status["previous"]
    assert active["version"] == "v6"
    assert active["runs"] == 247
    assert active["budget"] == 360
    assert active["state"] == "running"
    assert active["identity_status"] == "current"
    assert active["complete"] is False
    assert previous["runs"] == 360
    assert previous["state"] == "completed"
    assert previous["identity_status"] == "frozen_historical"
    assert active["promotion_supported"] is False
    assert previous["promotion_supported"] is False


def test_release_status_does_not_claim_readiness():
    status = release_status(PROJECT_ROOT)
    assert status["release_ready"] is False
    assert status["code_ready"] is current_verification(PROJECT_ROOT)["code_ready"]
    assert status["status"] == "failed"
    assert "paper_package_still_passes" in status["open_gates"]
    assert status["scores"] == {"product": 83, "academic": 70}


def test_research_cli_returns_machine_readable_status():
    output = StudioApplication(PROJECT_ROOT).run_cli(["research", "status"])
    assert json.loads(output)["active"]["runs"] == 247
    output = StudioApplication(PROJECT_ROOT).run_cli(["release", "status"])
    assert json.loads(output)["release_ready"] is False
