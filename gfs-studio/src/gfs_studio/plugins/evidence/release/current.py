from __future__ import annotations

import json
import math
from pathlib import Path

from gfs_core.kernel.project import data_identity, source_identity

REQUIRED_CHECKS = (
    "core_tests",
    "studio_tests",
    "core_ruff",
    "studio_ruff",
    "core_wheel",
    "studio_wheel",
    "static_deployment",
)


def current_verification(root: str | Path) -> dict:
    root = Path(root).resolve()
    path = root / "evidence" / "current" / "build_verification.json"
    if not path.is_file():
        return {
            "code_ready": False,
            "runtime_verified": False,
            "source_identity": source_identity(root),
            "report": None,
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("current build verification must be an object")
    checks = report.get("checks", {})
    code_ready = (
        report.get("code_ready") is True
        and report.get("source_identity") == source_identity(root)
        and all(checks.get(name) is True for name in REQUIRED_CHECKS)
    )
    deployment = report.get("deployment", {})
    return {
        "code_ready": code_ready,
        "runtime_verified": bool(deployment.get("runtime_verified", False)),
        "source_identity": source_identity(root),
        "report": report,
    }


def current_simulation_verification(root: str | Path) -> dict:
    root = Path(root).resolve()
    path = root / "evidence" / "current" / "simulation_evaluation.json"
    if not path.is_file():
        return {"available": False, "source_identity": source_identity(root), "report": None}
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("current simulation evidence must be an object")
    metrics = report.get("metrics", {})
    finite = bool(metrics) and all(
        isinstance(row.get("observed"), (int, float)) and math.isfinite(row["observed"])
        for row in metrics.values()
    )
    available = (
        report.get("state") == "completed"
        and report.get("protocol_complete") is True
        and report.get("source_identity") == source_identity(root)
        and report.get("data_identity") == data_identity(root)
        and report.get("matches_executed") == report.get("protocol", {}).get("runs_total")
        and report.get("matches_executed", 0) > 0
        and finite
    )
    return {
        "available": available,
        "source_identity": source_identity(root),
        "data_identity": data_identity(root),
        "report": report,
    }
