"""One read/write control plane over product runs and research lifecycle state."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from src.infrastructure import FileLease


DECISION_ARTIFACTS = {
    "world_model_formal_m1": "data/evaluation/formal_ablation/M1_calibrated_decision.json",
    "frozen_shot_head": "data/evaluation/frozen_shot_head_decision.json",
    "llm_prospective_pilot": "data/evaluation/llm_prospective_pilot_v1.json",
    "formal_confirmatory_v2": "data/evaluation/formal_confirmatory_v2/decision.json",
}


class ProductControlPlane:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.runs_root = self.root / "data/training/runs"

    @staticmethod
    def _observed_state(
        payload: dict[str, Any], run_dir: Path | None = None,
    ) -> tuple[str, str | None]:
        state = str(payload.get("state", "unknown"))
        if state != "running":
            return state, None
        lease_path = run_dir / ".run.lock" if run_dir is not None else None
        if lease_path is not None and lease_path.is_file():
            if FileLease.is_held(lease_path):
                return state, None
            return "stale", "training_lease_not_held"
        pid = payload.get("pid")
        hostname = payload.get("hostname")
        if not isinstance(pid, int) or not hostname:
            return state, None
        if hostname != socket.gethostname():
            return state, "remote_host_not_probed"
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return "stale", "recorded_process_not_alive"
        return state, None

    def training_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        if not self.runs_root.is_dir():
            return jobs
        for path in sorted(self.runs_root.glob("*/status.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {"state": "invalid", "error": "unreadable_status"}
            observed_state, observation = self._observed_state(payload, path.parent)
            jobs.append({
                **payload,
                "run_id": path.parent.name,
                "status_path": path.relative_to(self.root).as_posix(),
                "observed_state": observed_state,
                "stop_requested": (path.parent / "STOP").is_file(),
                **({"observation": observation} if observation else {}),
            })
        return jobs

    def decisions(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, relative in DECISION_ARTIFACTS.items():
            path = self.root / relative
            if path.is_file():
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                    result[name] = {
                        "available": True,
                        "path": relative,
                        "decision": report.get("decision") or report.get("state"),
                        "accepted": report.get("accepted"),
                        "promotion_supported": report.get("promotion_supported"),
                        "samples_total": report.get("samples_total"),
                        "successful_provider_calls": report.get("successful_provider_calls"),
                    }
                except (OSError, ValueError):
                    result[name] = {
                        "available": False, "path": relative,
                        "error": "invalid_json",
                    }
            else:
                result[name] = {"available": False, "path": relative}
        return result

    def excellence(self) -> dict[str, Any]:
        path = self.root / "data/evaluation/excellence_roadmap_v1.json"
        if not path.is_file():
            return {"available": False}
        roadmap = json.loads(path.read_text(encoding="utf-8"))
        tracks = {}
        for name, gates in (roadmap.get("tracks") or {}).items():
            tracks[name] = {
                "score": sum(int(gate.get("current_points", 0)) for gate in gates),
                "maximum": sum(int(gate.get("weight", 0)) for gate in gates),
                "critical_gates_remaining": [
                    gate["id"] for gate in gates
                    if gate.get("critical") and gate.get("status") != "verified"
                ],
            }
        return {
            "available": True,
            "path": path.relative_to(self.root).as_posix(),
            "current_phase": roadmap.get("current_phase"),
            "tracks": tracks,
            "security_action": (
                roadmap.get("security") or {}
            ).get("exposed_key_status"),
        }

    def snapshot(self) -> dict[str, Any]:
        jobs = self.training_jobs()
        return {
            "schema_version": 1,
            "training_jobs": jobs,
            "training_summary": {
                "total": len(jobs),
                "running": sum(job.get("observed_state") == "running" for job in jobs),
                "stale": sum(job.get("observed_state") == "stale" for job in jobs),
                "stopping": sum(
                    job.get("observed_state") == "running"
                    and job.get("stop_requested") for job in jobs
                ),
                "resumable": sum(
                    job.get("observed_state") in {
                        "stopped", "interrupted", "failed", "stale",
                    }
                    for job in jobs
                ),
                "completed": sum(job.get("state") == "completed" for job in jobs),
            },
            "decisions": self.decisions(),
            "excellence": self.excellence(),
        }

    def request_stop(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("invalid training run id")
        run_dir = (self.runs_root / run_id).resolve()
        if run_dir.parent != self.runs_root.resolve() or not run_dir.is_dir():
            raise FileNotFoundError(f"training run not found: {run_id}")
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"training status not found: {run_id}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "running":
            raise RuntimeError(f"training run is not running: {status.get('state')}")
        observed, _ = self._observed_state(status, run_dir)
        if observed == "stale":
            raise RuntimeError("training run status is stale; resume it instead")
        stop = run_dir / "STOP"
        stop.write_text("requested_by_gfs_studio\n", encoding="utf-8")
        return stop
