"""One read/write control plane over product runs and research lifecycle state."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from src.infrastructure import FileLease, file_sha256


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
        baseline = roadmap.get("baseline_assessment") or {}
        tracks = {}
        for name, gates in (roadmap.get("tracks") or {}).items():
            tracks[name] = {
                "score": sum(int(gate.get("current_points", 0)) for gate in gates),
                "baseline_score": int((baseline.get(name) or {}).get("score", 0)),
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

    def _report(self, relative: str) -> dict[str, Any]:
        path = self.root / relative
        if not path.is_file():
            return {"available": False, "path": relative}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"available": False, "path": relative, "error": "invalid_json"}
        if not isinstance(payload, dict):
            return {"available": False, "path": relative, "error": "invalid_report"}
        return {"available": True, "path": relative, **payload}

    def _artifact_hashes_current(self, expected: dict[str, Any]) -> bool:
        if not expected:
            return False
        for relative, digest in expected.items():
            candidate = (self.root / str(relative)).resolve()
            try:
                candidate.relative_to(self.root)
            except ValueError:
                return False
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or not candidate.is_file()
                or file_sha256(candidate) != digest
            ):
                return False
        return True

    def release_readiness(self) -> dict[str, Any]:
        paper = self._report("data/evaluation/paper_package_verification_v1.json")
        release = self._report(
            "data/evaluation/reproduction_release_verification_v1.json"
        )
        target = self._report("data/evaluation/target_lock_validation_v1.json")
        deployment = self._report(
            "data/evaluation/deployment_contract_verification_v1.json"
        )
        readiness = release.get("readiness") or {}
        checks = release.get("checks") or {}
        release_identity_current = self._artifact_hashes_current(
            release.get("artifact_sha256") or {}
        )
        paper_identity_current = self._artifact_hashes_current({
            "data/evaluation/formal_experiment_protocol_v2.json": paper.get(
                "protocol_sha256"
            ),
            "data/evaluation/paper_claims_v1.json": paper.get(
                "claim_registry_sha256"
            ),
            "docs/PAPER_DRAFT.md": paper.get("manuscript_sha256"),
            "data/evaluation/reproduction_manifest_v1.json": paper.get(
                "reproduction_manifest_sha256"
            ),
            "data/evaluation/reproduction_environment_v1.json": paper.get(
                "environment_snapshot_sha256"
            ),
        })
        excellence = self.excellence()
        scores = {
            name: row.get("score")
            for name, row in (excellence.get("tracks") or {}).items()
        }
        gate_values = [
            (
                "paper_package",
                "Registered-report package audit",
                paper.get("passed") is True and paper_identity_current,
                paper.get("path"),
            ),
            (
                "target_hash_lock",
                "Python 3.12 Linux transitive hash lock",
                release.get("passed") is True
                and release_identity_current
                and readiness.get("full_transitive_hash_lock") is True
                and target.get("passed") is True,
                target.get("path"),
            ),
            (
                "cyclonedx_sbom",
                "Current CycloneDX runtime SBOM",
                readiness.get("cyclonedx_sbom_current") is True,
                "data/evaluation/supply_chain_sbom_v1.cdx.json",
            ),
            (
                "known_source_decisions",
                "Known external-data release decisions",
                checks.get("known_provider_coverage_is_exact") is True
                and checks.get("license_policy_is_default_deny") is True,
                "data/evaluation/data_license_registry_v1.json",
            ),
            (
                "local_runtime",
                "Imported local direct dependency runtime",
                readiness.get("runtime_direct_dependencies_match") is True,
                "data/evaluation/local_runtime_verification_v1.json",
            ),
            (
                "cross_platform_lock_matrix",
                "Linux/Windows reference lock matrix",
                readiness.get("cross_platform_lock_matrix") is True,
                "data/evaluation/reproducibility_matrix_verification_v1.json",
            ),
            (
                "container_image_digests",
                "Container image digest pins",
                readiness.get("container_images_digest_pinned") is True,
                "data/evaluation/container_image_verification_v1.json",
            ),
            (
                "container_build",
                "Built and verified deployment container",
                deployment.get("image_built") is True,
                deployment.get("path"),
            ),
            (
                "external_data_archive",
                "Licensed external-data release subset",
                readiness.get("external_data_archive_approved") is True,
                "data/evaluation/data_release_verification_v1.json",
            ),
            (
                "confirmatory_results",
                "Complete confirmatory experiment result",
                readiness.get("confirmatory_results_available") is True,
                "data/evaluation/formal_confirmatory_v2/decision.json",
            ),
            (
                "independent_reproduction",
                "Independent reproduction report",
                readiness.get("independent_reproduction_available") is True,
                "docs/REPRODUCTION_GUIDE.md",
            ),
        ]
        gates = [
            {"id": identifier, "label": label, "passed": passed, "evidence": evidence}
            for identifier, label, passed, evidence in gate_values
        ]
        code_gate_ids = {
            "paper_package",
            "target_hash_lock",
            "cyclonedx_sbom",
            "known_source_decisions",
            "local_runtime",
            "cross_platform_lock_matrix",
            "container_image_digests",
            "external_data_archive",
        }
        code_ready = all(
            gate["passed"] for gate in gates if gate["id"] in code_gate_ids
        )
        release_ready = all(gate["passed"] for gate in gates)
        open_gates = [gate for gate in gates if not gate["passed"]]
        return {
            "schema_version": 1,
            "status": (
                "release_ready"
                if release_ready
                else "code_contract_ready_external_gates_open"
                if code_ready
                else "code_contract_incomplete"
            ),
            "code_ready": code_ready,
            "release_ready": release_ready,
            "scores": scores,
            "passed_gate_count": len(gates) - len(open_gates),
            "open_gate_count": len(open_gates),
            "next_action": open_gates[0]["id"] if open_gates else None,
            "gates": gates,
            "external_calls_made": False,
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
            "release": self.release_readiness(),
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
