"""One read/write control plane over product runs and research lifecycle state."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from src.infrastructure import FileLease, file_sha256
from src.product.excellence import derive_excellence
from src.product.completion_plan import build_completion_plan


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
        payload: dict[str, Any],
        run_dir: Path | None = None,
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
            jobs.append(
                {
                    **payload,
                    "run_id": path.parent.name,
                    "status_path": path.relative_to(self.root).as_posix(),
                    "observed_state": observed_state,
                    "stop_requested": (path.parent / "STOP").is_file(),
                    **({"observation": observation} if observation else {}),
                }
            )
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
                        "successful_provider_calls": report.get(
                            "successful_provider_calls"
                        ),
                    }
                except (OSError, ValueError):
                    result[name] = {
                        "available": False,
                        "path": relative,
                        "error": "invalid_json",
                    }
            else:
                result[name] = {"available": False, "path": relative}
        return result

    def excellence(
        self,
        passed_gates: dict[str, bool],
    ) -> dict[str, Any]:
        path = self.root / "data/evaluation/excellence_roadmap_v1.json"
        contract_path = (
            self.root / "data/evaluation/excellence_scoring_contract_v1.json"
        )
        if not path.is_file() or not contract_path.is_file():
            return {"available": False}
        roadmap = json.loads(path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        baseline = roadmap.get("baseline_assessment") or {}
        derived = derive_excellence(
            contract,
            passed_gates,
            baseline_scores={
                name: int((baseline.get(name) or {}).get("score", 0))
                for name in ("product", "academic")
            },
        )
        return {
            "available": True,
            "path": path.relative_to(self.root).as_posix(),
            "scoring_contract": contract_path.relative_to(self.root).as_posix(),
            "contract_id": derived["contract_id"],
            "current_phase": roadmap.get("current_phase"),
            "tracks": derived["tracks"],
            "all_tracks_full_maturity": derived["all_tracks_full_maturity"],
            "contract_checks": derived["contract_checks"],
            "security_action": (
                "credential_incident_closed"
                if passed_gates.get("credential_security_closure") is True
                else (roadmap.get("security") or {}).get("exposed_key_status")
            ),
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

    def _execution_identity_current(
        self,
        identity: dict[str, Any],
        *,
        protocol: str,
        checkpoint: str,
        prerequisite_decision: str | None = None,
    ) -> bool:
        if not isinstance(identity, dict):
            return False
        expected = {
            protocol: identity.get("protocol_sha256"),
            checkpoint: identity.get("checkpoint_sha256"),
            **(identity.get("code_sha256") or {}),
            **(identity.get("data_sha256") or {}),
        }
        if prerequisite_decision is not None:
            expected[prerequisite_decision] = identity.get(
                "confirmatory_decision_sha256"
            )
        return self._artifact_hashes_current(expected)

    @staticmethod
    def _academic_replication_is_complete(result: dict[str, Any]) -> bool:
        gates = result.get("gates") or {}
        external = result.get("external_validity") or {}
        external_checks = external.get("checks") or {}
        return (
            result.get("status")
            in {"passed_academic_replication", "failed_academic_replication"}
            and result.get("runs_executed") == 72
            and result.get("pairs_per_arm") == 24
            and result.get("material_deviations") == []
            and gates.get("fixed_complete_paired_budget") is True
            and gates.get("execution_identity_verified") is True
            and gates.get("no_material_deviations") is True
            and external.get("source") == "sportec_idsse"
            and set(external_checks)
            == {
                "passes_inside_idsse_observed_range",
                "shots_inside_idsse_observed_range",
            }
            and all(isinstance(value, bool) for value in external_checks.values())
        )

    def release_readiness(self) -> dict[str, Any]:
        paper = self._report("data/evaluation/paper_package_verification_v1.json")
        release = self._report(
            "data/evaluation/reproduction_release_verification_v1.json"
        )
        target = self._report("data/evaluation/target_lock_validation_v1.json")
        deployment = self._report(
            "data/evaluation/deployment_contract_verification_v1.json"
        )
        product_protocol = self._report(
            "data/evaluation/product_validation_protocol_verification_v1.json"
        )
        production_protocol = self._report(
            "data/evaluation/production_validation_protocol_verification_v1.json"
        )
        value_protocol = self._report(
            "data/evaluation/product_value_validation_protocol_verification_v1.json"
        )
        independent_protocol = self._report(
            "data/evaluation/independent_reproduction_protocol_verification_v1.json"
        )
        academic_protocol = self._report(
            "data/evaluation/academic_replication_protocol_verification_v1.json"
        )
        paper_finalization_protocol = self._report(
            "data/evaluation/paper_finalization_protocol_verification_v1.json"
        )
        security_protocol = self._report(
            "data/evaluation/security_closure_protocol_verification_v1.json"
        )
        security_closure = self._report(
            "data/evaluation/security_closure_verification_v1.json"
        )
        product_validation = self._report(
            "data/evaluation/product_validation_v1/decision.json"
        )
        independent_review = self._report(
            "data/evaluation/independent_reproduction_verification_v1.json"
        )
        production_operations = self._report(
            "data/evaluation/production_validation_v1/decision.json"
        )
        user_value = self._report(
            "data/evaluation/product_value_validation_v1/decision.json"
        )
        confirmatory = self._report(
            "data/evaluation/formal_confirmatory_v2/decision.json"
        )
        academic_replication = self._report(
            "data/evaluation/academic_replication_v1/decision.json"
        )
        completed_paper = self._report(
            "data/evaluation/paper_finalization_v1/verification.json"
        )
        readiness = release.get("readiness") or {}
        checks = release.get("checks") or {}
        release_identity_current = self._artifact_hashes_current(
            release.get("artifact_sha256") or {}
        )
        paper_identity_current = self._artifact_hashes_current(
            {
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
            }
        )
        product_protocol_current = self._artifact_hashes_current(
            product_protocol.get("artifact_sha256") or {}
        )
        production_protocol_current = self._artifact_hashes_current(
            production_protocol.get("artifact_sha256") or {}
        )
        value_protocol_current = self._artifact_hashes_current(
            value_protocol.get("artifact_sha256") or {}
        )
        independent_protocol_current = self._artifact_hashes_current(
            independent_protocol.get("artifact_sha256") or {}
        )
        academic_protocol_current = self._artifact_hashes_current(
            academic_protocol.get("artifact_sha256") or {}
        )
        paper_finalization_protocol_current = self._artifact_hashes_current(
            paper_finalization_protocol.get("artifact_sha256") or {}
        )
        security_protocol_current = self._artifact_hashes_current(
            security_protocol.get("artifact_sha256") or {}
        )
        security_closure_current = self._artifact_hashes_current(
            security_closure.get("artifact_sha256") or {}
        )
        product_validation_current = self._artifact_hashes_current(
            product_validation.get("artifact_sha256") or {}
        )
        independent_review_current = self._artifact_hashes_current(
            independent_review.get("artifact_sha256") or {}
        )
        completed_paper_current = self._artifact_hashes_current(
            completed_paper.get("artifact_sha256") or {}
        )
        product_gates = product_validation.get("gates") or {}
        external_review_checks = product_validation.get("external_review_checks") or {}
        gate_values = [
            (
                "paper_package",
                "Registered-report package audit",
                paper.get("passed") is True
                and paper_identity_current
                and paper_finalization_protocol.get("passed") is True
                and paper_finalization_protocol_current,
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
                "security_closure_protocol",
                "Frozen exposed-credential closure protocol",
                security_protocol.get("passed") is True
                and security_protocol.get("closure_complete") is False
                and security_protocol.get("boundary_aware_secret_match_count") == 0
                and security_protocol_current,
                "data/evaluation/security_closure_protocol_verification_v1.json",
            ),
            (
                "product_validation_protocol",
                "Frozen target-user, production, and product-value protocols",
                product_protocol.get("passed") is True
                and product_protocol.get("study_executed") is False
                and product_protocol_current
                and production_protocol.get("passed") is True
                and production_protocol.get("study_executed") is False
                and production_protocol.get("matches_executed") == 0
                and production_protocol_current
                and value_protocol.get("passed") is True
                and value_protocol.get("study_executed") is False
                and value_protocol.get("participants_observed") == 0
                and value_protocol_current,
                "data/evaluation/reproduction_release_verification_v1.json",
            ),
            (
                "independent_reproduction_protocol",
                "Frozen independent-reproduction handoff",
                independent_protocol.get("passed") is True
                and independent_protocol_current,
                independent_protocol.get("path"),
            ),
            (
                "target_user_validation",
                "Target-user workflow and SUS validation",
                product_validation.get("passed") is True
                and product_validation_current
                and all(
                    product_gates.get(key) is True
                    for key in (
                        "minimum_valid_sample",
                        "minimum_role_coverage",
                        "core_workflow_success_threshold",
                        "mean_sus_threshold",
                        "zero_critical_task_errors",
                    )
                ),
                "data/evaluation/product_validation_v1/decision.json",
            ),
            (
                "external_accessibility_review",
                "Independent WCAG 2.2 AA review",
                product_validation_current
                and external_review_checks.get("accessibility_review_passes_wcag_22_aa")
                is True
                and external_review_checks.get("reviewers_are_distinct_and_independent")
                is True,
                "data/evaluation/product_validation_v1/external_review.json",
            ),
            (
                "external_security_review",
                "Independent application security review",
                product_validation_current
                and external_review_checks.get(
                    "security_review_passes_without_critical_findings"
                )
                is True
                and external_review_checks.get(
                    "critical_issues_are_closed_not_accepted"
                )
                is True,
                "data/evaluation/product_validation_v1/external_review.json",
            ),
            (
                "credential_security_closure",
                "Exposed provider credential revoked with redacted evidence",
                security_closure.get("passed") is True
                and security_closure.get("closure_complete") is True
                and security_closure.get("boundary_aware_secret_match_count") == 0
                and security_closure.get("credential_value_stored") is False
                and security_closure_current,
                "data/evaluation/security_closure_verification_v1.json",
            ),
            (
                "production_operations_validation",
                "100-match soak and deployment-volume recovery drill",
                production_operations.get("passed") is True
                and int(production_operations.get("matches_executed") or 0) >= 100
                and production_operations.get("zero_lost_transactions") is True
                and production_operations.get("zero_duplicate_transactions") is True
                and production_operations.get("deployment_volume_recovery_drill_passed")
                is True,
                "data/evaluation/production_validation_v1/decision.json",
            ),
            (
                "user_value_validation",
                "Target-user comparative value validation",
                user_value.get("passed") is True
                and user_value.get("minimum_sample_met") is True
                and user_value.get("primary_value_threshold_met") is True,
                "data/evaluation/product_value_validation_v1/decision.json",
            ),
            (
                "confirmatory_results",
                "Confirmatory result plus mechanism and external replication",
                academic_protocol.get("passed") is True
                and academic_protocol_current
                and confirmatory.get("decision") in {
                    "promotion_candidate_pending_release_review",
                    "no_meaningful_difference_keep_research_only",
                    "inconclusive_keep_research_only",
                }
                and confirmatory.get("pairs_total") == 30
                and self._execution_identity_current(
                    confirmatory.get("execution_identity") or {},
                    protocol="data/evaluation/formal_experiment_protocol_v2.json",
                    checkpoint="data/world_model/latent_wm_rollout_calibrated_candidate.pt",
                )
                and self._academic_replication_is_complete(academic_replication)
                and self._execution_identity_current(
                    academic_replication.get("execution_identity") or {},
                    protocol="data/evaluation/academic_replication_protocol_v1.json",
                    checkpoint="data/world_model/latent_wm_rollout_calibrated_candidate.pt",
                    prerequisite_decision="data/evaluation/formal_confirmatory_v2/decision.json",
                ),
                "data/evaluation/academic_replication_v1/decision.json",
            ),
            (
                "independent_reproduction",
                "Independent reproduction report",
                independent_review.get("passed") is True
                and independent_review.get("independent_review_available") is True
                and independent_review.get("scope") == "full_study_132_runs"
                and independent_review.get("stages_verified")
                == ["confirmatory", "academic_replication"]
                and set(independent_review.get("computed_comparison") or {})
                == {"confirmatory", "academic_replication"}
                and independent_review_current,
                "data/evaluation/independent_reproduction_verification_v1.json",
            ),
            (
                "completed_manuscript",
                "Evidence-locked completed results manuscript",
                completed_paper.get("passed") is True
                and completed_paper.get("completed_manuscript_available") is True
                and completed_paper.get("status")
                == "verified_completed_manuscript"
                and completed_paper_current,
                "data/evaluation/paper_finalization_v1/verification.json",
            ),
        ]
        gates = [
            {"id": identifier, "label": label, "passed": passed, "evidence": evidence}
            for identifier, label, passed, evidence in gate_values
        ]
        excellence = self.excellence(
            {gate["id"]: gate["passed"] for gate in gates}
        )
        completion_plan = build_completion_plan(gates)
        scores = {
            name: row.get("score")
            for name, row in (excellence.get("tracks") or {}).items()
        }
        code_gate_ids = {
            "paper_package",
            "target_hash_lock",
            "cyclonedx_sbom",
            "known_source_decisions",
            "local_runtime",
            "cross_platform_lock_matrix",
            "container_image_digests",
            "external_data_archive",
            "product_validation_protocol",
            "independent_reproduction_protocol",
            "security_closure_protocol",
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
            "excellence": excellence,
            "passed_gate_count": len(gates) - len(open_gates),
            "open_gate_count": len(open_gates),
            "next_action": completion_plan["recommended_gate_id"],
            "gates": gates,
            "completion_plan": completion_plan,
            "external_calls_made": False,
        }

    def snapshot(self) -> dict[str, Any]:
        jobs = self.training_jobs()
        release = self.release_readiness()
        return {
            "schema_version": 1,
            "training_jobs": jobs,
            "training_summary": {
                "total": len(jobs),
                "running": sum(job.get("observed_state") == "running" for job in jobs),
                "stale": sum(job.get("observed_state") == "stale" for job in jobs),
                "stopping": sum(
                    job.get("observed_state") == "running" and job.get("stop_requested")
                    for job in jobs
                ),
                "resumable": sum(
                    job.get("observed_state")
                    in {
                        "stopped",
                        "interrupted",
                        "failed",
                        "stale",
                    }
                    for job in jobs
                ),
                "completed": sum(job.get("state") == "completed" for job in jobs),
            },
            "decisions": self.decisions(),
            "excellence": release["excellence"],
            "release": release,
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
