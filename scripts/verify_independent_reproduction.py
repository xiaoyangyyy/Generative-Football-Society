#!/usr/bin/env python3
"""Audit the independent-reproduction protocol or a signed reviewer report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "data/evaluation/independent_reproduction_protocol_v1.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
ORCID = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
REVIEWER_ATTESTATION = (
    "I independently reproduced the frozen GFS protocol and disclosed all deviations"
)
CONTAINER_CHECKS = {
    "docker_cli_available",
    "execution_explicitly_authorized",
    "source_revision_is_commit",
    "docker_daemon_reachable",
    "image_build_passed",
    "image_id_is_content_addressed",
    "image_revision_label_matches_source",
    "image_declares_non_root_user",
    "runtime_uid_gid_are_non_root",
    "direct_dependency_import_smoke_passed",
    "runtime_security_options_verified",
    "health_response_contract_passed",
    "docker_health_status_passed",
    "ephemeral_container_removed",
    "verification_image_removed",
}
BRANCH_BY_DECISION = {
    "promotion_candidate_pending_release_review": "mechanism_confirmation",
    "no_meaningful_difference_keep_research_only": "adoption_path_diagnosis",
    "inconclusive_keep_research_only": "variance_diagnosis",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confined(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    try:
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return path


def _timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_protocol(protocol: dict[str, Any]) -> dict[str, bool]:
    prerequisites = protocol.get("prerequisites") or {}
    reviewer = protocol.get("reviewer") or {}
    execution = protocol.get("execution_contract") or {}
    comparison = protocol.get("comparison") or {}
    outputs = protocol.get("outputs") or {}
    state = protocol.get("execution") or {}
    return {
        "schema_and_state_are_registered": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id")
            == "gfs-independent-full-study-reproduction-v1"
            and protocol.get("state") == "registered_waiting_for_complete_study"
        ),
        "prerequisites_are_exact_and_fail_closed": (
            prerequisites
            == {
                "confirmatory_protocol": "data/evaluation/formal_experiment_protocol_v2.json",
                "confirmatory_decision": "data/evaluation/formal_confirmatory_v2/decision.json",
                "academic_replication_protocol": "data/evaluation/academic_replication_protocol_v1.json",
                "academic_replication_decision": "data/evaluation/academic_replication_v1/decision.json",
                "container_runtime_evidence": "data/evaluation/container_runtime_verification_v1.json",
                "reproduction_release_evidence": "data/evaluation/reproduction_release_verification_v1.json",
                "all_must_pass_before_reproduction": True,
            }
        ),
        "reviewer_independence_is_required": (
            reviewer.get("independent_of_implementation_required") is True
            and reviewer.get("prior_code_contribution_to_frozen_commit_allowed")
            is False
            and reviewer.get("conflict_disclosure_required") is True
            and reviewer.get("public_identity_or_orcid_required") is True
            and reviewer.get("signed_attestation_required") is True
        ),
        "execution_budget_is_exact_and_zero_training": (
            execution.get("runs_total") == 132
            and execution.get("stages")
            == {
                "confirmatory": {
                    "pairs_total": 30,
                    "runs_total": 60,
                    "arms": ["M0", "M1"],
                    "runs_per_arm": 30,
                },
                "academic_replication": {
                    "pairs_per_arm": 24,
                    "runs_total": 72,
                    "arms": ["M0", "M1_predict_only", "M1"],
                    "runs_per_arm": 24,
                },
            }
            and execution.get("interim_analysis") is False
            and execution.get("optional_stopping") is False
            and execution.get("provider_calls_required") is False
            and execution.get("training_required") is False
        ),
        "comparison_contract_matches_formal_output": (
            comparison.get("confirmatory", {}).get("required_exact_fields")
            == [
                "schema_version",
                "protocol_id",
                "decision",
                "promotion_supported",
                "pairs_total",
                "primary.method",
                "primary.draws",
                "primary.seed",
                "behavior.changed_pairs",
                "promotion_gates",
                "secondary_metrics",
                "execution_identity",
            ]
            and comparison.get("confirmatory", {}).get("required_numeric_fields")
            == [
                "primary.point_delta",
                "primary.ci95_low",
                "primary.ci95_high",
                "behavior.changed_pair_fraction",
                "minimum_meaningful_delta_loss",
            ]
            and comparison.get("numeric_absolute_tolerance") == 1e-12
            and comparison.get("all_validity_gates_must_match") is True
            and comparison.get("material_deviation_forces_inconclusive") is True
            and comparison.get("negative_or_failed_results_must_not_be_relabelled")
            is True
            and comparison.get("academic_replication", {}).get(
                "required_exact_fields"
            )
            == [
                "schema_version",
                "protocol_id",
                "status",
                "passed",
                "branch",
                "confirmatory_decision",
                "conclusion",
                "runs_executed",
                "pairs_per_arm",
                "comparisons.M1_minus_M0.method",
                "comparisons.M1_minus_M0.draws",
                "comparisons.M1_minus_M0.seed",
                "comparisons.M1_minus_M0.candidate_all_external_gates",
                "comparisons.M1_minus_M1_predict_only.method",
                "comparisons.M1_minus_M1_predict_only.draws",
                "comparisons.M1_minus_M1_predict_only.seed",
                "comparisons.M1_minus_M1_predict_only.candidate_all_external_gates",
                "mechanism.changed_pairs",
                "mechanism.replicated_promotion",
                "mechanism.replicated_equivalence",
                "mechanism.replicated_harm",
                "mechanism.planning_contribution",
                "mechanism.planning_nonadoption",
                "mechanism.bounded_inconclusive",
                "external_validity.source",
                "external_validity.checks",
                "material_deviations",
                "gates",
                "execution_identity",
                "training_executed",
                "provider_calls_made",
            ]
            and comparison.get("academic_replication", {}).get(
                "required_numeric_fields"
            )
            == [
                "comparisons.M1_minus_M0.point_delta",
                "comparisons.M1_minus_M0.ci95_low",
                "comparisons.M1_minus_M0.ci95_high",
                "comparisons.M1_minus_M1_predict_only.point_delta",
                "comparisons.M1_minus_M1_predict_only.ci95_low",
                "comparisons.M1_minus_M1_predict_only.ci95_high",
                "mechanism.changed_pair_fraction",
                "external_validity.full_candidate_mean_passes_per_team_match",
                "external_validity.full_candidate_mean_shots_per_team_match",
            ]
        ),
        "outputs_are_confined_and_exact": (
            set(outputs)
            == {
                "review",
                "reproduced_confirmatory_decision",
                "reproduced_academic_replication_decision",
                "execution_log",
            }
            and all(_confined(ROOT, value) is not None for value in outputs.values())
        ),
        "execution_remains_not_performed": (
            state.get("reviewer_assigned") is False
            and state.get("runs_executed") == 0
            and state.get("confirmatory_pairs_completed") == 0
            and state.get("academic_replication_pairs_per_arm_completed") == 0
            and state.get("review_available") is False
            and state.get("reproduction_status") == "not_performed"
        ),
    }


def _prerequisite_checks(root: Path, protocol: dict[str, Any]) -> dict[str, bool]:
    paths = {
        key: _confined(root, value)
        for key, value in (protocol.get("prerequisites") or {}).items()
        if key != "all_must_pass_before_reproduction"
    }
    loaded: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        if path is not None and path.is_file():
            try:
                loaded[key] = _read_json(path)
            except (OSError, ValueError):
                pass
    container = loaded.get("container_runtime_evidence") or {}
    release = loaded.get("reproduction_release_evidence") or {}
    decision = loaded.get("confirmatory_decision") or {}
    formal = loaded.get("confirmatory_protocol") or {}
    replication_protocol = loaded.get("academic_replication_protocol") or {}
    replication = loaded.get("academic_replication_decision") or {}
    container_path = paths.get("container_runtime_evidence")
    release_path = paths.get("reproduction_release_evidence")
    container_checks = container.get("checks") or {}
    return {
        "confirmatory_protocol_is_frozen": (
            formal.get("schema_version") == 2
            and formal.get("state") == "preregistered_not_executed"
            and formal.get("design", {}).get("runs_total") == 60
        ),
        "confirmatory_decision_is_complete": (
            decision.get("schema_version") == 2
            and decision.get("protocol_id") == formal.get("protocol_id")
            and decision.get("pairs_total") == 30
            and decision.get("decision") in BRANCH_BY_DECISION
        ),
        "academic_replication_protocol_is_frozen": (
            replication_protocol.get("schema_version") == 1
            and replication_protocol.get("protocol_id")
            == "gfs-result-contingent-mechanism-replication-v1"
            and replication_protocol.get("state")
            == "preregistered_waiting_for_confirmatory_decision"
            and replication_protocol.get("design", {}).get("runs_total") == 72
        ),
        "academic_replication_decision_is_complete": (
            replication.get("schema_version") == 1
            and replication.get("protocol_id")
            == replication_protocol.get("protocol_id")
            and replication.get("runs_executed") == 72
            and replication.get("pairs_per_arm") == 24
            and replication.get("confirmatory_decision") == decision.get("decision")
            and replication.get("branch")
            == BRANCH_BY_DECISION.get(str(decision.get("decision") or ""))
            and isinstance(replication.get("execution_identity"), dict)
            and bool(replication.get("execution_identity"))
            and replication.get("training_executed") is False
            and replication.get("provider_calls_made") is False
        ),
        "container_runtime_gate_passed": (
            container.get("schema_version") == 2
            and container.get("status") == "passed"
            and container.get("passed") is True
            and container.get("image_built") is True
            and container.get("deployment_started") is True
            and COMMIT.fullmatch(str(container.get("source_revision") or ""))
            is not None
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(container.get("image_id") or ""),
            )
            is not None
            and container.get("registry_access_attempted") is True
            and set(container_checks) == CONTAINER_CHECKS
            and all(container_checks.values())
            and container_path is not None
            and _artifact_checks(root, container.get("artifact_sha256"))
        ),
        "reproduction_release_contract_passed": (
            release.get("passed") is True
            and release.get("status") == "passed_code_contract"
            and release_path is not None
            and _artifact_checks(root, release.get("artifact_sha256"))
        ),
    }


def protocol_report(
    root: Path = ROOT,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol)
    prerequisites = _prerequisite_checks(root, protocol)
    review_path = _confined(root, protocol.get("outputs", {}).get("review"))
    ready = all(prerequisites.values())
    return {
        "schema_version": 1,
        "verification": "gfs_independent_reproduction_preregistration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ready_for_independent_reviewer"
            if all(checks.values()) and ready
            else "registered_waiting_for_prerequisites"
            if all(checks.values())
            else "failed"
        ),
        "passed": all(checks.values()),
        "ready_to_start": ready,
        "review_available": bool(review_path and review_path.is_file()),
        "checks": checks,
        "prerequisite_checks": prerequisites,
        "protocol_sha256": _sha256(protocol_path),
        "artifact_sha256": {
            "data/evaluation/independent_reproduction_protocol_v1.json": _sha256(
                protocol_path
            ),
            "docs/INDEPENDENT_REPRODUCTION.md": _sha256(
                root / "docs/INDEPENDENT_REPRODUCTION.md"
            ),
            "scripts/verify_independent_reproduction.py": _sha256(Path(__file__)),
        },
        "external_calls_made": False,
        "runs_executed": 0,
        "matches_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
    }


def _dotted(payload: dict[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def compare_decisions(
    reference: dict[str, Any],
    reproduced: dict[str, Any],
    protocol: dict[str, Any],
    *,
    stage: str = "confirmatory",
) -> dict[str, Any]:
    settings = protocol["comparison"]
    if stage not in {"confirmatory", "academic_replication"}:
        raise ValueError(f"unknown reproduction stage: {stage}")
    fields = settings[stage]
    exact_mismatches: list[str] = []
    for field in fields["required_exact_fields"]:
        try:
            if _dotted(reference, field) != _dotted(reproduced, field):
                exact_mismatches.append(field)
        except KeyError:
            exact_mismatches.append(field)
    numeric_deltas: dict[str, float | None] = {}
    tolerance = float(settings["numeric_absolute_tolerance"])
    for field in fields["required_numeric_fields"]:
        try:
            left = float(_dotted(reference, field))
            right = float(_dotted(reproduced, field))
            delta = abs(left - right)
            numeric_deltas[field] = delta if math.isfinite(delta) else None
        except (KeyError, TypeError, ValueError):
            numeric_deltas[field] = None
    numeric_match = all(
        value is not None and value <= tolerance for value in numeric_deltas.values()
    )
    return {
        "stage": stage,
        "exact_fields_match": not exact_mismatches,
        "exact_field_mismatches": exact_mismatches,
        "numeric_fields_match": numeric_match,
        "numeric_absolute_deltas": numeric_deltas,
        "numeric_absolute_tolerance": tolerance,
        "reproduction_matches": not exact_mismatches and numeric_match,
    }


def _artifact_checks(root: Path, artifacts: Any) -> bool:
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for relative, digest in artifacts.items():
        path = _confined(root, relative)
        if (
            path is None
            or not path.is_file()
            or not SHA256.fullmatch(str(digest or ""))
            or _sha256(path) != digest
        ):
            return False
    return True


def verify_review(
    review_path: Path,
    *,
    root: Path = ROOT,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    protocol_checks = validate_protocol(protocol)
    prerequisite_checks = _prerequisite_checks(root, protocol)
    review = _read_json(review_path)
    reference_paths = {
        "confirmatory": _confined(
            root, protocol["prerequisites"]["confirmatory_decision"]
        ),
        "academic_replication": _confined(
            root, protocol["prerequisites"]["academic_replication_decision"]
        ),
    }
    reproduced_paths = {
        "confirmatory": _confined(
            root, protocol["outputs"]["reproduced_confirmatory_decision"]
        ),
        "academic_replication": _confined(
            root,
            protocol["outputs"]["reproduced_academic_replication_decision"],
        ),
    }
    for stage, path in reference_paths.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{stage} reference decision is unavailable")
    for stage, path in reproduced_paths.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{stage} reproduced decision is unavailable")
    references = {stage: _read_json(path) for stage, path in reference_paths.items()}
    reproduced = {stage: _read_json(path) for stage, path in reproduced_paths.items()}
    comparisons = {
        stage: compare_decisions(
            references[stage], reproduced[stage], protocol, stage=stage
        )
        for stage in ("confirmatory", "academic_replication")
    }
    reviewer = review.get("reviewer") or {}
    source = review.get("source") or {}
    execution = review.get("execution") or {}
    identities = review.get("identities") or {}
    deviations = review.get("deviations")
    if not isinstance(deviations, list):
        deviations = []
    deviation_shape = all(
        isinstance(row, dict)
        and set(row) == {"classification", "description"}
        and row.get("classification") in {"none", "nonmaterial", "material"}
        and isinstance(row.get("description"), str)
        and bool(row["description"].strip())
        for row in deviations
    )
    material_deviation = any(
        row.get("classification") == "material"
        for row in deviations
        if isinstance(row, dict)
    )
    stage_conclusions = {
        stage: "reproduced" if result["reproduction_matches"] else "not_reproduced"
        for stage, result in comparisons.items()
    }
    derived_conclusion = (
        "inconclusive"
        if material_deviation
        else "reproduced"
        if all(result["reproduction_matches"] for result in comparisons.values())
        else "not_reproduced"
    )
    formal_path = _confined(
        root,
        protocol["prerequisites"]["confirmatory_protocol"],
    )
    replication_protocol_path = _confined(
        root,
        protocol["prerequisites"]["academic_replication_protocol"],
    )
    container_path = _confined(
        root,
        protocol["prerequisites"]["container_runtime_evidence"],
    )
    release_path = _confined(
        root,
        protocol["prerequisites"]["reproduction_release_evidence"],
    )
    formal = _read_json(formal_path) if formal_path and formal_path.is_file() else {}
    replication_protocol = (
        _read_json(replication_protocol_path)
        if replication_protocol_path and replication_protocol_path.is_file()
        else {}
    )
    checkpoint_path = _confined(root, formal.get("candidate", {}).get("checkpoint"))
    code_paths = list(
        dict.fromkeys(
            [
                *(formal.get("integrity", {}).get("code_identity_files") or []),
                *(
                    replication_protocol.get("integrity", {}).get(
                        "code_identity_files"
                    )
                    or []
                ),
            ]
        )
    )
    data_paths = (
        replication_protocol.get("integrity", {}).get("data_identity_files") or []
    )
    expected_code = {
        relative: _sha256(path)
        for relative in code_paths
        if (path := _confined(root, relative)) is not None and path.is_file()
    }
    expected_data = {
        relative: _sha256(path)
        for relative in data_paths
        if (path := _confined(root, relative)) is not None and path.is_file()
    }
    checks = {
        "protocol_is_valid": all(protocol_checks.values()),
        "all_prerequisites_pass": all(prerequisite_checks.values()),
        "review_schema_and_protocol": (
            review.get("schema_version") == 1
            and review.get("protocol_id") == protocol["protocol_id"]
            and set(review)
            == {
                "schema_version",
                "protocol_id",
                "reviewer",
                "source",
                "execution",
                "identities",
                "comparison",
                "deviations",
                "artifact_sha256",
                "signed_at",
                "attestation",
            }
        ),
        "reviewer_is_public_independent_and_signed": (
            isinstance(reviewer.get("public_name"), str)
            and bool(reviewer["public_name"].strip())
            and ORCID.fullmatch(str(reviewer.get("orcid") or "")) is not None
            and isinstance(reviewer.get("affiliation"), str)
            and bool(reviewer["affiliation"].strip())
            and reviewer.get("independent_of_implementation") is True
            and reviewer.get("contributed_to_frozen_commit") is False
            and isinstance(reviewer.get("conflict_disclosure"), str)
            and bool(reviewer["conflict_disclosure"].strip())
            and _timestamp(review.get("signed_at"))
            and review.get("attestation") == REVIEWER_ATTESTATION
        ),
        "source_checkout_is_clean_and_identified": (
            COMMIT.fullmatch(str(source.get("commit_sha") or "")) is not None
            and source.get("commit_sha")
            == (
                _read_json(container_path).get("source_revision")
                if container_path
                else None
            )
            and source.get("clean_checkout") is True
            and isinstance(source.get("platform"), str)
            and bool(source["platform"].strip())
            and isinstance(source.get("python_version"), str)
            and bool(source["python_version"].strip())
        ),
        "execution_budget_and_boundaries_match": (
            execution.get("runs_total") == 132
            and execution.get("stages")
            == {
                "confirmatory": {
                    "pairs_completed": 30,
                    "runs_total": 60,
                    "runs_by_arm": {"M0": 30, "M1": 30},
                },
                "academic_replication": {
                    "pairs_per_arm_completed": 24,
                    "runs_total": 72,
                    "runs_by_arm": {
                        "M0": 24,
                        "M1_predict_only": 24,
                        "M1": 24,
                    },
                },
            }
            and execution.get("interim_analysis_performed") is False
            and execution.get("optional_stopping_performed") is False
            and execution.get("training_executed") is False
            and execution.get("provider_calls_made") is False
        ),
        "frozen_identities_match_current_artifacts": (
            formal_path is not None
            and formal_path.is_file()
            and identities.get("formal_protocol_sha256") == _sha256(formal_path)
            and replication_protocol_path is not None
            and replication_protocol_path.is_file()
            and identities.get("academic_replication_protocol_sha256")
            == _sha256(replication_protocol_path)
            and checkpoint_path is not None
            and checkpoint_path.is_file()
            and identities.get("checkpoint_sha256") == _sha256(checkpoint_path)
            and formal.get("candidate", {}).get("checkpoint_sha256")
            == _sha256(checkpoint_path)
            and len(expected_code) == len(code_paths)
            and identities.get("critical_code_sha256") == expected_code
            and len(expected_data) == len(data_paths)
            and identities.get("replication_data_sha256") == expected_data
            and container_path is not None
            and container_path.is_file()
            and identities.get("container_runtime_report_sha256")
            == _sha256(container_path)
            and release_path is not None
            and release_path.is_file()
            and identities.get("reproduction_release_report_sha256")
            == _sha256(release_path)
        ),
        "decisions_are_content_addressed": (
            review.get("comparison", {}).get("reference_decision_sha256")
            == {
                stage: _sha256(path) for stage, path in reference_paths.items()
            }
            and review.get("comparison", {}).get("reproduced_decision_sha256")
            == {
                stage: _sha256(path) for stage, path in reproduced_paths.items()
            }
        ),
        "deviations_are_complete_and_classified": (
            deviation_shape and bool(deviations)
        ),
        "reviewer_conclusion_matches_computed_result": (
            review.get("comparison", {}).get("stage_conclusions")
            == stage_conclusions
            and
            review.get("comparison", {}).get("reviewer_conclusion")
            == derived_conclusion
        ),
        "review_artifacts_are_current_and_confined": _artifact_checks(
            root,
            review.get("artifact_sha256"),
        ),
    }
    valid = all(checks.values())
    evidence_paths = {
        "review": review_path,
        "reference_confirmatory": reference_paths["confirmatory"],
        "reference_academic_replication": reference_paths[
            "academic_replication"
        ],
        "reproduced_confirmatory": reproduced_paths["confirmatory"],
        "reproduced_academic_replication": reproduced_paths[
            "academic_replication"
        ],
    }
    artifact_sha256 = {
        path.resolve().relative_to(root.resolve()).as_posix(): _sha256(path)
        for path in evidence_paths.values()
    }
    return {
        "schema_version": 1,
        "verification": "gfs_independent_reproduction_review",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (f"verified_{derived_conclusion}" if valid else "invalid_review"),
        "passed": valid,
        "independent_review_available": valid,
        "scope": "full_study_132_runs",
        "stages_verified": ["confirmatory", "academic_replication"],
        "reproduction_matches": valid
        and all(result["reproduction_matches"] for result in comparisons.values()),
        "reviewer_conclusion": (
            derived_conclusion
            if valid
            else review.get("comparison", {}).get("reviewer_conclusion")
        ),
        "checks": checks,
        "prerequisite_checks": prerequisite_checks,
        "computed_comparison": comparisons,
        "material_deviation": material_deviation,
        "review_sha256": _sha256(review_path),
        "artifact_sha256": artifact_sha256,
        "external_calls_made": False,
        "runs_executed_by_verifier": 0,
        "matches_executed_by_verifier": 0,
        "training_executed_by_verifier": False,
        "provider_calls_made_by_verifier": False,
    }


def _atomic_write(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}-",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = (
        verify_review(args.review) if args.review is not None else protocol_report()
    )
    if args.out is not None:
        _atomic_write(args.out, report, overwrite=args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
