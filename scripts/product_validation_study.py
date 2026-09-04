#!/usr/bin/env python3
"""Validate or analyze the preregistered GFS target-user study."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product.human_study import (  # noqa: E402
    ARCHIVE_LAYOUT,
    bind_records_to_registry,
    register_participant,
    validate_registry,
    verify_content_addressed_archive,
)


PROTOCOL_PATH = ROOT / "data/evaluation/product_validation_protocol_v1.json"
DEFAULT_REGISTRY_PATH = (
    ROOT / "data/evaluation/product_validation_v1/session_registry.json"
)
DEFAULT_RECORDS_PATH = (
    ROOT / "data/evaluation/product_validation_v1/participant_records.jsonl"
)
PARTICIPANT_ID = re.compile(r"^P-[A-Z0-9]{12}$")
MODERATOR_ID = re.compile(r"^M-[A-Z0-9]{8}$")
REVIEWER_ID = re.compile(r"^R-[A-Z0-9]{10}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
OBSERVER_ATTESTATION = "recorded contemporaneously under the frozen protocol"
REVIEWER_ATTESTATION = "I independently reviewed the recorded GFS evidence"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an RFC 3339 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _confined_output(relative: Any) -> bool:
    if not isinstance(relative, str) or not relative:
        return False
    try:
        (ROOT / relative).resolve().relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return False
    return not Path(relative).is_absolute()


def validate_protocol(protocol: dict[str, Any]) -> dict[str, bool]:
    population = protocol.get("population") or {}
    measurement = protocol.get("measurement") or {}
    review = protocol.get("external_review") or {}
    integrity = protocol.get("integrity") or {}
    execution = protocol.get("execution") or {}
    tasks = protocol.get("tasks") or []
    outputs = protocol.get("outputs") or {}
    registration = protocol.get("registration") or {}
    archive = protocol.get("evidence_archive") or {}
    amendments = protocol.get("preexecution_amendments") or []
    return {
        "schema_and_state_are_preregistered": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id") == "gfs-studio-target-user-validation-v1"
            and protocol.get("state") == "preregistered_not_executed"
        ),
        "population_and_sample_are_frozen": (
            population.get("target_roles")
            == ["football_analyst", "research_engineer", "product_operator"]
            and population.get("minimum_valid_participants") == 10
            and population.get("minimum_per_role") == 2
            and population.get("predeclared_exclusions")
            == ["consent_withdrawn", "infrastructure_failure_before_first_task"]
            and population.get("performance_based_exclusion") is False
            and population.get("optional_stopping") is False
        ),
        "task_order_and_success_contract_are_frozen": (
            [row.get("task_id") for row in tasks]
            == [
                "create_or_load_studio",
                "interpret_release_readiness",
                "run_stable_match",
                "review_auditable_report",
                "backup_verify_and_recover",
            ]
            and len(tasks) == 5
            and all(row.get("core") is True for row in tasks)
            and all(row.get("success_definition") for row in tasks)
        ),
        "decision_thresholds_are_frozen": (
            measurement.get("primary_metric")
            == "participant_core_workflow_success_rate"
            and measurement.get("primary_threshold") == 0.9
            and measurement.get("sus_threshold") == 80.0
            and measurement.get("critical_error_threshold") == 0
            and measurement.get("sus_items") == 10
            and measurement.get("confidence_interval")
            == "wilson_95_percent_for_success_proportions"
        ),
        "external_reviews_are_independent_and_fail_closed": (
            review.get("required") is True
            and review.get("accessibility_standard") == "WCAG-2.2-AA"
            and review.get("independent_accessibility_reviewer") is True
            and review.get("independent_security_reviewer") is True
            and review.get("unresolved_critical_findings_allowed") == 0
            and review.get("critical_findings_may_be_risk_accepted") is False
        ),
        "privacy_and_analysis_integrity_are_explicit": (
            integrity.get("protocol_changes_after_first_participant") is False
            and integrity.get("participant_ids_are_pseudonymous") is True
            and integrity.get("raw_free_text_in_analysis_records") is False
            and integrity.get("unknown_record_fields_forbidden") is True
            and integrity.get("completed_tasks_require_sha256_evidence") is True
            and integrity.get("duplicate_participant_ids_forbidden") is True
            and integrity.get("all_valid_records_analyzed") is True
            and integrity.get("interim_release_decisions") is False
            and integrity.get("authoritative_session_registry_required") is True
            and integrity.get("all_registrations_analyzed") is True
            and integrity.get("record_identity_bound_to_registration") is True
            and integrity.get("evidence_hashes_verified_against_real_files") is True
        ),
        "registration_is_authoritative_and_premeasurement": (
            registration.get("registry_kind")
            == "gfs_human_study_session_registry_v1"
            and registration.get("participant_id_pattern")
            == r"^P-[A-Z0-9]{12}$"
            and registration.get("moderator_id_pattern") == r"^M-[A-Z0-9]{8}$"
            and registration.get("registration_before_measurement") is True
            and registration.get("consent_required") is True
            and registration.get("duplicate_registration_policy")
            == "idempotent_exact_match"
        ),
        "evidence_archive_requires_real_unique_bytes": (
            archive.get("layout") == ARCHIVE_LAYOUT
            and archive.get("file_name_is_exact_sha256") is True
            and archive.get("symlinks_allowed") is False
            and archive.get("all_referenced_content_verified_during_analysis")
            is True
            and archive.get("duplicate_digest_across_observations_allowed")
            is False
            and archive.get("archive_path_recorded_in_decision") is False
        ),
        "preexecution_amendment_is_zero_participant": (
            len(amendments) == 1
            and amendments[0].get("amendment_id")
            == "authoritative-human-study-intake-v1"
            and amendments[0].get("participants_observed_before_amendment") == 0
            and amendments[0].get("sessions_executed_before_amendment") == 0
            and amendments[0].get("results_available_before_amendment") is False
        ),
        "outputs_are_confined_and_exact": (
            set(outputs)
            == {"session_registry", "raw_records", "external_review", "decision"}
            and all(_confined_output(value) for value in outputs.values())
        ),
        "execution_remains_zero": (
            execution.get("participants_observed") == 0
            and execution.get("sessions_executed") == 0
            and execution.get("matches_executed") == 0
            and execution.get("training_executed") is False
            and execution.get("provider_calls_authorized") is False
            and execution.get("results_available") is False
        ),
    }


def protocol_report(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol)
    decision = ROOT / str((protocol.get("outputs") or {}).get("decision", ""))
    return {
        "schema_version": 1,
        "verification": "gfs_product_validation_preregistration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_preregistered_not_executed"
        if all(checks.values())
        else "failed",
        "passed": all(checks.values()),
        "study_executed": False,
        "result_available": decision.is_file(),
        "checks": checks,
        "protocol_sha256": _file_sha256(protocol_path),
        "artifact_sha256": {
            "data/evaluation/product_validation_protocol_v1.json": _file_sha256(
                protocol_path
            ),
            "docs/PRODUCT_VALIDATION_STUDY.md": _file_sha256(
                ROOT / "docs/PRODUCT_VALIDATION_STUDY.md"
            ),
            "scripts/product_validation_study.py": _file_sha256(Path(__file__)),
            "src/product/human_study.py": _file_sha256(
                ROOT / "src/product/human_study.py"
            ),
        },
        "external_calls_made": False,
        "participants_observed": 0,
        "matches_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"invalid JSON on record line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"record line {line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError("participant record file is empty")
    return records


def _validate_record(
    record: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "protocol_id",
        "registration_id",
        "participant_id",
        "target_role",
        "consent",
        "session_started_at",
        "session_ended_at",
        "moderator_id",
        "tasks",
        "sus_responses",
        "qualitative_tags",
        "excluded",
        "exclusion_reason",
        "observer_attestation",
    }
    if set(record) != expected_fields:
        raise ValueError("participant record fields must match the privacy allowlist")
    if record.get("schema_version") != 1:
        raise ValueError("participant record schema_version must be 1")
    if record.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("participant protocol_id mismatch")
    participant_id = str(record.get("participant_id") or "")
    if not PARTICIPANT_ID.fullmatch(participant_id):
        raise ValueError("participant_id must be pseudonymous")
    registration_id = str(record.get("registration_id") or "")
    if re.fullmatch(r"REG-[0-9A-F]{24}", registration_id) is None:
        raise ValueError(f"registration_id is invalid: {participant_id}")
    if record.get("observer_attestation") != OBSERVER_ATTESTATION:
        raise ValueError(f"missing observer attestation for {participant_id}")
    excluded = record.get("excluded")
    if not isinstance(excluded, bool):
        raise ValueError(f"excluded must be boolean for {participant_id}")
    if not isinstance(record.get("consent"), bool):
        raise ValueError(f"consent must be boolean for {participant_id}")
    role = record.get("target_role")
    if role not in protocol["population"]["target_roles"]:
        raise ValueError(f"invalid target role for {participant_id}")
    moderator_id = str(record.get("moderator_id") or "")
    if not MODERATOR_ID.fullmatch(moderator_id):
        raise ValueError(f"moderator_id must be pseudonymous: {participant_id}")
    started = _timestamp(record.get("session_started_at"))
    ended = _timestamp(record.get("session_ended_at"))
    if ended <= started:
        raise ValueError(f"non-positive session duration: {participant_id}")
    allowed_exclusions = set(protocol["population"]["predeclared_exclusions"])
    reason = record.get("exclusion_reason")
    if excluded:
        if reason not in allowed_exclusions:
            raise ValueError(f"invalid exclusion reason for {participant_id}")
        if any(
            record.get(field) != []
            for field in ("tasks", "sus_responses", "qualitative_tags")
        ):
            raise ValueError(
                f"excluded record contains behavioral data: {participant_id}"
            )
        return {
            "participant_id": participant_id,
            "registration_id": registration_id,
            "moderator_id": moderator_id,
            "session_started_at": record["session_started_at"],
            "role": role,
            "sequence": None,
            "excluded": True,
            "reason": reason,
        }
    if reason is not None:
        raise ValueError(f"valid participant has exclusion reason: {participant_id}")
    if record.get("consent") is not True:
        raise ValueError(f"valid participant lacks consent: {participant_id}")
    tasks = record.get("tasks")
    expected_ids = [row["task_id"] for row in protocol["tasks"]]
    if (
        not isinstance(tasks, list)
        or [row.get("task_id") for row in tasks] != expected_ids
    ):
        raise ValueError(f"task set/order mismatch: {participant_id}")
    normalized_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if set(task) != {
            "task_id",
            "completed",
            "critical_error",
            "duration_seconds",
            "assistance_count",
            "evidence_sha256",
        }:
            raise ValueError(f"task fields violate privacy allowlist: {participant_id}")
        completed = task.get("completed")
        critical = task.get("critical_error")
        duration = task.get("duration_seconds")
        assistance = task.get("assistance_count")
        evidence = task.get("evidence_sha256")
        if not isinstance(completed, bool) or not isinstance(critical, bool):
            raise ValueError(f"task booleans invalid: {participant_id}")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError(f"task duration invalid: {participant_id}")
        if (
            isinstance(assistance, bool)
            or not isinstance(assistance, int)
            or assistance < 0
        ):
            raise ValueError(f"assistance count invalid: {participant_id}")
        if completed and not SHA256.fullmatch(str(evidence or "")):
            raise ValueError(f"completed task lacks evidence hash: {participant_id}")
        if not completed and evidence is not None:
            raise ValueError(f"incomplete task has evidence hash: {participant_id}")
        normalized_tasks.append(
            {
                "task_id": task["task_id"],
                "completed": completed,
                "critical_error": critical,
                "duration_seconds": float(duration),
                "assistance_count": assistance,
                "evidence_sha256": evidence,
            }
        )
    elapsed = (ended - started).total_seconds()
    if sum(row["duration_seconds"] for row in normalized_tasks) > elapsed:
        raise ValueError(f"task durations exceed session window: {participant_id}")

    responses = record.get("sus_responses")
    if (
        not isinstance(responses, list)
        or len(responses) != protocol["measurement"]["sus_items"]
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in responses
        )
        or any(value < 1 or value > 5 for value in responses)
    ):
        raise ValueError(f"SUS responses invalid: {participant_id}")
    tags = record.get("qualitative_tags")
    allowed_tags = set(protocol["measurement"]["qualitative_tags"])
    if (
        not isinstance(tags, list)
        or len(tags) != len(set(tags))
        or any(tag not in allowed_tags for tag in tags)
    ):
        raise ValueError(f"qualitative tags invalid: {participant_id}")
    return {
        "participant_id": participant_id,
        "registration_id": registration_id,
        "moderator_id": moderator_id,
        "session_started_at": record["session_started_at"],
        "excluded": False,
        "role": role,
        "sequence": None,
        "tasks": normalized_tasks,
        "sus": _sus_score(responses),
        "duration_seconds": elapsed,
        "qualitative_tags": tags,
    }


def _sus_score(responses: list[int]) -> float:
    contribution = sum(
        value - 1 if index % 2 == 0 else 5 - value
        for index, value in enumerate(responses)
    )
    return contribution * 2.5


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _validate_artifact_map(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(name, str) and bool(name) and SHA256.fullmatch(str(digest or ""))
            for name, digest in value.items()
        )
    )


def _external_review_checks(
    review: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, bool]:
    accessibility = review.get("accessibility") or {}
    security = review.get("security") or {}
    issues = review.get("issue_resolution") or {}
    reviewer_ids = [
        accessibility.get("reviewer_id"),
        security.get("reviewer_id"),
    ]
    signed = True
    for section in (accessibility, security):
        try:
            _timestamp(section.get("signed_at"))
        except (TypeError, ValueError):
            signed = False
    return {
        "external_review_schema_and_protocol": (
            review.get("schema_version") == 1
            and review.get("protocol_id") == protocol["protocol_id"]
        ),
        "reviewers_are_distinct_and_independent": (
            len(set(reviewer_ids)) == 2
            and all(REVIEWER_ID.fullmatch(str(value or "")) for value in reviewer_ids)
            and accessibility.get("independent_of_implementation") is True
            and security.get("independent_of_implementation") is True
            and all(
                isinstance(section.get("organization"), str)
                and bool(section["organization"].strip())
                for section in (accessibility, security)
            )
        ),
        "reviews_are_signed_and_evidence_addressed": (
            signed
            and all(
                section.get("attestation") == REVIEWER_ATTESTATION
                and _validate_artifact_map(section.get("artifact_sha256"))
                for section in (accessibility, security)
            )
        ),
        "accessibility_review_passes_wcag_22_aa": (
            accessibility.get("standard") == "WCAG-2.2-AA"
            and accessibility.get("passed") is True
            and accessibility.get("unresolved_critical_findings") == 0
        ),
        "security_review_passes_without_critical_findings": (
            security.get("method") == "independent_application_security_review"
            and security.get("passed") is True
            and security.get("unresolved_critical_findings") == 0
        ),
        "critical_issues_are_closed_not_accepted": (
            isinstance(issues.get("findings_total"), int)
            and issues.get("findings_total") >= 0
            and issues.get("unresolved_critical_count") == 0
            and issues.get("risk_accepted_critical_count") == 0
            and issues.get("all_critical_findings_closed") is True
        ),
    }


def analyze(
    records_path: Path,
    external_review_path: Path,
    registry_path: Path,
    evidence_root: Path,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    protocol_checks = validate_protocol(protocol)
    if not all(protocol_checks.values()):
        raise ValueError("product validation protocol is invalid or has drifted")
    raw_records = _load_records(records_path)
    normalized = [_validate_record(row, protocol) for row in raw_records]
    registry_report = validate_registry(
        protocol_path=protocol_path,
        registry=registry_path,
    )
    registry_binding = bind_records_to_registry(normalized, registry_report)
    valid = [row for row in normalized if not row["excluded"]]
    excluded = [row for row in normalized if row["excluded"]]
    if not valid:
        raise ValueError("no valid participant records")

    role_counts = Counter(row["role"] for row in valid)
    workflow_successes = sum(
        all(task["completed"] and not task["critical_error"] for task in row["tasks"])
        for row in valid
    )
    task_successes = sum(task["completed"] for row in valid for task in row["tasks"])
    task_total = len(valid) * len(protocol["tasks"])
    critical_errors = sum(
        task["critical_error"] for row in valid for task in row["tasks"]
    )
    sus_scores = [row["sus"] for row in valid]
    tags = Counter(tag for row in valid for tag in row["qualitative_tags"])
    review = _read_json(external_review_path)
    review_checks = _external_review_checks(review, protocol)
    evidence_digests = [
        str(task["evidence_sha256"])
        for row in valid
        for task in row["tasks"]
        if task["evidence_sha256"] is not None
    ]
    for section_name in ("accessibility", "security"):
        artifact_map = (review.get(section_name) or {}).get("artifact_sha256") or {}
        evidence_digests.extend(str(value) for value in artifact_map.values())
    archive_report = verify_content_addressed_archive(
        evidence_root,
        evidence_digests,
    )
    workflow_rate = workflow_successes / len(valid)
    task_rate = task_successes / task_total
    mean_sus = mean(sus_scores)
    gates = {
        "minimum_valid_sample": (
            len(valid) >= protocol["population"]["minimum_valid_participants"]
        ),
        "minimum_role_coverage": all(
            role_counts[role] >= protocol["population"]["minimum_per_role"]
            for role in protocol["population"]["target_roles"]
        ),
        "core_workflow_success_threshold": (
            workflow_rate >= protocol["measurement"]["primary_threshold"]
        ),
        "mean_sus_threshold": mean_sus >= protocol["measurement"]["sus_threshold"],
        "zero_critical_task_errors": (
            critical_errors <= protocol["measurement"]["critical_error_threshold"]
        ),
        "external_review_bundle_passes": all(review_checks.values()),
        "all_records_bound_to_registry": all(
            value is True
            for key, value in registry_binding.items()
            if key.startswith("all_")
        ),
        "all_evidence_bytes_verified": (
            archive_report["verified_artifacts"] == len(evidence_digests)
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_product_validation"
        if passed
        else "failed_product_validation",
        "passed": passed,
        "participants_recorded": len(normalized),
        "participants_valid": len(valid),
        "participants_excluded": len(excluded),
        "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
        "role_counts": dict(sorted(role_counts.items())),
        "metrics": {
            "participant_core_workflow_success_rate": workflow_rate,
            "participant_core_workflow_success_wilson_95": _wilson(
                workflow_successes,
                len(valid),
            ),
            "aggregate_task_completion_rate": task_rate,
            "aggregate_task_completion_wilson_95": _wilson(
                task_successes,
                task_total,
            ),
            "mean_sus": mean_sus,
            "median_sus": median(sus_scores),
            "critical_task_errors": critical_errors,
            "median_session_duration_seconds": median(
                row["duration_seconds"] for row in valid
            ),
            "qualitative_tag_counts": dict(sorted(tags.items())),
        },
        "gates": gates,
        "external_review_checks": review_checks,
        "registry_checks": registry_binding,
        "evidence_archive": {
            "layout": archive_report["layout"],
            "verified_artifacts": archive_report["verified_artifacts"],
            "inventory_sha256": archive_report["inventory_sha256"],
        },
        "artifact_sha256": {
            "protocol": _file_sha256(protocol_path),
            "session_registry": _file_sha256(registry_path),
            "participant_records": _file_sha256(records_path),
            "external_review": _file_sha256(external_review_path),
            "analyzer": _file_sha256(Path(__file__)),
        },
        "external_calls_made_by_analyzer": False,
        "participants_observed_by_analyzer": 0,
        "matches_executed_by_analyzer": 0,
        "training_executed_by_analyzer": False,
        "provider_calls_made_by_analyzer": False,
        "limitations": [
            "The study evaluates the frozen tasks and recruited target-role sample, not every possible user or deployment context.",
            "The analyzer verifies referenced evidence bytes but does not independently prove the real-world identity of a pseudonymous participant or reviewer.",
            "Passing this study does not substitute for confirmatory scientific results or production deployment approval.",
        ],
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
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--external-review", type=Path)
    parser.add_argument("--participant-id")
    parser.add_argument("--target-role")
    parser.add_argument("--moderator-id")
    parser.add_argument("--confirm-consent", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.analyze and args.register:
        parser.error("--analyze and --register are mutually exclusive")
    if args.register:
        if args.out is not None or args.external_review is not None:
            parser.error("--register does not accept --out or --external-review")
        if args.evidence_root is not None or args.overwrite:
            parser.error("--register does not accept evidence or overwrite options")
        if not all((args.participant_id, args.target_role, args.moderator_id)):
            parser.error(
                "--register requires --participant-id, --target-role, and --moderator-id"
            )
        if not args.confirm_consent:
            parser.error("--register requires --confirm-consent")
        if not all(validate_protocol(_read_json(PROTOCOL_PATH)).values()):
            raise ValueError("cannot register under an invalid or drifted protocol")
        result = register_participant(
            project_root=ROOT,
            protocol_path=PROTOCOL_PATH,
            registry_path=args.registry or DEFAULT_REGISTRY_PATH,
            records_path=args.records or DEFAULT_RECORDS_PATH,
            participant_id=args.participant_id,
            target_role=args.target_role,
            moderator_id=args.moderator_id,
            consent_recorded=True,
        )
        public = {
            "schema_version": 1,
            "status": "registered" if result["created"] else "already_registered",
            "created": result["created"],
            "registration": result["registration"],
            "external_calls_made": False,
            "matches_executed": 0,
            "training_executed": False,
        }
        print(json.dumps(public, ensure_ascii=False, indent=2))
        return 0
    if not args.analyze:
        if any(
            value is not None
            for value in (
                args.records,
                args.registry,
                args.evidence_root,
                args.external_review,
                args.participant_id,
                args.target_role,
                args.moderator_id,
            )
        ) or args.confirm_consent:
            parser.error("study inputs require --register or --analyze")
        report = protocol_report()
        if args.out is not None:
            _atomic_write(args.out, report, overwrite=args.overwrite)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    if any((args.participant_id, args.target_role, args.moderator_id)) or args.confirm_consent:
        parser.error("participant registration options require --register")
    if (
        args.records is None
        or args.registry is None
        or args.evidence_root is None
        or args.external_review is None
        or args.out is None
    ):
        parser.error(
            "--analyze requires --records, --registry, --evidence-root, "
            "--external-review, and --out"
        )
    decision = analyze(
        args.records,
        args.external_review,
        args.registry,
        args.evidence_root,
    )
    _atomic_write(args.out, decision, overwrite=args.overwrite)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
