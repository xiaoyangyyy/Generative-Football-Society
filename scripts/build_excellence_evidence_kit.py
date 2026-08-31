#!/usr/bin/env python3
"""Build a deterministic, template-only kit for outstanding excellence gates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.product_validation_study import (  # noqa: E402
    OBSERVER_ATTESTATION as PRODUCT_OBSERVER_ATTESTATION,
    REVIEWER_ATTESTATION,
)
from scripts.product_value_study import (  # noqa: E402
    OBSERVER_ATTESTATION as VALUE_OBSERVER_ATTESTATION,
)
from scripts.run_production_validation import ATTESTATION as PRODUCTION_ATTESTATION  # noqa: E402
from scripts.verify_independent_reproduction import REVIEWER_ATTESTATION as REPRO_ATTESTATION  # noqa: E402
from scripts.verify_security_closure import ATTESTATION as SECURITY_ATTESTATION  # noqa: E402
from src.infrastructure import file_sha256  # noqa: E402

KIT_ID = "gfs-excellence-evidence-kit-v1"
KIT_ROOT = Path("build/evidence-kits")
SECRET_PATTERN = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")
PLACEHOLDER = "REPLACE_WITH_VERIFIED_VALUE"
PROTOCOLS = {
    "security": "data/evaluation/security_closure_protocol_v1.json",
    "product": "data/evaluation/product_validation_protocol_v1.json",
    "production": "data/evaluation/production_validation_protocol_v1.json",
    "value": "data/evaluation/product_value_validation_protocol_v1.json",
    "independent": "data/evaluation/independent_action_reproduction_protocol_v2.json",
    "paper": "data/evaluation/action_paper_finalization_protocol_v2.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


def _template_header(protocol_id: str) -> dict[str, Any]:
    return {
        "template_only": True,
        "template_notice": (
            "NOT EVIDENCE. Replace every placeholder and remove template-only "
            "fields before validation."
        ),
        "schema_version": 1,
        "protocol_id": protocol_id,
    }


def build_template_files(root: Path = ROOT) -> dict[str, bytes]:
    protocols = {
        name: _read_json(root / relative) for name, relative in PROTOCOLS.items()
    }
    product = protocols["product"]
    value = protocols["value"]
    independent = protocols["independent"]
    security = protocols["security"]
    production = protocols["production"]

    security_receipt = {
        "template_only": True,
        "provider": "deepseek",
        "status": "REPLACE_WITH_REVOKED_STATUS",
        "redacted": True,
        "non_secret_receipt_reference": PLACEHOLDER,
    }
    security_attestation = {
        **_template_header(security["protocol_id"]),
        "provider": security["incident"]["provider"],
        "credential_revoked": False,
        "revoked_at": PLACEHOLDER,
        "revocation_evidence": "security/receipt.json",
        "revocation_evidence_sha256": PLACEHOLDER,
        "evidence_contains_secret": False,
        "replacement_credential_generated": False,
        "replacement_storage": "not_generated",
        "repository_secret_scan": {
            "commit_sha": PLACEHOLDER,
            "boundary_aware_secret_match_count": None,
        },
        "signed_at": PLACEHOLDER,
        "attestation": SECURITY_ATTESTATION,
    }
    product_record = {
        **_template_header(product["protocol_id"]),
        "participant_id": "REPLACE_WITH_PSEUDONYMOUS_PARTICIPANT_ID",
        "target_role": product["population"]["target_roles"][0],
        "consent": False,
        "session_started_at": PLACEHOLDER,
        "session_ended_at": PLACEHOLDER,
        "moderator_id": "REPLACE_WITH_PSEUDONYMOUS_MODERATOR_ID",
        "tasks": [
            {
                "task_id": task["task_id"],
                "completed": False,
                "critical_error": False,
                "duration_seconds": None,
                "assistance_count": None,
                "evidence_sha256": None,
            }
            for task in product["tasks"]
        ],
        "sus_responses": [],
        "qualitative_tags": [],
        "excluded": False,
        "exclusion_reason": None,
        "observer_attestation": PRODUCT_OBSERVER_ATTESTATION,
    }
    review_common = {
        "independent_of_implementation": False,
        "organization": PLACEHOLDER,
        "signed_at": PLACEHOLDER,
        "attestation": REVIEWER_ATTESTATION,
        "artifact_sha256": {},
        "passed": False,
        "unresolved_critical_findings": None,
    }
    external_review = {
        **_template_header(product["protocol_id"]),
        "accessibility": {
            **review_common,
            "reviewer_id": "REPLACE_WITH_PSEUDONYMOUS_ACCESSIBILITY_REVIEWER_ID",
            "standard": product["external_review"]["accessibility_standard"],
        },
        "security": {
            **review_common,
            "reviewer_id": "REPLACE_WITH_PSEUDONYMOUS_SECURITY_REVIEWER_ID",
            "method": "independent_application_security_review",
        },
        "issue_resolution": {
            "findings_total": None,
            "unresolved_critical_count": None,
            "risk_accepted_critical_count": None,
            "all_critical_findings_closed": False,
        },
    }
    value_record = {
        **_template_header(value["protocol_id"]),
        "participant_id": "REPLACE_WITH_PSEUDONYMOUS_PARTICIPANT_ID",
        "target_role": value["population"]["target_roles"][0],
        "consent": False,
        "moderator_id": "REPLACE_WITH_PSEUDONYMOUS_MODERATOR_ID",
        "sequence": "AB",
        "session_started_at": PLACEHOLDER,
        "session_ended_at": PLACEHOLDER,
        "conditions": [
            {
                "condition": "gfs_studio",
                "case_pack": "case_pack_alpha",
                "completed": False,
                "correct": False,
                "critical_error": False,
                "duration_seconds": None,
                "evidence_sha256": None,
            },
            {
                "condition": "manual_baseline",
                "case_pack": "case_pack_beta",
                "completed": False,
                "correct": False,
                "critical_error": False,
                "duration_seconds": None,
                "evidence_sha256": None,
            },
        ],
        "excluded": False,
        "exclusion_reason": None,
        "observer_attestation": VALUE_OBSERVER_ATTESTATION,
    }
    deployment_attestation = {
        **_template_header(production["protocol_id"]),
        "operator_id": "REPLACE_WITH_PSEUDONYMOUS_OPERATOR_ID",
        "signed_at": PLACEHOLDER,
        "attestation": PRODUCTION_ATTESTATION,
        "volume_id": "REPLACE_WITH_PSEUDONYMOUS_VOLUME_ID",
        "container_image_digest": PLACEHOLDER,
        "deployment_instance_ids": [],
        "forced_termination_observed": False,
        "same_volume_persisted": False,
        "raw_host_identifiers_included": False,
    }
    independent_review = {
        **_template_header(independent["protocol_id"]),
        "reviewer": {
            "public_name": PLACEHOLDER,
            "orcid": PLACEHOLDER,
            "affiliation": PLACEHOLDER,
            "independent_of_implementation": False,
            "contributed_to_frozen_commit": False,
            "conflict_disclosure": PLACEHOLDER,
        },
        "source": {
            "commit_sha": PLACEHOLDER,
            "clean_checkout": False,
            "platform": PLACEHOLDER,
            "python_version": PLACEHOLDER,
        },
        "execution": {
            "runs_total": 0,
            "stages": {
                "confirmatory": {
                    "pairs_completed": 0,
                    "runs_total": 0,
                    "runs_by_arm": {"M0": 0, "M1": 0},
                },
                "academic_replication": {
                    "pairs_per_arm_completed": 0,
                    "runs_total": 0,
                    "runs_by_arm": {"M0": 0, "M1_predict_only": 0, "M1": 0},
                },
            },
            "interim_analysis_performed": False,
            "optional_stopping_performed": False,
            "training_executed": False,
            "provider_calls_made": False,
        },
        "identities": {
            "formal_protocol_sha256": PLACEHOLDER,
            "academic_replication_protocol_sha256": PLACEHOLDER,
            "checkpoint_sha256": PLACEHOLDER,
            "critical_code_sha256": {},
            "replication_data_sha256": {},
            "container_runtime_report_sha256": PLACEHOLDER,
            "reproduction_release_report_sha256": PLACEHOLDER,
        },
        "comparison": {
            "reference_decision_sha256": {},
            "reproduced_decision_sha256": {},
            "stage_conclusions": {
                "confirmatory": PLACEHOLDER,
                "academic_replication": PLACEHOLDER,
            },
            "reviewer_conclusion": PLACEHOLDER,
        },
        "deviations": [],
        "artifact_sha256": {},
        "signed_at": PLACEHOLDER,
        "attestation": REPRO_ATTESTATION,
    }
    manuscript = """# GFS completed-results manuscript template

TEMPLATE ONLY — this is not a completed manuscript and contains no result.
Copy the preexecution draft only after both study stages and independent
reproduction are complete. Build the result ledger, replace every placeholder,
and then run the finalization verifier.

REPLACE_WITH_COMPLETED_STAGE_MARKERS

## Abstract

REPLACE_WITH_EVIDENCE_LOCKED_ABSTRACT

## 5. Confirmatory results

REPLACE_WITH_LEDGER_DERIVED_CONFIRMATORY_RESULTS

## 7. Discussion

REPLACE_WITH_RESULT_CONTINGENT_DISCUSSION

## 8. Limitations, validity, and ethics

REPLACE_WITH_LIMITATIONS_NEGATIVE_RESULTS_AND_DEVIATIONS

## 9. Reproducibility and independent review

REPLACE_WITH_SIGNED_REPRODUCTION_SUMMARY_AND_RESULT_LEDGER

## References

REPLACE_WITH_RESOLVED_REFERENCES
"""
    readme = f"""# GFS Excellence Evidence Kit

Kit: `{KIT_ID}`

Every JSON file in this archive is a deliberately invalid template, not
evidence. Never add names, email addresses, access tokens, API keys, raw free
text, host names, or other direct identifiers. Use pseudonymous IDs and hashes
only where the frozen protocol allows them.

Workflow:

1. Read the matching protocol and study guide in the repository.
2. Copy a `.template.json` file outside this archive.
3. Replace every placeholder using contemporaneously observed evidence.
4. Remove `template_only` and `template_notice`; validators reject unknown
   fields and must continue to do so.
5. Keep failures, exclusions, negative results, and deviations. Never improve
   values to satisfy a threshold.
6. Run the exact command shown by `python gfs.py studio excellence`.
7. Treat a passing verifier with current hashes as the only machine gate.

The kit generator makes no provider call and runs no match, study, container,
training job, or formal experiment.
"""
    files = {
        "README.md": readme.encode(),
        "security/receipt.template.json": _json_bytes(security_receipt),
        "security/attestation.template.json": _json_bytes(security_attestation),
        "product_validation/participant_record.template.json": _json_bytes(product_record),
        "product_validation/external_review.template.json": _json_bytes(external_review),
        "product_value/participant_record.template.json": _json_bytes(value_record),
        "production/deployment_attestation.template.json": _json_bytes(deployment_attestation),
        "independent_reproduction/review.template.json": _json_bytes(independent_review),
        "paper/PAPER_FINAL.template.md": manuscript.encode(),
    }
    manifest = {
        "schema_version": 1,
        "kit_id": KIT_ID,
        "template_only": True,
        "files": {
            path: hashlib.sha256(content).hexdigest()
            for path, content in sorted(files.items())
        },
        "protocol_sha256": {
            relative: file_sha256(root / relative)
            for relative in PROTOCOLS.values()
        },
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
    }
    files["manifest.json"] = _json_bytes(manifest)
    return files


def build_archive(root: Path = ROOT) -> tuple[bytes, dict[str, bytes]]:
    files = build_template_files(root)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(files.items()):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return output.getvalue(), files


def verify_kit(root: Path = ROOT) -> dict[str, Any]:
    archive, files = build_archive(root)
    json_templates = [
        json.loads(content) for path, content in files.items()
        if path.endswith(".template.json")
    ]
    checks = {
        "template_inventory_is_exact": set(files) == {
            "README.md", "manifest.json",
            "security/receipt.template.json",
            "security/attestation.template.json",
            "product_validation/participant_record.template.json",
            "product_validation/external_review.template.json",
            "product_value/participant_record.template.json",
            "production/deployment_attestation.template.json",
            "independent_reproduction/review.template.json",
            "paper/PAPER_FINAL.template.md",
        },
        "all_json_templates_are_explicitly_non_evidence": (
            len(json_templates) == 7
            and all(row.get("template_only") is True for row in json_templates)
        ),
        "templates_contain_placeholders": all(
            b"REPLACE_WITH_" in content
            for path, content in files.items() if path.endswith(".template.json")
        ),
        "archive_is_secret_free": SECRET_PATTERN.search(archive) is None,
        "archive_paths_are_relative_and_normalized": all(
            not Path(path).is_absolute()
            and ".." not in Path(path).parts
            and "\\" not in path
            for path in files
        ),
        "archive_is_deterministic": archive == build_archive_once(files),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_excellence_evidence_kit",
        "status": "passed_template_only_kit" if passed else "failed",
        "passed": passed,
        "kit_id": KIT_ID,
        "template_count": 7,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "checks": checks,
        "artifact_sha256": {
            **{
                relative: file_sha256(root / relative)
                for relative in PROTOCOLS.values()
            },
            "scripts/build_excellence_evidence_kit.py": file_sha256(Path(__file__)),
            "docs/EXCELLENCE_EVIDENCE_KIT.md": file_sha256(
                root / "docs/EXCELLENCE_EVIDENCE_KIT.md"
            ),
        },
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
    }


def build_archive_once(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(files.items()):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return output.getvalue()


def _confined_output(root: Path, output: Path) -> Path:
    target = (root / output).resolve() if not output.is_absolute() else output.resolve()
    allowed = (root / KIT_ROOT).resolve()
    try:
        target.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"evidence kit output must stay inside {KIT_ROOT.as_posix()}") from exc
    if target.suffix.lower() != ".zip":
        raise ValueError("evidence kit output must be a .zip file")
    return target


def _atomic_write(path: Path, content: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def materialize_kit(output: Path, *, root: Path = ROOT, overwrite: bool = False) -> Path:
    target = _confined_output(root, output)
    archive, _ = build_archive(root)
    _atomic_write(target, archive, overwrite=overwrite)
    return target


def _write_report(path: Path, report: dict[str, Any]) -> None:
    content = _json_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = verify_kit()
    if args.materialize is not None:
        target = materialize_kit(args.materialize, overwrite=args.overwrite)
        report["materialized_path"] = target.relative_to(ROOT).as_posix()
    if args.report is not None:
        _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
