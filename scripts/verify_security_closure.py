#!/usr/bin/env python3
"""Audit or verify closure of an exposed provider credential without using it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "data/evaluation/security_closure_protocol_v2.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SECRET_PATTERN_VERSION = "gfs_known_credentials_v2"
SECRET_PATTERNS = {
    "openai_compatible_api_key": re.compile(
        rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"
    ),
    "github_classic_pat": re.compile(
        rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9_]{20,}"
    ),
    "github_fine_grained_pat": re.compile(
        rb"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}"
    ),
}
KNOWN_INCIDENTS = (
    {
        "incident_id": "deepseek_api_credential",
        "provider": "deepseek",
        "credential_family": "openai_compatible_api_key",
        "credential_value_must_not_be_recorded": True,
        "revocation_required": True,
        "replacement_storage": "local_environment_only",
    },
    {
        "incident_id": "github_classic_pat",
        "provider": "github",
        "credential_family": "github_classic_pat",
        "credential_value_must_not_be_recorded": True,
        "revocation_required": True,
        "replacement_storage": "local_environment_only",
    },
)
ATTESTATION = (
    "I revoked every known exposed credential, stored no credential value in "
    "evidence, and confined every replacement to the local environment"
)


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


def _current_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and COMMIT.fullmatch(revision) else None


def _origin_has_embedded_credential(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    url = completed.stdout.strip()
    if not re.match(r"https?://", url, flags=re.IGNORECASE):
        return False
    authority = url.split("://", 1)[1].split("/", 1)[0]
    if "@" not in authority:
        return False
    userinfo = unquote(authority.rsplit("@", 1)[0])
    if ":" in userinfo:
        return True
    encoded = userinfo.encode("utf-8", errors="ignore")
    return any(pattern.search(encoded) for pattern in SECRET_PATTERNS.values())


def _secret_match_counts(root: Path) -> dict[str, int]:
    counts = {name: 0 for name in SECRET_PATTERNS}
    excluded = {".git", ".pytest_cache", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            counts[name] += len(pattern.findall(data))
    return counts


def _secret_match_count(root: Path) -> int:
    return sum(_secret_match_counts(root).values())


def validate_protocol(protocol: dict[str, Any], root: Path = ROOT) -> dict[str, bool]:
    incidents = protocol.get("incidents") or []
    evidence = protocol.get("evidence_contract") or {}
    execution = protocol.get("current_execution") or {}
    output = _confined(root, protocol.get("output"))
    return {
        "schema_state_and_known_incidents_are_frozen": (
            protocol.get("schema_version") == 2
            and protocol.get("protocol_id")
            == "gfs-all-known-exposed-credential-closure-v2"
            and protocol.get("state")
            == "registered_waiting_for_all_user_revocations"
            and incidents == list(KNOWN_INCIDENTS)
        ),
        "attestation_allowlists_are_exact": (
            protocol.get("required_attestation_fields")
            == [
                "schema_version", "protocol_id", "incidents",
                "repository_secret_scan", "signed_at", "attestation",
            ]
            and protocol.get("required_incident_attestation_fields")
            == [
                "incident_id", "provider", "credential_revoked", "revoked_at",
                "revocation_evidence", "revocation_evidence_sha256",
                "evidence_contains_secret", "replacement_credential_generated",
                "replacement_storage",
            ]
        ),
        "redacted_evidence_contract_is_fail_closed": (
            evidence
            == {
                "revocation_evidence_root": "data/evaluation/security_closure_v2",
                "allowed_extensions": [".json", ".pdf", ".png"],
                "redaction_required": True,
                "distinct_receipt_per_incident_required": True,
                "current_commit_scan_required": True,
                "origin_remote_credential_forbidden": True,
                "secret_pattern_version": SECRET_PATTERN_VERSION,
                "boundary_aware_secret_matches_required": 0,
                "attestation_text": ATTESTATION,
            }
        ),
        "output_is_confined": output is not None,
        "execution_is_zero_and_all_incidents_remain_open": (
            execution
            == {
                "known_incident_count": len(KNOWN_INCIDENTS),
                "incidents_attested": 0,
                "credential_value_stored": False,
                "provider_calls_made": False,
                "training_executed": False,
                "matches_executed": 0,
            }
        ),
    }


def protocol_report(
    root: Path = ROOT,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol, root)
    secret_counts = _secret_match_counts(root)
    origin_has_credential = _origin_has_embedded_credential(root)
    checks.update({
        "repository_has_zero_boundary_aware_secret_matches": (
            sum(secret_counts.values()) == 0
        ),
        "origin_remote_contains_no_embedded_credential": (
            origin_has_credential is False
        ),
    })
    return {
        "schema_version": 2,
        "verification": "gfs_all_known_credential_closure_preregistration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "registered_waiting_for_all_user_revocations"
            if all(checks.values()) else "failed"
        ),
        "passed": all(checks.values()),
        "closure_complete": False,
        "known_incident_count": len(KNOWN_INCIDENTS),
        "revoked_incident_count": 0,
        "known_incidents_complete": False,
        "incident_ids": [
            incident["incident_id"] for incident in KNOWN_INCIDENTS
        ],
        "checks": checks,
        "artifact_sha256": {
            "data/evaluation/security_closure_protocol_v2.json": _sha256(protocol_path),
            "docs/SECURITY_CREDENTIAL_CLOSURE.md": _sha256(
                root / "docs/SECURITY_CREDENTIAL_CLOSURE.md"
            ),
            "scripts/verify_security_closure.py": _sha256(Path(__file__)),
        },
        "secret_pattern_version": SECRET_PATTERN_VERSION,
        "boundary_aware_secret_match_count": sum(secret_counts.values()),
        "boundary_aware_secret_matches_by_family": secret_counts,
        "origin_has_embedded_credential": origin_has_credential,
        "credential_value_stored": False,
        "provider_calls_made": False,
        "training_executed": False,
        "matches_executed": 0,
    }


def verify_closure(
    attestation_path: Path,
    *,
    root: Path = ROOT,
    protocol_path: Path = PROTOCOL_PATH,
    revision: str | None = None,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    protocol_checks = validate_protocol(protocol, root)
    evidence = protocol["evidence_contract"]
    evidence_root = _confined(root, evidence["revocation_evidence_root"])
    resolved_attestation = attestation_path.resolve()
    if evidence_root is None:
        raise ValueError("security evidence root is invalid")
    try:
        resolved_attestation.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError("attestation must stay inside the security evidence root") from exc
    attestation = _read_json(resolved_attestation)
    raw_incidents = attestation.get("incidents")
    raw_incidents = raw_incidents if isinstance(raw_incidents, list) else []
    expected_fields = set(protocol["required_attestation_fields"])
    expected_incident_fields = set(
        protocol["required_incident_attestation_fields"]
    )
    expected_by_id = {
        row["incident_id"]: row for row in KNOWN_INCIDENTS
    }
    observed_by_id: dict[str, Mapping[str, Any]] = {}
    incidents_have_exact_fields = True
    for row in raw_incidents:
        if not isinstance(row, Mapping) or set(row) != expected_incident_fields:
            incidents_have_exact_fields = False
            continue
        incident_id = row.get("incident_id")
        if not isinstance(incident_id, str) or incident_id in observed_by_id:
            incidents_have_exact_fields = False
            continue
        observed_by_id[incident_id] = row
    all_known_incidents_present = (
        incidents_have_exact_fields
        and len(raw_incidents) == len(KNOWN_INCIDENTS)
        and set(observed_by_id) == set(expected_by_id)
        and all(
            observed_by_id[incident_id].get("provider")
            == expected["provider"]
            for incident_id, expected in expected_by_id.items()
        )
    )
    receipts: list[Path] = []
    revoked_incident_count = 0
    every_receipt_valid = all_known_incidents_present
    every_replacement_safe = all_known_incidents_present
    for incident_id, expected in expected_by_id.items():
        row = observed_by_id.get(incident_id) or {}
        if (
            row.get("provider") == expected["provider"]
            and row.get("credential_revoked") is True
            and _timestamp(row.get("revoked_at"))
        ):
            revoked_incident_count += 1
        evidence_path = _confined(root, row.get("revocation_evidence"))
        evidence_is_bounded = False
        if evidence_path is not None:
            try:
                evidence_path.relative_to(evidence_root)
                evidence_is_bounded = (
                    evidence_path.suffix.casefold()
                    in set(evidence["allowed_extensions"])
                    and evidence_path != resolved_attestation
                )
            except ValueError:
                pass
        receipt_valid = bool(
            evidence_is_bounded
            and evidence_path is not None
            and evidence_path.is_file()
            and SHA256.fullmatch(
                str(row.get("revocation_evidence_sha256") or "")
            )
            is not None
            and _sha256(evidence_path)
            == row.get("revocation_evidence_sha256")
            and row.get("evidence_contains_secret") is False
        )
        every_receipt_valid = every_receipt_valid and receipt_valid
        if receipt_valid and evidence_path is not None:
            receipts.append(evidence_path)
        replacement_safe = (
            isinstance(row.get("replacement_credential_generated"), bool)
            and row.get("replacement_storage")
            in {"local_environment_only", "not_generated"}
            and (
                row.get("replacement_credential_generated") is True
                or row.get("replacement_storage") == "not_generated"
            )
        )
        every_replacement_safe = (
            every_replacement_safe and replacement_safe
        )
    receipts_are_distinct = (
        len(receipts) == len(KNOWN_INCIDENTS)
        and len({path.resolve() for path in receipts}) == len(KNOWN_INCIDENTS)
    )
    scan = attestation.get("repository_secret_scan") or {}
    actual_revision = revision if revision is not None else _current_revision(root)
    actual_match_counts = _secret_match_counts(root)
    actual_matches = sum(actual_match_counts.values())
    origin_has_credential = _origin_has_embedded_credential(root)
    zero_counts = {name: 0 for name in SECRET_PATTERNS}
    checks = {
        "protocol_is_valid": all(protocol_checks.values()),
        "attestation_schema_and_fields_are_exact": (
            set(attestation) == expected_fields
            and attestation.get("schema_version") == 2
            and attestation.get("protocol_id") == protocol["protocol_id"]
        ),
        "all_known_incidents_are_present_exactly_once": (
            all_known_incidents_present
        ),
        "every_incident_is_explicitly_revoked_and_signed": (
            revoked_incident_count == len(KNOWN_INCIDENTS)
            and _timestamp(attestation.get("signed_at"))
            and attestation.get("attestation") == ATTESTATION
        ),
        "every_redacted_receipt_is_distinct_confined_and_content_addressed": (
            every_receipt_valid and receipts_are_distinct
        ),
        "every_replacement_location_is_safe": every_replacement_safe,
        "current_revision_has_zero_boundary_aware_matches": (
            actual_revision is not None
            and scan
            == {
                "commit_sha": actual_revision,
                "secret_pattern_version": SECRET_PATTERN_VERSION,
                "boundary_aware_secret_match_count": 0,
                "boundary_aware_secret_matches_by_family": zero_counts,
            }
            and actual_matches == 0
        ),
        "origin_remote_contains_no_embedded_credential": (
            origin_has_credential is False
        ),
    }
    passed = all(checks.values())
    artifacts = {
        resolved_attestation.relative_to(root.resolve()).as_posix(): _sha256(
            resolved_attestation
        )
    }
    for evidence_path in receipts:
        artifacts[
            evidence_path.resolve().relative_to(root.resolve()).as_posix()
        ] = _sha256(evidence_path)
    return {
        "schema_version": 2,
        "verification": "gfs_all_known_credential_closure",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified_closed" if passed else "failed",
        "passed": passed,
        "closure_complete": passed,
        "known_incident_count": len(KNOWN_INCIDENTS),
        "revoked_incident_count": revoked_incident_count,
        "known_incidents_complete": passed,
        "incident_ids": [
            incident["incident_id"] for incident in KNOWN_INCIDENTS
        ],
        "providers": sorted({
            incident["provider"] for incident in KNOWN_INCIDENTS
        }),
        "checks": checks,
        "artifact_sha256": artifacts,
        "secret_pattern_version": SECRET_PATTERN_VERSION,
        "boundary_aware_secret_match_count": actual_matches,
        "boundary_aware_secret_matches_by_family": actual_match_counts,
        "origin_has_embedded_credential": origin_has_credential,
        "credential_value_stored": False,
        "provider_calls_made": False,
        "training_executed": False,
        "matches_executed": 0,
    }


def _atomic_write(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
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
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = (
        verify_closure(args.attestation)
        if args.attestation is not None
        else protocol_report()
    )
    if args.out is not None:
        _atomic_write(args.out, result, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
