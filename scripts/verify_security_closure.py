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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "data/evaluation/security_closure_protocol_v1.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SECRET_PATTERN = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")
ATTESTATION = (
    "I revoked the exposed provider credential, stored no credential value in "
    "evidence, and confined any replacement to the local environment"
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


def _secret_match_count(root: Path) -> int:
    count = 0
    excluded = {".git", ".pytest_cache", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        count += len(SECRET_PATTERN.findall(data))
    return count


def validate_protocol(protocol: dict[str, Any], root: Path = ROOT) -> dict[str, bool]:
    incident = protocol.get("incident") or {}
    evidence = protocol.get("evidence_contract") or {}
    execution = protocol.get("current_execution") or {}
    output = _confined(root, protocol.get("output"))
    return {
        "schema_state_and_provider_are_frozen": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id")
            == "gfs-exposed-provider-credential-closure-v1"
            and protocol.get("state") == "registered_waiting_for_user_revocation"
            and incident
            == {
                "provider": "deepseek",
                "credential_value_must_not_be_recorded": True,
                "revocation_required": True,
                "replacement_storage": "local_environment_only",
            }
        ),
        "attestation_allowlist_is_exact": (
            protocol.get("required_attestation_fields")
            == [
                "schema_version", "protocol_id", "provider",
                "credential_revoked", "revoked_at", "revocation_evidence",
                "revocation_evidence_sha256", "evidence_contains_secret",
                "replacement_credential_generated", "replacement_storage",
                "repository_secret_scan", "signed_at", "attestation",
            ]
        ),
        "redacted_evidence_contract_is_fail_closed": (
            evidence
            == {
                "revocation_evidence_root": "data/evaluation/security_closure_v1",
                "allowed_extensions": [".json", ".pdf", ".png"],
                "redaction_required": True,
                "current_commit_scan_required": True,
                "boundary_aware_secret_matches_required": 0,
                "attestation_text": ATTESTATION,
            }
        ),
        "output_is_confined": output is not None,
        "execution_is_zero_and_contains_no_credential": (
            execution
            == {
                "attestation_available": False,
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
    return {
        "schema_version": 1,
        "verification": "gfs_security_credential_closure_preregistration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "registered_waiting_for_user_revocation" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "closure_complete": False,
        "checks": checks,
        "artifact_sha256": {
            "data/evaluation/security_closure_protocol_v1.json": _sha256(protocol_path),
            "docs/SECURITY_CREDENTIAL_CLOSURE.md": _sha256(
                root / "docs/SECURITY_CREDENTIAL_CLOSURE.md"
            ),
            "scripts/verify_security_closure.py": _sha256(Path(__file__)),
        },
        "boundary_aware_secret_match_count": _secret_match_count(root),
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
    evidence_path = _confined(root, attestation.get("revocation_evidence"))
    scan = attestation.get("repository_secret_scan") or {}
    actual_revision = revision if revision is not None else _current_revision(root)
    actual_matches = _secret_match_count(root)
    expected_fields = set(protocol["required_attestation_fields"])
    evidence_is_bounded = False
    if evidence_path is not None and evidence_root is not None:
        try:
            evidence_path.relative_to(evidence_root)
            evidence_is_bounded = evidence_path.suffix.casefold() in set(
                evidence["allowed_extensions"]
            )
        except ValueError:
            pass
    checks = {
        "protocol_is_valid": all(protocol_checks.values()),
        "attestation_schema_and_fields_are_exact": (
            set(attestation) == expected_fields
            and attestation.get("schema_version") == 1
            and attestation.get("protocol_id") == protocol["protocol_id"]
            and attestation.get("provider") == "deepseek"
        ),
        "revocation_is_explicitly_signed": (
            attestation.get("credential_revoked") is True
            and _timestamp(attestation.get("revoked_at"))
            and _timestamp(attestation.get("signed_at"))
            and attestation.get("attestation") == ATTESTATION
        ),
        "redacted_receipt_is_confined_and_content_addressed": (
            evidence_is_bounded
            and evidence_path is not None
            and evidence_path.is_file()
            and SHA256.fullmatch(
                str(attestation.get("revocation_evidence_sha256") or "")
            )
            is not None
            and _sha256(evidence_path)
            == attestation.get("revocation_evidence_sha256")
            and attestation.get("evidence_contains_secret") is False
        ),
        "replacement_location_is_safe": (
            isinstance(attestation.get("replacement_credential_generated"), bool)
            and attestation.get("replacement_storage")
            in {"local_environment_only", "not_generated"}
            and (
                attestation.get("replacement_credential_generated") is True
                or attestation.get("replacement_storage") == "not_generated"
            )
        ),
        "current_revision_has_zero_boundary_aware_matches": (
            actual_revision is not None
            and scan
            == {
                "commit_sha": actual_revision,
                "boundary_aware_secret_match_count": 0,
            }
            and actual_matches == 0
        ),
    }
    passed = all(checks.values())
    artifacts = {
        resolved_attestation.relative_to(root.resolve()).as_posix(): _sha256(
            resolved_attestation
        )
    }
    if evidence_path is not None and evidence_path.is_file():
        artifacts[evidence_path.resolve().relative_to(root.resolve()).as_posix()] = (
            _sha256(evidence_path)
        )
    return {
        "schema_version": 1,
        "verification": "gfs_security_credential_closure",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified_closed" if passed else "failed",
        "passed": passed,
        "closure_complete": passed,
        "provider": "deepseek",
        "checks": checks,
        "artifact_sha256": artifacts,
        "boundary_aware_secret_match_count": actual_matches,
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
