"""Authoritative registration and evidence primitives for human studies.

The study analyzers are deliberately read-only.  This module owns the smaller
pre-measurement mutation boundary: it registers a pseudonymous participant
under an OS lease and, where required, assigns a counterbalanced sequence.
It also verifies that evidence digests name real immutable files rather than
merely looking like SHA-256 values.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.infrastructure import FileLease, file_sha256, fsync_directory


REGISTRY_KIND = "gfs_human_study_session_registry_v1"
ARCHIVE_LAYOUT = "content_addressed_flat_sha256_v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an RFC 3339 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _confined(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"study artifact is outside the project root: {path}") from exc
    return resolved


def _registration_contract(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = protocol.get("registration")
    if not isinstance(contract, Mapping):
        raise ValueError("protocol has no registration contract")
    required = {
        "registry_kind",
        "participant_id_pattern",
        "moderator_id_pattern",
        "registration_before_measurement",
        "consent_required",
        "duplicate_registration_policy",
    }
    if set(contract) != required:
        raise ValueError("registration contract fields are not exact")
    if contract.get("registry_kind") != REGISTRY_KIND:
        raise ValueError("unsupported registry kind")
    if contract.get("registration_before_measurement") is not True:
        raise ValueError("registration must precede measurement")
    if contract.get("consent_required") is not True:
        raise ValueError("registration requires recorded consent")
    if contract.get("duplicate_registration_policy") != "idempotent_exact_match":
        raise ValueError("unsupported duplicate registration policy")
    return contract


def _pattern(contract: Mapping[str, Any], name: str) -> re.Pattern[str]:
    value = contract.get(name)
    if not isinstance(value, str) or not value.startswith("^") or not value.endswith("$"):
        raise ValueError(f"invalid anchored pattern: {name}")
    return re.compile(value)


def _allocation_contract(protocol: Mapping[str, Any]) -> Mapping[str, Any] | None:
    allocation = protocol.get("allocation")
    if allocation is None:
        return None
    if not isinstance(allocation, Mapping):
        raise ValueError("allocation contract must be an object")
    required = {
        "method",
        "seed",
        "block_size",
        "sequences",
        "assignments_per_sequence_per_block",
    }
    if set(allocation) != required:
        raise ValueError("allocation contract fields are not exact")
    sequences = allocation.get("sequences")
    block_size = allocation.get("block_size")
    per_sequence = allocation.get("assignments_per_sequence_per_block")
    if (
        allocation.get("method") != "role_stratified_permuted_blocks_v1"
        or isinstance(allocation.get("seed"), bool)
        or not isinstance(allocation.get("seed"), int)
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
        or sequences != ["AB", "BA"]
        or isinstance(per_sequence, bool)
        or not isinstance(per_sequence, int)
        or per_sequence <= 0
        or per_sequence * len(sequences) != block_size
    ):
        raise ValueError("allocation contract is invalid")
    return allocation


def _sequence_for_slot(
    allocation: Mapping[str, Any],
    role: str,
    role_slot: int,
) -> str:
    block_size = int(allocation["block_size"])
    block_index, within_block = divmod(role_slot, block_size)
    values = [
        sequence
        for sequence in allocation["sequences"]
        for _ in range(int(allocation["assignments_per_sequence_per_block"]))
    ]
    seed_material = (
        f"{allocation['seed']}:{role}:{block_index}:"
        "role_stratified_permuted_blocks_v1"
    )
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest(), "big")
    random.Random(seed).shuffle(values)
    return str(values[within_block])


def _registration_id(
    protocol_id: str,
    protocol_sha256: str,
    participant_id: str,
    target_role: str,
    moderator_id: str,
    role_slot: int,
    sequence: str | None,
    case_pack_manifest_sha256: str | None,
    registered_at: str,
) -> str:
    digest = _canonical_sha256(
        {
            "protocol_id": protocol_id,
            "protocol_sha256": protocol_sha256,
            "participant_id": participant_id,
            "target_role": target_role,
            "moderator_id": moderator_id,
            "role_slot": role_slot,
            "sequence": sequence,
            "case_pack_manifest_sha256": case_pack_manifest_sha256,
            "registered_at": registered_at,
        }
    )
    return f"REG-{digest[:24].upper()}"


def _new_registry(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    case_pack_manifest_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry_kind": REGISTRY_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "case_pack_manifest_sha256": case_pack_manifest_sha256,
        "created_at": _now(),
        "registrations": [],
    }


def register_participant(
    *,
    project_root: Path,
    protocol_path: Path,
    registry_path: Path,
    records_path: Path,
    participant_id: str,
    target_role: str,
    moderator_id: str,
    consent_recorded: bool,
    case_pack_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Register one pseudonymous session before any behavioral record exists."""

    root = project_root.resolve()
    protocol_path = protocol_path.resolve()
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise ValueError("protocol must be a real file")
    registry_path = _confined(root, registry_path)
    records_path = _confined(root, records_path)
    if case_pack_manifest_path is not None:
        case_pack_manifest_path = case_pack_manifest_path.resolve()
        if case_pack_manifest_path.is_symlink() or not case_pack_manifest_path.is_file():
            raise ValueError("case-pack manifest must be a real file")
    protocol = _read_json(protocol_path)
    contract = _registration_contract(protocol)
    participant_pattern = _pattern(contract, "participant_id_pattern")
    moderator_pattern = _pattern(contract, "moderator_id_pattern")
    if participant_pattern.fullmatch(participant_id) is None:
        raise ValueError("participant_id does not satisfy the frozen pseudonym pattern")
    if moderator_pattern.fullmatch(moderator_id) is None:
        raise ValueError("moderator_id does not satisfy the frozen pseudonym pattern")
    roles = (protocol.get("population") or {}).get("target_roles") or []
    if target_role not in roles:
        raise ValueError("target_role is not in the frozen population")
    if consent_recorded is not True:
        raise ValueError("recorded informed consent is required before registration")
    protocol_digest = file_sha256(protocol_path)
    case_digest = (
        file_sha256(case_pack_manifest_path)
        if case_pack_manifest_path is not None
        else None
    )
    allocation = _allocation_contract(protocol)
    with FileLease(
        registry_path.with_suffix(registry_path.suffix + ".lock"),
        timeout=5.0,
    ):
        registry = (
            _read_json(registry_path)
            if registry_path.is_file()
            else _new_registry(protocol, protocol_digest, case_digest)
        )
        validated = validate_registry(
            protocol_path=protocol_path,
            registry=registry,
            case_pack_manifest_path=case_pack_manifest_path,
        )
        existing = validated["by_participant"].get(participant_id)
        if existing is not None:
            if (
                existing["target_role"] != target_role
                or existing["moderator_id"] != moderator_id
                or existing["consent_recorded"] is not True
            ):
                raise ValueError("duplicate participant registration conflicts with frozen data")
            return {"created": False, "registration": existing, "registry": registry}
        if records_path.is_file() and records_path.read_text(encoding="utf-8").strip():
            raise ValueError("cannot register after measurement records exist")
        role_slot = sum(
            row["target_role"] == target_role
            for row in registry["registrations"]
        )
        sequence = (
            _sequence_for_slot(allocation, target_role, role_slot)
            if allocation is not None
            else None
        )
        registered_at = _now()
        registration = {
            "registration_id": _registration_id(
                str(protocol["protocol_id"]),
                protocol_digest,
                participant_id,
                target_role,
                moderator_id,
                role_slot,
                sequence,
                case_digest,
                registered_at,
            ),
            "participant_id": participant_id,
            "target_role": target_role,
            "moderator_id": moderator_id,
            "consent_recorded": True,
            "registered_at": registered_at,
            "role_slot": role_slot,
            "sequence": sequence,
        }
        registry["registrations"].append(registration)
        validate_registry(
            protocol_path=protocol_path,
            registry=registry,
            case_pack_manifest_path=case_pack_manifest_path,
        )
        _atomic_write_json(registry_path, registry)
        return {"created": True, "registration": registration, "registry": registry}


def validate_registry(
    *,
    protocol_path: Path,
    registry: Mapping[str, Any] | Path,
    case_pack_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate every registration and recompute deterministic allocations."""

    protocol = _read_json(protocol_path)
    contract = _registration_contract(protocol)
    value = _read_json(registry) if isinstance(registry, Path) else dict(registry)
    expected_fields = {
        "schema_version",
        "registry_kind",
        "protocol_id",
        "protocol_sha256",
        "case_pack_manifest_sha256",
        "created_at",
        "registrations",
    }
    if set(value) != expected_fields:
        raise ValueError("session registry fields are not exact")
    if value.get("schema_version") != 1 or value.get("registry_kind") != REGISTRY_KIND:
        raise ValueError("unsupported session registry schema")
    if value.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("session registry protocol_id mismatch")
    protocol_digest = file_sha256(protocol_path)
    if value.get("protocol_sha256") != protocol_digest:
        raise ValueError("session registry protocol hash mismatch")
    expected_case_digest = (
        file_sha256(case_pack_manifest_path)
        if case_pack_manifest_path is not None
        else None
    )
    if value.get("case_pack_manifest_sha256") != expected_case_digest:
        raise ValueError("session registry case-pack identity mismatch")
    _timestamp(value.get("created_at"))
    registrations = value.get("registrations")
    if not isinstance(registrations, list):
        raise ValueError("session registry registrations must be a list")
    participant_pattern = _pattern(contract, "participant_id_pattern")
    moderator_pattern = _pattern(contract, "moderator_id_pattern")
    roles = (protocol.get("population") or {}).get("target_roles") or []
    allocation = _allocation_contract(protocol)
    counts: Counter[str] = Counter()
    by_participant: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for raw in registrations:
        if not isinstance(raw, dict) or set(raw) != {
            "registration_id",
            "participant_id",
            "target_role",
            "moderator_id",
            "consent_recorded",
            "registered_at",
            "role_slot",
            "sequence",
        }:
            raise ValueError("registration fields are not exact")
        participant_id = str(raw.get("participant_id") or "")
        moderator_id = str(raw.get("moderator_id") or "")
        role = str(raw.get("target_role") or "")
        if participant_pattern.fullmatch(participant_id) is None:
            raise ValueError("registry participant_id is not pseudonymous")
        if moderator_pattern.fullmatch(moderator_id) is None:
            raise ValueError("registry moderator_id is not pseudonymous")
        if participant_id in by_participant:
            raise ValueError("duplicate participant_id in session registry")
        if role not in roles:
            raise ValueError("registry target_role is invalid")
        role_slot = raw.get("role_slot")
        if isinstance(role_slot, bool) or not isinstance(role_slot, int):
            raise ValueError("registry role_slot must be an integer")
        if role_slot != counts[role]:
            raise ValueError("registry role slots are not contiguous")
        expected_sequence = (
            _sequence_for_slot(allocation, role, role_slot)
            if allocation is not None
            else None
        )
        if raw.get("sequence") != expected_sequence:
            raise ValueError("registry sequence does not match frozen allocation")
        if raw.get("consent_recorded") is not True:
            raise ValueError("registry entry lacks recorded consent")
        _timestamp(raw.get("registered_at"))
        expected_id = _registration_id(
            str(protocol["protocol_id"]),
            protocol_digest,
            participant_id,
            role,
            moderator_id,
            role_slot,
            expected_sequence,
            expected_case_digest,
            str(raw["registered_at"]),
        )
        registration_id = str(raw.get("registration_id") or "")
        if registration_id != expected_id or registration_id in ids:
            raise ValueError("registry registration_id is invalid or duplicated")
        counts[role] += 1
        ids.add(registration_id)
        by_participant[participant_id] = raw
    return {
        "registry": value,
        "by_participant": by_participant,
        "role_counts": dict(sorted(counts.items())),
        "registration_count": len(registrations),
        "registry_sha256": _canonical_sha256(value),
    }


def bind_records_to_registry(
    normalized_records: Iterable[Mapping[str, Any]],
    registry_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact record coverage and matching immutable session identity."""

    records = list(normalized_records)
    by_participant = registry_report["by_participant"]
    seen: set[str] = set()
    for record in records:
        participant_id = str(record["participant_id"])
        registration = by_participant.get(participant_id)
        if registration is None:
            raise ValueError(f"unregistered participant record: {participant_id}")
        if participant_id in seen:
            raise ValueError("duplicate participant_id")
        seen.add(participant_id)
        if record.get("registration_id") != registration["registration_id"]:
            raise ValueError(f"registration_id mismatch: {participant_id}")
        if record.get("role") != registration["target_role"]:
            raise ValueError(f"registered role mismatch: {participant_id}")
        if record.get("moderator_id") != registration["moderator_id"]:
            raise ValueError(f"registered moderator mismatch: {participant_id}")
        if record.get("sequence") != registration["sequence"]:
            raise ValueError(f"registered sequence mismatch: {participant_id}")
        if _timestamp(record["session_started_at"]) < _timestamp(
            registration["registered_at"]
        ):
            raise ValueError(f"session starts before registration: {participant_id}")
    missing = set(by_participant) - seen
    if missing:
        raise ValueError(
            "registered sessions are missing records: " + ", ".join(sorted(missing))
        )
    return {
        "all_records_registered": True,
        "all_registrations_analyzed": True,
        "registration_count": len(by_participant),
        "record_count": len(records),
    }


def verify_content_addressed_archive(
    evidence_root: Path,
    digests: Iterable[str],
    *,
    retain_verified_bytes: bool = False,
    max_bytes_per_artifact: int | None = None,
) -> dict[str, Any]:
    """Verify exact, non-symlink evidence files named by their SHA-256."""

    if retain_verified_bytes and (
        isinstance(max_bytes_per_artifact, bool)
        or not isinstance(max_bytes_per_artifact, int)
        or max_bytes_per_artifact <= 0
    ):
        raise ValueError("retained evidence bytes require a positive size limit")
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ValueError("evidence root must be a real directory, not a symlink")
    root = evidence_root.resolve()
    requested = list(digests)
    if not requested:
        raise ValueError("no evidence digests were supplied")
    if len(requested) != len(set(requested)):
        raise ValueError("evidence digests must be unique across observations")
    verified_bytes: dict[str, bytes] = {}
    sizes: dict[str, int] = {}
    for digest in requested:
        if SHA256.fullmatch(str(digest)) is None:
            raise ValueError("invalid evidence SHA-256")
        candidate = evidence_root / digest
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"missing content-addressed evidence: {digest}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("evidence file escapes the archive root") from exc
        hasher = hashlib.sha256()
        size = 0
        retained = bytearray() if retain_verified_bytes else None
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                if (
                    max_bytes_per_artifact is not None
                    and size > max_bytes_per_artifact
                ):
                    raise ValueError(f"evidence exceeds frozen size limit: {digest}")
                hasher.update(chunk)
                if retained is not None:
                    retained.extend(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError(f"evidence content hash mismatch: {digest}")
        if retained is not None:
            verified_bytes[digest] = bytes(retained)
        sizes[digest] = size
    inventory = [
        {"sha256": digest, "size_bytes": sizes[digest]}
        for digest in sorted(sizes)
    ]
    return {
        "layout": ARCHIVE_LAYOUT,
        "verified_artifacts": len(sizes),
        "inventory_sha256": _canonical_sha256(inventory),
        # Internal-only byte snapshots close the hash-then-reopen race.  Callers
        # must never serialize this field into a study decision.
        "verified_bytes": verified_bytes,
    }
