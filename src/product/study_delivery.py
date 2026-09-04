"""Blinded, identity-bound delivery for the comparative product-value study.

This product boundary never returns scoring keys.  It validates the separate
moderator-only scoring seal, resolves a preregistered participant allocation,
and builds deterministic participant packets from the frozen public cases.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.infrastructure import file_sha256, fsync_directory
from src.product.human_study import register_participant, validate_registry


PROTOCOL_RELATIVE = Path("data/evaluation/product_value_validation_protocol_v1.json")
CASE_PACK_RELATIVE = Path("data/evaluation/product_value_case_packs_v1.json")
SCORING_SEAL_RELATIVE = Path("data/evaluation/product_value_scoring_seal_v1.json")
REGISTRY_RELATIVE = Path("data/evaluation/product_value_validation_v1/session_registry.json")
RECORDS_RELATIVE = Path("data/evaluation/product_value_validation_v1/participant_records.jsonl")
DELIVERY_RELATIVE = Path("build/study-delivery/product-value-v1")
REGISTRATION_ID = re.compile(r"^REG-[0-9A-F]{24}$")
FORBIDDEN_DELIVERY_FIELDS = frozenset({
    "scoring_key",
    "scoring_keys",
    "expected_answer",
    "derived_correct",
    "correct",
})


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"study authority must be a real file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"study authority must be a JSON object: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_existing_symlink_components(root: Path, path: Path) -> None:
    """Reject a lexically confined path if any existing component is a link."""
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError("participant packet path is outside its authority root") from exc
    cursor = lexical_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(
                f"participant packet path contains a symbolic link: {cursor}"
            )


def _forbidden_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in FORBIDDEN_DELIVERY_FIELDS:
                found.add(str(key))
            found.update(_forbidden_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_fields(child))
    return found


def _file_has_non_whitespace(path: Path) -> bool:
    if path.is_symlink():
        raise ValueError("measurement record path cannot be a symbolic link")
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError("measurement record path must be a file")
    with path.open("rb") as handle:
        return any(chunk.strip() for chunk in iter(lambda: handle.read(65536), b""))


def _condition_plan(
    sequence: str,
    conditions: list[str],
    case_packs: list[str],
) -> list[tuple[str, str]]:
    if conditions != ["gfs_studio", "manual_baseline"]:
        raise ValueError("unsupported product-value conditions")
    if len(case_packs) != 2 or len(set(case_packs)) != 2:
        raise ValueError("product-value delivery requires two distinct case packs")
    if sequence == "AB":
        return [(conditions[0], case_packs[0]), (conditions[1], case_packs[1])]
    if sequence == "BA":
        return [(conditions[1], case_packs[0]), (conditions[0], case_packs[1])]
    raise ValueError("unsupported product-value sequence")


def _derived_winner(packet: Mapping[str, Any]) -> str:
    branches = packet.get("branches")
    rules = packet.get("decision_contract")
    if not isinstance(branches, list) or not branches or not isinstance(rules, list):
        raise ValueError("participant decision contract is invalid")
    winners: list[str] = []
    for branch in branches:
        if not isinstance(branch, Mapping) or not isinstance(branch.get("id"), str):
            raise ValueError("participant branch is invalid")
        satisfied = True
        for rule in rules:
            if not isinstance(rule, Mapping) or set(rule) != {
                "field", "operator", "threshold",
            }:
                raise ValueError("participant decision rule is invalid")
            value = branch.get(rule.get("field"))
            threshold = rule.get("threshold")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(value))
                or not math.isfinite(float(threshold))
            ):
                raise ValueError("participant decision values are invalid")
            if rule.get("operator") == "gte":
                satisfied = satisfied and float(value) >= float(threshold)
            elif rule.get("operator") == "lte":
                satisfied = satisfied and float(value) <= float(threshold)
            else:
                raise ValueError("participant decision operator is invalid")
        if satisfied:
            winners.append(branch["id"])
    if len(winners) != 1:
        raise ValueError("participant decision contract has no unique winner")
    return winners[0]


class ProductValueStudyDelivery:
    """Moderator-side study operations with participant-safe output."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.protocol_path = self.root / PROTOCOL_RELATIVE
        self.case_pack_path = self.root / CASE_PACK_RELATIVE
        self.scoring_seal_path = self.root / SCORING_SEAL_RELATIVE
        self.registry_path = self.root / REGISTRY_RELATIVE
        self.records_path = self.root / RECORDS_RELATIVE
        self.delivery_root = self.root / DELIVERY_RELATIVE

    def _authority(self) -> dict[str, Any]:
        for path in (
            self.protocol_path,
            self.case_pack_path,
            self.scoring_seal_path,
        ):
            _reject_existing_symlink_components(self.root, path)
        protocol = _read_object(self.protocol_path)
        cases = _read_object(self.case_pack_path)
        seal = _read_object(self.scoring_seal_path)
        if (
            protocol.get("schema_version") != 1
            or protocol.get("protocol_id")
            != "gfs-studio-comparative-value-validation-v1"
            or protocol.get("state") != "preregistered_not_executed"
        ):
            raise ValueError("product-value protocol is not frozen")
        if (protocol.get("outputs") or {}).get("case_pack_manifest") != str(
            CASE_PACK_RELATIVE
        ).replace("\\", "/"):
            raise ValueError("protocol case-pack path does not match product authority")
        if (protocol.get("outputs") or {}).get("scoring_seal") != str(
            SCORING_SEAL_RELATIVE
        ).replace("\\", "/"):
            raise ValueError("protocol scoring-seal path does not match product authority")
        if _forbidden_fields(cases):
            raise ValueError("participant case manifest contains scoring material")
        packs = cases.get("packs")
        if not isinstance(packs, list) or len(packs) != 2:
            raise ValueError("participant case manifest must contain two packs")
        expected_ids = (protocol.get("design") or {}).get("case_packs")
        observed_ids = [pack.get("case_pack_id") for pack in packs]
        if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
            raise ValueError("participant case-pack identity or order mismatch")
        if any(
            not isinstance(pack, dict)
            or set(pack) != {"case_pack_id", "revision", "participant_packet"}
            or pack.get("revision") != 1
            for pack in packs
        ):
            raise ValueError("participant case-pack fields are not exact")
        case_digest = file_sha256(self.case_pack_path)
        expected_seal_fields = {
            "schema_version",
            "seal_id",
            "protocol_id",
            "state",
            "case_pack_manifest_sha256",
            "scoring_keys",
            "access_contract",
            "current_execution",
        }
        if (
            set(seal) != expected_seal_fields
            or seal.get("schema_version") != 1
            or seal.get("seal_id")
            != "gfs-studio-product-value-scoring-seal-v1"
            or seal.get("protocol_id") != protocol.get("protocol_id")
            or seal.get("state") != "frozen_preexecution"
            or seal.get("case_pack_manifest_sha256") != case_digest
            or seal.get("access_contract")
            != {
                "participant_delivery_forbidden": True,
                "moderator_analysis_only": True,
                "repository_is_not_a_participant_delivery_channel": True,
            }
            or seal.get("current_execution")
            != {
                "participants_exposed": 0,
                "measured_sessions": 0,
                "results_available": False,
            }
        ):
            raise ValueError("scoring seal is invalid, exposed, or case-drifted")
        receipt = cases.get("receipt_contract") or {}
        answer_fields = receipt.get("answer_fields")
        scoring_keys = seal.get("scoring_keys")
        if (
            not isinstance(answer_fields, list)
            or not answer_fields
            or not isinstance(scoring_keys, dict)
            or set(scoring_keys) != set(observed_ids)
        ):
            raise ValueError("scoring-seal key coverage is invalid")
        for pack in packs:
            questions = {
                row.get("id"): row
                for row in (pack.get("participant_packet") or {}).get(
                    "questions", []
                )
                if isinstance(row, dict)
            }
            key = scoring_keys[pack["case_pack_id"]]
            if set(questions) != set(answer_fields) or set(key) != set(answer_fields):
                raise ValueError("question and scoring-seal fields do not match")
            if any(
                key[field] not in (questions[field].get("allowed_answers") or [])
                for field in answer_fields
            ):
                raise ValueError("scoring seal contains an answer outside frozen choices")
            if (
                key.get("recommended_intervention")
                != _derived_winner(pack["participant_packet"])
                or
                key.get("supported_claim") != "best_frozen_simulator_branch"
                or key.get("next_evidence_action")
                != "preregistered_paired_validation"
            ):
                raise ValueError("scoring seal violates the frozen claim boundary")
        return {
            "protocol": protocol,
            "cases": cases,
            "seal": seal,
            "case_pack_sha256": case_digest,
            "protocol_sha256": file_sha256(self.protocol_path),
            "scoring_seal_sha256": file_sha256(self.scoring_seal_path),
            "packs": {pack["case_pack_id"]: pack for pack in packs},
        }

    def status(self) -> dict[str, Any]:
        authority = self._authority()
        protocol = authority["protocol"]
        roles = protocol["population"]["target_roles"]
        sequences = protocol["design"]["sequences"]
        _reject_existing_symlink_components(self.root, self.registry_path)
        _reject_existing_symlink_components(self.root, self.records_path)
        if self.registry_path.is_file():
            registry = validate_registry(
                protocol_path=self.protocol_path,
                registry=self.registry_path,
                case_pack_manifest_path=self.case_pack_path,
            )
            registrations = list(registry["by_participant"].values())
        else:
            registrations = []
        measurement_records_present = _file_has_non_whitespace(self.records_path)
        role_counts = Counter(row["target_role"] for row in registrations)
        sequence_counts = Counter(row["sequence"] for row in registrations)
        return {
            "schema_version": 1,
            "study": "product_value_v1",
            "state": protocol["state"],
            "authority_valid": True,
            "registration_count": len(registrations),
            "role_counts": {role: role_counts[role] for role in roles},
            "sequence_counts": {
                sequence: sequence_counts[sequence] for sequence in sequences
            },
            "minimum_valid_participants": protocol["population"][
                "minimum_valid_participants"
            ],
            "registration_open": not measurement_records_present,
            "delivery_ready": bool(registrations),
            "blinded_delivery": True,
            "participant_manifest_contains_scoring_keys": False,
            "case_pack_manifest_sha256": authority["case_pack_sha256"],
            "scoring_seal_sha256": authority["scoring_seal_sha256"],
            "participant_identifiers_exposed": False,
            "scoring_material_exposed": False,
            "measurement_records_present": measurement_records_present,
            "participants_observed": 0 if not measurement_records_present else None,
            "matches_executed": 0,
            "training_executed": False,
            "provider_calls_made": False,
        }

    def register(
        self,
        *,
        participant_id: str,
        target_role: str,
        moderator_id: str,
        consent_recorded: bool,
    ) -> dict[str, Any]:
        self._authority()
        _reject_existing_symlink_components(self.root, self.registry_path)
        _reject_existing_symlink_components(self.root, self.records_path)
        return register_participant(
            project_root=self.root,
            protocol_path=self.protocol_path,
            registry_path=self.registry_path,
            records_path=self.records_path,
            participant_id=participant_id,
            target_role=target_role,
            moderator_id=moderator_id,
            consent_recorded=consent_recorded,
            case_pack_manifest_path=self.case_pack_path,
        )

    def packet(self, registration_id: str) -> dict[str, Any]:
        if REGISTRATION_ID.fullmatch(registration_id) is None:
            raise ValueError("registration_id is malformed")
        authority = self._authority()
        if not self.registry_path.is_file():
            raise ValueError("session registry does not exist")
        registry = validate_registry(
            protocol_path=self.protocol_path,
            registry=self.registry_path,
            case_pack_manifest_path=self.case_pack_path,
        )
        matches = [
            row
            for row in registry["by_participant"].values()
            if row["registration_id"] == registration_id
        ]
        if len(matches) != 1:
            raise ValueError("registration_id is not present in the registry")
        registration = matches[0]
        protocol = authority["protocol"]
        plan = _condition_plan(
            registration["sequence"],
            protocol["design"]["conditions"],
            protocol["design"]["case_packs"],
        )
        conditions = [
            {
                "position": index + 1,
                "condition": condition,
                "case_pack": case_pack,
                "participant_packet": authority["packs"][case_pack][
                    "participant_packet"
                ],
            }
            for index, (condition, case_pack) in enumerate(plan)
        ]
        base = {
            "schema_version": 1,
            "packet_kind": "gfs_product_value_blinded_participant_packet_v1",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": authority["protocol_sha256"],
            "case_pack_manifest_sha256": authority["case_pack_sha256"],
            "registration_id": registration_id,
            "participant_id": registration["participant_id"],
            "target_role": registration["target_role"],
            "sequence": registration["sequence"],
            "conditions": conditions,
            "receipt_contract": authority["cases"]["receipt_contract"],
            "delivery_boundary": {
                "scoring_material_included": False,
                "moderator_identity_included": False,
                "direct_personal_identifiers_allowed": False,
                "repository_is_not_a_participant_delivery_channel": True,
            },
            "claim_boundary": (
                "Frozen simulator-local decision cases only; this packet does not "
                "establish a real-football causal effect or product benefit."
            ),
            "external_calls_made": False,
            "matches_executed": 0,
            "training_executed": False,
        }
        packet = {
            **base,
            "delivery_id": "DEL-" + _canonical_sha256(base)[:24].upper(),
        }
        forbidden = _forbidden_fields(packet)
        if forbidden:
            raise ValueError(
                "participant packet contains forbidden fields: "
                + ", ".join(sorted(forbidden))
            )
        return packet

    def packet_bytes(self, registration_id: str) -> bytes:
        return _canonical_bytes(self.packet(registration_id))

    def materialize_packet(
        self,
        registration_id: str,
        output: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        _reject_existing_symlink_components(self.root, self.delivery_root)
        self.delivery_root.mkdir(parents=True, exist_ok=True)
        _reject_existing_symlink_components(self.root, self.delivery_root)
        root = self.delivery_root.resolve()
        try:
            root.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                "participant packet delivery root must stay in the project"
            ) from exc
        requested = (
            Path(output)
            if output is not None
            else DELIVERY_RELATIVE / f"{registration_id}.json"
        )
        lexical_target = Path(os.path.abspath(
            requested if requested.is_absolute() else self.root / requested
        ))
        try:
            lexical_target.relative_to(Path(os.path.abspath(root)))
        except ValueError as exc:
            raise ValueError(
                "participant packet output must stay in build/study-delivery"
            ) from exc
        _reject_existing_symlink_components(root, lexical_target)
        target = lexical_target.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("participant packet output must stay in build/study-delivery") from exc
        if target.suffix.lower() != ".json":
            raise ValueError("participant packet output must be a .json file")
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite participant packet: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _reject_existing_symlink_components(root, lexical_target)
        if lexical_target.resolve() != target:
            raise ValueError("participant packet output changed during validation")
        content = self.packet_bytes(registration_id)
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}-",
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(temporary, target)
            else:
                try:
                    os.link(temporary, target)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"refusing to overwrite participant packet: {target}"
                    ) from exc
                os.remove(temporary)
            fsync_directory(target.parent)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return target
