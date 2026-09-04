"""Isolated execution boundary for the preregistered product-value study.

The moderator provisions a capability-bound session from the frozen authority.
The participant process reads only that sanitized external session file: it has
no repository access, moderator identity, future condition, or scoring seal.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote, urlsplit
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from src.infrastructure import FileLease, fsync_directory
from src.product.human_study import validate_registry, verify_content_addressed_archive
from src.product.study_delivery import ProductValueStudyDelivery, _condition_plan
from src.product.web_security import WebAccessPolicy, is_loopback_host


SESSION_KIND = "gfs_product_value_participant_session_v1"
SESSION_ID = re.compile(r"^SES-[0-9A-F]{24}$")
SUBMISSION_ID = re.compile(r"^SUB-[0-9a-f]{32}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_BODY_BYTES = 16 * 1024
OBSERVER_ATTESTATION = (
    "I attest that this record reflects the observed session and contains no "
    "direct participant identifiers or raw free text."
)
TRAINING_PACKET = {
    "title": "Evidence-bound decision tutorial",
    "scenario": (
        "This unscored tutorial confirms how to distinguish simulator evidence "
        "from claims about real football."
    ),
    "prompt": "Which statement stays inside the evidence boundary?",
    "allowed_answers": [
        "simulator_local_comparison_only",
        "proven_real_world_effect",
        "guaranteed_match_result",
    ],
    "unscored": True,
}
CRITICAL_CLAIMS = frozenset({"proven_real_world_tactic", "guaranteed_match_win"})


class StudySessionError(Exception):
    """A safe, expected participant-session failure."""

    def __init__(self, code: str, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("session clock must return timezone-aware values")
    return value.astimezone(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"immutable study artifact cannot be a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"immutable study artifact already differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise
        os.remove(temporary)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _reject_symlink_chain(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    for candidate in reversed((absolute, *absolute.parents)):
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"study path contains a symbolic link: {candidate}")


def _external_root(project_root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be an absolute external path")
    _reject_symlink_chain(candidate)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{label} must be outside the project repository")


def _safe_origin(origin: str) -> str:
    parsed = urlsplit(origin)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("participant origin must be a bare HTTP(S) origin")
    if parsed.scheme == "http" and not is_loopback_host(parsed.hostname):
        raise ValueError("non-loopback participant origin must use HTTPS")
    return origin.rstrip("/")


def _find_registration(
    service: ProductValueStudyDelivery, registration_id: str,
) -> dict[str, Any]:
    registry = validate_registry(
        protocol_path=service.protocol_path,
        registry=service.registry_path,
        case_pack_manifest_path=service.case_pack_path,
    )
    matches = [
        row for row in registry["by_participant"].values()
        if row["registration_id"] == registration_id
    ]
    if len(matches) != 1:
        raise ValueError("registration_id is not present in the registry")
    return dict(matches[0])


def provision_participant_session(
    project_root: str | Path,
    *,
    registration_id: str,
    session_root: str | Path,
    origin: str,
    clock: Callable[[], datetime] = _now,
    token_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Provision one external session and return its one-time launch capability."""
    project = Path(project_root).resolve()
    external = _external_root(project, session_root, "session_root")
    service = ProductValueStudyDelivery(project)
    if service.status()["registration_open"] is not True:
        raise ValueError("cannot provision after measurement records exist")
    session_contract = service._authority()["protocol"].get("session_execution")
    if not isinstance(session_contract, dict) or (
        session_contract.get("runner_kind") != SESSION_KIND
        or session_contract.get("capability_token_minimum_characters") != 32
        or session_contract.get(
            "capability_token_minimum_distinct_characters"
        ) != 8
        or session_contract.get("server_authoritative_deadline_seconds") != 900
        or session_contract.get(
            "idempotent_retry_requires_exact_answers_and_verified_receipt"
        ) is not True
        or session_contract.get("completion_commit_policy")
        != "atomic_state_first_then_idempotent_derived_record"
        or session_contract.get("remote_transport_policy")
        != "loopback_service_same_host_trusted_https_proxy"
        or (session_contract.get("critical_error_policy") or {}).get("field")
        != "supported_claim"
        or set(
            (session_contract.get("critical_error_policy") or {}).get("values", [])
        ) != CRITICAL_CLAIMS
    ):
        raise ValueError("participant session execution contract is invalid")
    packet = service.packet(registration_id)
    registration = _find_registration(service, registration_id)
    safe_origin = _safe_origin(origin)
    external.mkdir(parents=True, exist_ok=True)
    _reject_symlink_chain(external)
    if not external.is_dir():
        raise ValueError("session_root must be a directory")
    token = (token_factory or (lambda: secrets.token_urlsafe(32)))()
    if (
        not isinstance(token, str)
        or len(token) < 32
        or len(token) > 160
        or any(char.isspace() or not char.isprintable() for char in token)
        or len(set(token)) < 8
    ):
        raise ValueError(
            "capability token must contain 32 to 160 high-entropy characters"
        )
    capability_sha256 = _digest(token.encode("utf-8"))
    session_id = "SES-" + _digest(
        f"{registration_id}:{capability_sha256}".encode("utf-8")
    )[:24].upper()
    conditions = []
    for row in packet["conditions"]:
        conditions.append({
            "position": row["position"],
            "condition": row["condition"],
            "case_pack": row["case_pack"],
            "participant_packet": row["participant_packet"],
            "state": "locked",
            "started_at": None,
            "deadline_at": None,
            "submitted_at": None,
            "duration_seconds": None,
            "receipt_sha256": None,
            "submission_id": None,
            "declared_correct": None,
            "critical_error": None,
        })
    state = {
        "schema_version": 1,
        "session_kind": SESSION_KIND,
        "session_id": session_id,
        "protocol_id": packet["protocol_id"],
        "protocol_sha256": packet["protocol_sha256"],
        "case_pack_manifest_sha256": packet["case_pack_manifest_sha256"],
        "registration": registration,
        "capability_sha256": capability_sha256,
        "created_at": _iso(clock()),
        "phase": "training",
        "training": {
            "state": "available", "packet": TRAINING_PACKET,
            "completed_at": None, "response": None,
        },
        "maximum_seconds_per_condition": 900,
        "current_position": None,
        "session_started_at": None,
        "session_ended_at": None,
        "conditions": conditions,
        "completion_record_sha256": None,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
    }
    path = external / f"{session_id}.json"
    with FileLease(external / ".provision.lock", timeout=5.0):
        for existing in external.glob("SES-*.json"):
            _reject_symlink_chain(existing)
            observed = json.loads(existing.read_text(encoding="utf-8"))
            if (
                isinstance(observed, dict)
                and (observed.get("registration") or {}).get("registration_id")
                == registration_id
            ):
                raise ValueError(
                    "registration already has a provisioned participant session"
                )
        _write_once(path, _canonical_bytes(state))
    return {
        "schema_version": 1,
        "status": "participant_session_provisioned",
        "session_id": session_id,
        "session_file": str(path),
        "launch_url": safe_origin + "/#access=" + quote(token, safe=""),
        "capability_returned_once": True,
        "capability_stored_in_plaintext": False,
        "participant_process_requires_repository_access": False,
        "scoring_material_exposed": False,
        "sessions_executed": 0,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
    }


def _derived_winner(packet: Mapping[str, Any]) -> str:
    winners: list[str] = []
    for branch in packet["branches"]:
        accepted = True
        for rule in packet["decision_contract"]:
            value = float(branch[rule["field"]])
            threshold = float(rule["threshold"])
            accepted = accepted and (
                value >= threshold
                if rule["operator"] == "gte"
                else value <= threshold
            )
        if accepted:
            winners.append(str(branch["id"]))
    if len(winners) != 1:
        raise ValueError("participant decision contract must have one derived winner")
    return winners[0]


def _answer_contract(packet: Mapping[str, Any]) -> dict[str, set[str]]:
    questions = packet.get("questions")
    if not isinstance(questions, list) or len(questions) != 3:
        raise ValueError("participant packet questions are invalid")
    result: dict[str, set[str]] = {}
    for question in questions:
        if not isinstance(question, dict) or set(question) != {
            "id", "prompt", "allowed_answers",
        }:
            raise ValueError("participant question is invalid")
        field = question["id"]
        choices = question["allowed_answers"]
        if not isinstance(field, str) or not isinstance(choices, list) or not choices:
            raise ValueError("participant question choices are invalid")
        result[field] = {str(value) for value in choices}
    expected = {
        "recommended_intervention", "supported_claim", "next_evidence_action",
    }
    if set(result) != expected:
        raise ValueError("participant question fields are invalid")
    return result


def _normalize_answers(
    packet: Mapping[str, Any], answers: Mapping[str, Any],
) -> dict[str, str]:
    contract = _answer_contract(packet)
    if not isinstance(answers, Mapping) or set(answers) != set(contract):
        raise StudySessionError(
            "invalid_answers", "Answer fields must match the frozen form", 422,
        )
    normalized: dict[str, str] = {}
    for field, choices in contract.items():
        value = answers[field]
        if not isinstance(value, str) or value not in choices:
            raise StudySessionError(
                "invalid_answers",
                "Select one allowed answer for every field",
                422,
            )
        normalized[field] = value
    return normalized


def _score_without_seal(
    packet: Mapping[str, Any], answers: Mapping[str, str],
) -> bool:
    return (
        answers["recommended_intervention"] == _derived_winner(packet)
        and answers["supported_claim"] == "best_frozen_simulator_branch"
        and answers["next_evidence_action"] == "preregistered_paired_validation"
    )


def _integrated_presentation(packet: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for branch in packet["branches"]:
        checks = []
        for rule in packet["decision_contract"]:
            value = float(branch[rule["field"]])
            threshold = float(rule["threshold"])
            passed = (
                value >= threshold
                if rule["operator"] == "gte"
                else value <= threshold
            )
            checks.append({
                "field": rule["field"],
                "value": value,
                "operator": rule["operator"],
                "threshold": threshold,
                "passed": passed,
            })
        rows.append({
            "branch": branch["id"],
            "checks": checks,
            "all_thresholds_met": all(item["passed"] for item in checks),
        })
    return {
        "interface": "integrated_evidence_workspace_v1",
        "workflow": [
            "inspect_state", "compare_thresholds", "check_claim_boundary", "submit",
        ],
        "provenance": "same_world_state_shared_random_identity",
        "threshold_matrix": rows,
        "limitation_surface": packet["evidence_boundary"],
        "computed_assistance": True,
    }


def _baseline_presentation(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interface": "flat_read_only_document_v1",
        "workflow": None,
        "provenance": None,
        "threshold_matrix": None,
        "limitation_surface": packet["evidence_boundary"],
        "computed_assistance": False,
    }


class ParticipantSessionRuntime:
    """Capability-authenticated state machine over one external session file."""

    def __init__(
        self,
        project_root: str | Path,
        session_file: str | Path,
        evidence_root: str | Path,
        *,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.session_file = Path(session_file)
        if not self.session_file.is_absolute():
            raise ValueError("session_file must be an absolute external path")
        _external_root(
            self.project_root, self.session_file.parent, "session_file parent",
        )
        _reject_symlink_chain(self.session_file)
        if not self.session_file.is_file():
            raise ValueError("participant session file does not exist")
        self.evidence_root = _external_root(
            self.project_root, evidence_root, "evidence_root",
        )
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        _reject_symlink_chain(self.evidence_root)
        self.clock = clock
        self.lease_path = self.session_file.with_suffix(".lock")
        with FileLease(self.lease_path, timeout=5.0):
            state = self._load()
            self._validate_state(state)
            self._ensure_completion_record(state)

    def _load(self) -> dict[str, Any]:
        _reject_symlink_chain(self.session_file)
        value = json.loads(self.session_file.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("participant session state must be an object")
        return value

    def _save(self, state: Mapping[str, Any]) -> None:
        self._validate_state(state)
        _atomic_replace(self.session_file, _canonical_bytes(state))

    def _commit(self, state: Mapping[str, Any]) -> None:
        """Commit recoverable state before materializing its derived record."""
        self._save(state)
        self._ensure_completion_record(state)

    @staticmethod
    def _validate_state(state: Mapping[str, Any]) -> None:
        required = {
            "schema_version", "session_kind", "session_id", "protocol_id",
            "protocol_sha256", "case_pack_manifest_sha256", "registration",
            "capability_sha256", "created_at", "phase", "training",
            "maximum_seconds_per_condition", "current_position",
            "session_started_at", "session_ended_at", "conditions",
            "completion_record_sha256", "external_calls_made",
            "matches_executed", "training_executed",
        }
        if set(state) != required:
            raise ValueError("participant session fields are not exact")
        if (
            state.get("schema_version") != 1
            or state.get("session_kind") != SESSION_KIND
        ):
            raise ValueError("participant session schema is invalid")
        if SESSION_ID.fullmatch(str(state.get("session_id") or "")) is None:
            raise ValueError("participant session_id is invalid")
        if any(
            SHA256.fullmatch(str(state.get(field) or "")) is None
            for field in (
                "protocol_sha256", "case_pack_manifest_sha256",
                "capability_sha256",
            )
        ):
            raise ValueError("participant session authority digest is invalid")
        _timestamp(state.get("created_at"))
        if state.get("maximum_seconds_per_condition") != 900:
            raise ValueError("participant session time budget drifted")
        if state.get("external_calls_made") is not False:
            raise ValueError("participant session cannot make external calls")
        if (
            state.get("matches_executed") != 0
            or state.get("training_executed") is not False
        ):
            raise ValueError(
                "participant session cannot execute matches or model training"
            )
        training = state.get("training")
        if (
            not isinstance(training, dict)
            or set(training) != {"state", "packet", "completed_at", "response"}
            or training.get("packet") != TRAINING_PACKET
        ):
            raise ValueError("participant training contract is invalid")
        if training.get("state") not in {"available", "completed"}:
            raise ValueError("participant training state is invalid")
        if training.get("state") == "completed":
            _timestamp(training.get("completed_at"))
            if training.get("response") not in TRAINING_PACKET["allowed_answers"]:
                raise ValueError("participant training response is invalid")
        elif training.get("completed_at") is not None or training.get("response") is not None:
            raise ValueError("available participant training contains a response")
        conditions = state.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 2:
            raise ValueError("participant session requires two conditions")
        expected_fields = {
            "position", "condition", "case_pack", "participant_packet", "state",
            "started_at", "deadline_at", "submitted_at", "duration_seconds",
            "receipt_sha256", "submission_id", "declared_correct",
            "critical_error",
        }
        observed_pairs = []
        for index, row in enumerate(conditions, 1):
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise ValueError("participant condition fields are not exact")
            if (
                row.get("position") != index
                or row.get("condition") not in {
                    "gfs_studio", "manual_baseline",
                }
                or row.get("case_pack") not in {
                    "case_pack_alpha", "case_pack_beta",
                }
            ):
                raise ValueError("participant condition identity is invalid")
            _answer_contract(row["participant_packet"])
            observed_pairs.append((row["condition"], row["case_pack"]))
            status = row.get("state")
            if status not in {
                "locked", "available", "started", "submitted", "timed_out",
            }:
                raise ValueError("participant condition state is invalid")
            if status == "started":
                started = _timestamp(row.get("started_at"))
                deadline = _timestamp(row.get("deadline_at"))
                if deadline - started != timedelta(seconds=900):
                    raise ValueError("participant condition deadline is invalid")
                if any(
                    row.get(field) is not None
                    for field in (
                        "submitted_at", "duration_seconds", "receipt_sha256",
                        "submission_id", "declared_correct", "critical_error",
                    )
                ):
                    raise ValueError("started participant condition has final fields")
            if status == "submitted":
                started = _timestamp(row.get("started_at"))
                deadline = _timestamp(row.get("deadline_at"))
                submitted = _timestamp(row.get("submitted_at"))
                expected_duration = min(
                    900.0,
                    max(0.000001, (submitted - started).total_seconds()),
                )
                if (
                    deadline - started != timedelta(seconds=900)
                    or not started <= submitted <= deadline
                    or row.get("duration_seconds") != expected_duration
                ):
                    raise ValueError("submitted participant timing is invalid")
                if SHA256.fullmatch(str(row.get("receipt_sha256") or "")) is None:
                    raise ValueError("submitted participant condition lacks receipt")
                if SUBMISSION_ID.fullmatch(str(row.get("submission_id") or "")) is None:
                    raise ValueError(
                        "submitted participant condition lacks idempotency identity"
                    )
                if (
                    not isinstance(row.get("declared_correct"), bool)
                    or not isinstance(row.get("critical_error"), bool)
                ):
                    raise ValueError(
                        "submitted participant condition lacks derived flags"
                    )
            if status == "timed_out":
                started = _timestamp(row.get("started_at"))
                deadline = _timestamp(row.get("deadline_at"))
                if (
                    deadline - started != timedelta(seconds=900)
                    or row.get("duration_seconds") != 900.0
                    or any(
                    row.get(field) is not None
                    for field in (
                        "submitted_at", "receipt_sha256", "submission_id",
                        "declared_correct", "critical_error",
                    )
                    )
                ):
                    raise ValueError("timed-out participant condition is invalid")
            if status in {"locked", "available"} and any(
                row.get(field) is not None
                for field in (
                    "started_at", "deadline_at", "submitted_at",
                    "duration_seconds", "receipt_sha256", "submission_id",
                    "declared_correct", "critical_error",
                )
            ):
                raise ValueError("unstarted participant condition has evidence")
            duration = row.get("duration_seconds")
            if duration is not None and (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or not 0 < float(duration) <= 900
            ):
                raise ValueError("participant condition duration is invalid")
        if (
            len(set(observed_pairs)) != 2
            or len({pair[1] for pair in observed_pairs}) != 2
        ):
            raise ValueError("participant conditions are not distinct")
        phase = state.get("phase")
        if phase not in {"training", "ready", "measuring", "completed"}:
            raise ValueError("participant phase is invalid")
        if phase == "training" and training.get("state") != "available":
            raise ValueError("training phase is inconsistent")
        if phase != "training" and training.get("state") != "completed":
            raise ValueError("measurement cannot precede training")
        statuses = [row["state"] for row in conditions]
        position = state.get("current_position")
        started_at = state.get("session_started_at")
        ended_at = state.get("session_ended_at")
        digest = state.get("completion_record_sha256")
        if phase == "training" and not (
            position is None
            and statuses == ["locked", "locked"]
            and started_at is None
            and ended_at is None
            and digest is None
        ):
            raise ValueError("training session state is inconsistent")
        if phase == "ready":
            valid_ready = (
                position == 1
                and statuses == ["available", "locked"]
                and started_at is None
            ) or (
                position == 2
                and statuses[0] in {"submitted", "timed_out"}
                and statuses[1] == "available"
                and started_at is not None
            )
            if not valid_ready or ended_at is not None or digest is not None:
                raise ValueError("ready session state is inconsistent")
        if phase == "measuring":
            valid_measuring = (
                position == 1
                and statuses == ["started", "locked"]
            ) or (
                position == 2
                and statuses[0] in {"submitted", "timed_out"}
                and statuses[1] == "started"
            )
            if (
                not valid_measuring
                or started_at is None
                or ended_at is not None
                or digest is not None
            ):
                raise ValueError("measuring session state is inconsistent")
        if phase in {"ready", "measuring"} and started_at is not None:
            _timestamp(started_at)
        if phase == "completed":
            started = _timestamp(started_at)
            ended = _timestamp(ended_at)
            if (
                position is not None
                or any(
                    status not in {"submitted", "timed_out"}
                    for status in statuses
                )
                or ended <= started
            ):
                raise ValueError("completed session state is inconsistent")
            if SHA256.fullmatch(
                str(digest or "")
            ) is None:
                raise ValueError("completed session lacks a record digest")

    def _authorized(self, token: str) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) > 160:
            raise StudySessionError(
                "authentication_required", "Valid session access is required", 401,
            )
        state = self._load()
        observed = _digest(token.encode("utf-8"))
        if not hmac.compare_digest(observed, str(state["capability_sha256"])):
            raise StudySessionError(
                "authentication_required", "Valid session access is required", 401,
            )
        self._validate_state(state)
        return state

    @staticmethod
    def _current(state: Mapping[str, Any]) -> dict[str, Any] | None:
        position = state.get("current_position")
        if isinstance(position, int) and 1 <= position <= 2:
            return state["conditions"][position - 1]
        return None

    @staticmethod
    def _completion_record(state: Mapping[str, Any]) -> dict[str, Any]:
        if state.get("phase") != "completed":
            raise ValueError("completion record requires a completed session")
        registration = state["registration"]
        conditions = []
        for row in state["conditions"]:
            submitted = row["state"] == "submitted"
            conditions.append({
                "condition": row["condition"],
                "case_pack": row["case_pack"],
                "completed": submitted,
                "correct": bool(row["declared_correct"]) if submitted else False,
                "critical_error": (
                    bool(row["critical_error"]) if submitted else False
                ),
                "duration_seconds": float(row["duration_seconds"]),
                "evidence_sha256": row["receipt_sha256"] if submitted else None,
            })
        return {
            "schema_version": 1,
            "protocol_id": state["protocol_id"],
            "registration_id": registration["registration_id"],
            "participant_id": registration["participant_id"],
            "target_role": registration["target_role"],
            "consent": True,
            "moderator_id": registration["moderator_id"],
            "sequence": registration["sequence"],
            "session_started_at": state["session_started_at"],
            "session_ended_at": state["session_ended_at"],
            "conditions": conditions,
            "excluded": False,
            "exclusion_reason": None,
        }

    def _ensure_completion_record(self, state: Mapping[str, Any]) -> None:
        """Idempotently recover the record derived from committed final state."""
        if state.get("phase") != "completed":
            return
        content = _canonical_bytes(self._completion_record(state))
        if _digest(content) != state.get("completion_record_sha256"):
            raise ValueError("completed session record digest is inconsistent")
        record_path = (
            self.session_file.parent / "records" / f"{state['session_id']}.json"
        )
        _reject_symlink_chain(record_path)
        _write_once(record_path, content)

    def _finalize(self, state: dict[str, Any], now: datetime) -> None:
        if state["phase"] == "completed":
            return
        if any(
            row["state"] not in {"submitted", "timed_out"}
            for row in state["conditions"]
        ):
            return
        state["phase"] = "completed"
        state["current_position"] = None
        state["session_ended_at"] = _iso(now)
        state["completion_record_sha256"] = _digest(
            _canonical_bytes(self._completion_record(state))
        )

    def _advance_expired(
        self, state: dict[str, Any], now: datetime,
    ) -> bool:
        current = self._current(state)
        if current is None or current["state"] != "started":
            return False
        if now <= _timestamp(current["deadline_at"]):
            return False
        current["state"] = "timed_out"
        current["duration_seconds"] = 900.0
        if current["position"] == 1:
            state["current_position"] = 2
            state["conditions"][1]["state"] = "available"
            state["phase"] = "ready"
        else:
            self._finalize(state, now)
        return True

    def _public(self, state: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            "schema_version": 1,
            "session_id": state["session_id"],
            "phase": state["phase"],
            "condition_count": 2,
            "scoring_material_exposed": False,
            "moderator_identity_exposed": False,
            "future_condition_exposed": False,
            "external_calls_made": False,
            "matches_executed": 0,
            "training_executed": False,
        }
        if state["phase"] == "training":
            return {
                **base, "training": state["training"]["packet"], "current": None,
            }
        current = self._current(state)
        if current is None:
            return {
                **base,
                "training": {"completed": True, "unscored": True},
                "current": None,
                "completion": {
                    "submitted_conditions": sum(
                        row["state"] == "submitted" for row in state["conditions"]
                    ),
                    "timed_out_conditions": sum(
                        row["state"] == "timed_out" for row in state["conditions"]
                    ),
                    "answers_scored_for_participant": False,
                    "moderator_import_required": True,
                },
            }
        public_current: dict[str, Any] = {
            "position": current["position"],
            "state": current["state"],
            "maximum_seconds": state["maximum_seconds_per_condition"],
        }
        if current["state"] == "started":
            packet = current["participant_packet"]
            public_current.update({
                "started_at": current["started_at"],
                "deadline_at": current["deadline_at"],
                "participant_packet": packet,
                "presentation": (
                    _integrated_presentation(packet)
                    if current["condition"] == "gfs_studio"
                    else _baseline_presentation(packet)
                ),
            })
        return {
            **base,
            "training": {"completed": True, "unscored": True},
            "current": public_current,
        }

    def view(self, token: str) -> dict[str, Any]:
        with FileLease(self.lease_path, timeout=5.0):
            state = self._authorized(token)
            if self._advance_expired(state, self.clock()):
                self._commit(state)
            return self._public(state)

    def complete_training(self, token: str, response: str) -> dict[str, Any]:
        with FileLease(self.lease_path, timeout=5.0):
            state = self._authorized(token)
            if state["phase"] != "training":
                if state["training"]["response"] == response:
                    return self._public(state)
                raise StudySessionError(
                    "training_already_completed", "Training is already complete",
                )
            if response not in TRAINING_PACKET["allowed_answers"]:
                raise StudySessionError(
                    "invalid_training_response",
                    "Select one allowed tutorial answer",
                    422,
                )
            state["training"].update({
                "state": "completed",
                "completed_at": _iso(self.clock()),
                "response": response,
            })
            state["phase"] = "ready"
            state["current_position"] = 1
            state["conditions"][0]["state"] = "available"
            self._commit(state)
            return self._public(state)

    def start_current(self, token: str) -> dict[str, Any]:
        with FileLease(self.lease_path, timeout=5.0):
            state = self._authorized(token)
            now = self.clock()
            changed = self._advance_expired(state, now)
            current = self._current(state)
            if current is None:
                if changed:
                    self._commit(state)
                raise StudySessionError(
                    "session_complete", "No measured condition remains",
                )
            if current["state"] == "started":
                if changed:
                    self._commit(state)
                return self._public(state)
            if current["state"] != "available":
                raise StudySessionError(
                    "condition_unavailable", "The current condition is unavailable",
                )
            current["state"] = "started"
            current["started_at"] = _iso(now)
            current["deadline_at"] = _iso(now + timedelta(seconds=900))
            state["phase"] = "measuring"
            if state["session_started_at"] is None:
                state["session_started_at"] = current["started_at"]
            self._commit(state)
            return self._public(state)

    def submit_current(
        self,
        token: str,
        *,
        submission_id: str,
        answers: Mapping[str, Any],
    ) -> dict[str, Any]:
        if SUBMISSION_ID.fullmatch(submission_id) is None:
            raise StudySessionError(
                "invalid_submission_id",
                "A valid submission identity is required",
                422,
            )
        with FileLease(self.lease_path, timeout=5.0):
            state = self._authorized(token)
            for row in state["conditions"]:
                if row["submission_id"] == submission_id:
                    normalized = _normalize_answers(
                        row["participant_packet"], answers,
                    )
                    receipt_path = self.evidence_root / row["receipt_sha256"]
                    _reject_symlink_chain(receipt_path)
                    if not receipt_path.is_file():
                        raise ValueError("idempotent submission receipt is missing")
                    receipt_content = receipt_path.read_bytes()
                    if _digest(receipt_content) != row["receipt_sha256"]:
                        raise ValueError("idempotent submission receipt hash mismatch")
                    receipt = json.loads(receipt_content.decode("utf-8"))
                    if receipt.get("answers") != normalized:
                        raise StudySessionError(
                            "idempotency_conflict",
                            "Submission identity was already used for other answers",
                        )
                    return {
                        "schema_version": 1,
                        "status": "already_submitted",
                        "receipt_sha256": row["receipt_sha256"],
                        "session": self._public(state),
                    }
            now = self.clock()
            if self._advance_expired(state, now):
                self._commit(state)
                raise StudySessionError(
                    "condition_timed_out",
                    "The measured condition reached its frozen 900-second limit",
                )
            current = self._current(state)
            if current is None or current["state"] != "started":
                raise StudySessionError(
                    "condition_not_started", "Start the current condition first",
                )
            normalized = _normalize_answers(current["participant_packet"], answers)
            submitted_at = _iso(now)
            receipt = {
                "schema_version": 1,
                "protocol_id": state["protocol_id"],
                "registration_id": state["registration"]["registration_id"],
                "participant_id": state["registration"]["participant_id"],
                "condition": current["condition"],
                "case_pack": current["case_pack"],
                "answers": normalized,
                "submitted_at": submitted_at,
            }
            content = _canonical_bytes(receipt)
            if len(content) > 65536:
                raise ValueError(
                    "structured receipt exceeds the frozen byte limit"
                )
            receipt_sha256 = _digest(content)
            receipt_path = self.evidence_root / receipt_sha256
            _reject_symlink_chain(receipt_path)
            _write_once(receipt_path, content)
            duration = (
                now - _timestamp(current["started_at"])
            ).total_seconds()
            duration = min(900.0, max(0.000001, float(duration)))
            current.update({
                "state": "submitted",
                "submitted_at": submitted_at,
                "duration_seconds": duration,
                "receipt_sha256": receipt_sha256,
                "submission_id": submission_id,
                "declared_correct": _score_without_seal(
                    current["participant_packet"], normalized,
                ),
                "critical_error": (
                    normalized["supported_claim"] in CRITICAL_CLAIMS
                ),
            })
            if current["position"] == 1:
                state["current_position"] = 2
                state["conditions"][1]["state"] = "available"
                state["phase"] = "ready"
            else:
                self._finalize(state, now)
            self._commit(state)
            return {
                "schema_version": 1,
                "status": "submitted",
                "receipt_sha256": receipt_sha256,
                "correctness_disclosed": False,
                "session": self._public(state),
            }


def _validate_completed_record(
    state: Mapping[str, Any],
    record: Mapping[str, Any],
    verified_bytes: Mapping[str, bytes],
) -> None:
    expected_record_fields = {
        "schema_version", "protocol_id", "registration_id", "participant_id",
        "target_role", "consent", "moderator_id", "sequence",
        "session_started_at", "session_ended_at", "conditions", "excluded",
        "exclusion_reason",
    }
    if set(record) != expected_record_fields:
        raise ValueError("completed record fields violate the privacy allowlist")
    registration = state["registration"]
    expected_identity = {
        "schema_version": 1,
        "protocol_id": state["protocol_id"],
        "registration_id": registration["registration_id"],
        "participant_id": registration["participant_id"],
        "target_role": registration["target_role"],
        "consent": True,
        "moderator_id": registration["moderator_id"],
        "sequence": registration["sequence"],
        "excluded": False,
        "exclusion_reason": None,
    }
    if any(record.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("completed record identity does not match session authority")
    started = _timestamp(record.get("session_started_at"))
    ended = _timestamp(record.get("session_ended_at"))
    if started < _timestamp(registration["registered_at"]) or ended <= started:
        raise ValueError("completed record session window is invalid")
    conditions = record.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 2:
        raise ValueError("completed record must contain exactly two conditions")
    total_duration = 0.0
    condition_fields = {
        "condition", "case_pack", "completed", "correct", "critical_error",
        "duration_seconds", "evidence_sha256",
    }
    for observed, source in zip(conditions, state["conditions"]):
        if not isinstance(observed, dict) or set(observed) != condition_fields:
            raise ValueError("completed condition fields violate the privacy allowlist")
        submitted = source["state"] == "submitted"
        expected = {
            "condition": source["condition"],
            "case_pack": source["case_pack"],
            "completed": submitted,
            "correct": bool(source["declared_correct"]) if submitted else False,
            "critical_error": bool(source["critical_error"]) if submitted else False,
            "duration_seconds": float(source["duration_seconds"]),
            "evidence_sha256": source["receipt_sha256"] if submitted else None,
        }
        if observed != expected:
            raise ValueError("completed condition does not match session state")
        total_duration += float(observed["duration_seconds"])
        if not submitted:
            continue
        digest = str(observed["evidence_sha256"])
        content = verified_bytes.get(digest)
        if content is None:
            raise ValueError("submitted condition receipt bytes are unavailable")
        try:
            receipt = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("submitted condition receipt is not UTF-8 JSON") from exc
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema_version", "protocol_id", "registration_id", "participant_id",
            "condition", "case_pack", "answers", "submitted_at",
        }:
            raise ValueError("submitted receipt fields violate the frozen contract")
        receipt_identity = {
            "schema_version": 1,
            "protocol_id": state["protocol_id"],
            "registration_id": registration["registration_id"],
            "participant_id": registration["participant_id"],
            "condition": source["condition"],
            "case_pack": source["case_pack"],
            "submitted_at": source["submitted_at"],
        }
        if any(
            receipt.get(key) != value
            for key, value in receipt_identity.items()
        ):
            raise ValueError("submitted receipt identity does not match session state")
        submitted_at = _timestamp(receipt["submitted_at"])
        if not started <= submitted_at <= ended:
            raise ValueError("submitted receipt timestamp is outside the session")
        answers = receipt.get("answers")
        contract = _answer_contract(source["participant_packet"])
        if (
            not isinstance(answers, dict)
            or set(answers) != set(contract)
            or any(
                not isinstance(answers[field], str)
                or answers[field] not in choices
                for field, choices in contract.items()
            )
        ):
            raise ValueError("submitted receipt answers violate the frozen form")
        if observed["correct"] is not _score_without_seal(
            source["participant_packet"], answers,
        ):
            raise ValueError("declared correctness differs from participant receipt")
        if observed["critical_error"] is not (
            answers["supported_claim"] in CRITICAL_CLAIMS
        ):
            raise ValueError("critical-error flag differs from participant receipt")
    if total_duration > (ended - started).total_seconds():
        raise ValueError("condition durations exceed the measured session window")


def import_completed_session(
    project_root: str | Path,
    *,
    session_file: str | Path,
    evidence_root: str | Path,
    observer_attested: bool,
) -> dict[str, Any]:
    """Verify an external completed session and append its attested raw record."""
    if observer_attested is not True:
        raise ValueError("explicit moderator observer attestation is required")
    project = Path(project_root).resolve()
    runtime = ParticipantSessionRuntime(project, session_file, evidence_root)
    state = runtime._load()
    runtime._validate_state(state)
    if state["phase"] != "completed":
        raise ValueError("participant session is not complete")
    record_path = (
        runtime.session_file.parent / "records" / f"{state['session_id']}.json"
    )
    _reject_symlink_chain(record_path)
    if not record_path.is_file():
        raise ValueError("completed participant session record is missing")
    content = record_path.read_bytes()
    if _digest(content) != state["completion_record_sha256"]:
        raise ValueError("completed participant session record hash mismatch")
    record = json.loads(content.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("completed participant record must be an object")
    service = ProductValueStudyDelivery(project)
    authority = service._authority()
    if (
        state["protocol_sha256"] != authority["protocol_sha256"]
        or state["case_pack_manifest_sha256"] != authority["case_pack_sha256"]
    ):
        raise ValueError("participant session authority identity drifted")
    registration = _find_registration(
        service, str(record.get("registration_id") or ""),
    )
    if state["registration"] != registration:
        raise ValueError("participant session registration authority drifted")
    identity = {
        "participant_id": "participant_id",
        "target_role": "target_role",
        "moderator_id": "moderator_id",
        "sequence": "sequence",
    }
    if any(
        record.get(target) != registration[source]
        for target, source in identity.items()
    ):
        raise ValueError(
            "participant session record does not match its registration"
        )
    expected_plan = _condition_plan(
        registration["sequence"],
        authority["protocol"]["design"]["conditions"],
        authority["protocol"]["design"]["case_packs"],
    )
    if [
        (row.get("condition"), row.get("case_pack"))
        for row in record.get("conditions", [])
    ] != expected_plan:
        raise ValueError("participant session condition order drifted")
    authoritative_packet = service.packet(registration["registration_id"])
    expected_conditions = authoritative_packet["conditions"]
    for observed, expected in zip(state["conditions"], expected_conditions):
        if any(
            observed.get(field) != expected.get(field)
            for field in (
                "position", "condition", "case_pack", "participant_packet",
            )
        ):
            raise ValueError("participant session case authority drifted")
    digests = [
        row["evidence_sha256"]
        for row in record["conditions"]
        if row["completed"]
    ]
    verified_bytes: Mapping[str, bytes] = {}
    if digests:
        archive = verify_content_addressed_archive(
            runtime.evidence_root,
            digests,
            retain_verified_bytes=True,
            max_bytes_per_artifact=65536,
        )
        verified_bytes = archive["verified_bytes"]
    _validate_completed_record(state, record, verified_bytes)
    record["observer_attestation"] = OBSERVER_ATTESTATION
    final_content = _canonical_bytes(record)
    target = service.records_path
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_chain(target)
    imported = True
    with FileLease(target.with_suffix(target.suffix + ".lock"), timeout=5.0):
        existing_records: list[dict[str, Any]] = []
        if target.is_file():
            for line in target.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(
                            "existing participant record is not an object"
                        )
                    existing_records.append(value)
        matches = [
            row for row in existing_records
            if row.get("registration_id") == record["registration_id"]
        ]
        if matches:
            if len(matches) != 1 or matches[0] != record:
                raise ValueError(
                    "participant record import conflicts with existing evidence"
                )
            imported = False
        else:
            _atomic_replace(
                target,
                b"".join(
                    _canonical_bytes(row)
                    for row in (*existing_records, record)
                ),
            )
    return {
        "schema_version": 1,
        "status": (
            "participant_record_imported" if imported else "already_imported"
        ),
        "imported": imported,
        "registration_id": record["registration_id"],
        "record_sha256": _digest(final_content),
        "receipt_count": len(digests),
        "scoring_performed": False,
        "observer_attested": True,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
    }


_STATUS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    413: "Payload Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Entity",
    426: "Upgrade Required",
    500: "Internal Server Error",
}


class ParticipantStudyWebApp:
    """Narrow WSGI surface that cannot dispatch into the Studio application."""

    def __init__(
        self,
        runtime: ParticipantSessionRuntime,
        *,
        allow_remote: bool = False,
        allowed_hosts: tuple[str, ...] = (),
    ) -> None:
        self.runtime = runtime
        self.allow_remote = bool(allow_remote)
        self.allowed_hosts = tuple(dict.fromkeys(
            WebAccessPolicy.request_host({"HTTP_HOST": host})
            for host in allowed_hosts
            if str(host).strip()
        ))
        if self.allow_remote and not self.allowed_hosts:
            raise ValueError(
                "remote participant service requires an exact Host allowlist"
            )

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: Callable[
            [str, list[tuple[str, str]]], Any,
        ],
    ) -> Iterable[bytes]:
        try:
            status, headers, body = self._dispatch(environ)
        except StudySessionError as exc:
            status = exc.status
            headers = [("Content-Type", "application/json; charset=utf-8")]
            body = json.dumps({
                "error": {"code": exc.code, "message": exc.message},
            }).encode("utf-8")
        except Exception:
            status = 500
            headers = [("Content-Type", "application/json; charset=utf-8")]
            body = (
                b'{"error":{"code":"internal_error","message":'
                b'"Participant session request failed"}}'
            )
        common = [
            ("Cache-Control", "no-store"),
            ("Pragma", "no-cache"),
            ("Referrer-Policy", "no-referrer"),
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Content-Length", str(len(body))),
        ]
        start_response(f"{status} {_STATUS[status]}", headers + common)
        return [body]

    def _host_guard(self, environ: Mapping[str, Any]) -> None:
        try:
            host = WebAccessPolicy.request_host(dict(environ))
        except ValueError as exc:
            raise StudySessionError(
                "invalid_host", "Host is not allowed", 400,
            ) from exc
        if self.allow_remote:
            if not is_loopback_host(str(environ.get("REMOTE_ADDR") or "")):
                raise StudySessionError(
                    "trusted_proxy_required",
                    "Remote access must arrive through the local trusted proxy",
                    400,
                )
            if host not in self.allowed_hosts:
                raise StudySessionError(
                    "invalid_host", "Host is not allowed", 400,
                )
            if str(environ.get("HTTP_X_FORWARDED_PROTO", "")).lower() != "https":
                raise StudySessionError(
                    "https_required", "Remote access requires HTTPS", 426,
                )
        elif not is_loopback_host(host):
            raise StudySessionError(
                "invalid_host", "Participant service is loopback-only", 400,
            )

    @staticmethod
    def _token(environ: Mapping[str, Any]) -> str:
        value = str(environ.get("HTTP_AUTHORIZATION") or "")
        if not value.startswith("Bearer "):
            raise StudySessionError(
                "authentication_required", "Valid session access is required", 401,
            )
        return value.removeprefix("Bearer ")

    @staticmethod
    def _read_json(environ: Mapping[str, Any]) -> dict[str, Any]:
        media_type = str(environ.get("CONTENT_TYPE", "")).split(";", 1)[0]
        if media_type != "application/json":
            raise StudySessionError(
                "json_required", "Content-Type must be application/json", 415,
            )
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except (TypeError, ValueError) as exc:
            raise StudySessionError(
                "invalid_length", "Invalid Content-Length", 400,
            ) from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            status = 413 if length > MAX_BODY_BYTES else 400
            raise StudySessionError(
                "invalid_body_size", "JSON body size is invalid", status,
            )
        try:
            value = json.loads(
                environ["wsgi.input"].read(length).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StudySessionError(
                "invalid_json", "Malformed UTF-8 JSON", 400,
            ) from exc
        if not isinstance(value, dict):
            raise StudySessionError(
                "object_required", "JSON body must be an object", 422,
            )
        return value

    @staticmethod
    def _json(
        payload: Mapping[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        return (
            200,
            [("Content-Type", "application/json; charset=utf-8")],
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    def _dispatch(
        self, environ: dict[str, Any],
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        self._host_guard(environ)
        if method == "GET" and path == "/healthz":
            return self._json({
                "schema_version": 1,
                "status": "ok",
                "service": "gfs-isolated-participant-session",
                "repository_routes_exposed": False,
            })
        if method == "GET" and path == "/":
            nonce = secrets.token_urlsafe(18)
            document = _PARTICIPANT_HTML.replace("__NONCE__", nonce)
            return (
                200,
                [
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Security-Policy", (
                        "default-src 'none'; connect-src 'self'; "
                        f"script-src 'nonce-{nonce}'; "
                        f"style-src 'nonce-{nonce}'; "
                        "base-uri 'none'; form-action 'self'; "
                        "frame-ancestors 'none'"
                    )),
                ],
                document.encode("utf-8"),
            )
        token = self._token(environ)
        if method == "GET" and path == "/api/v1/session":
            return self._json(self.runtime.view(token))
        if method == "POST" and path == "/api/v1/training/complete":
            payload = self._read_json(environ)
            if (
                set(payload) != {"response"}
                or not isinstance(payload["response"], str)
            ):
                raise StudySessionError(
                    "invalid_training_payload",
                    "Training fields are not exact",
                    422,
                )
            return self._json(
                self.runtime.complete_training(token, payload["response"])
            )
        if (
            method == "POST"
            and path == "/api/v1/conditions/current/start"
        ):
            payload = self._read_json(environ)
            if payload:
                raise StudySessionError(
                    "invalid_start_payload", "Start payload must be empty", 422,
                )
            return self._json(self.runtime.start_current(token))
        if (
            method == "POST"
            and path == "/api/v1/conditions/current/submit"
        ):
            payload = self._read_json(environ)
            if (
                set(payload) != {"submission_id", "answers"}
                or not isinstance(payload["submission_id"], str)
                or not isinstance(payload["answers"], dict)
            ):
                raise StudySessionError(
                    "invalid_submission", "Submission fields are not exact", 422,
                )
            return self._json(self.runtime.submit_current(
                token,
                submission_id=payload["submission_id"],
                answers=payload["answers"],
            ))
        known = {
            "/api/v1/session",
            "/api/v1/training/complete",
            "/api/v1/conditions/current/start",
            "/api/v1/conditions/current/submit",
        }
        if path in known:
            raise StudySessionError(
                "method_not_allowed", "Method not allowed", 405,
            )
        raise StudySessionError("not_found", "Resource not found", 404)


class _ParticipantWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            lease = getattr(self, "participant_server_lease", None)
            if lease is not None:
                lease.release()


def create_participant_study_server(
    project_root: str | Path,
    *,
    session_file: str | Path,
    evidence_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8876,
    allow_remote: bool = False,
    allowed_hosts: tuple[str, ...] = (),
):
    if not is_loopback_host(host):
        raise ValueError(
            "participant service must bind to loopback behind a trusted proxy"
        )
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 <= port <= 65535
    ):
        raise ValueError("port must be 0 or between 1 and 65535")
    runtime = ParticipantSessionRuntime(
        project_root, session_file, evidence_root,
    )
    lease = FileLease(
        runtime.session_file.with_suffix(".server.lock"), timeout=0.0,
    ).acquire()
    try:
        server = make_server(
            host,
            port,
            ParticipantStudyWebApp(
                runtime,
                allow_remote=allow_remote,
                allowed_hosts=allowed_hosts,
            ),
            server_class=_ParticipantWSGIServer,
            handler_class=WSGIRequestHandler,
        )
        server.participant_server_lease = lease
        return server
    except BaseException:
        lease.release()
        raise


_PARTICIPANT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GFS blinded study session</title>
<style nonce="__NONCE__">
:root{color-scheme:dark;--bg:#07110e;--panel:#10221c;--ink:#f4fbf7;--muted:#a7bbb1;--accent:#73e2a7;--warn:#ffd166;--line:#2c4d40}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#15362a,var(--bg) 52%);color:var(--ink);font:16px/1.55 system-ui,sans-serif}
main{width:min(920px,calc(100% - 2rem));margin:auto;padding:3rem 0}section{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:clamp(1rem,4vw,2rem)}
h1{font-size:clamp(2rem,6vw,4rem);line-height:1;margin:.2rem 0}.eyebrow{color:var(--accent);letter-spacing:.14em;text-transform:uppercase;font-weight:750}.muted{color:var(--muted)}
button,select{min-height:44px;padding:.7rem .9rem;border-radius:9px;border:1px solid var(--line);font:inherit}button{background:var(--accent);color:#052013;font-weight:750;cursor:pointer}button:disabled{opacity:.45}select{width:100%;background:#08140f;color:var(--ink)}
fieldset{border:1px solid var(--line);border-radius:12px;margin:1rem 0;padding:1rem}legend{font-weight:750}.card{padding:1rem;margin:.7rem 0;background:#08140f;border:1px solid var(--line);border-radius:12px}.pass{border-color:var(--accent)}.fail{border-color:var(--warn)}
:focus-visible{outline:3px solid var(--warn);outline-offset:3px}[hidden]{display:none!important}
</style></head><body><main><p class="eyebrow">Blinded participant surface</p>
<h1>One condition at a time.</h1>
<p class="muted">No scoring key, future case, moderator workspace, match execution, or model training is available here.</p>
<section id="app" aria-live="polite"><p>Opening the capability-bound session...</p></section>
</main><script nonce="__NONCE__">
const app=document.querySelector('#app');let token='';
async function api(path,options={}){const headers={'Authorization':'Bearer '+token,...(options.body?{'Content-Type':'application/json'}:{})};const response=await fetch(path,{...options,headers});const data=await response.json();if(!response.ok)throw new Error(data.error?.message||'Request failed');return data}
function action(label,fn){const node=document.createElement('button');node.type='button';node.textContent=label;node.addEventListener('click',async()=>{node.disabled=true;try{await fn()}catch(error){showError(error)}finally{node.disabled=false}});return node}
function paragraph(text,cls=''){const node=document.createElement('p');node.textContent=String(text??'');if(cls)node.className=cls;return node}
function showError(error){const title=document.createElement('h2');title.textContent='Session action unavailable';app.replaceChildren(title,paragraph(error.message),action('Retry',load))}
function renderTraining(state){const packet=state.training,title=document.createElement('h2'),form=document.createElement('form'),label=document.createElement('label'),select=document.createElement('select'),submit=document.createElement('button');title.textContent=packet.title;label.textContent='Tutorial response';for(const value of packet.allowed_answers){const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option)}label.append(select);submit.type='submit';submit.textContent='Complete unscored tutorial';form.append(paragraph(packet.scenario),paragraph(packet.prompt),label,submit);form.addEventListener('submit',async event=>{event.preventDefault();submit.disabled=true;try{await api('/api/v1/training/complete',{method:'POST',body:JSON.stringify({response:select.value})});await load()}catch(error){showError(error)}});app.replaceChildren(title,form,paragraph('This tutorial is required but is not scored.','muted'))}
function renderMatrix(presentation){if(!presentation?.computed_assistance)return null;const host=document.createElement('div');host.append(paragraph('Integrated threshold workspace'));for(const row of presentation.threshold_matrix){const card=document.createElement('div');card.className='card '+(row.all_thresholds_met?'pass':'fail');card.append(paragraph(row.branch+' - '+(row.all_thresholds_met?'all thresholds met':'one or more thresholds missed')));for(const check of row.checks)card.append(paragraph(check.field+': '+check.value+' '+check.operator+' '+check.threshold+' = '+(check.passed?'pass':'miss'),'muted'));host.append(card)}return host}
function answerForm(packet){const form=document.createElement('form'),submit=document.createElement('button');for(const question of packet.questions){const field=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=question.prompt;field.append(legend);for(const value of question.allowed_answers){const label=document.createElement('label'),input=document.createElement('input');input.type='radio';input.name=question.id;input.value=value;input.required=true;label.append(input,document.createTextNode(' '+value));field.append(label)}form.append(field)}submit.type='submit';submit.textContent='Submit this condition';form.append(submit);form.addEventListener('submit',async event=>{event.preventDefault();submit.disabled=true;try{const values=new FormData(form),answers={};for(const question of packet.questions)answers[question.id]=values.get(question.id);const id='SUB-'+crypto.randomUUID().replaceAll('-','');await api('/api/v1/conditions/current/submit',{method:'POST',body:JSON.stringify({submission_id:id,answers})});await load()}catch(error){showError(error)}});return form}
function renderCurrent(state){const current=state.current;if(!current){const title=document.createElement('h2');title.textContent='Session complete';app.replaceChildren(title,paragraph('Your structured evidence is sealed. A moderator must attest and import it before analysis.'),paragraph('No correctness result is disclosed in-session.','muted'));return}if(current.state==='available'){const title=document.createElement('h2');title.textContent='Condition '+current.position+' of '+state.condition_count;app.replaceChildren(title,paragraph('The timer begins only when the case is revealed.'),action('Start condition',async()=>{await api('/api/v1/conditions/current/start',{method:'POST',body:'{}'});await load()}));return}const packet=current.participant_packet,title=document.createElement('h2');title.textContent=packet.title;const parts=[title,paragraph(packet.scenario),paragraph(packet.decision_rule),paragraph('Deadline: '+current.deadline_at,'muted')],matrix=renderMatrix(current.presentation);if(matrix)parts.push(matrix);else for(const branch of packet.branches)parts.push(paragraph(JSON.stringify(branch),'card'));parts.push(paragraph(packet.evidence_boundary,'muted'),answerForm(packet));app.replaceChildren(...parts)}
async function load(){const state=await api('/api/v1/session');if(state.phase==='training')renderTraining(state);else renderCurrent(state)}
try{const fragment=new URLSearchParams(location.hash.slice(1));token=fragment.get('access')||sessionStorage.getItem('gfs-study-access')||'';if(token){sessionStorage.setItem('gfs-study-access',token);history.replaceState(null,'',location.pathname)}if(!token)throw new Error('Open the one-time moderator launch link.');load().catch(showError)}catch(error){showError(error)}
</script></body></html>"""
