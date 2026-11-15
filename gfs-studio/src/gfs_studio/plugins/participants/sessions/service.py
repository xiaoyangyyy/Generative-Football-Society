from __future__ import annotations

import copy
import hashlib
import hmac
import random
import re
from datetime import datetime, timezone

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.workspace import WorkspaceRepository

PROTOCOL = "product_value_validation_protocol_v1.json"


class SessionService:
    def __init__(self, project_root, evidence) -> None:
        self.workspace = WorkspaceRepository(project_root)
        self.protocol = evidence.read_frozen(PROTOCOL)

    def register(self, participant_id: str, role: str, moderator_id: str, capability_token: str, *, consent: bool) -> dict:
        registration = self.protocol["registration"]
        population = self.protocol["population"]
        if not re.fullmatch(registration["participant_id_pattern"], participant_id):
            raise ValueError("participant ID does not match the frozen protocol")
        if not re.fullmatch(registration["moderator_id_pattern"], moderator_id):
            raise ValueError("moderator ID does not match the frozen protocol")
        if role not in population["target_roles"]:
            raise ValueError("participant role is outside the frozen protocol")
        if not consent:
            raise ValueError("informed consent is required")
        if len(capability_token) < 32 or len(set(capability_token)) < 8:
            raise ValueError("participant capability token is too weak")
        with self.workspace.lease("participant-sessions.lock"):
            document = self._read()
            existing = self._find(document, participant_id)
            if existing is not None:
                if (existing["role"], existing["moderator_id"]) != (role, moderator_id):
                    raise ValueError("participant registration conflicts with its identity")
                if not hmac.compare_digest(existing["capability_sha256"], _hash(capability_token)):
                    raise ValueError("participant capability does not match registration")
                return _public(existing)
            role_index = sum(
                participant["role"] == role for participant in document["participants"]
            )
            record = {
                "registration_id": _hash(participant_id)[:24],
                "participant_id": participant_id,
                "role": role,
                "moderator_id": moderator_id,
                "capability_sha256": _hash(capability_token),
                "protocol_id": self.protocol["protocol_id"],
                "consent": True,
                "state": "registered",
                "training_completed": False,
                "sequence": _allocation(role_index, self.protocol["allocation"]),
                "condition_index": None,
                "condition_starts": {},
                "receipts": [],
                "moderator_attested": False,
                "registered_at": _now(),
            }
            document["participants"].append(record)
            self.workspace.write("participant_sessions.json", document)
            return _public(record)

    def complete_training(self, participant_id: str, capability_token: str) -> dict:
        def transition(record):
            if record["state"] != "registered":
                raise ValueError("training can only precede measured conditions")
            record["training_completed"] = True
            record["state"] = "training_completed"
        return self._transition(participant_id, capability_token, transition)

    def begin(self, participant_id: str, capability_token: str) -> dict:
        def transition(record):
            if record["state"] == "registered":
                raise ValueError("unscored training must be completed first")
            if record["state"] == "completed":
                raise ValueError("participant has completed both conditions")
            if record["state"] == "active":
                return
            record["condition_index"] = len(record["condition_starts"])
            record["condition_starts"][str(record["condition_index"])] = _now()
            record["state"] = "active"
        return self._transition(participant_id, capability_token, transition)

    def session_view(self, participant_id: str, capability_token: str) -> dict:
        with self.workspace.lease("participant-sessions.lock"):
            document = self._read()
            record = self._authenticated(document, participant_id, capability_token)
            if record["state"] != "active":
                raise ValueError("participant condition has not been started")
            condition, case_pack = self._condition(record)
            start = record["condition_starts"][str(record["condition_index"])]
            elapsed = (_parse(_now()) - _parse(start)).total_seconds()
            maximum = self.protocol["session_execution"]["server_authoritative_deadline_seconds"]
            return {
                **_public(record),
                "condition": condition,
                "case_pack": case_pack,
                "started_at": start,
                "deadline_seconds": max(0, maximum - int(elapsed)),
                "expired": elapsed > maximum,
            }

    def submit_receipt(self, participant_id: str, capability_token: str, receipt: dict) -> dict:
        with self.workspace.lease("participant-sessions.lock"):
            document = self._read()
            record = self._authenticated(document, participant_id, capability_token)
            view_condition, _ = self._condition(record)
            existing = next(
                (item for item in record["receipts"] if item["submission_id"] == receipt["submission_id"]),
                None,
            )
            if existing is not None:
                if existing != receipt:
                    raise ValueError("submission ID cannot be reused with different content")
                return copy.deepcopy(existing)
            if record["receipts"] and record["condition_index"] is None:
                raise ValueError("first valid submission wins for this condition")
            if record["state"] != "active" or receipt["condition"] != view_condition:
                raise ValueError("submission does not match the active condition")
            started = _parse(record["condition_starts"][str(record["condition_index"])])
            elapsed = (_parse(receipt["submitted_at"]) - started).total_seconds()
            if elapsed > self.protocol["session_execution"]["server_authoritative_deadline_seconds"]:
                raise TimeoutError("participant condition deadline has expired")
            record["receipts"].append(copy.deepcopy(receipt))
            if len(record["receipts"]) == 2:
                record["state"] = "completed"
                record["condition_index"] = None
            else:
                record["state"] = "condition_submitted"
                record["condition_index"] = None
            self.workspace.write("participant_sessions.json", document)
            return copy.deepcopy(receipt)

    def attest(self, participant_id: str, moderator_id: str) -> dict:
        with self.workspace.lease("participant-sessions.lock"):
            document = self._read()
            record = self._find(document, participant_id)
            if record is None or record["moderator_id"] != moderator_id:
                raise ValueError("moderator attestation does not match registration")
            if record["state"] != "completed":
                raise ValueError("attestation requires two completed conditions")
            record["moderator_attested"] = True
            self.workspace.write("participant_sessions.json", document)
            return _public(record)

    def find_submission(self, participant_id: str, capability_token: str, submission_id: str) -> dict | None:
        with self.workspace.lease("participant-sessions.lock"):
            document = self._read()
            record = self._authenticated(document, participant_id, capability_token)
            receipt = next(
                (item for item in record["receipts"] if item["submission_id"] == submission_id),
                None,
            )
            return copy.deepcopy(receipt)

    def records(self) -> list[dict]:
        return copy.deepcopy(self._read()["participants"])

    def status(self) -> dict:
        records = self.records()
        completed = sum(record["state"] == "completed" for record in records)
        return {
            "protocol_id": self.protocol["protocol_id"],
            "registered_participants": len(records),
            "completed_participants": completed,
            "study_complete": False,
            "user_value_validated": False,
            "transport": "local_workflow_not_remote_participant_service",
        }

    def _transition(self, participant_id, token, transition) -> dict:
        with self.workspace.lease("participant-sessions.lock"):
            document = self._read()
            record = self._authenticated(document, participant_id, token)
            transition(record)
            self.workspace.write("participant_sessions.json", document)
            return _public(record)

    def _read(self) -> dict:
        try:
            document = self.workspace.read("participant_sessions.json")
        except FileNotFoundError:
            return {"protocol_id": self.protocol["protocol_id"], "participants": []}
        if document.get("protocol_id") != self.protocol["protocol_id"] or not isinstance(document.get("participants"), list):
            raise ValueError("participant session registry does not match the protocol")
        return document

    def _find(self, document, participant_id):
        return next((item for item in document["participants"] if item["participant_id"] == participant_id), None)

    def _authenticated(self, document, participant_id, token) -> dict:
        record = self._find(document, participant_id)
        if record is None or not hmac.compare_digest(record["capability_sha256"], _hash(token)):
            raise PermissionError("participant capability is invalid")
        return record

    @staticmethod
    def _condition(record) -> tuple[str, str]:
        index = record["condition_index"]
        if index not in (0, 1):
            raise ValueError("no active condition")
        first = "gfs_studio" if record["sequence"][index] == "A" else "manual_baseline"
        return first, "case_pack_alpha" if index == 0 else "case_pack_beta"


class SessionsPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.participants.sessions",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.participants",
        provides=("participant-sessions",),
        requires=("gfs.studio.evidence.receipts",),
    )

    def __init__(self, project_root) -> None:
        self._project_root = project_root

    def activate(self, context: PluginContext) -> None:
        context.provide("participant-sessions", SessionService(
            self._project_root, context.require("receipts"),
        ))

    def deactivate(self) -> None:
        pass


def _allocation(index: int, allocation: dict) -> str:
    block, _ = divmod(index, allocation["block_size"])
    sequences = allocation["sequences"] * 2
    random.Random(allocation["seed"] + block).shuffle(sequences)
    return sequences[index % allocation["block_size"]]


def _public(record: dict) -> dict:
    return {key: copy.deepcopy(value) for key, value in record.items() if key != "capability_sha256"}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("participant timestamps must include a timezone")
    return parsed
