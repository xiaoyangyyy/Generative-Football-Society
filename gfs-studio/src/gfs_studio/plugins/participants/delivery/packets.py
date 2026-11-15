import json
import re
from datetime import datetime, timezone

from gfs_core.contracts import PluginContext, PluginMetadata


class DeliveryService:
    def __init__(self, sessions, evidence) -> None:
        self.sessions = sessions
        self.manifest = evidence.read_frozen("product_value_case_packs_v1.json")

    def current(self, participant_id: str, capability_token: str) -> dict:
        view = self.sessions.session_view(participant_id, capability_token)
        if view["expired"]:
            raise TimeoutError("participant condition deadline has expired")
        pack = self._pack(view["case_pack"])
        return {
            "schema_version": 1,
            "protocol_id": self.manifest["protocol_id"],
            "registration_id": view["registration_id"],
            "participant_id": participant_id,
            "condition": view["condition"],
            "case_pack": view["case_pack"],
            "participant_packet": pack["participant_packet"],
            "started_at": view["started_at"],
            "deadline_seconds": view["deadline_seconds"],
        }

    def submit(self, participant_id: str, capability_token: str, submission_id: str, answers: dict) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", submission_id):
            raise ValueError("submission ID is invalid")
        existing = self.sessions.find_submission(
            participant_id, capability_token, submission_id,
        )
        if existing is not None:
            if existing["answers"] != answers:
                raise ValueError("submission ID cannot be reused with different content")
            return existing
        packet = self.current(participant_id, capability_token)
        allowed = {
            question["id"]: question["allowed_answers"]
            for question in packet["participant_packet"]["questions"]
        }
        if set(answers) != set(allowed):
            raise ValueError("answers must exactly match the frozen questions")
        if any(answers[key] not in values for key, values in allowed.items()):
            raise ValueError("answer is outside the frozen choices")
        receipt = {
            key: packet[key]
            for key in self.manifest["receipt_contract"]["required_fields"]
            if key not in {"answers", "submitted_at"}
        }
        receipt.update({
            "submission_id": submission_id,
            "answers": dict(answers),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "started_at": packet["started_at"],
        })
        size = len(json.dumps(receipt, allow_nan=False).encode("utf-8"))
        if size > self.manifest["receipt_contract"]["maximum_bytes"]:
            raise ValueError("participant receipt is oversized")
        return self.sessions.submit_receipt(participant_id, capability_token, receipt)

    def _pack(self, case_pack: str) -> dict:
        return next(
            pack for pack in self.manifest["packs"] if pack["case_pack_id"] == case_pack
        )


class DeliveryPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.participants.delivery",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.participants",
        provides=("participant-delivery",),
        requires=("gfs.studio.participants.sessions",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("participant-delivery", DeliveryService(
            context.require("participant-sessions"), context.require("receipts"),
        ))

    def deactivate(self) -> None:
        pass
