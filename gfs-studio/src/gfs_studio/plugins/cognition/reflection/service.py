from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.workspace import WorkspaceRepository


class ReflectionService:
    def __init__(self, project_root, cognition, evidence) -> None:
        self.workspace = WorkspaceRepository(project_root)
        self.cognition = cognition
        self.evidence = evidence

    def reflect(self, *, role: str, topic: str, match_id: str, seed: int = 42) -> dict:
        request = {"role": role, "topic": topic, "match_id": match_id, "seed": seed}
        reflection_id = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode("utf-8")
        ).hexdigest()
        with self.workspace.lease(f"reflections/{reflection_id}.lock"):
            record = self._read(reflection_id)
            if record is None:
                content = self.cognition.reflect(
                    role=role, topic=topic, match_id=match_id, seed=seed,
                )
                receipt = self._receipt(reflection_id, request, content)
                record = {
                    "reflection_id": reflection_id,
                    "request": request,
                    "provider_response": content,
                    "receipt": receipt,
                    "state": "response_persisted",
                    "application": None,
                }
                self.workspace.write(self._name(reflection_id), record)
            return self._apply(record)

    def resume(self, reflection_id: str) -> dict:
        if len(reflection_id) != 64 or any(character not in "0123456789abcdef" for character in reflection_id):
            raise ValueError("reflection identity is invalid")
        with self.workspace.lease(f"reflections/{reflection_id}.lock"):
            record = self._read(reflection_id)
            if record is None:
                raise KeyError(reflection_id)
            return self._apply(record)

    def _apply(self, record: dict) -> dict:
        if record["state"] == "response_persisted":
            content = record["provider_response"]
            action = content.get("recommended_action", "retain_current_vector") if isinstance(content, dict) else "retain_current_vector"
            record["application"] = {
                "application_id": f"apply-{record['reflection_id']}",
                "recommended_action": action,
                "applied_to": "advisory_workspace",
                "official_simulation_changed": False,
                "causal_claim": False,
            }
            record["state"] = "applied"
            self.workspace.write(self._name(record["reflection_id"]), record)
        if record["state"] != "applied":
            raise ValueError("unknown reflection state")
        content = record["provider_response"]
        response = dict(content) if isinstance(content, dict) else {"content": content}
        return {
            **response,
            "reflection_id": record["reflection_id"],
            "receipt": record["receipt"],
            "application": record["application"],
            "application_idempotent": True,
        }

    def _read(self, reflection_id: str) -> dict | None:
        try:
            return self.workspace.read(self._name(reflection_id))
        except FileNotFoundError:
            return None

    def _receipt(self, reflection_id: str, request: dict, content) -> dict:
        identity = f"reflection-{reflection_id}"
        document = {"reflection_id": reflection_id, "request": request, "response": content}
        try:
            receipt = self.evidence.issue_receipt(identity, document)
        except FileExistsError:
            receipt = self.evidence.read_receipt(identity, document)
        return asdict(receipt)

    @staticmethod
    def _name(reflection_id: str) -> str:
        return f"reflections/{reflection_id}.json"


class ReflectionPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.cognition.reflection",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.cognition",
        provides=("reflection",),
        requires=(
            "gfs.studio.cognition.gateway",
            "gfs.studio.evidence.receipts",
        ),
    )

    def __init__(self, project_root) -> None:
        self._project_root = project_root

    def activate(self, context: PluginContext) -> None:
        context.provide("reflection", ReflectionService(
            self._project_root,
            context.require("provider-gateway"),
            context.require("receipts"),
        ))

    def deactivate(self) -> None:
        pass
