"""Audited, time-scale-aware application of LLM reflection proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import uuid
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetaParameter:
    time_scale: str
    max_delta: float
    low: float
    high: float


PARAMETERS = {
    "risk_budget": MetaParameter("fast", 0.10, 0.0, 1.0),
    "pressing_intensity": MetaParameter("fast", 0.10, 0.0, 1.0),
    "line_height": MetaParameter("fast", 0.10, 0.0, 1.0),
    "rotation_aggressiveness": MetaParameter("fast", 0.10, 0.0, 1.0),
    "icon_patience": MetaParameter("medium", 0.06, 0.05, 1.2),
    "w_h_delta": MetaParameter("slow", 0.025, 0.15, 1.5),
    "w_x_delta": MetaParameter("slow", 0.025, 0.05, 1.5),
}


@dataclass
class MetaProposal:
    proposal_id: str
    agent: str
    reflection: str
    evidence_ids: tuple[str, ...]
    changes: dict[str, dict[str, float]]
    status: str = "shadow"
    evaluation: dict[str, float] = field(default_factory=dict)
    scope: str = "match"
    expires_after: int = 1
    created_step: int = 0


def _read_parameter(agent: Any, key: str) -> float:
    if key in agent.tactical_controls:
        return float(agent.tactical_controls[key])
    if key == "icon_patience":
        return float(agent.roles["Icon"]["patience"])
    return float(agent.W_h if key == "w_h_delta" else agent.W_x)


def _write_parameter(agent: Any, key: str, value: float) -> None:
    if key in agent.tactical_controls:
        agent.tactical_controls[key] = value
    elif key == "icon_patience":
        agent.roles["Icon"]["patience"] = value
    elif key == "w_h_delta":
        agent.W_h = value
    else:
        agent.W_x = value


class MetaLearningController:
    def __init__(self, audit_path: str | Path | None = None) -> None:
        self.proposals: dict[str, MetaProposal] = {}
        self.audit_path = Path(audit_path) if audit_path else None

    def _audit(self, proposal: MetaProposal, action: str) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "proposal_id": proposal.proposal_id, "agent": proposal.agent,
            "action": action, "status": proposal.status, "scope": proposal.scope,
            "expires_after": proposal.expires_after, "evidence_ids": list(proposal.evidence_ids),
            "changes": proposal.changes, "evaluation": proposal.evaluation,
        }
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def propose(
        self, agent: Any, reflection: dict[str, Any] | None, *,
        operation_id: str | None = None,
    ) -> MetaProposal:
        reflection = reflection or {}
        if reflection.get("uses_sealed_data"):
            raise ValueError("sealed data cannot be used for meta-adaptation")
        suggestion = dict(reflection.get("suggested_adjustments") or {})
        legacy = {
            "new_wh": ("w_h_delta", agent.W_h),
            "new_wx": ("w_x_delta", agent.W_x),
            "new_icon_patience": ("icon_patience", agent.roles["Icon"]["patience"]),
        }
        for source, (target, current) in legacy.items():
            if source in reflection:
                try:
                    suggestion[target] = float(reflection[source]) - float(current)
                except (TypeError, ValueError):
                    pass
        known = {
            str(record.get("id"))
            for record in getattr(agent, "memory_event_log", [])
            if isinstance(record, dict) and record.get("id")
        }
        requested = tuple(str(value) for value in reflection.get("evidence_memory_ids", []))
        verified = tuple(value for value in requested if value in known)
        confidence = float(np.clip(reflection.get("confidence", 0.5), 0.0, 1.0))
        evidence_factor = 1.0 if verified else 0.65
        changes = {}
        for key, raw_delta in suggestion.items():
            spec = PARAMETERS.get(key)
            if spec is None:
                continue
            try:
                proposed = float(raw_delta)
            except (TypeError, ValueError):
                continue
            before = _read_parameter(agent, key)
            bounded = spec.max_delta * float(np.tanh(proposed / spec.max_delta))
            after = float(np.clip(before + bounded * confidence * evidence_factor, spec.low, spec.high))
            changes[key] = {"before": before, "after": after, "delta": after - before}
        proposal_id = (
            hashlib.sha256(
                f"reflection:{agent.name}:{operation_id}".encode("utf-8")
            ).hexdigest()[:32]
            if operation_id else uuid.uuid4().hex
        )
        proposal = MetaProposal(
            proposal_id=proposal_id,
            agent=str(agent.name),
            reflection=str(reflection.get("reflection", reflection.get("diary", ""))),
            evidence_ids=verified,
            changes=changes,
            scope=str(reflection.get("scope", "match")),
            expires_after=max(1, int(reflection.get("expires_after", 1))),
            created_step=max(0, int(reflection.get("created_step", 0))),
        )
        if proposal.scope not in {"match", "tournament", "long_term"}:
            raise ValueError("invalid meta-adaptation scope")
        self.proposals[proposal.proposal_id] = proposal
        self._audit(proposal, "proposed")
        return proposal

    def commit(self, agent: Any, proposal: MetaProposal) -> MetaProposal:
        if proposal.status != "shadow":
            raise ValueError(f"Proposal is not committable: {proposal.status}")
        for key, change in proposal.changes.items():
            if not np.isclose(_read_parameter(agent, key), change["before"]):
                raise RuntimeError(f"Parameter changed since proposal: {key}")
        for key, change in proposal.changes.items():
            _write_parameter(agent, key, change["after"])
        proposal.status = "committed"
        self._audit(proposal, "committed")
        return proposal

    def rollback(self, agent: Any, proposal: MetaProposal) -> MetaProposal:
        if proposal.status != "committed":
            raise ValueError(f"Proposal is not rollbackable: {proposal.status}")
        for key, change in proposal.changes.items():
            _write_parameter(agent, key, change["before"])
        proposal.status = "rolled_back"
        self._audit(proposal, "rolled_back")
        return proposal

    def expire(self, agent: Any, *, current_step: int) -> list[str]:
        rolled_back = []
        for proposal in self.proposals.values():
            if proposal.status == "committed" and current_step >= proposal.created_step + proposal.expires_after:
                self.rollback(agent, proposal)
                rolled_back.append(proposal.proposal_id)
        return rolled_back

    def evaluate_and_commit(
        self,
        agent: Any,
        proposal: MetaProposal,
        evaluator: Any,
        *,
        minimum_effect: float = 0.0,
    ) -> MetaProposal:
        report = evaluator(proposal)
        proposal.evaluation = {
            "ate": float(report.average_treatment_effect),
            "ci_low": float(report.ci_low),
            "ci_high": float(report.ci_high),
        }
        if report.ci_low > minimum_effect:
            return self.commit(agent, proposal)
        proposal.status = "rejected"
        self._audit(proposal, "rejected")
        return proposal

    def apply(
        self, agent: Any, reflection: dict[str, Any] | None, *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility entry point implemented through the canonical proposal path."""
        reflection = reflection or {}
        diary = str(reflection.get("reflection", reflection.get("diary", "")))
        agent.reflection_diary = diary
        proposal = self.propose(agent, reflection, operation_id=operation_id)
        self.commit(agent, proposal)
        requested_evidence = [str(value) for value in reflection.get("evidence_memory_ids", [])]
        applied = {
            key: {
                "time_scale": PARAMETERS[key].time_scale,
                "proposed_delta": float((reflection.get("suggested_adjustments") or {}).get(key, change["delta"])),
                "applied_delta": change["delta"],
                "before": change["before"],
                "after": change["after"],
            }
            for key, change in proposal.changes.items()
        }
        suggestion = dict(reflection.get("suggested_adjustments") or {})
        rejected = {
            key: ("unknown_parameter" if key not in PARAMETERS else "non_numeric_delta")
            for key, value in suggestion.items()
            if key not in proposal.changes and (key not in PARAMETERS or not isinstance(value, (int, float)))
        }
        return {
            "agent": agent.name,
            "operation_id": operation_id,
            "reflection": diary,
            "proposal_id": proposal.proposal_id,
            "proposal_status": proposal.status,
            "confidence": float(np.clip(reflection.get("confidence", 0.5), 0.0, 1.0)),
            "evidence_requested": requested_evidence,
            "evidence_verified": list(proposal.evidence_ids),
            "evidence_factor": 1.0 if proposal.evidence_ids else 0.65,
            "applied_adjustments": applied,
            "rejected_adjustments": rejected,
        }
