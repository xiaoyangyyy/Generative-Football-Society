from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class M2StudyStatus:
    protocol_id: str
    version: str
    state: str
    runs: int
    budget: int
    promotion_supported: bool
    next_action: str
    identity_status: str

    @property
    def complete(self) -> bool:
        return self.runs == self.budget

    def as_dict(self) -> dict:
        return {
            "protocol_id": self.protocol_id,
            "version": self.version,
            "state": self.state,
            "runs": self.runs,
            "budget": self.budget,
            "promotion_supported": self.promotion_supported,
            "next_action": self.next_action,
            "identity_status": self.identity_status,
            "complete": self.complete,
        }
