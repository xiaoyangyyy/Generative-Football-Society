from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DecisionCase:
    case_id: str
    home: str
    away: str
    seed: int
    candidate_ids: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "home": self.home,
            "away": self.away,
            "seed": self.seed,
            "candidate_ids": list(self.candidate_ids),
        }


@runtime_checkable
class ModelAssessment(Protocol):
    def assess_case(self, case: DecisionCase, observation, action) -> dict: ...


@runtime_checkable
class ProductWorkflow(Protocol):
    def overview(self) -> dict: ...

    def micro_match(self, home: str, away: str, seed: int) -> dict: ...

    def compare_tactics(self, home: str, away: str, seed: int) -> dict: ...


@runtime_checkable
class TaskStore(Protocol):
    def submit(self, idempotency_key: str, payload: dict) -> dict: ...

    def get(self, task_id: str) -> dict | None: ...

    def list(self) -> list[dict]: ...

    def wait(self, task_id: str, timeout: float = 30.0) -> dict: ...

    def recover(self) -> int: ...

    def resume(self, task_id: str | None = None) -> int: ...


@runtime_checkable
class Comparison(Protocol):
    def compare(self, baseline: dict, treatment: dict) -> dict: ...

    def aggregate(self, comparisons: list[dict]) -> dict: ...


@runtime_checkable
class Recovery(Protocol):
    def create(self) -> dict: ...

    def verify(self, backup_id: str) -> dict: ...

    def restore(self, backup_id: str, destination, *, replace: bool = False) -> dict: ...


@runtime_checkable
class CandidateEvaluation(Protocol):
    def evaluate(self, observations: list[dict]) -> dict: ...
