from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train_groups: int
    validation_groups: int
    train_rows: int
    validation_rows: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    status: str
    epochs: int
    checkpoint: str
    validation: dict
    qualification: dict
    resumed: bool = False
    dataset_identity: str | None = None
    code_identity: str | None = None


@runtime_checkable
class Trainer(Protocol):
    def run(self, model: object, dataset: object, run_name: str, **configuration) -> TrainingResult: ...


@runtime_checkable
class ModelEvaluator(Protocol):
    def evaluate(self, model: object, dataset: object) -> dict: ...


@runtime_checkable
class CheckpointRegistry(Protocol):
    def register(self, run_name: str) -> dict: ...

    def get(self, digest: str) -> dict: ...

    def artifacts(self) -> list[str]: ...


@runtime_checkable
class TraceExporter(Protocol):
    def export(self, output, **options) -> dict: ...
