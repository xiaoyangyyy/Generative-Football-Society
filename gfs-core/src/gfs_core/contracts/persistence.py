from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


class TransactionalResource(Protocol):
    def snapshot(self) -> Mapping[str, Any]: ...

    def restore(self, snapshot: Mapping[str, Any]) -> None: ...

    def verify(self, result: Any) -> None: ...


@runtime_checkable
class MicroCheckpointStore(Protocol):
    def save(self, name: str, document: dict) -> dict: ...

    def load(self, name: str) -> dict: ...


class RollbackError(RuntimeError):
    def __init__(self, original_error: BaseException, rollback_error: BaseException):
        self.original_error_type = type(original_error).__name__
        self.rollback_error_type = type(rollback_error).__name__
        super().__init__(
            f"transaction rollback failed: {self.original_error_type}, "
            f"{self.rollback_error_type}"
        )
