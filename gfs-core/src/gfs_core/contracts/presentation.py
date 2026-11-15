from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from gfs_core.contracts.status import StatusSnapshot


class ReportPresenter(Protocol):
    def status(self, snapshot: StatusSnapshot, limit: int) -> list[dict[str, object]]: ...

    def monte_carlo(self, rows: list[dict[str, object]]) -> list[dict[str, object]]: ...


class CommandService(Protocol):
    def execute(self, arguments: Sequence[str]) -> str: ...
