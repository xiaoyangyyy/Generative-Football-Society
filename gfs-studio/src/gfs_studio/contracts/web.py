from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WebResponse:
    status: int
    body: bytes
    content_type: str
    headers: tuple[tuple[str, str], ...]


class HttpError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)
