from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CognitionRequest:
    role: str
    system_prompt: str
    user_prompt: str
    match_id: str
    seed: int
    json_mode: bool = True
    temperature: float = 0.2


@dataclass(frozen=True, slots=True)
class ProviderResult:
    content: Any
    provider_request_id: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ProviderAdapter(Protocol):
    provider: str

    def complete(self, request: CognitionRequest, request_id: str) -> ProviderResult: ...


class TransientProviderError(RuntimeError):
    pass


class PermanentProviderError(RuntimeError):
    pass


class CircuitOpenError(RuntimeError):
    pass
