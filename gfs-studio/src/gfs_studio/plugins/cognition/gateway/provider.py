from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass

from gfs_studio.contracts.cognition import (
    CircuitOpenError,
    CognitionRequest,
    PermanentProviderError,
    ProviderResult,
    TransientProviderError,
)


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    provider: str = "rule_based"
    model: str = "gfs-offline-v1"
    base_url: str = ""
    max_retries: int = 3
    failure_threshold: int = 3
    reset_seconds: float = 30.0
    telemetry_capacity: int = 256
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


class LLMGateway:
    def __init__(self, provider, config: GatewayConfig | None = None, *, clock=time.monotonic) -> None:
        self.provider = provider
        self.config = config or GatewayConfig()
        self.clock = clock
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0
        self.telemetry = {}
        self.history = deque(maxlen=self.config.telemetry_capacity)

    def generate(self, request: CognitionRequest) -> ProviderResult:
        self._allow_request()
        request_id = _request_id(request)
        last_error = None
        for _ in range(self.config.max_retries):
            try:
                result = self.provider.complete(request, request_id)
                self._record_success(request.match_id, request_id, result)
                return result
            except PermanentProviderError:
                raise
            except TransientProviderError as error:
                last_error = error
                counters = self.telemetry.setdefault(request.match_id, _counters())
                counters["failed"] += 1
        error = last_error or TransientProviderError("provider failed")
        self._record_failure(request.match_id, request_id, error)
        raise error

    def _allow_request(self) -> None:
        if self.state == "open":
            if self.clock() - self.opened_at < self.config.reset_seconds:
                raise CircuitOpenError("provider circuit is open")
            self.state = "half-open"

    def _record_success(self, match_id, request_id, result) -> None:
        self.state = "closed"
        self.failures = 0
        counters = self.telemetry.setdefault(match_id, _counters())
        counters["accepted"] += 1
        counters["input_tokens"] += result.input_tokens
        counters["output_tokens"] += result.output_tokens
        counters["total_tokens"] += result.total_tokens
        counters["cost"] += (
            result.input_tokens * self.config.input_cost_per_million
            + result.output_tokens * self.config.output_cost_per_million
        ) / 1_000_000
        self.history.append({
            "match_id": match_id,
            "request_id": request_id,
            "accepted": True,
        })

    def _record_failure(self, match_id, request_id, error) -> None:
        self.failures += 1
        if self.failures >= self.config.failure_threshold:
            self.state = "open"
            self.opened_at = self.clock()
        self.history.append({
            "match_id": match_id,
            "request_id": request_id,
            "accepted": False,
            "error_type": type(error).__name__,
        })

    def status(self) -> dict:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "circuit_state": self.state,
            "failures": self.failures,
            "matches": self.telemetry,
            "recent": list(self.history),
        }


class RuleBasedProvider:
    provider = "rule_based"

    def complete(self, request: CognitionRequest, request_id: str) -> ProviderResult:
        content = {
            "role": request.role,
            "summary": f"Review {request.user_prompt} with bounded tactical evidence.",
            "recommended_action": "retain_current_vector",
            "authority": "advisory_only",
        }
        return ProviderResult(content, request_id, 0, 0, 0)


class OpenAICompatibleProvider:
    provider = "openai_compatible"

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def complete(self, request: CognitionRequest, request_id: str) -> ProviderResult:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                temperature=request.temperature,
                response_format={"type": "json_object"} if request.json_mode else None,
                extra_headers={"X-Request-ID": request_id},
            )
        except Exception as error:
            raise TransientProviderError(type(error).__name__) from error
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return ProviderResult(
            choice.message.content,
            getattr(response, "id", None),
            int(getattr(usage, "prompt_tokens", 0)),
            int(getattr(usage, "completion_tokens", 0)),
            int(getattr(usage, "total_tokens", 0)),
        )


def _request_id(request: CognitionRequest) -> str:
    payload = "\x1f".join([
        request.role,
        request.match_id,
        str(request.seed),
        request.system_prompt,
        request.user_prompt,
    ])
    return "gfs-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _counters() -> dict:
    return {
        "accepted": 0,
        "failed": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }
