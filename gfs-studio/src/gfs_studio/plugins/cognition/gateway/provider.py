from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict, deque
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
    telemetry_match_capacity: int = 256
    retry_base_seconds: float = 0.05
    max_retry_delay_seconds: float = 1.0
    max_cost_per_match: float = 0.0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_retries, self.failure_threshold, self.telemetry_capacity,
            self.telemetry_match_capacity,
        )
        if any(isinstance(value, bool) or value < 1 for value in integer_fields):
            raise ValueError("retry, circuit, and telemetry limits must be positive")
        numeric = (
            self.reset_seconds, self.retry_base_seconds,
            self.max_retry_delay_seconds, self.max_cost_per_match,
            self.input_cost_per_million, self.output_cost_per_million,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("timing and cost limits must be finite and non-negative")
        if self.max_retry_delay_seconds < self.retry_base_seconds:
            raise ValueError("maximum retry delay is below the base delay")


class LLMGateway:
    def __init__(
        self, provider, config: GatewayConfig | None = None, *,
        clock=time.monotonic, sleeper=time.sleep,
    ) -> None:
        self.provider = provider
        self.config = config or GatewayConfig()
        self.clock = clock
        self.sleeper = sleeper
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0
        self._probe_inflight = False
        self._lock = threading.RLock()
        self.telemetry = OrderedDict()
        self.history = deque(maxlen=self.config.telemetry_capacity)

    def generate(self, request: CognitionRequest) -> ProviderResult:
        self._allow_request()
        request_id = _request_id(request)
        last_error = None
        try:
            for attempt in range(self.config.max_retries):
                self._check_budget(request.match_id)
                try:
                    result = self.provider.complete(request, request_id)
                    self._record_success(request.match_id, request_id, result)
                    return result
                except PermanentProviderError:
                    self._release_probe()
                    raise
                except TransientProviderError as error:
                    last_error = error
                    counters = self._counters(request.match_id)
                    counters["failed"] += 1
                    if attempt + 1 < self.config.max_retries:
                        self.sleeper(_retry_delay(request_id, attempt, self.config))
            error = last_error or TransientProviderError("provider failed")
            self._record_failure(request.match_id, request_id, error)
            raise error
        except BaseException:
            self._release_probe()
            raise

    def _allow_request(self) -> None:
        with self._lock:
            if self.state == "open":
                if self.clock() - self.opened_at < self.config.reset_seconds:
                    raise CircuitOpenError("provider circuit is open")
                self.state = "half-open"
                self._probe_inflight = False
            if self.state == "half-open":
                if self._probe_inflight:
                    raise CircuitOpenError("a half-open provider probe is already running")
                self._probe_inflight = True

    def _release_probe(self) -> None:
        with self._lock:
            self._probe_inflight = False

    def _check_budget(self, match_id: str) -> None:
        limit = self.config.max_cost_per_match
        if limit > 0 and self._counters(match_id)["cost"] >= limit:
            raise PermanentProviderError("recorded provider cost limit reached")

    def _counters(self, match_id: str) -> dict:
        with self._lock:
            counters = self.telemetry.pop(match_id, None)
            if counters is None:
                counters = _counters()
            self.telemetry[match_id] = counters
            while len(self.telemetry) > self.config.telemetry_match_capacity:
                self.telemetry.popitem(last=False)
            return counters

    def _record_success(self, match_id, request_id, result) -> None:
        with self._lock:
            self.state = "closed"
            self.failures = 0
            self._probe_inflight = False
            counters = self._counters(match_id)
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
        with self._lock:
            self.failures += 1
            self._probe_inflight = False
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
        with self._lock:
            return {
                "provider": self.config.provider,
                "model": self.config.model,
                "circuit_state": self.state,
                "failures": self.failures,
                "matches": dict(self.telemetry),
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

    def __init__(self, client, model: str, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive")
        self.client = client
        self.model = model
        self.timeout_seconds = timeout_seconds

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
                timeout=self.timeout_seconds,
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status in {400, 401, 403, 404, 422}:
                raise PermanentProviderError(type(error).__name__) from error
            raise TransientProviderError(type(error).__name__) from error
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as error:
            raise PermanentProviderError("provider response has no content") from error
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0))
        output_tokens = int(getattr(usage, "completion_tokens", 0))
        total_tokens = int(getattr(usage, "total_tokens", 0))
        return ProviderResult(
            content,
            getattr(response, "id", None),
            input_tokens,
            output_tokens,
            total_tokens or input_tokens + output_tokens,
        )


def _retry_delay(request_id: str, attempt: int, config: GatewayConfig) -> float:
    base = min(
        config.max_retry_delay_seconds,
        config.retry_base_seconds * 2**attempt,
    )
    if base == 0:
        return 0.0
    digest = hashlib.sha256(f"{request_id}:{attempt}".encode()).digest()[0]
    return base * (0.9 + 0.2 * digest / 255)


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
