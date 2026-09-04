"""Shared transport for all LLM-backed simulation roles."""

from __future__ import annotations

import json
import hashlib
import math
import re
import secrets
import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

from src.simulation.runtime import (
    environment_snapshot,
    env_bool,
    env_float,
    env_int,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4-turbo-preview"
RETIRED_DEEPSEEK_MODELS = {"deepseek-chat", "deepseek-reasoner"}
PLACEHOLDER_KEYS = {
    "", "your_default_key", "your_api_key", "your_api_key_here",
    "replace_with_rotated_key",
}
_REQUEST_SCOPE: ContextVar[str | None] = ContextVar(
    "gfs_llm_request_scope", default=None,
)


def resolve_llm_credential(values: dict[str, str]) -> tuple[str, str]:
    """Resolve a provider credential without persisting or exposing its value."""
    for name in ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        value = str(values.get(name) or "").strip()
        if value and value not in PLACEHOLDER_KEYS:
            return value, name
    return "", ""


def llm_credentials_available(values: dict[str, str]) -> bool:
    return bool(resolve_llm_credential(values)[0])


def provider_preflight() -> dict[str, Any]:
    """Validate provider configuration without constructing a client or calling it."""
    try:
        config = LLMGatewayConfig.from_env()
    except (TypeError, ValueError) as exc:
        return {
            "ready": False,
            "external_calls_made": False,
            "provider": None,
            "error": str(exc),
        }
    summary = config.public_summary()
    return {
        "ready": bool(summary["credentials_available"]),
        "external_calls_made": False,
        "provider": summary,
        "error": None if summary["credentials_available"] else (
            "missing valid DEEPSEEK_API_KEY/API_KEY/OPENAI_API_KEY"
        ),
    }


@dataclass(frozen=True)
class LLMGatewayConfig:
    api_key: str = field(default="", repr=False)
    api_key_source: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: int = 90
    max_retries: int = 6
    circuit_failure_threshold: int = 3
    circuit_reset_s: float = 30.0
    telemetry_capacity: int = 256
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    send_request_id_header: bool = True

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("LLM base URL must not be empty")
        parsed_url = urlparse(self.base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or bool(parsed_url.query)
            or bool(parsed_url.fragment)
        ):
            raise ValueError(
                "LLM base URL must be an http(s) endpoint without credentials, "
                "query parameters, or fragments"
            )
        if not self.model.strip():
            raise ValueError("LLM model must not be empty")
        if self.timeout_s < 1:
            raise ValueError("LLM timeout must be at least one second")
        if self.max_retries < 1:
            raise ValueError("LLM max retries must be at least one")
        if self.circuit_failure_threshold < 1:
            raise ValueError("LLM circuit failure threshold must be positive")
        if (
            not math.isfinite(self.circuit_reset_s)
            or self.circuit_reset_s < 0.0
        ):
            raise ValueError("LLM circuit reset must be finite and non-negative")
        if not 1 <= self.telemetry_capacity <= 10_000:
            raise ValueError("LLM telemetry capacity must be between 1 and 10000")
        for name, value in (
            ("input", self.input_cost_per_million),
            ("output", self.output_cost_per_million),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"LLM {name} cost must be finite and non-negative"
                )
        host = (parsed_url.hostname or "").lower()
        if host == "api.deepseek.com" and self.model in RETIRED_DEEPSEEK_MODELS:
            raise ValueError(
                "retired DeepSeek model name; use deepseek-v4-flash or deepseek-v4-pro"
            )

    @property
    def provider(self) -> str:
        host = (urlparse(self.base_url).hostname or "").lower()
        return "deepseek" if host == "api.deepseek.com" else "openai_compatible"

    def public_summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "credential_source": self.api_key_source or None,
            "credentials_available": bool(self.api_key),
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "request_ids_enabled": self.send_request_id_header,
            "circuit_breaker": {
                "failure_threshold": self.circuit_failure_threshold,
                "reset_s": self.circuit_reset_s,
            },
            "usage_telemetry": {
                "capacity": self.telemetry_capacity,
                "pricing_configured": bool(
                    self.input_cost_per_million
                    or self.output_cost_per_million
                ),
            },
        }

    @classmethod
    def from_env(cls) -> "LLMGatewayConfig":
        values = environment_snapshot()
        api_key, api_key_source = resolve_llm_credential(values)
        return cls(
            api_key=api_key,
            api_key_source=api_key_source,
            base_url=values.get("BASE_URL", DEFAULT_BASE_URL),
            model=values.get("MODEL_NAME", DEFAULT_MODEL),
            timeout_s=env_int(values, "LLM_TIMEOUT_S", 90),
            max_retries=env_int(values, "LLM_MAX_RETRIES", 6),
            circuit_failure_threshold=env_int(
                values, "LLM_CIRCUIT_FAILURE_THRESHOLD", 3,
            ),
            circuit_reset_s=env_float(values, "LLM_CIRCUIT_RESET_S", 30.0),
            telemetry_capacity=env_int(
                values, "LLM_TELEMETRY_CAPACITY", 256,
            ),
            input_cost_per_million=env_float(
                values, "LLM_INPUT_COST_PER_MILLION", 0.0,
            ),
            output_cost_per_million=env_float(
                values, "LLM_OUTPUT_COST_PER_MILLION", 0.0,
            ),
            send_request_id_header=env_bool(
                values, "LLM_SEND_REQUEST_ID_HEADER", True,
            ),
        )


@dataclass(frozen=True)
class ProviderResult:
    content: Any
    provider_request_id: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ProviderAdapter(Protocol):
    provider: str

    def create(
        self,
        client: Any,
        config: LLMGatewayConfig,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
        temperature: float,
        request_id: str,
    ) -> ProviderResult: ...


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _bounded_token_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(count, 1_000_000_000))


def _safe_provider_request_id(value: Any) -> str | None:
    if value is None:
        return None
    identifier = str(value).strip()
    if not identifier:
        return None
    if re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", identifier):
        return identifier
    return "sha256:" + hashlib.sha256(
        identifier.encode("utf-8", errors="replace")
    ).hexdigest()


class OpenAICompatibleAdapter:
    """Translate the internal request contract to OpenAI-compatible chat."""

    provider = "openai_compatible"

    def create(
        self,
        client: Any,
        config: LLMGatewayConfig,
        *,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
        temperature: float,
        request_id: str,
    ) -> ProviderResult:
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "timeout": config.timeout_s,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if config.send_request_id_header:
            kwargs["extra_headers"] = {"X-GFS-Request-ID": request_id}
        response = client.chat.completions.create(**kwargs)
        choices = _field(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise RuntimeError("LLM provider response has no choices.")
        message = _field(choices[0], "message")
        content = _field(message, "content")
        usage = _field(response, "usage")
        input_tokens = _bounded_token_count(
            _field(usage, "prompt_tokens", _field(usage, "input_tokens", 0))
        )
        output_tokens = _bounded_token_count(
            _field(
                usage,
                "completion_tokens",
                _field(usage, "output_tokens", 0),
            )
        )
        total_tokens = _bounded_token_count(
            _field(usage, "total_tokens", input_tokens + output_tokens)
        )
        total_tokens = max(total_tokens, input_tokens + output_tokens)
        return ProviderResult(
            content=content,
            provider_request_id=_safe_provider_request_id(
                _field(response, "id")
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """Explicit DeepSeek boundary over its OpenAI-compatible transport."""

    provider = "deepseek"


def provider_adapter(config: LLMGatewayConfig) -> ProviderAdapter:
    if config.provider == "deepseek":
        return DeepSeekAdapter()
    return OpenAICompatibleAdapter()


class LLMCircuitOpenError(RuntimeError):
    """Raised before transport when the provider circuit is open."""


class LLMGateway:
    """One provider client shared by independent role agents."""

    def __init__(
        self,
        config: LLMGatewayConfig | None = None,
        *,
        adapter: ProviderAdapter | None = None,
    ):
        self.config = config or LLMGatewayConfig.from_env()
        if self.config.api_key.strip() in PLACEHOLDER_KEYS:
            raise ValueError(
                "Missing valid DEEPSEEK_API_KEY/API_KEY/OPENAI_API_KEY. "
                "LLM simulation runs in strict mode and will not fallback."
            )
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "OpenAI SDK is not installed. Run `pip install -r requirements.txt` "
                "before enabling LLM features."
            ) from exc
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        self.call_count = 0
        self.adapter = adapter or provider_adapter(self.config)
        self._initialize_runtime_state()

    def _initialize_runtime_state(self) -> None:
        self._state_lock = threading.RLock()
        self._telemetry = deque(maxlen=self.config.telemetry_capacity)
        self._telemetry_sequence = 0
        self._logical_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._circuit_rejections = 0
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._half_open_in_flight = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._estimated_cost_usd = 0.0
        self._scope_successful_requests: dict[str, int] = {}
        self._gateway_id = "gateway-" + secrets.token_hex(8)
        self._clock = time.monotonic
        self._sleeper = time.sleep

    def _ensure_runtime_state(self) -> None:
        """Keep object.__new__ test doubles and old in-process objects compatible."""
        if not hasattr(self, "adapter"):
            self.adapter = provider_adapter(self.config)
        if not hasattr(self, "_state_lock"):
            self._initialize_runtime_state()
            if not hasattr(self, "call_count"):
                self.call_count = 0

    @staticmethod
    def _scope_id(value: str) -> str:
        scope_id = str(value).strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{4,96}", scope_id):
            raise ValueError(
                "LLM request scope must contain 4-96 safe identifier characters"
            )
        return scope_id

    @contextmanager
    def request_scope(self, scope_id: str):
        """Bind calls in this context to one product-operation identity."""
        self._ensure_runtime_state()
        normalized = self._scope_id(scope_id)
        token = _REQUEST_SCOPE.set(normalized)
        try:
            yield normalized
        finally:
            _REQUEST_SCOPE.reset(token)

    def _request_id(self, value: str | None) -> str:
        scope_id = _REQUEST_SCOPE.get()
        request_id = value or (
            f"{scope_id}.gfs-{secrets.token_hex(8)}"
            if scope_id else "gfs-" + secrets.token_hex(12)
        )
        if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", request_id):
            raise ValueError(
                "LLM request_id must contain 8-128 safe identifier characters"
            )
        return request_id

    def _cost(self, input_tokens: int, output_tokens: int) -> float | None:
        if not (
            self.config.input_cost_per_million
            or self.config.output_cost_per_million
        ):
            return None
        return (
            input_tokens * self.config.input_cost_per_million
            + output_tokens * self.config.output_cost_per_million
        ) / 1_000_000.0

    def _record_event(
        self,
        *,
        request_id: str,
        attempt: int,
        status: str,
        duration_ms: float,
        result: ProviderResult | None = None,
        error_type: str | None = None,
    ) -> None:
        input_tokens = result.input_tokens if result is not None else 0
        output_tokens = result.output_tokens if result is not None else 0
        total_tokens = result.total_tokens if result is not None else 0
        cost = self._cost(input_tokens, output_tokens)
        with self._state_lock:
            self._telemetry_sequence += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._total_tokens += total_tokens
            if cost is not None:
                self._estimated_cost_usd += cost
            self._telemetry.append({
                "sequence": self._telemetry_sequence,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "scope_id": _REQUEST_SCOPE.get(),
                "attempt": int(attempt),
                "status": status,
                "provider": self.adapter.provider,
                "model": str(self.config.model)[:160],
                "provider_request_id": (
                    result.provider_request_id if result is not None else None
                ),
                "duration_ms": round(max(0.0, float(duration_ms)), 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": (
                    round(cost, 10) if cost is not None else None
                ),
                "error_type": str(error_type)[:120] if error_type else None,
            })

    def _circuit_state(self, now: float | None = None) -> dict[str, Any]:
        current = self._clock() if now is None else now
        if self._circuit_open_until > current:
            state = "open"
        elif self._half_open_in_flight:
            state = "half_open"
        else:
            state = "closed"
        return {
            "state": state,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self.config.circuit_failure_threshold,
            "reset_s": self.config.circuit_reset_s,
            "retry_after_s": round(
                max(0.0, self._circuit_open_until - current), 3,
            ),
        }

    def _before_request(self, request_id: str) -> None:
        with self._state_lock:
            self._logical_requests += 1
            now = self._clock()
            if self._circuit_open_until > now:
                self._circuit_rejections += 1
                self._record_event(
                    request_id=request_id,
                    attempt=0,
                    status="circuit_rejected",
                    duration_ms=0.0,
                    error_type="LLMCircuitOpenError",
                )
                raise LLMCircuitOpenError(
                    f"LLM circuit open for request_id={request_id}"
                )
            if self._circuit_open_until > 0.0:
                if self._half_open_in_flight:
                    self._circuit_rejections += 1
                    self._record_event(
                        request_id=request_id,
                        attempt=0,
                        status="circuit_rejected",
                        duration_ms=0.0,
                        error_type="LLMCircuitOpenError",
                    )
                    raise LLMCircuitOpenError(
                        f"LLM circuit half-open probe active for request_id={request_id}"
                    )
                self._half_open_in_flight = True

    def _request_succeeded(self) -> None:
        with self._state_lock:
            self.call_count += 1
            self._successful_requests += 1
            scope_id = _REQUEST_SCOPE.get()
            if scope_id is not None:
                self._scope_successful_requests[scope_id] = (
                    self._scope_successful_requests.get(scope_id, 0) + 1
                )
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._half_open_in_flight = False

    def _request_failed(self) -> None:
        with self._state_lock:
            self._failed_requests += 1
            self._consecutive_failures += 1
            if (
                self._half_open_in_flight
                or self._consecutive_failures
                >= self.config.circuit_failure_threshold
            ):
                self._circuit_open_until = (
                    self._clock() + self.config.circuit_reset_s
                )
            self._half_open_in_flight = False

    def telemetry_cursor(self) -> dict[str, Any]:
        self._ensure_runtime_state()
        with self._state_lock:
            scope_id = _REQUEST_SCOPE.get()
            return {
                "schema_version": 1,
                "gateway_id": self._gateway_id,
                "sequence": self._telemetry_sequence,
                "scope_id": scope_id,
                "successful_calls": (
                    self._scope_successful_requests.get(scope_id, 0)
                    if scope_id is not None else self._successful_requests
                ),
            }

    def telemetry_since(self, cursor: Mapping[str, Any]) -> dict[str, Any]:
        self._ensure_runtime_state()
        if (
            cursor.get("schema_version") != 1
            or cursor.get("gateway_id") != self._gateway_id
            or isinstance(cursor.get("sequence"), bool)
            or not isinstance(cursor.get("sequence"), int)
            or cursor["sequence"] < 0
            or (
                cursor.get("scope_id") is not None
                and (
                    not isinstance(cursor.get("scope_id"), str)
                    or self._scope_id(cursor["scope_id"]) != cursor["scope_id"]
                )
            )
            or isinstance(cursor.get("successful_calls"), bool)
            or not isinstance(cursor.get("successful_calls"), int)
            or cursor["successful_calls"] < 0
        ):
            raise ValueError("invalid LLM telemetry cursor")
        with self._state_lock:
            sequence = int(cursor["sequence"])
            scope_id = cursor.get("scope_id")
            events = [
                dict(row) for row in self._telemetry
                if int(row["sequence"]) > sequence
                and (scope_id is None or row.get("scope_id") == scope_id)
            ]
            first_retained = (
                int(self._telemetry[0]["sequence"])
                if self._telemetry else self._telemetry_sequence + 1
            )
            pricing = bool(
                self.config.input_cost_per_million
                or self.config.output_cost_per_million
            )
            aggregate_successful_calls = (
                self._scope_successful_requests.get(scope_id, 0)
                if scope_id is not None else self._successful_requests
            ) - int(cursor["successful_calls"])
            return {
                "schema_version": 1,
                "gateway_id": self._gateway_id,
                "from_sequence": sequence,
                "to_sequence": self._telemetry_sequence,
                "scope_id": scope_id,
                "truncated": sequence < first_retained - 1,
                "logical_request_ids": list(dict.fromkeys(
                    str(row["request_id"]) for row in events
                )),
                "provider_attempts": sum(
                    row["status"] in {"accepted", "failed"} for row in events
                ),
                "successful_calls": sum(
                    row["status"] == "accepted" for row in events
                ),
                "aggregate_successful_calls": max(
                    0, aggregate_successful_calls,
                ),
                "failed_attempts": sum(
                    row["status"] == "failed" for row in events
                ),
                "circuit_rejections": sum(
                    row["status"] == "circuit_rejected" for row in events
                ),
                "input_tokens": sum(
                    int(row["input_tokens"]) for row in events
                ),
                "output_tokens": sum(
                    int(row["output_tokens"]) for row in events
                ),
                "total_tokens": sum(
                    int(row["total_tokens"]) for row in events
                ),
                "cost_estimate_available": pricing,
                "estimated_cost_usd": (
                    round(sum(
                        float(row["estimated_cost_usd"] or 0.0)
                        for row in events
                    ), 10) if pricing else None
                ),
                "circuit": self._circuit_state(),
                "events": events,
                "contains_prompts_or_credentials": False,
            }

    def telemetry_snapshot(self) -> dict[str, Any]:
        self._ensure_runtime_state()
        with self._state_lock:
            events = [dict(row) for row in self._telemetry]
            pricing = bool(
                self.config.input_cost_per_million
                or self.config.output_cost_per_million
            )
            return {
                "schema_version": 1,
                "gateway_id": self._gateway_id,
                "provider": self.adapter.provider,
                "model": str(self.config.model)[:160],
                "logical_requests": self._logical_requests,
                "successful_calls": self._successful_requests,
                "failed_requests": self._failed_requests,
                "provider_attempts": sum(
                    row["status"] in {"accepted", "failed"} for row in events
                ),
                "circuit_rejections": self._circuit_rejections,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._total_tokens,
                "cost_estimate_available": pricing,
                "estimated_cost_usd": (
                    round(self._estimated_cost_usd, 10) if pricing else None
                ),
                "circuit": self._circuit_state(),
                "retained_event_count": len(events),
                "telemetry_capacity": self.config.telemetry_capacity,
                "events": events,
                "contains_prompts_or_credentials": False,
            }

    @staticmethod
    def extract_json_object(content: Any) -> dict[str, Any]:
        if content is None or not str(content).strip():
            raise RuntimeError("Empty LLM response content.")
        text = str(content).strip()
        candidates = [text]
        candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE))
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate.strip())
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError("Unable to extract JSON object from model output.")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.8,
        request_id: str | None = None,
    ) -> str:
        self._ensure_runtime_state()
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ValueError("LLM temperature must be finite and between 0 and 2")
        logical_request_id = self._request_id(request_id)
        self._before_request(logical_request_id)
        for attempt in range(self.config.max_retries):
            started = self._clock()
            provider_result: ProviderResult | None = None
            try:
                provider_result = self.adapter.create(
                    self.client,
                    self.config,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=json_mode,
                    temperature=temperature,
                    request_id=logical_request_id,
                )
                content = provider_result.content
                if json_mode:
                    accepted = json.dumps(
                        self.extract_json_object(content), ensure_ascii=False,
                    )
                else:
                    if content is None:
                        raise RuntimeError("Empty LLM response content.")
                    accepted = str(content)
                with self._state_lock:
                    self._record_event(
                        request_id=logical_request_id,
                        attempt=attempt + 1,
                        status="accepted",
                        duration_ms=(self._clock() - started) * 1000.0,
                        result=provider_result,
                    )
                    self._request_succeeded()
                return accepted
            except Exception as exc:
                final_attempt = attempt == self.config.max_retries - 1
                with self._state_lock:
                    self._record_event(
                        request_id=logical_request_id,
                        attempt=attempt + 1,
                        status="failed",
                        duration_ms=(self._clock() - started) * 1000.0,
                        result=provider_result,
                        error_type=type(exc).__name__,
                    )
                    if final_attempt:
                        self._request_failed()
                if final_attempt:
                    raise RuntimeError(
                        "LLM call failed after retries "
                        f"request_id={logical_request_id} "
                        f"model={str(self.config.model)[:160]} "
                        f"error_type={type(exc).__name__}"
                    ) from exc
                self._sleeper(min(16.0, 1.5 * (2**attempt)))
        raise RuntimeError("LLM call failed unexpectedly.")


_GATEWAYS: dict[LLMGatewayConfig, LLMGateway] = {}
_GATEWAY_LOCK = threading.Lock()


def get_shared_llm_gateway(config: LLMGatewayConfig | None = None) -> LLMGateway:
    """Share transport only between callers with exactly the same config."""
    resolved = config or LLMGatewayConfig.from_env()
    gateway = _GATEWAYS.get(resolved)
    if gateway is None:
        with _GATEWAY_LOCK:
            gateway = _GATEWAYS.get(resolved)
            if gateway is None:
                gateway = LLMGateway(resolved)
                _GATEWAYS[resolved] = gateway
    return gateway


def clear_shared_llm_gateways() -> None:
    """Drop process-local transports between isolated product/test sessions."""
    with _GATEWAY_LOCK:
        _GATEWAYS.clear()
