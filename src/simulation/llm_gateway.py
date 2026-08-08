"""Shared transport for all LLM-backed simulation roles."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.simulation.runtime import environment_snapshot, env_int

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4-turbo-preview"


@dataclass(frozen=True)
class LLMGatewayConfig:
    api_key: str = field(default="", repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_s: int = 90
    max_retries: int = 6

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("LLM base URL must not be empty")
        if not self.model.strip():
            raise ValueError("LLM model must not be empty")
        if self.timeout_s < 1:
            raise ValueError("LLM timeout must be at least one second")
        if self.max_retries < 1:
            raise ValueError("LLM max retries must be at least one")

    @classmethod
    def from_env(cls) -> "LLMGatewayConfig":
        values = environment_snapshot()
        return cls(
            api_key=values.get("API_KEY") or values.get("OPENAI_API_KEY", ""),
            base_url=values.get("BASE_URL", DEFAULT_BASE_URL),
            model=values.get("MODEL_NAME", DEFAULT_MODEL),
            timeout_s=env_int(values, "LLM_TIMEOUT_S", 90),
            max_retries=env_int(values, "LLM_MAX_RETRIES", 6),
        )


class LLMGateway:
    """One provider client shared by independent role agents."""

    def __init__(self, config: LLMGatewayConfig | None = None):
        self.config = config or LLMGatewayConfig.from_env()
        if self.config.api_key.strip() in {"", "your_default_key", "your_api_key_here"}:
            raise ValueError(
                "Missing valid API_KEY/OPENAI_API_KEY. "
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
        raise RuntimeError(f"Unable to extract JSON object from model output: {text[:260]}")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.8,
    ) -> str:
        for attempt in range(self.config.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "timeout": self.config.timeout_s,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if json_mode:
                    accepted = json.dumps(
                        self.extract_json_object(content), ensure_ascii=False,
                    )
                    self.call_count += 1
                    return accepted
                if content is None:
                    raise RuntimeError("Empty LLM response content.")
                self.call_count += 1
                return str(content)
            except Exception as exc:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(
                        f"LLM call failed after retries on model={self.config.model}: {exc}"
                    ) from exc
                time.sleep(min(16.0, 1.5 * (2**attempt)))
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
