import json

import pytest

from gfs_studio.bootstrap.application import StudioApplication
from gfs_studio.contracts.cognition import (
    CircuitOpenError,
    CognitionRequest,
    ProviderResult,
    TransientProviderError,
)
from gfs_studio.plugins.cognition.gateway.provider import GatewayConfig, LLMGateway


class FlakyProvider:
    provider = "mock"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.request_ids = []

    def complete(self, request, request_id):
        self.request_ids.append(request_id)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def request(match_id="match-1"):
    return CognitionRequest("coach", "system", "topic", match_id, 42)


def test_retry_uses_stable_request_id_and_records_attempts():
    provider = FlakyProvider([
        TransientProviderError("first"),
        TransientProviderError("second"),
        ProviderResult({"ok": True}, "provider-1", 10, 5, 15),
    ])
    gateway = LLMGateway(provider, GatewayConfig(max_retries=3))
    result = gateway.generate(request())
    assert result.content == {"ok": True}
    assert len(set(provider.request_ids)) == 1
    counters = gateway.status()["matches"]["match-1"]
    assert counters["accepted"] == 1
    assert counters["failed"] == 2
    assert counters["total_tokens"] == 15


def test_circuit_opens_and_recovers_after_reset():
    provider = FlakyProvider([
        TransientProviderError("first"),
        TransientProviderError("second"),
        ProviderResult({"ok": True}, "provider-2", 1, 1, 2),
    ])
    now = [0.0]
    gateway = LLMGateway(
        provider,
        GatewayConfig(max_retries=1, failure_threshold=2, reset_seconds=30),
        clock=lambda: now[0],
    )
    with pytest.raises(TransientProviderError):
        gateway.generate(request())
    with pytest.raises(TransientProviderError):
        gateway.generate(request())
    assert gateway.status()["circuit_state"] == "open"
    with pytest.raises(CircuitOpenError):
        gateway.generate(request())
    now[0] = 31.0
    assert gateway.generate(request()).content == {"ok": True}
    assert gateway.status()["circuit_state"] == "closed"


def test_match_counters_are_isolated_and_prompt_free():
    provider = FlakyProvider([
        ProviderResult({"ok": True}, "provider-3", 1, 2, 3),
        ProviderResult({"ok": True}, "provider-4", 2, 3, 5),
    ])
    gateway = LLMGateway(provider)
    gateway.generate(request("match-a"))
    gateway.generate(request("match-b"))
    status = gateway.status()
    assert status["matches"]["match-a"]["total_tokens"] == 3
    assert status["matches"]["match-b"]["total_tokens"] == 5
    assert "topic" not in json.dumps(status)
    assert "system" not in json.dumps(status)


def test_offline_cognition_service_returns_bounded_reflection(tmp_path):
    app = StudioApplication(tmp_path)
    result = app.reflect(role="coach", topic="review match", match_id="match-1")
    assert result["authority"] == "advisory_only"
    status = app.cognition_status()
    assert status["provider"] == "rule_based"
    assert status["matches"]["match-1"]["accepted"] == 1
    assert json.loads(app.run_cli(["cognition", "status"]))["provider"] == "rule_based"
