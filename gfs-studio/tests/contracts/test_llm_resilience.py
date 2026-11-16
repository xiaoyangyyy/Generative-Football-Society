import threading

import pytest

from gfs_studio.contracts.cognition import (
    CircuitOpenError,
    CognitionRequest,
    PermanentProviderError,
    ProviderResult,
    TransientProviderError,
)
from gfs_studio.plugins.cognition.gateway.provider import (
    GatewayConfig,
    LLMGateway,
    OpenAICompatibleProvider,
    RuleBasedProvider,
)


def request(match_id="match-1"):
    return CognitionRequest("coach", "system", "topic", match_id, 42)


def result():
    return ProviderResult({"ok": True}, "provider", 1, 1, 2)


def test_retry_backoff_is_bounded_and_does_not_sleep_after_success():
    class Provider:
        def __init__(self):
            self.calls = 0

        def complete(self, request, request_id):
            self.calls += 1
            if self.calls == 1:
                raise TransientProviderError("temporary")
            return result()

    delays = []
    gateway = LLMGateway(
        Provider(),
        GatewayConfig(max_retries=2, retry_base_seconds=0.25),
        sleeper=delays.append,
    )
    assert gateway.generate(request()).content == {"ok": True}
    assert len(delays) == 1
    assert 0.225 <= delays[0] <= 0.275


def test_half_open_circuit_allows_only_one_probe():
    entered = threading.Event()
    release = threading.Event()

    class Provider:
        def __init__(self):
            self.calls = 0

        def complete(self, request, request_id):
            self.calls += 1
            if self.calls == 1:
                raise TransientProviderError("failed")
            entered.set()
            assert release.wait(2)
            return result()

    now = [0.0]
    gateway = LLMGateway(
        Provider(),
        GatewayConfig(max_retries=1, failure_threshold=1, reset_seconds=1),
        clock=lambda: now[0],
    )
    with pytest.raises(TransientProviderError):
        gateway.generate(request())
    now[0] = 2.0
    completed = []
    thread = threading.Thread(target=lambda: completed.append(gateway.generate(request())))
    thread.start()
    assert entered.wait(1)
    with pytest.raises(CircuitOpenError, match="already running"):
        gateway.generate(request())
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert completed[0].content == {"ok": True}


def test_authentication_error_is_permanent_and_not_retried():
    class AuthenticationError(Exception):
        status_code = 401

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **options):
            self.calls += 1
            assert options["timeout"] == 30.0
            raise AuthenticationError("unauthorized")

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    provider = OpenAICompatibleProvider(client, "test-model")
    gateway = LLMGateway(provider, GatewayConfig(max_retries=3, retry_base_seconds=0))
    with pytest.raises(PermanentProviderError):
        gateway.generate(request())
    assert completions.calls == 1


def test_match_telemetry_is_bounded_without_recording_prompts():
    gateway = LLMGateway(
        RuleBasedProvider(), GatewayConfig(telemetry_match_capacity=2),
    )
    for match_id in ("match-a", "match-b", "match-c"):
        gateway.generate(request(match_id))
    status = gateway.status()
    assert set(status["matches"]) == {"match-b", "match-c"}
    assert "topic" not in str(status)
