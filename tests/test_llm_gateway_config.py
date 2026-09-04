from src.simulation import llm_gateway
import pytest
import threading
from types import SimpleNamespace


def test_llm_gateway_config_is_read_at_call_time(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "first")
    monkeypatch.setenv("MODEL_NAME", "model-a")
    first = llm_gateway.LLMGatewayConfig.from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "second")
    monkeypatch.setenv("MODEL_NAME", "model-b")
    second = llm_gateway.LLMGatewayConfig.from_env()
    assert first.api_key == "first" and first.model == "model-a"
    assert second.api_key == "second" and second.model == "model-b"


def test_deepseek_specific_key_has_explicit_priority(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("API_KEY", "generic-key")
    monkeypatch.setenv("BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("MODEL_NAME", "deepseek-v4-flash")
    config = llm_gateway.LLMGatewayConfig.from_env()
    assert config.api_key == "deepseek-key"
    assert config.api_key_source == "DEEPSEEK_API_KEY"
    assert config.provider == "deepseek"
    assert config.public_summary()["credentials_available"]
    assert "deepseek-key" not in repr(config.public_summary())


@pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
def test_retired_deepseek_model_names_fail_preflight(model):
    with pytest.raises(ValueError, match="retired DeepSeek"):
        llm_gateway.LLMGatewayConfig(
            api_key="key", base_url="https://api.deepseek.com", model=model,
        )


def test_gateway_pool_is_scoped_by_complete_config(monkeypatch):
    created = []

    class _Gateway:
        def __init__(self, config):
            self.config = config
            created.append(self)

    monkeypatch.setattr(llm_gateway, "LLMGateway", _Gateway)
    llm_gateway.clear_shared_llm_gateways()
    a = llm_gateway.LLMGatewayConfig(api_key="a", model="one")
    b = llm_gateway.LLMGatewayConfig(api_key="b", model="two")
    assert llm_gateway.get_shared_llm_gateway(a) is llm_gateway.get_shared_llm_gateway(a)
    assert llm_gateway.get_shared_llm_gateway(b) is not llm_gateway.get_shared_llm_gateway(a)
    assert len(created) == 2
    llm_gateway.clear_shared_llm_gateways()


@pytest.mark.parametrize("field,value", [
    ("timeout_s", 0),
    ("max_retries", 0),
    ("base_url", ""),
    ("model", ""),
    ("circuit_failure_threshold", 0),
    ("circuit_reset_s", -1.0),
    ("telemetry_capacity", 0),
    ("telemetry_capacity", 10_001),
    ("input_cost_per_million", -0.1),
    ("output_cost_per_million", float("inf")),
    ("base_url", "ftp://provider.invalid/v1"),
    ("base_url", "https://user:secret@provider.invalid/v1"),
    ("base_url", "https://provider.invalid/v1?api_key=secret"),
])
def test_gateway_config_rejects_invalid_operational_values(field, value):
    values = {"api_key": "key", field: value}
    with pytest.raises(ValueError):
        llm_gateway.LLMGatewayConfig(**values)


def test_gateway_counts_only_schema_accepted_provider_responses():
    gateway = object.__new__(llm_gateway.LLMGateway)
    gateway.config = llm_gateway.LLMGatewayConfig(
        api_key="secret", model="test", max_retries=2,
    )
    responses = iter(("not json", '{"action":"press"}'))

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses)),
            )])

    gateway.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )
    gateway.call_count = 0
    result = gateway.complete("system", "user", json_mode=True)
    assert result == '{"action": "press"}'
    assert gateway.call_count == 1


def test_gateway_config_repr_does_not_expose_api_key():
    config = llm_gateway.LLMGatewayConfig(api_key="super-secret")
    assert "super-secret" not in repr(config)


def test_provider_preflight_is_zero_call_and_secret_free(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "local-test-secret")
    monkeypatch.setenv("BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("MODEL_NAME", "deepseek-v4-flash")
    result = llm_gateway.provider_preflight()
    assert result["ready"]
    assert not result["external_calls_made"]
    assert result["provider"]["credential_source"] == "DEEPSEEK_API_KEY"
    assert "local-test-secret" not in repr(result)


def test_provider_preflight_reports_missing_key_without_client(monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = llm_gateway.provider_preflight()
    assert not result["ready"]
    assert not result["external_calls_made"]
    assert result["provider"]["credentials_available"] is False


def _response(content, *, request_id, input_tokens=0, output_tokens=0):
    return SimpleNamespace(
        id=request_id,
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
        )],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def test_gateway_retries_one_logical_request_with_safe_usage_and_cost_telemetry():
    gateway = object.__new__(llm_gateway.LLMGateway)
    gateway.config = llm_gateway.LLMGatewayConfig(
        api_key="super-secret-key",
        model="test-model",
        max_retries=2,
        input_cost_per_million=2.0,
        output_cost_per_million=4.0,
    )
    responses = iter((
        _response(
            "invalid-json",
            request_id="provider-first",
            input_tokens=10,
            output_tokens=5,
        ),
        _response(
            '{"action":"press"}',
            request_id="provider-second",
            input_tokens=20,
            output_tokens=7,
        ),
    ))
    requests = []

    class _Completions:
        @staticmethod
        def create(**kwargs):
            requests.append(kwargs)
            return next(responses)

    gateway.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )
    gateway.call_count = 0
    cursor = gateway.telemetry_cursor()
    gateway._sleeper = lambda _delay: None

    result = gateway.complete(
        "private-system-prompt",
        "private-user-prompt",
        json_mode=True,
        request_id="request-retry-001",
    )
    delta = gateway.telemetry_since(cursor)

    assert result == '{"action": "press"}'
    assert gateway.call_count == 1
    assert len(requests) == 2
    assert {
        request["extra_headers"]["X-GFS-Request-ID"]
        for request in requests
    } == {"request-retry-001"}
    assert delta["logical_request_ids"] == ["request-retry-001"]
    assert delta["provider_attempts"] == 2
    assert delta["failed_attempts"] == 1
    assert delta["successful_calls"] == 1
    assert delta["input_tokens"] == 30
    assert delta["output_tokens"] == 12
    assert delta["total_tokens"] == 42
    assert delta["estimated_cost_usd"] == pytest.approx(0.000108)
    serialized = repr(delta)
    assert "private-system-prompt" not in serialized
    assert "private-user-prompt" not in serialized
    assert "super-secret-key" not in serialized
    assert delta["contains_prompts_or_credentials"] is False


def test_gateway_circuit_opens_without_provider_call_and_recovers_half_open():
    gateway = object.__new__(llm_gateway.LLMGateway)
    gateway.config = llm_gateway.LLMGatewayConfig(
        api_key="secret",
        model="test",
        max_retries=1,
        circuit_failure_threshold=2,
        circuit_reset_s=10.0,
    )
    state = {"calls": 0, "fail": True, "clock": 100.0}

    class _Completions:
        @staticmethod
        def create(**_kwargs):
            state["calls"] += 1
            if state["fail"]:
                raise TimeoutError("provider unavailable")
            return _response(
                "recovered",
                request_id="provider-recovered",
                input_tokens=3,
                output_tokens=2,
            )

    gateway.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )
    gateway.call_count = 0
    gateway.telemetry_cursor()
    gateway._clock = lambda: state["clock"]
    gateway._sleeper = lambda _delay: None

    for request_id in ("request-fail-01", "request-fail-02"):
        with pytest.raises(RuntimeError, match=request_id):
            gateway.complete("system", "user", request_id=request_id)
    assert state["calls"] == 2
    assert gateway.telemetry_snapshot()["circuit"]["state"] == "open"

    with pytest.raises(llm_gateway.LLMCircuitOpenError):
        gateway.complete(
            "system", "user", request_id="request-blocked-03",
        )
    assert state["calls"] == 2
    assert gateway.telemetry_snapshot()["circuit_rejections"] == 1

    state["clock"] = 111.0
    state["fail"] = False
    assert gateway.complete(
        "system", "user", request_id="request-probe-004",
    ) == "recovered"
    snapshot = gateway.telemetry_snapshot()
    assert state["calls"] == 3
    assert snapshot["circuit"]["state"] == "closed"
    assert snapshot["circuit"]["consecutive_failures"] == 0
    assert snapshot["successful_calls"] == 1
    assert snapshot["failed_requests"] == 2


def test_gateway_telemetry_is_bounded_but_aggregate_counts_are_not():
    gateway = object.__new__(llm_gateway.LLMGateway)
    gateway.config = llm_gateway.LLMGatewayConfig(
        api_key="secret",
        model="test",
        max_retries=1,
        telemetry_capacity=2,
    )
    state = {"calls": 0}

    class _Completions:
        @staticmethod
        def create(**_kwargs):
            state["calls"] += 1
            return _response(
                "ok",
                request_id=f"provider-{state['calls']}",
                input_tokens=1,
                output_tokens=1,
            )

    gateway.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )
    gateway.call_count = 0
    cursor = gateway.telemetry_cursor()
    for index in range(3):
        gateway.complete(
            "system", "user", request_id=f"request-bounded-{index}",
        )

    snapshot = gateway.telemetry_snapshot()
    delta = gateway.telemetry_since(cursor)
    assert snapshot["successful_calls"] == 3
    assert snapshot["total_tokens"] == 6
    assert snapshot["retained_event_count"] == 2
    assert delta["truncated"] is True
    assert len(delta["events"]) == 2


def test_gateway_rejects_unsafe_request_identity_before_transport():
    gateway = object.__new__(llm_gateway.LLMGateway)
    gateway.config = llm_gateway.LLMGatewayConfig(
        api_key="secret", model="test", max_retries=1,
    )
    called = []
    gateway.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: called.append(kwargs),
        )),
    )
    gateway.call_count = 0

    with pytest.raises(ValueError, match="request_id"):
        gateway.complete("system", "user", request_id="bad id")
    with pytest.raises(ValueError, match="temperature"):
        gateway.complete("system", "user", temperature=float("nan"))
    assert called == []


def test_provider_adapter_is_explicit_and_public_summary_is_secret_free():
    config = llm_gateway.LLMGatewayConfig(
        api_key="deep-secret",
        api_key_source="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
    )
    adapter = llm_gateway.provider_adapter(config)
    summary = config.public_summary()

    assert isinstance(adapter, llm_gateway.DeepSeekAdapter)
    assert adapter.provider == "deepseek"
    assert summary["request_ids_enabled"] is True
    assert summary["circuit_breaker"]["failure_threshold"] == 3
    assert summary["usage_telemetry"]["capacity"] == 256
    assert "deep-secret" not in repr(summary)


def test_concurrent_request_scopes_keep_provider_evidence_separate():
    gateway = object.__new__(llm_gateway.LLMGateway)
    gateway.config = llm_gateway.LLMGatewayConfig(
        api_key="secret", model="test", max_retries=1,
    )
    barrier = threading.Barrier(2)
    requests = []
    results = {}

    class _Completions:
        @staticmethod
        def create(**kwargs):
            requests.append(kwargs)
            return _response(
                "ok",
                request_id="provider-" + str(len(requests)),
                input_tokens=2,
                output_tokens=1,
            )

    gateway.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )
    gateway.call_count = 0
    gateway.telemetry_cursor()

    def run(scope_id):
        with gateway.request_scope(scope_id):
            cursor = gateway.telemetry_cursor()
            barrier.wait()
            gateway.complete("private-system", "private-user")
            barrier.wait()
            results[scope_id] = gateway.telemetry_since(cursor)

    threads = [
        threading.Thread(target=run, args=(scope_id,))
        for scope_id in ("match-scope-a", "match-scope-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert set(results) == {"match-scope-a", "match-scope-b"}
    for scope_id, delta in results.items():
        assert delta["scope_id"] == scope_id
        assert delta["successful_calls"] == 1
        assert delta["aggregate_successful_calls"] == 1
        assert len(delta["logical_request_ids"]) == 1
        assert delta["logical_request_ids"][0].startswith(scope_id + ".gfs-")
        assert all(event["scope_id"] == scope_id for event in delta["events"])
    assert len(requests) == 2
