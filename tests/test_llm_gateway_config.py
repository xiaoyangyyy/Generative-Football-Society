from src.simulation import llm_gateway
import pytest
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
