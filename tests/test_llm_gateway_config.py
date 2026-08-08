from src.simulation import llm_gateway
import pytest
from types import SimpleNamespace


def test_llm_gateway_config_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "first")
    monkeypatch.setenv("MODEL_NAME", "model-a")
    first = llm_gateway.LLMGatewayConfig.from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "second")
    monkeypatch.setenv("MODEL_NAME", "model-b")
    second = llm_gateway.LLMGatewayConfig.from_env()
    assert first.api_key == "first" and first.model == "model-a"
    assert second.api_key == "second" and second.model == "model-b"


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
