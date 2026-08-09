import json

from src.cli import main


def test_provider_cli_is_zero_call_and_redacts_secret(monkeypatch, capsys):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "cli-test-secret")
    monkeypatch.setenv("BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("MODEL_NAME", "deepseek-v4-flash")
    assert main(["studio", "provider"]) == 0
    output = capsys.readouterr().out
    payload, _ = json.JSONDecoder().raw_decode(output)
    assert payload["ready"]
    assert not payload["external_calls_made"]
    assert "cli-test-secret" not in output


def test_provider_cli_fails_when_no_credential(monkeypatch, capsys):
    for name in ("DEEPSEEK_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert main(["studio", "provider"]) == 2
    output = capsys.readouterr().out
    assert '"ready": false' in output
