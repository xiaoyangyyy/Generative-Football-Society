import pytest

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import CognitiveTriggerEvent
from src.match_engine.cognitive.executor import CognitiveExecutor


class _FailingLLM:
    model = "failing-provider"

    def coach_in_match_plan(self, *args, **kwargs):
        raise RuntimeError("provider unavailable")


def _trigger():
    return CognitiveTriggerEvent(
        t_sec=10.0,
        kind="momentum_shift",
        entity_tier="coach",
        entity_id="coach:home",
        team_id="home",
        salience=1.0,
        facts={},
    )


def test_strict_cognitive_mode_propagates_provider_failure(monkeypatch):
    monkeypatch.setenv("COGNITIVE_PLAN_RETRIES", "1")
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True, require_llm=True), _FailingLLM(),
    )
    with pytest.raises(RuntimeError, match="strict cognitive provider failed"):
        executor._call_llm_for_trigger(_trigger())


def test_research_fallback_remains_explicitly_available(monkeypatch):
    monkeypatch.setenv("COGNITIVE_PLAN_RETRIES", "1")
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True, require_llm=False), _FailingLLM(),
    )
    plan = executor._call_llm_for_trigger(_trigger())
    assert plan["cognitive_source"] == "rule_fallback"
