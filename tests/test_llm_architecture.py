from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.simulation.generation_pipeline import GenerationPipeline
from src.simulation.llm_engine import SimulationLLM
from src.simulation.meta_learning import MetaLearningController
from src.simulation.narrative_events import NarrativeEventBus


class _FakeGateway:
    def __init__(self, response: dict | None = None):
        self.client = object()
        self.config = SimpleNamespace(model="fake", timeout_s=1, max_retries=1)
        self.response = response or {}

    def complete(self, *args, **kwargs):
        return json.dumps(self.response)

    @staticmethod
    def extract_json_object(content):
        return json.loads(content)


def test_role_wrappers_can_share_one_gateway():
    gateway = _FakeGateway()
    first = SimulationLLM(gateway=gateway)
    second = SimulationLLM(gateway=gateway)
    assert first.gateway is second.gateway
    assert first.client is second.client


def test_atmosphere_is_structured_and_bounded():
    gateway = _FakeGateway(
        {
            "key_event": "The simulation turns after a 4-2 glitch.",
            "signals": {"coach_pressure": 8, "crowd_hostility": -4},
            "persistence": 3,
        }
    )
    llm = SimulationLLM(gateway=gateway)
    payload = json.loads(llm.predict_match_result("A", "calm", "B", "tense", "Group A"))
    assert "simulation" not in payload["key_event"].lower()
    assert "4-2" not in payload["key_event"]
    assert payload["signals"]["coach_pressure"] == pytest.approx(1.0)
    assert payload["persistence"] == pytest.approx(1.0)
    assert llm.generation_audits[-1].original["signals"]["coach_pressure"] == 8


class _MetaAgent:
    name = "A"
    W_h = 0.8
    W_x = 0.3
    tactical_controls = {
        "risk_budget": 0.5,
        "pressing_intensity": 0.5,
        "line_height": 0.5,
        "rotation_aggressiveness": 0.5,
    }
    roles = {"Icon": {"patience": 0.8}}
    memory_event_log = [{"id": "A-7"}]
    reflection_diary = ""


def test_meta_learning_uses_slower_step_for_model_weights():
    agent = _MetaAgent()
    audit = MetaLearningController().apply(
        agent,
        {
            "reflection": "Press more, learn slowly.",
            "confidence": 1.0,
            "evidence_memory_ids": ["A-7"],
            "suggested_adjustments": {"pressing_intensity": 1.0, "w_h_delta": 1.0},
        },
    )
    fast = audit["applied_adjustments"]["pressing_intensity"]
    slow = audit["applied_adjustments"]["w_h_delta"]
    assert fast["time_scale"] == "fast"
    assert slow["time_scale"] == "slow"
    assert abs(slow["applied_delta"]) < abs(fast["applied_delta"])
    assert audit["evidence_verified"] == ["A-7"]


class _NarrativeAgent:
    def __init__(self, name):
        self.name = name
        self.team_name = name
        self.roles = {"Manager": {"pressure": 0.0}}
        self.events = []

    def apply_social_signal(self, signal, stage_pressure=0.2):
        self.last_signal = signal
        return {"chaos_delta": signal["provocation_level"]}

    def record_social_dialogue_event(self, *args):
        self.events.append(args)


def test_narrative_event_changes_targeted_agent_state():
    a, b = _NarrativeAgent("A"), _NarrativeAgent("B")
    record = NarrativeEventBus().publish(
        {
            "key_event": "The crowd turns on the coach.",
            "signals": {
                "coach_pressure": 0.8,
                "crowd_hostility": 0.7,
                "media_amplification": 0.9,
            },
            "targets": ["A"],
            "persistence": 0.75,
        },
        [a, b],
        stage="Quarter-Finals",
    )
    assert a.roles["Manager"]["pressure"] > 0.0
    assert a.events
    assert "A" in record["applied"]
    assert "B" not in record["applied"]
    assert not b.events


def test_media_pipeline_preserves_original_and_bounds_metrics():
    audit = GenerationPipeline().validate_media(
        {
            "mainstream": "A clean tactical account.",
            "social_chaos_post": "The simulation engine is broken.",
            "metrics": {"professional_score": 99, "social_chaos": -99},
        },
        {"score": "1-0", "winner": "A"},
    )
    assert not audit.accepted
    assert "simulation" in audit.original["social_chaos_post"].lower()
    assert "simulation" not in audit.published["social_chaos_post"].lower()
    assert audit.published["metrics"] == {"professional_score": 2.0, "social_chaos": -5.0}
