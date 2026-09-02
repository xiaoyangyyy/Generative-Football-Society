import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_engine.coach_loader import CoachProfile
from src.data_engine.roster_loader import _abilities_from_dict, load_roster_json
from src.simulation.agent import SocietyAgent
from src.simulation.engine import SocialMediaFeed
from src.simulation.narrative_events import NarrativeEvent
from src.simulation.psychological_state import PsychologicalState
from src.simulation.random_control import derive_seed
from src.simulation.tournament_2026 import TournamentManager
from src.simulation.tournament_checkpoint import (
    load_checkpoint, validate_reflection_journal,
    validate_reflection_world_consistency,
)
from src.simulation.world_state import restore_world_state, snapshot_world_state
from src.simulation.world_cup_runner import _attach_team_dynamics_from_rosters


RUN_IDENTITY = "c" * 64


def _agent(name: str, seed: int = 1) -> SocietyAgent:
    return SocietyAgent(
        name,
        {
            "tier": "Semi-Core", "final_status_score": 55.0,
            "c1_win_rate": 52.0, "c3_major_exp": 45.0,
            "c5_pressure": 40.0,
        },
        initialization_rng=random.Random(seed),
        random_root_seed=derive_seed(91, "agent", name),
    )


def _manager(tmp_path, names=("Alpha",)) -> TournamentManager:
    agents = {name: _agent(name, index + 1) for index, name in enumerate(names)}
    world = SimpleNamespace(
        root_seed=91,
        agents=agents,
        feed=SocialMediaFeed(),
        current_date=pd.Timestamp("2026-01-07"),
    )
    return TournamentManager(
        world, base_dir=tmp_path, run_identity_sha256=RUN_IDENTITY,
    )


class _ReflectionLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def perform_agent_reflection(self, *args, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _valid_reflection(label="settled"):
    return json.dumps({
        "reflection": label,
        "confidence": 0.8,
        "suggested_adjustments": {"risk_budget": -0.04},
        "evidence_memory_ids": [],
        "scope": "tournament",
    })


def test_world_snapshot_round_trips_agent_social_and_narrative_state(tmp_path):
    manager = _manager(tmp_path)
    agent = manager.world.agents["Alpha"]
    agent.momentum = 1.25
    agent.roles["Manager"]["pressure"] = 0.44
    agent.psychological_state = PsychologicalState(morale=0.3, pressure=0.2)
    agent.tactical_vector = np.array([0.1, 0.2, 0.3])
    agent.coach_profile = CoachProfile(
        team_id="Alpha", name="Coach A", preferred_preset="high_press",
        preset_affinities={"high_press": 0.8, "balanced": 0.2},
    )
    manager.world.feed.publish("Alpha", "A durable post", tags=["test"])
    manager.dialogue_engine.market_step = 7
    manager.dialogue_engine.topic_market = {
        "pressure": {"heat": 0.8, "last_step": 7},
    }
    manager.narrative_event_bus.history = [{
        "event": NarrativeEvent(
            text="Pressure rises", signals={"coach_pressure": 0.4},
            targets=("Alpha",), persistence=0.5, stage="Final",
        ),
        "applied": {"Alpha": {"psychology_after": agent.psychological_state}},
    }]

    snapshot = json.loads(json.dumps(snapshot_world_state(manager)))
    agent.momentum = -9.0
    agent.roles["Manager"]["pressure"] = 0.0
    agent.coach_profile.preferred_preset = "balanced"
    agent.coach_profile.preset_affinities = {"balanced": 1.0}
    del agent.psychological_state
    manager.world.feed.posts.clear()
    manager.dialogue_engine.market_step = 0
    manager.dialogue_engine.topic_market.clear()
    manager.narrative_event_bus.history.clear()

    restore_world_state(manager, snapshot)

    assert agent.momentum == 1.25
    assert agent.roles["Manager"]["pressure"] == 0.44
    assert agent.psychological_state == PsychologicalState(
        morale=0.3, pressure=0.2,
    )
    assert np.array_equal(agent.tactical_vector, np.array([0.1, 0.2, 0.3]))
    assert agent.coach_profile.preferred_preset == "high_press"
    assert agent.coach_profile.preset_affinities["high_press"] == 0.8
    assert manager.world.feed.posts[0]["content"] == "A durable post"
    assert manager.dialogue_engine.market_step == 7
    assert manager.dialogue_engine.topic_market["pressure"]["heat"] == 0.8
    event = manager.narrative_event_bus.history[0]["event"]
    assert isinstance(event, NarrativeEvent)
    assert event.targets == ("Alpha",)


def test_world_snapshot_fails_closed_on_agent_or_numeric_corruption(tmp_path):
    manager = _manager(tmp_path)
    original_date = manager.world.current_date
    snapshot = snapshot_world_state(manager)
    snapshot["agents"]["Ghost"] = snapshot["agents"].pop("Alpha")
    snapshot["agents"]["Ghost"]["team_name"] = "Ghost"
    with pytest.raises(ValueError, match="agent identity mismatch"):
        restore_world_state(manager, snapshot)
    assert manager.world.current_date == original_date
    assert manager.world.feed.posts == []

    manager.world.agents["Alpha"].momentum = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        snapshot_world_state(manager)


def test_world_snapshot_rejects_stored_nonfinite_and_preserves_reserved_key(
    tmp_path,
):
    manager = _manager(tmp_path)
    manager.world.agents["Alpha"].beliefs = {
        "__gfs_snapshot_type__": "ordinary football belief",
    }
    snapshot = json.loads(json.dumps(snapshot_world_state(manager)))
    restore_world_state(manager, snapshot)
    assert manager.world.agents["Alpha"].beliefs == {
        "__gfs_snapshot_type__": "ordinary football belief",
    }

    snapshot["agents"]["Alpha"]["state"]["momentum"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        restore_world_state(manager, snapshot)

    snapshot = snapshot_world_state(manager)
    snapshot["current_date"] = "NaT"
    with pytest.raises(ValueError, match="world-state date"):
        restore_world_state(manager, snapshot)


def test_world_snapshot_rejects_unclassified_future_agent_state(tmp_path):
    manager = _manager(tmp_path)
    manager.world.agents["Alpha"].new_mutable_subsystem = {"value": 1}
    with pytest.raises(ValueError, match="Unclassified mutable Agent fields"):
        snapshot_world_state(manager)


def test_manager_restore_preflight_leaves_existing_state_untouched(tmp_path):
    first = _manager(tmp_path)
    first._save_checkpoint()
    checkpoint = load_checkpoint(str(tmp_path))
    second = _manager(tmp_path)
    second.standings = {"sentinel": {}}
    second.world.agents["Alpha"].random_root_seed += 1

    with pytest.raises(ValueError, match="random-world mismatch"):
        second._restore_from_checkpoint(checkpoint)

    assert second.standings == {"sentinel": {}}


def test_roster_loader_treats_nonfinite_observations_as_missing(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(
        '{"team_id":"A","note":null,"players":[{"abilities":'
        '{"tech":NaN,"vision":0.7}}],"team_dynamics":'
        '{"attack":NaN,"institutional_pressure":-0.2}}',
        encoding="utf-8",
    )

    roster = load_roster_json(str(path))
    assert roster["note"] is None
    assert "tech" not in roster["players"][0]["abilities"]
    assert "attack" not in roster["team_dynamics"]
    abilities = _abilities_from_dict(roster["players"][0]["abilities"])
    assert abilities.tech == 0.5
    assert abilities.vision == 0.7
    assert all(np.isfinite(value) for value in vars(abilities).values())


def test_incomplete_team_dynamics_uses_status_fallback(monkeypatch):
    complete = {
        "attack": 0.1, "defense": 0.2, "press": 0.3,
        "morale_field": 0.4, "institutional_pressure": 0.5,
    }
    rosters = {
        "Incomplete": {"team_dynamics": {"attack": 0.1}},
        "Complete": {"team_dynamics": complete},
    }
    monkeypatch.setattr(
        "src.simulation.squad_registry.load_effective_roster",
        lambda _root, team: rosters[team],
    )
    agents = {
        "Incomplete": SimpleNamespace(), "Complete": SimpleNamespace(),
    }

    _attach_team_dynamics_from_rosters(".", agents)

    assert not hasattr(agents["Incomplete"], "team_dynamics")
    assert agents["Complete"].team_dynamics == complete


def test_reflection_parsing_retries_then_commits_one_durable_operation(
    tmp_path, monkeypatch,
):
    manager = _manager(tmp_path)
    agent = manager.world.agents["Alpha"]
    llm = _ReflectionLLM(["not-json", "[]", _valid_reflection()])
    monkeypatch.setattr("src.simulation.tournament_2026.time.sleep", lambda _: None)

    manager._reflection_with_retry(
        agent, llm, "post_group:Alpha", attempts=3,
    )
    manager._reflection_with_retry(
        agent, llm, "post_group:Alpha", attempts=3,
    )

    assert llm.calls == 3
    assert manager.reflection_journal["applied"] == ["post_group:Alpha"]
    assert len(agent.llm_reflection_audit) == 1
    assert agent.llm_reflection_audit[0]["operation_id"] == "post_group:Alpha"
    assert load_checkpoint(str(tmp_path))["reflection_journal"] == (
        manager.reflection_journal
    )


def test_reflection_identity_and_receipt_size_fail_before_side_effects(tmp_path):
    manager = _manager(tmp_path)
    llm = _ReflectionLLM([_valid_reflection()])
    with pytest.raises(ValueError, match="operation ID"):
        manager._reflection_with_retry(
            manager.world.agents["Alpha"], llm, "", attempts=1,
        )
    assert llm.calls == 0

    oversized = {
        "schema_version": 1,
        "receipts": {
            "post_group:Alpha": {
                "agent": "Alpha", "payload": {"reflection": "x" * (257 * 1024)},
            },
        },
        "applied": [],
    }
    with pytest.raises(ValueError, match="exceeds 256 KiB"):
        validate_reflection_journal(oversized)


def test_reflection_journal_and_world_snapshot_must_commit_together(tmp_path):
    manager = _manager(tmp_path)
    snapshot = snapshot_world_state(manager)
    receipt = {
        "schema_version": 1,
        "receipts": {
            "post_group:Alpha": {
                "agent": "Alpha",
                "payload": json.loads(_valid_reflection()),
            },
        },
        "applied": ["post_group:Alpha"],
    }
    with pytest.raises(ValueError, match="missing from Agent state"):
        validate_reflection_world_consistency(receipt, snapshot)

    manager.world.agents["Alpha"].apply_reflection_payload(
        json.loads(_valid_reflection()), operation_id="post_group:Alpha",
    )
    mutated = snapshot_world_state(manager)
    validate_reflection_world_consistency(receipt, mutated)

    receipt["applied"] = []
    with pytest.raises(ValueError, match="already mutated Agent state"):
        validate_reflection_world_consistency(receipt, mutated)

    receipt["receipts"] = {}
    with pytest.raises(ValueError, match="lacks a durable receipt"):
        validate_reflection_world_consistency(receipt, mutated)


def test_reflection_receipt_rejects_unknown_agent(tmp_path):
    snapshot = snapshot_world_state(_manager(tmp_path))
    journal = {
        "schema_version": 1,
        "receipts": {
            "post_group:Ghost": {
                "agent": "Ghost", "payload": json.loads(_valid_reflection()),
            },
        },
        "applied": [],
    }
    with pytest.raises(ValueError, match="unknown Agent"):
        validate_reflection_world_consistency(journal, snapshot)


def test_receipted_reflection_resumes_without_second_provider_call(
    tmp_path, monkeypatch,
):
    first = _manager(tmp_path)
    first_agent = first.world.agents["Alpha"]
    llm = _ReflectionLLM([_valid_reflection("durable")])
    def crash_before_apply(*args, **kwargs):
        raise RuntimeError("injected crash")

    with monkeypatch.context() as crash_patch:
        crash_patch.setattr(
            SocietyAgent, "apply_reflection_payload", crash_before_apply,
        )
        with pytest.raises(RuntimeError, match="injected crash"):
            first._reflection_with_retry(
                first_agent, llm, "post_group:Alpha", attempts=1,
            )
    assert llm.calls == 1
    checkpoint = load_checkpoint(str(tmp_path))
    assert "post_group:Alpha" in checkpoint["reflection_journal"]["receipts"]
    assert checkpoint["reflection_journal"]["applied"] == []
    assert first_agent.llm_reflection_audit == []

    second = _manager(tmp_path)
    second._restore_from_checkpoint(checkpoint)
    forbidden = _ReflectionLLM([AssertionError("provider must not be called")])
    second._reflection_with_retry(
        second.world.agents["Alpha"], forbidden,
        "post_group:Alpha", attempts=1,
    )

    restored = second.world.agents["Alpha"]
    assert forbidden.calls == 0
    assert len(restored.llm_reflection_audit) == 1
    assert restored.reflection_diary == "durable"
    # The operation remains idempotent even when invoked again in memory.
    before_clock = restored.memory_clock
    restored.apply_reflection_payload(
        json.loads(_valid_reflection("ignored")),
        operation_id="post_group:Alpha",
    )
    assert restored.memory_clock == before_clock


def test_checkpoint_binds_dynamic_evidence_and_cognitive_cache(
    tmp_path, monkeypatch,
):
    cache = tmp_path / "cache/cognitive"
    cache.mkdir(parents=True)
    cache_file = cache / "decision.json"
    cache_file.write_text('{"plan":1}', encoding="utf-8")
    fusion = tmp_path / "data/persistence/world_model_fusion.jsonl"
    fusion.parent.mkdir(parents=True)
    fusion.write_text('{"decision_id":"one"}\n', encoding="utf-8")
    monkeypatch.setenv("MATCH_COGNITIVE", "1")
    monkeypatch.setenv("MATCH_COGNITIVE_CACHE", str(cache))
    manager = _manager(tmp_path)
    manager._save_checkpoint()

    checkpoint = load_checkpoint(str(tmp_path))
    assert "cache/cognitive" in checkpoint["state_artifacts"]
    assert "data/persistence/world_model_fusion.jsonl" in (
        checkpoint["state_artifacts"]
    )

    cache_file.write_text('{"plan":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="external state integrity mismatch"):
        load_checkpoint(str(tmp_path))


def test_skipped_completed_final_rebuilds_result_and_reflection(tmp_path):
    manager = _manager(tmp_path, names=("Alpha", "Beta"))
    manager.phase = "knockout"
    manager.qualified_teams = ["Alpha", "Beta"]
    match_key = manager._match_key("Final", "Alpha", "Beta")
    manager.completed_matches = [match_key]
    manager.match_results = {match_key: "Alpha"}
    llm = _ReflectionLLM([_valid_reflection("champion review")])

    manager.simulate_knockout_round("Final", 2, llm)

    assert manager.final_result == {"champion": "Alpha", "runner_up": "Beta"}
    assert manager.qualified_teams == ["Alpha"]
    assert manager.reflection_journal["applied"] == [
        f"knockout:{match_key}:Alpha",
    ]
    assert llm.calls == 1
