from dataclasses import replace

import pytest

from gfs_core.contracts.society import MemoryEvent, TeamProfile
from gfs_core.plugins.society.memory.service import SocietyMemory
from gfs_core.plugins.tournament.checkpointing.carryover import CrossMatchService, TeamCarryover


def event(event_id, match_index, valence):
    return MemoryEvent(
        event_id=event_id,
        match_id=f"match-{match_index}",
        team="Brazil",
        subject="Brazil",
        kind="match_result",
        summary=f"result {match_index}",
        match_index=match_index,
        importance=0.8,
        valence=valence,
        arousal=0.5,
        emotion={"pride": 0.8 if valence > 0 else 0.1},
        source="test",
    )


def test_event_identity_decay_and_structured_recall(tmp_path):
    memory = SocietyMemory(tmp_path, decay=0.5)
    first = memory.record(event("event-1", 1, 1.0))
    assert memory.record(event("event-1", 1, 1.0)) == first
    memory.record(event("event-2", 2, -1.0))
    recalled = memory.recall("Brazil", kind="match_result", as_of=2)
    assert len(recalled) == 2
    assert recalled[0]["event_id"] == "event-2"
    assert recalled[0]["strength"] > recalled[1]["strength"]
    assert [item["event_id"] for item in memory.recall("Brazil", as_of=1)] == ["event-1"]
    assert memory.status()["retention_policy"] == "most_recent_match_index"
    effects = memory.effects("Brazil", as_of=2)
    assert effects["pressure"] > 0
    assert set(effects["evidence_ids"]) == {"event-1", "event-2"}
    with pytest.raises(ValueError, match="conflicts"):
        memory.record(replace(event("event-1", 1, 1.0), summary="tampered"))
    snapshot = memory.snapshot()
    memory.restore(snapshot)
    assert memory.snapshot() == snapshot


def test_memory_affects_next_profile_without_duplicate_settlement(tmp_path):
    memory = SocietyMemory(tmp_path)
    service = CrossMatchService(memory=memory)
    carryover = TeamCarryover("Brazil")
    service.settle_outcome(
        carryover,
        won=True,
        lost=False,
        match_id="match-1",
        opponent="Argentina",
        score=(2, 1),
        xg=(1.4, 0.8),
        engine="test",
        emotion={"pride": 0.8},
    )
    service.settle_outcome(carryover, won=True, lost=False, match_id="match-1")
    profile = service.apply_to_profile(
        TeamProfile("Brazil", status_score=60.0, emotion_profile={"pride": 0.2}),
        carryover,
    )
    assert profile.status_score > 60.0
    assert profile.psychology_profile["memory_confidence"] == 1.0
    assert profile.emotion_profile["pride"] == pytest.approx(0.38)
    assert len(memory.recall("Brazil", kind="match_result")) == 1
    record = memory.recall("Brazil")[0]
    assert record["context"]["score"] == [2, 1]
    assert record["source"] == "test:match-1"
