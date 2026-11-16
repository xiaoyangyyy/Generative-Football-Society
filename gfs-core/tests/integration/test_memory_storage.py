from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from gfs_core.kernel.workspace import WorkspaceRepository
from gfs_core.plugins.society.memory.service import SocietyMemory
from tests.integration.test_society_memory import event


def test_capacity_does_not_change_historical_replay(tmp_path):
    memory = SocietyMemory(tmp_path, capacity=1)
    memory.record(event("event-1", 1, 1.0))
    memory.record(event("event-2", 2, -1.0))
    assert memory.status()["at_capacity"] is True
    assert memory.status()["capacity_enforced"] is False
    assert [
        item["event_id"] for item in memory.recall("Brazil", as_of=1)
    ] == ["event-1"]


def test_legacy_json_migrates_once_without_resurrecting_restored_state(tmp_path):
    WorkspaceRepository(tmp_path).write("society_memory.json", {
        "schema_version": 2,
        "records": [{**asdict(event("legacy-1", 1, 1.0)), "index": 0}],
    })
    memory = SocietyMemory(tmp_path)
    assert memory.recall("Brazil")[0]["event_id"] == "legacy-1"
    assert memory.status()["legacy_json_migrated"] is True
    memory.restore({"schema_version": 2, "records": []})
    reopened = SocietyMemory(tmp_path)
    assert reopened.snapshot()["records"] == []


def test_match_memories_are_one_atomic_transaction(tmp_path, monkeypatch):
    import gfs_core.plugins.society.memory.service as memory_module

    valid = event("event-1", 1, 1.0)
    invalid = replace(event("event-2", 1, -1.0), match_index=-1)
    monkeypatch.setattr(
        memory_module, "player_appraisals",
        lambda *args: [valid, invalid],
    )
    memory = SocietyMemory(tmp_path)
    with pytest.raises(ValueError, match="cannot be negative"):
        memory.record_match_summary(
            SimpleNamespace(home_team="Brazil", away_team="Argentina"),
            "match-1", 1,
        )
    assert memory.snapshot()["records"] == []
