from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.contracts.society import MemoryEvent
from gfs_core.kernel.workspace import WorkspaceRepository

from .appraisal import player_appraisals


class SocietyMemory:
    def __init__(self, project_root: str | Path, *, decay: float = 0.12, capacity: int = 1000) -> None:
        if decay < 0 or capacity < 1:
            raise ValueError("invalid memory retention configuration")
        self.workspace = WorkspaceRepository(project_root)
        self.decay = float(decay)
        self.capacity = int(capacity)

    def record(self, event: MemoryEvent) -> dict:
        _validate_event(event)
        record = asdict(event)
        record["tags"] = list(event.tags)
        with self.workspace.lease("society-memory.lock"):
            records = self._read()
            existing = next((item for item in records if item.get("event_id") == event.event_id), None)
            if existing is not None:
                comparable = {key: value for key, value in existing.items() if key != "index"}
                if comparable != record:
                    raise ValueError("memory event identity conflicts with its content")
                return copy.deepcopy(existing)
            record["index"] = max((int(item.get("index", -1)) for item in records), default=-1) + 1
            records.append(record)
            self._write(records)
            return copy.deepcopy(record)

    def record_match_result(self, *, team: str, opponent: str, match_id: str, match_index: int, won: bool, lost: bool, score, xg, engine: str, emotion=None, coach_stress: float | None = None) -> dict:
        outcome = "win" if won else "loss" if lost else "draw"
        event_id = hashlib.sha256(f"{match_id}\x1f{team}\x1fmatch_result".encode()).hexdigest()
        emotion = dict(emotion or {})
        summary = f"{team} {score[0]}-{score[1]} {opponent}: {outcome}; engine={engine}"
        return self.record(MemoryEvent(
            event_id=event_id,
            match_id=match_id,
            team=team,
            subject=team,
            kind="match_result",
            summary=summary,
            match_index=match_index,
            importance=0.7 if won or lost else 0.45,
            valence=1.0 if won else -1.0 if lost else 0.05,
            arousal=0.8 if won or lost else 0.35,
            emotion=emotion,
            source=f"{engine}:{match_id}",
            tags=(outcome,),
            context={
                "score": list(score),
                "xg": list(xg),
                "coach_stress": coach_stress,
            },
        ))

    def record_match_summary(self, summary, match_id: str, match_index: int) -> list[dict]:
        records = []
        for team in (summary.home_team, summary.away_team):
            for event in player_appraisals(summary, team, match_id, match_index):
                records.append(self.record(event))
        return records

    def append(self, team: str, content: str, *, importance: float = 0.5) -> dict:
        if not isinstance(content, str) or not content.strip() or len(content) > 512:
            raise ValueError("memory content must be non-empty and at most 512 characters")
        with self.workspace.lease("society-memory.lock"):
            records = self._read()
            index = max((int(item.get("index", -1)) for item in records), default=-1) + 1
            match_index = max((int(item.get("match_index", 0)) for item in records), default=0) + 1
            event = MemoryEvent(
                event_id=f"note-{index}",
                match_id=f"note-{index}",
                team=team,
                subject=team,
                kind="note",
                summary=content.strip(),
                match_index=match_index,
                importance=importance,
                source="manual",
            )
            _validate_event(event)
            record = asdict(event)
            record["index"] = index
            records.append(record)
            self._write(records)
            return copy.deepcopy(record)

    def recall(self, team: str, *, limit: int = 20, kind: str | None = None, subject: str | None = None, as_of: int | None = None) -> list[dict]:
        if limit < 1 or limit > 100:
            raise ValueError("memory recall limit must be between 1 and 100")
        records = [
            item for item in self._read()
            if item["team"] == team
            and (kind is None or item.get("kind", "note") == kind)
            and (subject is None or item.get("subject", item["team"]) == subject)
            and (as_of is None or int(item.get("match_index", 0)) <= as_of)
        ]
        current = _current_index(records, as_of)
        ranked = []
        for record in records:
            item = copy.deepcopy(record)
            item["strength"] = self._strength(record, current)
            ranked.append(item)
        ranked.sort(key=lambda item: (item["strength"], item.get("match_index", 0), item["event_id"]), reverse=True)
        return ranked[:limit]

    def effects(self, team: str, *, as_of: int | None = None, limit: int = 8, subject: str | None = None, kind: str | None = None) -> dict:
        records = self.recall(
            team, limit=limit, as_of=as_of, subject=subject, kind=kind,
        )
        if not records:
            return {"valence": 0.0, "arousal": 0.0, "pressure": 0.0, "emotion": {}, "evidence_ids": []}
        total = sum(record["strength"] for record in records)
        if total <= 0:
            return {"valence": 0.0, "arousal": 0.0, "pressure": 0.0, "emotion": {}, "evidence_ids": []}
        emotions = sorted({key for record in records for key in record.get("emotion", {})})
        return {
            "valence": sum(record["valence"] * record["strength"] for record in records) / total,
            "arousal": sum(record["arousal"] * record["strength"] for record in records) / total,
            "pressure": sum(max(0.0, -record["valence"]) * record["strength"] for record in records) / total,
            "emotion": {
                key: sum(record.get("emotion", {}).get(key, 0.0) * record["strength"] for record in records) / total
                for key in emotions
            },
            "evidence_ids": [record["event_id"] for record in records],
        }

    def apply_to_state(self, state, *, as_of: int | None = None) -> dict:
        affected = []
        evidence = []
        for team in (state.home, state.away):
            for player in team.players:
                effect = self.effects(
                    team.team_id, subject=f"player:{player.player_id}", as_of=as_of,
                )
                if not effect["evidence_ids"]:
                    continue
                player.morale_logit = min(6.0, max(-6.0, player.morale_logit + 0.15 * effect["valence"]))
                player.cognitive_load = min(1.0, max(0.0, player.cognitive_load + 0.1 * effect["pressure"]))
                affected.append(player.player_id)
                evidence.extend(effect["evidence_ids"])
            relation = self.effects(
                team.team_id, kind="coach_player_relation", as_of=as_of,
            )
            if relation["evidence_ids"]:
                team.coach.z_trust = min(6.0, max(-6.0, team.coach.z_trust + 0.15 * relation["valence"]))
                team.coach.z_stress = min(6.0, max(-6.0, team.coach.z_stress + 0.1 * relation["pressure"]))
                evidence.extend(relation["evidence_ids"])
        return {
            "players_affected": sorted(set(affected)),
            "evidence_ids": sorted(set(evidence)),
            "scope": "deterministic_memory_appraisal_only",
        }

    def has_match(self, match_id: str, team: str) -> bool:
        return any(record.get("match_id") == match_id and record["team"] == team for record in self._read())

    def status(self) -> dict:
        stored = len(self._read())
        return {
            "stored_events": stored,
            "capacity": self.capacity,
            "at_capacity": stored >= self.capacity,
            "retention_policy": "most_recent_match_index",
        }

    def snapshot(self) -> dict:
        return {
            "schema_version": 2,
            "records": copy.deepcopy(self._read()),
            "retention": self.status(),
        }

    def restore(self, snapshot: dict) -> None:
        if snapshot.get("schema_version") != 2 or not isinstance(snapshot.get("records"), list):
            raise ValueError("invalid society memory snapshot")
        with self.workspace.lease("society-memory.lock"):
            self._write(copy.deepcopy(snapshot["records"]))

    def _read(self) -> list[dict]:
        try:
            payload = self.workspace.read("society_memory.json")
        except FileNotFoundError:
            return []
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError("society memory registry must contain an array")
        return records

    def _write(self, records: list[dict]) -> None:
        encoded = json.dumps(records[-self.capacity:], allow_nan=False)
        self.workspace.write("society_memory.json", {
            "schema_version": 2,
            "records": json.loads(encoded),
        })

    def _strength(self, record: dict, as_of: int) -> float:
        age = max(0, as_of - int(record.get("match_index", 0)))
        importance = float(record.get("importance", 0.5))
        arousal = float(record.get("arousal", 0.0))
        return importance * (0.75 + 0.25 * arousal) * math.exp(-self.decay * age)


class MemoryPlugin:
    metadata = PluginMetadata(
        id="gfs.core.society.memory",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.society",
        provides=("memory",),
        requires=(),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._project_root = project_root

    def activate(self, context: PluginContext) -> None:
        context.provide("memory", SocietyMemory(self._project_root))

    def deactivate(self) -> None:
        pass


def _current_index(records: list[dict], as_of: int | None) -> int:
    if as_of is not None:
        if as_of < 0:
            raise ValueError("memory match index cannot be negative")
        return as_of
    return max((int(record.get("match_index", 0)) for record in records), default=0)


def _validate_event(event: MemoryEvent) -> None:
    for value in (event.event_id, event.match_id, event.team, event.subject, event.kind, event.summary):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("memory event identities and summary are required")
    if len(event.summary) > 512:
        raise ValueError("memory content must be at most 512 characters")
    if event.match_index < 0:
        raise ValueError("memory match index cannot be negative")
    if not 0 <= event.importance <= 1 or not -1 <= event.valence <= 1 or not 0 <= event.arousal <= 1:
        raise ValueError("memory appraisal values are outside their bounds")
    values = (event.importance, event.valence, event.arousal, *event.emotion.values())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("memory appraisal values must be finite")
