from __future__ import annotations

import copy
import hashlib
import json
import math
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.contracts.society import MemoryEvent
from gfs_core.kernel.workspace import WorkspaceRepository

from .appraisal import player_appraisals


class SocietyMemory:
    def __init__(
        self, project_root: str | Path, *, decay: float = 0.12,
        capacity: int = 1000,
    ) -> None:
        if decay < 0 or capacity < 1:
            raise ValueError("invalid memory retention configuration")
        self.workspace = WorkspaceRepository(project_root)
        self.path = self.workspace.persistence / "society_memory.sqlite3"
        self.legacy_path = self.workspace.persistence / "society_memory.json"
        self.decay = float(decay)
        self.capacity = int(capacity)
        self._initialization_lock = threading.RLock()
        self._initialized = False

    def record(self, event: MemoryEvent) -> dict:
        return self._record_many((event,))[0]

    def _record_many(self, events) -> list[dict]:
        documents = [_event_document(event) for event in events]
        with self._connect() as connection:
            results = []
            for document in documents:
                existing = self._insert(connection, document)
                results.append(copy.deepcopy(existing))
            return results

    def record_match_result(
        self, *, team: str, opponent: str, match_id: str, match_index: int,
        won: bool, lost: bool, score, xg, engine: str, emotion=None,
        coach_stress: float | None = None,
    ) -> dict:
        outcome = "win" if won else "loss" if lost else "draw"
        event_id = hashlib.sha256(
            f"{match_id}\x1f{team}\x1fmatch_result".encode()
        ).hexdigest()
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
                "score": list(score), "xg": list(xg),
                "coach_stress": coach_stress,
            },
        ))

    def record_match_summary(self, summary, match_id: str, match_index: int) -> list[dict]:
        events = [
            event
            for team in (summary.home_team, summary.away_team)
            for event in player_appraisals(summary, team, match_id, match_index)
        ]
        return self._record_many(events)

    def append(self, team: str, content: str, *, importance: float = 0.5) -> dict:
        if not isinstance(content, str) or not content.strip() or len(content) > 512:
            raise ValueError("memory content must be non-empty and at most 512 characters")
        with self._connect() as connection:
            index = connection.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 FROM events"
            ).fetchone()[0]
            match_index = connection.execute(
                "SELECT COALESCE(MAX(match_index), 0) + 1 FROM events"
            ).fetchone()[0]
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
            return self._insert(connection, _event_document(event))

    def recall(
        self, team: str, *, limit: int = 20, kind: str | None = None,
        subject: str | None = None, as_of: int | None = None,
    ) -> list[dict]:
        if limit < 1 or limit > 100:
            raise ValueError("memory recall limit must be between 1 and 100")
        clauses = ["team = ?"]
        values: list = [team]
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        if subject is not None:
            clauses.append("subject = ?")
            values.append(subject)
        if as_of is not None:
            if as_of < 0:
                raise ValueError("memory match index cannot be negative")
            clauses.append("match_index <= ?")
            values.append(as_of)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT document FROM events WHERE " + " AND ".join(clauses),
                values,
            ).fetchall()
        records = [json.loads(row[0]) for row in rows]
        current = _current_index(records, as_of)
        ranked = []
        for record in records:
            item = copy.deepcopy(record)
            item["strength"] = self._strength(record, current)
            ranked.append(item)
        ranked.sort(
            key=lambda item: (
                item["strength"], item.get("match_index", 0), item["event_id"],
            ),
            reverse=True,
        )
        return ranked[:limit]

    def effects(
        self, team: str, *, as_of: int | None = None, limit: int = 8,
        subject: str | None = None, kind: str | None = None,
    ) -> dict:
        records = self.recall(
            team, limit=limit, as_of=as_of, subject=subject, kind=kind,
        )
        if not records:
            return {
                "valence": 0.0, "arousal": 0.0, "pressure": 0.0,
                "emotion": {}, "evidence_ids": [],
            }
        total = sum(record["strength"] for record in records)
        if total <= 0:
            return {
                "valence": 0.0, "arousal": 0.0, "pressure": 0.0,
                "emotion": {}, "evidence_ids": [],
            }
        emotions = sorted({
            key for record in records for key in record.get("emotion", {})
        })
        return {
            "valence": sum(
                record["valence"] * record["strength"] for record in records
            ) / total,
            "arousal": sum(
                record["arousal"] * record["strength"] for record in records
            ) / total,
            "pressure": sum(
                max(0.0, -record["valence"]) * record["strength"]
                for record in records
            ) / total,
            "emotion": {
                key: sum(
                    record.get("emotion", {}).get(key, 0.0) * record["strength"]
                    for record in records
                ) / total
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
                player.morale_logit = min(
                    6.0, max(-6.0, player.morale_logit + 0.15 * effect["valence"])
                )
                player.cognitive_load = min(
                    1.0,
                    max(0.0, player.cognitive_load + 0.1 * effect["pressure"]),
                )
                affected.append(player.player_id)
                evidence.extend(effect["evidence_ids"])
            relation = self.effects(
                team.team_id, kind="coach_player_relation", as_of=as_of,
            )
            if relation["evidence_ids"]:
                team.coach.z_trust = min(
                    6.0,
                    max(-6.0, team.coach.z_trust + 0.15 * relation["valence"]),
                )
                team.coach.z_stress = min(
                    6.0,
                    max(-6.0, team.coach.z_stress + 0.1 * relation["pressure"]),
                )
                evidence.extend(relation["evidence_ids"])
        return {
            "players_affected": sorted(set(affected)),
            "evidence_ids": sorted(set(evidence)),
            "scope": "deterministic_memory_appraisal_only",
        }

    def has_match(self, match_id: str, team: str) -> bool:
        with self._connect() as connection:
            return connection.execute(
                "SELECT EXISTS(SELECT 1 FROM events WHERE match_id = ? AND team = ?)",
                (match_id, team),
            ).fetchone()[0] == 1

    def status(self) -> dict:
        with self._connect() as connection:
            stored = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            migrated = connection.execute(
                "SELECT value FROM metadata WHERE key = 'legacy_json_sha256'"
            ).fetchone()
        return {
            "stored_events": stored,
            "capacity": self.capacity,
            "at_capacity": stored >= self.capacity,
            "capacity_enforced": False,
            "storage": "sqlite_immutable_events",
            "legacy_json_migrated": bool(migrated and migrated[0] != "not_present"),
            "retention_policy": "immutable_history_with_indexed_recall",
        }

    def snapshot(self) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT document FROM events ORDER BY sequence"
            ).fetchall()
        return {
            "schema_version": 2,
            "records": [json.loads(row[0]) for row in rows],
            "retention": self.status(),
        }

    def restore(self, snapshot: dict) -> None:
        if snapshot.get("schema_version") != 2 or not isinstance(snapshot.get("records"), list):
            raise ValueError("invalid society memory snapshot")
        documents = [_snapshot_document(record) for record in snapshot["records"]]
        indices = [document["index"] for document in documents]
        if len(indices) != len(set(indices)):
            raise ValueError("memory snapshot contains duplicate sequence indices")
        with self._connect() as connection:
            connection.execute("DELETE FROM events")
            for document in documents:
                connection.execute(
                    "INSERT INTO events(sequence, event_id, match_id, match_index, team, subject, kind, document) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    _row(document),
                )

    def _insert(self, connection, document: dict) -> dict:
        index = connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM events"
        ).fetchone()[0]
        candidate = {**document, "index": index}
        connection.execute(
            "INSERT INTO events(sequence, event_id, match_id, match_index, team, subject, kind, document) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(event_id) DO NOTHING",
            _row(candidate),
        )
        stored = connection.execute(
            "SELECT document FROM events WHERE event_id = ?", (document["event_id"],)
        ).fetchone()[0]
        existing = json.loads(stored)
        comparable = {key: value for key, value in existing.items() if key != "index"}
        if comparable != document:
            raise ValueError("memory event identity conflicts with its content")
        return existing

    def _connect(self) -> sqlite3.Connection:
        with self._initialization_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30.0)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA busy_timeout = 30000")
            with connection:
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS events(
                        sequence INTEGER PRIMARY KEY,
                        event_id TEXT NOT NULL UNIQUE,
                        match_id TEXT NOT NULL,
                        match_index INTEGER NOT NULL,
                        team TEXT NOT NULL,
                        subject TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        document TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS memory_recall
                        ON events(team, subject, kind, match_index);
                    CREATE TABLE IF NOT EXISTS metadata(
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                """)
                if not self._initialized:
                    self._migrate_legacy(connection)
                    self._initialized = True
            return connection

    def _migrate_legacy(self, connection) -> None:
        migrated = connection.execute(
            "SELECT value FROM metadata WHERE key = 'legacy_migration'"
        ).fetchone()
        if migrated is not None:
            return
        if self.legacy_path.is_file():
            payload = self.workspace.read("society_memory.json")
            records = payload.get("records", [])
            if not isinstance(records, list):
                raise ValueError("society memory registry must contain an array")
            for record in records:
                document = _snapshot_document(record)
                connection.execute(
                    "INSERT INTO events(sequence, event_id, match_id, match_index, team, subject, kind, document) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    _row(document),
                )
            digest = hashlib.sha256(self.legacy_path.read_bytes()).hexdigest()
        else:
            digest = "not_present"
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('legacy_migration', 'completed')"
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('legacy_json_sha256', ?)",
            (digest,),
        )

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


def _event_document(event: MemoryEvent) -> dict:
    _validate_event(event)
    document = asdict(event)
    document["tags"] = list(event.tags)
    return json.loads(json.dumps(document, allow_nan=False))


def _snapshot_document(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("memory snapshot records must be objects")
    document = _event_document(MemoryEvent(**{
        key: value for key, value in record.items() if key != "index"
    }))
    index = record.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("memory sequence index must be a non-negative integer")
    document["index"] = index
    return document


def _row(document: dict) -> tuple:
    return (
        document["index"], document["event_id"], document["match_id"],
        document["match_index"], document["team"], document["subject"], document["kind"],
        json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False),
    )


def _current_index(records: list[dict], as_of: int | None) -> int:
    if as_of is not None:
        if as_of < 0:
            raise ValueError("memory match index cannot be negative")
        return as_of
    return max((int(record.get("match_index", 0)) for record in records), default=0)


def _validate_event(event: MemoryEvent) -> None:
    for value in (
        event.event_id, event.match_id, event.team, event.subject,
        event.kind, event.summary,
    ):
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
