"""Versioned, group-safe dataset manifests for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from src.match_engine.world_model.schema import SHOT_GOAL_INDEX


SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_partition(group: str, seed: int = 42) -> str:
    value = int.from_bytes(
        hashlib.blake2s(f"{seed}:{group}".encode("utf-8"), digest_size=8).digest(), "big"
    ) / float(2**64)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "dev"
    return "sealed_test"


@dataclass(frozen=True)
class DatasetFile:
    path: str
    group: str
    split: str
    sha256: str
    bytes: int
    rows: int
    passes: int
    shots: int
    goals: int


def _inspect_trace(path: Path) -> tuple[int, int, int, int]:
    rows = passes = shots = goals = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            action = row.get("action", [])
            rows += 1
            if len(action) > 0 and action[0] > 0.5:
                passes += 1
            if len(action) > 1 and action[1] > 0.5:
                shots += 1
                goals += int(len(action) > SHOT_GOAL_INDEX and action[SHOT_GOAL_INDEX] > 0.5)
    return rows, passes, shots, goals


def build_trace_manifest(trace_dir: str | Path, *, seed: int = 42) -> dict:
    root = Path(trace_dir).resolve()
    try:
        source_root = root.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        source_root = str(root)
    entries: list[DatasetFile] = []
    seen_hashes: dict[str, str] = {}
    for path in sorted(root.glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        digest = file_sha256(path)
        if digest in seen_hashes:
            raise ValueError(f"Duplicate trace content: {path.name} and {seen_hashes[digest]}")
        seen_hashes[digest] = path.name
        rows, passes, shots, goals = _inspect_trace(path)
        group = path.stem
        entries.append(DatasetFile(
            path=path.name, group=group, split=stable_partition(group, seed),
            sha256=digest, bytes=path.stat().st_size, rows=rows,
            passes=passes, shots=shots, goals=goals,
        ))
    counts = Counter(entry.split for entry in entries)
    if entries and any(counts[name] == 0 for name in ("train", "dev", "sealed_test")):
        raise ValueError(f"Dataset is too small for all splits: {dict(counts)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "gfs-world-model-transitions",
        "seed": seed,
        "partition": "blake2s-group-70-15-15",
        "sealed_test_policy": "evaluation-only; never train or tune",
        "source_root": source_root,
        "files": [asdict(entry) for entry in entries],
        "summary": {
            split: {
                "groups": counts[split],
                "rows": sum(e.rows for e in entries if e.split == split),
                "passes": sum(e.passes for e in entries if e.split == split),
                "shots": sum(e.shots for e in entries if e.split == split),
                "goals": sum(e.goals for e in entries if e.split == split),
            }
            for split in ("train", "dev", "sealed_test")
        },
    }


def write_json_atomic(path: str | Path, payload: Mapping) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def load_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset manifest schema")
    return payload


def verify_trace_manifest(
    manifest: Mapping, *, base_dir: str | Path | None = None,
) -> Path:
    """Verify that frozen trace files still match their manifest contract."""
    root = Path(str(manifest["source_root"]))
    if not root.is_absolute():
        root = Path(base_dir or Path.cwd()) / root
    root = root.resolve()
    seen_groups: set[str] = set()
    seen_hashes: set[str] = set()
    for entry in manifest["files"]:
        path = (root / str(entry["path"])).resolve()
        if path.parent != root:
            raise ValueError(f"Trace path escapes source root: {entry['path']}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing trace file: {path}")
        if entry["group"] in seen_groups:
            raise ValueError(f"Duplicate trace group: {entry['group']}")
        seen_groups.add(str(entry["group"]))
        digest = file_sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"Trace hash mismatch: {entry['path']}")
        if digest in seen_hashes:
            raise ValueError(f"Duplicate trace content: {entry['path']}")
        seen_hashes.add(digest)
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Trace size mismatch: {entry['path']}")
        actual = _inspect_trace(path)
        expected = tuple(
            int(entry[name]) for name in ("rows", "passes", "shots", "goals")
        )
        if actual != expected:
            raise ValueError(f"Trace statistics mismatch: {entry['path']}")
    return root


def files_for_split(manifest: Mapping, splits: Iterable[str]) -> set[str]:
    allowed = set(splits)
    if "sealed_test" in allowed and allowed != {"sealed_test"}:
        raise ValueError("sealed_test cannot be combined with train/dev")
    return {entry["path"] for entry in manifest["files"] if entry["split"] in allowed}
