"""Immutable run configuration and reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from types import MappingProxyType


def environment_snapshot(source: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Capture process configuration once as an immutable boundary object."""
    return MappingProxyType(dict(os.environ if source is None else source))


def env_bool(values: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def env_int(values: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(values.get(key, str(default)))
    except ValueError:
        return default


def env_float(values: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(values.get(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    micro_enabled: bool = True
    cognitive_enabled: bool = True
    world_model_enabled: bool = False
    world_model_planning: bool = False
    score_path: str = "physics_official"
    save_carryover: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score_path not in {"physics_official", "micro_replay", "macro_unified", "poisson_legacy"}:
            raise ValueError(f"Unknown score path: {self.score_path}")
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "micro_enabled": self.micro_enabled,
            "cognitive_enabled": self.cognitive_enabled,
            "world_model_enabled": self.world_model_enabled,
            "world_model_planning": self.world_model_planning,
            "score_path": self.score_path,
            "save_carryover": self.save_carryover,
            "extras": dict(self.extras),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "SimulationConfig":
        micro = env_bool(values, "MATCH_MICRO", True)
        legacy = env_bool(values, "MATCH_LEGACY_POISSON", False)
        micro_score = env_bool(values, "MATCH_MICRO_SCORE", True)
        score_path = "poisson_legacy" if legacy else ("physics_official" if micro and micro_score else ("micro_replay" if micro else "macro_unified"))
        return cls(
            seed=env_int(values, "GFS_SEED", 42),
            micro_enabled=micro,
            cognitive_enabled=env_bool(values, "MATCH_COGNITIVE", False),
            world_model_enabled=env_bool(values, "MATCH_WORLD_MODEL", False),
            world_model_planning=env_bool(values, "MATCH_WM_PLAN", True),
            score_path=score_path,
            save_carryover=env_bool(values, "SAVE_CARRYOVER", True),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_run_manifest(
    config: SimulationConfig,
    root: str | Path,
    *,
    data_paths: list[str | Path] | None = None,
    model_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    artifacts = {}
    for kind, paths in (("data", data_paths or []), ("models", model_paths or [])):
        artifacts[kind] = [
            {"path": str(Path(path)), "sha256": _sha256(Path(path))}
            for path in paths
            if Path(path).is_file()
        ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "root_seed": config.seed,
        "seed_derivation": "blake2s-v1",
        "git_revision": _git_revision(root_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "artifacts": artifacts,
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(manifest), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target
