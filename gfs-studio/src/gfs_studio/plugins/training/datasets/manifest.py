from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gfs_studio.contracts.training import DatasetSplit
from gfs_studio.contracts.world_model import ACTION_DIM, OBS_DIM


@dataclass(frozen=True, slots=True)
class TraceDataset:
    observations: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    identity: str | None = None
    source: str = "legacy_unverified"
    config_identity: str | None = None
    split_seed: int | None = None
    ticks: np.ndarray | None = None
    group_teams: dict[str, tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        rows = len(self.observations)
        if rows < 2:
            raise ValueError("training dataset requires at least two rows")
        if self.observations.shape != (rows, OBS_DIM):
            raise ValueError("invalid observation shape")
        if self.actions.shape != (rows, ACTION_DIM):
            raise ValueError("invalid action shape")
        if self.targets.shape != (rows, OBS_DIM):
            raise ValueError("invalid target shape")
        if self.groups.shape != (rows,):
            raise ValueError("invalid group shape")
        if self.ticks is not None and self.ticks.shape != (rows,):
            raise ValueError("invalid tick shape")
        for values in (self.observations, self.actions, self.targets):
            if not np.isfinite(values).all():
                raise ValueError("training values must be finite")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> TraceDataset:
        path = Path(path)
        observations = []
        actions = []
        targets = []
        groups = []
        tick_values = []
        last_ticks = {}
        group_teams = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                observations.append(row.get("observation", row.get("obs")))
                actions.append(row["action"])
                targets.append(row.get("next_observation", row.get("next_obs")))
                group = str(row.get("group_id", row.get("match_id", line_number)))
                groups.append(group)
                tick = row.get("tick", -1)
                tick_values.append(tick)
                if "tick" in row:
                    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0 or tick <= last_ticks.get(group, -1):
                        raise ValueError("trace ticks must be strictly increasing per match")
                    last_ticks[group] = tick
                if "home" in row and "away" in row:
                    group_teams[group] = (str(row["home"]), str(row["away"]))
        digest = _file_hash(path)
        manifest_path = path.with_suffix(".manifest.json")
        source = "legacy_unverified"
        config_identity = None
        split_seed = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("trace_sha256") != digest:
                raise ValueError("trace content identity mismatch")
            if manifest.get("rows") != len(observations):
                raise ValueError("trace row count does not match its manifest")
            if manifest.get("groups") != sorted(set(groups)):
                raise ValueError("trace groups do not match their manifest")
            if manifest.get("observation_dim") != OBS_DIM or manifest.get("action_dim") != ACTION_DIM:
                raise ValueError("trace dimensions do not match their manifest")
            source = str(manifest.get("source", "unknown"))
            config_identity = manifest.get("config_identity")
            split_seed = manifest.get("split_seed")
        return cls(
            np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            np.asarray(groups),
            identity=digest,
            source=source,
            config_identity=config_identity,
            split_seed=split_seed,
            ticks=None if any(tick < 0 for tick in tick_values) else np.asarray(tick_values, dtype=np.int64),
            group_teams=group_teams or None,
        )

    def split(self, *, validation_groups: int = 1, seed: int | None = None) -> tuple[TraceDataset, TraceDataset, DatasetSplit]:
        seed = self.split_seed if seed is None else seed
        unique = sorted(set(self.groups.tolist()))
        if seed is not None:
            unique.sort(key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
        if not 0 < validation_groups < len(unique):
            raise ValueError("validation group count must leave training groups")
        validation_ids = set(unique[-validation_groups:])
        validation_mask = np.asarray([group in validation_ids for group in self.groups])
        train = self._mask(~validation_mask)
        validation = self._mask(validation_mask)
        return train, validation, DatasetSplit(
            len(unique) - validation_groups,
            validation_groups,
            len(train.observations),
            len(validation.observations),
        )

    def split_by_teams(self, holdout_teams: list[str]) -> tuple[TraceDataset, TraceDataset, DatasetSplit]:
        if not holdout_teams:
            raise ValueError("holdout teams are required")
        if self.group_teams is None or not set(self.group_teams) >= set(self.groups):
            raise ValueError("team isolation requires complete group-to-team metadata")
        holdout = set(holdout_teams)
        validation_mask = np.asarray([
            bool(holdout.intersection(self.group_teams[group]))
            for group in self.groups
        ])
        return self._split_mask(validation_mask)

    def split_time(self, *, validation_fraction: float = 0.2) -> tuple[TraceDataset, TraceDataset, DatasetSplit]:
        if self.ticks is None:
            raise ValueError("time split requires tick metadata")
        if not 0 < validation_fraction < 1:
            raise ValueError("time validation fraction must be between zero and one")
        cutoff = float(np.quantile(self.ticks, 1 - validation_fraction))
        return self._split_mask(self.ticks >= cutoff)

    def _split_mask(self, validation_mask):
        if not validation_mask.any() or validation_mask.all():
            raise ValueError("split must leave both training and validation rows")
        train = self._mask(~validation_mask)
        validation = self._mask(validation_mask)
        return train, validation, DatasetSplit(
            len(set(train.groups.tolist())),
            len(set(validation.groups.tolist())),
            len(train.observations),
            len(validation.observations),
        )

    def _mask(self, mask) -> TraceDataset:
        groups = self.groups[mask].copy()
        return TraceDataset(
            self.observations[mask].copy(),
            self.actions[mask].copy(),
            self.targets[mask].copy(),
            groups,
            identity=self.identity,
            source=self.source,
            config_identity=self.config_identity,
            split_seed=self.split_seed,
            ticks=None if self.ticks is None else self.ticks[mask].copy(),
            group_teams=None if self.group_teams is None else {
                group: self.group_teams[group] for group in set(groups.tolist())
            },
        )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
