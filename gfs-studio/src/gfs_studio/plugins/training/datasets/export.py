from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.kernel.identity import canonical_bytes, content_identity
from gfs_core.kernel.random import named_rng

from gfs_studio.contracts.world_model import ACTION_DIM, OBS_DIM
from .manifest import TraceDataset


class TraceStream:
    def __init__(self, path: str | Path, codec, config_identity: str) -> None:
        self.path = Path(path)
        self.codec = codec
        self.config_identity = config_identity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp",
        )
        self.temporary = temporary
        self.handle = os.fdopen(descriptor, "wb")
        self.previous_observation = None
        self.previous_target = None
        self.rows = 0
        self.groups = set()

    def consume(self, record) -> None:
        observation = np.asarray(
            self.codec.observation(record.state), dtype=np.float32,
        )
        if observation.shape != (OBS_DIM,) or not np.isfinite(observation).all():
            raise ValueError("trace codec returned an invalid observation")
        if record.tick == 0:
            self.previous_observation = observation
            self.previous_target = record.state.ball.position.copy()
            return
        action = np.asarray(self.codec.action(
            record.action, self.previous_target, record.dt,
        ), dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("trace codec returned an invalid action")
        row = {
            "schema_version": 1,
            "record_type": "transition",
            "group_id": record.match_id,
            "match_id": record.match_id,
            "home": record.state.home.team_id,
            "away": record.state.away.team_id,
            "tick": record.tick - 1,
            "observation": self.previous_observation.tolist(),
            "action": action.tolist(),
            "next_observation": observation.tolist(),
            "events": [event.event_type.value for event in record.events],
        }
        self.handle.write(canonical_bytes(row) + b"\n")
        self.rows += 1
        self.groups.add(record.match_id)
        self.previous_observation = observation
        self.previous_target = record.state.ball.position.copy()

    def finish(self, status: str) -> dict:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temporary, self.path)
        digest = _file_hash(self.path)
        manifest = {
            "schema_version": 1,
            "source": "gfs.core.match",
            "trace_contract": "gfs.core.trace/v1",
            "codec_version": self.codec.version,
            "config_identity": self.config_identity,
            "trace_sha256": digest,
            "rows": self.rows,
            "groups": sorted(self.groups),
            "observation_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "split_seed": 42,
            "status": status,
        }
        manifest_path = self.path.with_suffix(".manifest.json")
        _atomic_json(manifest_path, manifest)
        return {**manifest, "trace": str(self.path), "manifest": str(manifest_path)}

    def abort(self) -> None:
        self.handle.close()
        if os.path.exists(self.temporary):
            os.remove(self.temporary)


class DatasetService:
    from_jsonl = staticmethod(TraceDataset.from_jsonl)

    def __init__(self, status, tactics, scoring, match_runner, codec) -> None:
        self.status = status
        self.tactics = tactics
        self.scoring = scoring
        self.match_runner = match_runner
        self.codec = codec

    def export(self, output: str | Path, *, home: str = "Brazil", away: str = "Argentina", seed: int = 42, fast: bool = True, stop_at_tick: int | None = None) -> dict:
        config = MicroMatchConfig.fast_demo() if fast else MicroMatchConfig()
        match_id = f"core:{home}:{away}:{seed}"
        request_identity = content_identity({
            "home": home,
            "away": away,
            "seed": seed,
            "fast": fast,
            "config": config,
        })
        stream = TraceStream(output, self.codec, request_identity)
        try:
            home_profile = self.tactics.profile(home, self.status.table.loc[home])
            away_profile = self.tactics.profile(away, self.status.table.loc[away])
            prior = self.scoring.simulate(
                home_profile.status_score,
                away_profile.status_score,
                is_knockout=False,
                rng=named_rng(seed, "micro_match", home, away, "macro_prior"),
            )
            result = self.match_runner.run(
                home_profile,
                away_profile,
                goals_home=prior.home_goals,
                goals_away=prior.away_goals,
                xg_home=prior.home_xg,
                xg_away=prior.away_xg,
                referee={"strictness": 0.58, "bias_t1": 0.04},
                config=config,
                seed=seed,
                trace_sink=stream.consume,
                trace_match_id=match_id,
                stop_at_tick=stop_at_tick,
            )
            status = "interrupted" if isinstance(result, dict) else "completed"
            return stream.finish(status)
        except BaseException:
            stream.abort()
            raise

    def export_cohort(self, specifications: list[dict], output_dir: str | Path, *, fast: bool = True, stop_at_tick: int | None = None, holdout_teams: list[str] | None = None, validation_time_fraction: float = 0.2) -> dict:
        if not specifications:
            raise ValueError("cohort requires at least one match")
        output_root = Path(output_dir).resolve()
        reports = []
        for index, specification in enumerate(specifications):
            home = specification.get("home", "Brazil")
            away = specification.get("away", "Argentina")
            seed = specification.get("seed", 42 + index)
            if not isinstance(home, str) or not isinstance(away, str) or isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("invalid cohort match specification")
            match_id = f"core:{home}:{away}:{seed}"
            suffix = hashlib.sha256(match_id.encode()).hexdigest()[:12]
            report = self.export(
                output_root / "matches" / f"{index:03d}-{suffix}.jsonl",
                home=home,
                away=away,
                seed=seed,
                fast=fast,
                stop_at_tick=stop_at_tick,
            )
            reports.append({
                **report,
                "trace": Path(report["trace"]).resolve().relative_to(output_root).as_posix(),
                "manifest": Path(report["manifest"]).resolve().relative_to(output_root).as_posix(),
                "home": home,
                "away": away,
            })
        cohort_path = output_root / "cohort.jsonl"
        _atomic_bytes(cohort_path, b"".join(
            (output_root / report["trace"]).read_bytes() for report in reports
        ))
        group_teams = {
            report["groups"][0]: (report["home"], report["away"])
            for report in reports
        }
        manifest = {
            "schema_version": 1,
            "cohort_identity": content_identity({
                "specifications": specifications,
                "fast": fast,
                "stop_at_tick": stop_at_tick,
            }),
            "source": "gfs.core.match",
            "trace_contract": "gfs.core.trace/v1",
            "codec_version": reports[0]["codec_version"],
            "config_identity": content_identity([
                report["config_identity"] for report in reports
            ]),
            "trace_sha256": _file_hash(cohort_path),
            "rows": sum(report["rows"] for report in reports),
            "groups": sorted(group_teams),
            "group_teams": group_teams,
            "observation_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "split_seed": 42,
            "holdout_teams": list(holdout_teams or []),
            "validation_time_fraction": validation_time_fraction,
            "status": "completed" if all(report["status"] == "completed" for report in reports) else "interrupted",
            "matches": reports,
        }
        manifest_path = output_root / "cohort.manifest.json"
        _atomic_json(manifest_path, manifest)
        return {
            **manifest,
            "trace": str(cohort_path),
            "manifest": str(manifest_path),
        }


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
