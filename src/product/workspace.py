"""Persistent studio sessions that compose the simulator and research layers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


PRODUCT_SCHEMA_VERSION = 1
PRODUCT_MODES = ("stable", "research", "cognitive")
MODE_ENVIRONMENT = {
    "stable": {
        "MATCH_WORLD_MODEL": "0", "MATCH_WM_PLAN": "0",
        "MATCH_COGNITIVE": "0",
    },
    "research": {
        "MATCH_WORLD_MODEL": "1", "MATCH_WM_PLAN": "1",
        "MATCH_COGNITIVE": "0",
        "MATCH_WM_CHECKPOINT": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
    },
    "cognitive": {
        "MATCH_WORLD_MODEL": "1", "MATCH_WM_PLAN": "1",
        "MATCH_COGNITIVE": "1",
        "MATCH_WM_CHECKPOINT": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return normalized.lower() or "gfs-studio"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


@dataclass(frozen=True)
class StudioConfig:
    name: str = "My GFS Studio"
    mode: str = "stable"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.mode not in PRODUCT_MODES:
            raise ValueError(f"unsupported product mode: {self.mode}")


class ProductWorkspace:
    """One user-facing workspace over simulation, evidence, and reports."""

    def __init__(self, root: str | Path, config: StudioConfig):
        self.root = Path(root).resolve()
        self.config = config
        self.slug = _slug(config.name)
        self.session_path = self.root / "data/persistence/product_session.json"
        self.output_root = self.root / "outputs/studio" / self.slug

    @classmethod
    def create(cls, root: str | Path, config: StudioConfig) -> "ProductWorkspace":
        workspace = cls(root, config)
        session = {
            "schema_version": PRODUCT_SCHEMA_VERSION,
            "product": "Generative Football Society Studio",
            "name": config.name,
            "slug": workspace.slug,
            "mode": config.mode,
            "seed": config.seed,
            "created_at": _now(),
            "updated_at": _now(),
            "matches": [],
        }
        _atomic_json(workspace.session_path, session)
        return workspace

    @classmethod
    def load(cls, root: str | Path) -> "ProductWorkspace":
        path = Path(root).resolve() / "data/persistence/product_session.json"
        if not path.is_file():
            raise FileNotFoundError("studio not initialized; run `gfs studio init`")
        session = json.loads(path.read_text(encoding="utf-8"))
        if session.get("schema_version") != PRODUCT_SCHEMA_VERSION:
            raise ValueError("unsupported studio session schema")
        return cls(root, StudioConfig(
            name=str(session["name"]), mode=str(session["mode"]),
            seed=int(session["seed"]),
        ))

    def _session(self) -> dict[str, Any]:
        return json.loads(self.session_path.read_text(encoding="utf-8"))

    def evidence(self) -> dict[str, Any]:
        def read(relative: str) -> dict:
            path = self.root / relative
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

        release = read("data/releases/current.json")
        staged = read("data/evaluation/staged_completion_v1.json")
        phase5 = read("data/evaluation/phase5_research_layer_validation_v1.json")
        candidate = phase5.get("world_model_candidate") or {}
        live_llm = phase5.get("live_llm_evidence") or {}
        return {
            "stable_release": release.get("active"),
            "rollback_release": (release.get("rollback") or {}).get("release"),
            "all_research_stages_complete": staged.get("all_stages_complete", False),
            "world_model_candidate_accepted": candidate.get("accepted", False),
            "world_model_checkpoint": candidate.get("checkpoint"),
            "shot_planner_active": candidate.get("shot_planner_active", False),
            "shot_fallback": candidate.get("shot_fallback"),
            "live_llm_evaluable": live_llm.get("evaluable", False),
            "production_promotion_ready": staged.get("production_promotion_ready", False),
        }

    def readiness(self) -> dict[str, Any]:
        evidence = self.evidence()
        checkpoint = self.root / str(
            evidence.get("world_model_checkpoint")
            or "data/world_model/latent_wm_rollout_calibrated_candidate.pt"
        )
        checks = {
            "stable_release_available": bool(evidence.get("stable_release")),
            "research_checkpoint_available": checkpoint.is_file(),
            "research_checkpoint_accepted": bool(evidence.get("world_model_candidate_accepted")),
            "llm_credentials_available": bool(os.environ.get("API_KEY")),
        }
        if self.config.mode == "stable":
            ready = checks["stable_release_available"]
            blockers: list[str] = [] if ready else ["stable_release_missing"]
        elif self.config.mode == "research":
            ready = checks["research_checkpoint_available"] and checks["research_checkpoint_accepted"]
            blockers = [] if ready else [
                name for name in ("research_checkpoint_available", "research_checkpoint_accepted")
                if not checks[name]
            ]
        else:
            ready = all(checks[name] for name in (
                "research_checkpoint_available", "research_checkpoint_accepted",
                "llm_credentials_available",
            ))
            blockers = [name for name in (
                "research_checkpoint_available", "research_checkpoint_accepted",
                "llm_credentials_available",
            ) if not checks[name]]
        return {"mode": self.config.mode, "ready": ready, "checks": checks, "blockers": blockers}

    def status(self) -> dict[str, Any]:
        session = self._session()
        return {
            "product": session["product"], "name": session["name"],
            "mode": session["mode"], "seed": session["seed"],
            "matches_played": len(session.get("matches") or []),
            "last_match": (session.get("matches") or [None])[-1],
            "readiness": self.readiness(), "evidence": self.evidence(),
        }

    @contextmanager
    def _mode_environment(self) -> Iterator[None]:
        values = dict(MODE_ENVIRONMENT[self.config.mode])
        checkpoint = values.get("MATCH_WM_CHECKPOINT")
        if checkpoint:
            values["MATCH_WM_CHECKPOINT"] = str((self.root / checkpoint).resolve())
        previous = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def run_match(self, home: str, away: str, *, fast: bool = False) -> dict[str, Any]:
        readiness = self.readiness()
        if not readiness["ready"]:
            raise RuntimeError("studio mode is not ready: " + ", ".join(readiness["blockers"]))
        from src import app

        session = self._session()
        match_index = len(session.get("matches") or []) + 1
        seed = self.config.seed + match_index - 1
        with self._mode_environment():
            summary = app.run_micro_match(
                home, away, base_dir=self.root, seed=seed, fast=fast,
            )
        raw = asdict(summary) if is_dataclass(summary) else dict(summary)
        match_id = f"{match_index:04d}-{_slug(home)}-vs-{_slug(away)}"
        report = {
            "schema_version": PRODUCT_SCHEMA_VERSION,
            "match_id": match_id,
            "created_at": _now(),
            "studio": {"name": self.config.name, "mode": self.config.mode},
            "fixture": {"home": home, "away": away, "seed": seed, "fast": fast},
            "result": {
                "score": {"home": raw.get("goals_micro_home"), "away": raw.get("goals_micro_away")},
                "xg": {"home": raw.get("micro_xg_home"), "away": raw.get("micro_xg_away")},
                "possession": {"home": raw.get("possession_home"), "away": 1.0 - float(raw.get("possession_home", 0.5))},
                "passes": {"home": raw.get("passes_home"), "away": raw.get("passes_away")},
                "shots": {"home": raw.get("shots_home"), "away": raw.get("shots_away")},
            },
            "layers": {
                "psychology": {
                    "crowd_field": raw.get("final_psi"),
                    "coach_stress": {"home": raw.get("home_coach_stress"), "away": raw.get("away_coach_stress")},
                    "tactical_drift": {"home": raw.get("tactical_drift_home"), "away": raw.get("tactical_drift_away")},
                },
                "world_model": {
                    "enabled": self.config.mode in {"research", "cognitive"},
                    "online_calibration": raw.get("world_model_online_calibration") or {},
                    "decision_adoption": raw.get("world_model_decision_adoption") or {},
                },
                "cognition": {
                    "enabled": self.config.mode == "cognitive",
                    "triggers": raw.get("cognitive_triggers") or [],
                    "plans": raw.get("cognitive_plans") or [],
                    "tier_usage": raw.get("cognitive_tier_usage") or {},
                },
            },
            "timeline": raw.get("timeline_snippet") or [],
            "artifacts": {"ball_log": raw.get("ball_log_path") or ""},
            "evidence_snapshot": self.evidence(),
            "raw_summary": raw,
        }
        report_path = self.output_root / "matches" / f"{match_id}.json"
        _atomic_json(report_path, report)
        from src.product.reporting import write_match_html
        html_path = write_match_html(report_path.with_suffix(".html"), report)
        session.setdefault("matches", []).append({
            "match_id": match_id, "home": home, "away": away,
            "score": report["result"]["score"],
            "report": report_path.relative_to(self.root).as_posix(),
            "dashboard": html_path.relative_to(self.root).as_posix(),
        })
        session["updated_at"] = _now()
        _atomic_json(self.session_path, session)
        return {
            **report, "report_path": str(report_path),
            "dashboard_path": str(html_path),
        }
