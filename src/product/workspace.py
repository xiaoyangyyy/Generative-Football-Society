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

from src.infrastructure import (
    FileLease, file_sha256, portable_text_hash_matches,
    verify_artifact_manifest,
)
from src.simulation.runtime import environment_snapshot


PRODUCT_SCHEMA_VERSION = 1
PRODUCT_MODES = ("stable", "research", "cognitive")
ACTIVE_RUN_STATES = {"running", "finalizing"}
MODE_ENVIRONMENT = {
    "stable": {
        "MATCH_WORLD_MODEL": "0", "MATCH_WM_PLAN": "0",
        "MATCH_WORLD_MODEL_REQUIRED": "0",
        "MATCH_COGNITIVE": "0",
    },
    "research": {
        "MATCH_WORLD_MODEL": "1", "MATCH_WM_PLAN": "1",
        "MATCH_WORLD_MODEL_REQUIRED": "1",
        "MATCH_COGNITIVE": "0",
        "MATCH_WM_CHECKPOINT": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
    },
    "cognitive": {
        "MATCH_WORLD_MODEL": "1", "MATCH_WM_PLAN": "1",
        "MATCH_WORLD_MODEL_REQUIRED": "1",
        "MATCH_COGNITIVE": "1",
        "MATCH_COGNITIVE_SYNC": "1",
        "MATCH_COGNITIVE_REQUIRE_LLM": "1",
        "MATCH_COGNITIVE_MAX_PER_TIER": "2,0,0,0,0",
        "MATCH_WM_LLM_TWO_STAGE_DELIBERATION": "0",
        "MATCH_WM_CHECKPOINT": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
    },
}
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return normalized.lower() or "gfs-studio"


def _artifact_path(root: Path, value: Any) -> Path | None:
    """Resolve an evidence-controlled artifact without allowing root escape."""
    if value is None or not str(value).strip():
        return None
    candidate = Path(str(value))
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


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
        if not self.name.strip():
            raise ValueError("studio name must not be empty")
        if self.mode not in PRODUCT_MODES:
            raise ValueError(f"unsupported product mode: {self.mode}")


class ProductWorkspace:
    """One user-facing workspace over simulation, evidence, and reports."""

    def __init__(self, root: str | Path, config: StudioConfig):
        self.root = Path(root).resolve()
        self.config = config
        self.slug = _slug(config.name)
        self.session_path = self.root / "data/persistence/product_session.json"
        self.session_lease_path = self.root / "data/persistence/product_session.lock"
        self.output_root = self.root / "outputs/studio" / self.slug

    @classmethod
    def create(
        cls, root: str | Path, config: StudioConfig, *, replace: bool = False,
    ) -> "ProductWorkspace":
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
            "next_match_index": 1,
            "runs": [],
            "matches": [],
        }
        with FileLease(workspace.session_lease_path, timeout=30.0):
            if workspace.session_path.exists() and not replace:
                raise FileExistsError(
                    "studio session already exists; pass --replace to reset it"
                )
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
            "stable_release_manifest": release.get("manifest"),
            "stable_release_manifest_sha256": release.get("manifest_sha256"),
            "rollback_release": (release.get("rollback") or {}).get("release"),
            "all_research_stages_complete": staged.get("all_stages_complete", False),
            "world_model_candidate_accepted": candidate.get("accepted", False),
            "world_model_checkpoint": candidate.get("checkpoint"),
            "world_model_checkpoint_sha256": candidate.get("checkpoint_sha256"),
            "shot_planner_active": candidate.get("shot_planner_active", False),
            "shot_fallback": candidate.get("shot_fallback"),
            "frozen_shot_head": read(
                "data/evaluation/frozen_shot_head_decision.json"
            ),
            "live_llm_evaluable": live_llm.get("evaluable", False),
            "production_promotion_ready": staged.get("production_promotion_ready", False),
        }

    def readiness(self) -> dict[str, Any]:
        evidence = self.evidence()
        checkpoint = _artifact_path(self.root, (
            evidence.get("world_model_checkpoint")
            or "data/world_model/latent_wm_rollout_calibrated_candidate.pt"
        ))
        release_manifest = _artifact_path(
            self.root, evidence.get("stable_release_manifest"),
        )
        expected_release_sha256 = str(
            evidence.get("stable_release_manifest_sha256") or ""
        )
        release_identity_verified = bool(
            release_manifest is not None
            and release_manifest.is_file()
            and expected_release_sha256
            and portable_text_hash_matches(
                release_manifest, expected_release_sha256,
            )
        )
        release_artifact_verification: dict[str, Any] = {
            "ok": False, "artifacts": 0,
            "failures": [{"reason": "release_manifest_identity_unverified"}],
        }
        if release_identity_verified and release_manifest is not None:
            try:
                release_payload = json.loads(
                    release_manifest.read_text(encoding="utf-8-sig")
                )
                release_artifact_verification = verify_artifact_manifest(
                    self.root, release_payload,
                )
            except (OSError, ValueError, TypeError) as exc:
                release_artifact_verification = {
                    "ok": False, "artifacts": 0,
                    "failures": [{
                        "reason": "invalid_release_manifest",
                        "error_type": type(exc).__name__,
                    }],
                }
        expected_checkpoint_sha256 = str(
            evidence.get("world_model_checkpoint_sha256") or ""
        )
        actual_checkpoint_sha256 = (
            file_sha256(checkpoint)
            if checkpoint is not None and checkpoint.is_file() else None
        )
        shot_decision = evidence.get("frozen_shot_head") or {}
        shot_artifact_value = shot_decision.get("promotion_artifact")
        shot_artifact = _artifact_path(self.root, shot_artifact_value)
        shot_identity_verified = (
            not shot_decision.get("accepted")
            or bool(
                shot_artifact is not None
                and shot_artifact.is_file()
                and shot_decision.get("promotion_artifact_sha256")
                and file_sha256(shot_artifact)
                == shot_decision.get("promotion_artifact_sha256")
            )
        )
        from src.simulation.llm_gateway import (
            LLMGatewayConfig, llm_credentials_available,
        )

        llm_values = environment_snapshot()
        llm_config_error = None
        try:
            llm_config = LLMGatewayConfig.from_env()
            llm_provider = llm_config.public_summary()
        except (TypeError, ValueError) as exc:
            llm_config_error = str(exc)
            llm_provider = None
        checks = {
            "stable_release_available": bool(evidence.get("stable_release")),
            "stable_release_identity_verified": release_identity_verified,
            "stable_release_artifacts_verified": bool(
                release_artifact_verification.get("ok")
            ),
            "research_checkpoint_available": bool(
                checkpoint is not None and checkpoint.is_file()
            ),
            "research_checkpoint_accepted": bool(evidence.get("world_model_candidate_accepted")),
            "research_checkpoint_identity_verified": bool(
                expected_checkpoint_sha256
                and actual_checkpoint_sha256 == expected_checkpoint_sha256
            ),
            "shot_head_identity_verified": shot_identity_verified,
            "llm_credentials_available": llm_credentials_available(llm_values),
            "llm_config_valid": llm_config_error is None,
        }
        if self.config.mode == "stable":
            ready = all(checks[name] for name in (
                "stable_release_available", "stable_release_identity_verified",
                "stable_release_artifacts_verified",
            ))
            blockers: list[str] = [] if ready else [
                name for name in (
                    "stable_release_available", "stable_release_identity_verified",
                    "stable_release_artifacts_verified",
                ) if not checks[name]
            ]
        elif self.config.mode == "research":
            ready = all(checks[name] for name in (
                "research_checkpoint_available",
                "research_checkpoint_accepted",
                "research_checkpoint_identity_verified",
                "shot_head_identity_verified",
            ))
            blockers = [] if ready else [
                name for name in (
                    "research_checkpoint_available",
                    "research_checkpoint_accepted",
                    "research_checkpoint_identity_verified",
                    "shot_head_identity_verified",
                )
                if not checks[name]
            ]
        else:
            ready = all(checks[name] for name in (
                "research_checkpoint_available", "research_checkpoint_accepted",
                "research_checkpoint_identity_verified",
                "shot_head_identity_verified",
                "llm_credentials_available",
                "llm_config_valid",
            ))
            blockers = [name for name in (
                "research_checkpoint_available", "research_checkpoint_accepted",
                "research_checkpoint_identity_verified",
                "shot_head_identity_verified",
                "llm_credentials_available",
                "llm_config_valid",
            ) if not checks[name]]
        return {
            "mode": self.config.mode, "ready": ready,
            "checks": checks, "blockers": blockers,
            "release_artifact_verification": release_artifact_verification,
            "llm_provider": llm_provider,
            "llm_config_error": llm_config_error,
        }

    def status(self) -> dict[str, Any]:
        session = self._session()
        from src.product.control_plane import ProductControlPlane
        lease_held = FileLease.is_held(self.session_lease_path)
        runs = list(session.get("runs") or [])
        observed_runs = [
            {
                **run,
                "observed_state": (
                    "running" if lease_held else "abandoned"
                ) if run.get("state") in ACTIVE_RUN_STATES else run.get("state"),
            }
            for run in runs
        ]
        status = {
            "product": session["product"], "name": session["name"],
            "mode": session["mode"], "seed": session["seed"],
            "matches_played": len(session.get("matches") or []),
            "last_match": (session.get("matches") or [None])[-1],
            "runs_total": len(runs),
            "failed_runs": sum(run.get("state") == "failed" for run in runs),
            "abandoned_runs": sum(
                run.get("observed_state") == "abandoned" for run in observed_runs
            ),
            "last_run": (observed_runs or [None])[-1],
            "readiness": self.readiness(), "evidence": self.evidence(),
            "control_plane": ProductControlPlane(self.root).snapshot(),
        }
        status["workflow"] = self.workflow(status=status)
        return status

    def workflow(self, *, status: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the single guided product journey derived from persisted state."""
        if status is None:
            session = self._session()
            readiness = self.readiness()
            lease_held = FileLease.is_held(self.session_lease_path)
            runs = list(session.get("runs") or [])
            matches = list(session.get("matches") or [])
            last_match = (matches or [None])[-1]
            active = any(
                run.get("state") in ACTIVE_RUN_STATES and lease_held for run in runs
            )
            failed = sum(run.get("state") == "failed" for run in runs)
        else:
            readiness = status["readiness"]
            matches = [None] * int(status.get("matches_played", 0))
            last_match = status.get("last_match")
            active = (
                (status.get("last_run") or {}).get("observed_state")
                in ACTIVE_RUN_STATES
            )
            failed = int(status.get("failed_runs", 0))

        if not readiness["ready"]:
            state = "blocked"
            next_action = {
                "id": "resolve_readiness",
                "command": "gfs studio status",
                "reason": ", ".join(readiness["blockers"]),
            }
        elif active:
            state = "running"
            next_action = {
                "id": "wait_for_match",
                "command": "gfs studio status",
                "reason": "a match transaction is active",
            }
        elif not matches:
            state = "ready_to_run"
            next_action = {
                "id": "run_first_match",
                "command": "gfs studio match --home Brazil --away Argentina --fast",
                "reason": "readiness passed; no match has been completed",
            }
        else:
            state = "review"
            next_action = {
                "id": "open_dashboard",
                "target": (last_match or {}).get("dashboard"),
                "reason": "at least one auditable report is available",
            }

        return {
            "schema_version": 1,
            "state": state,
            "progress": {
                "workspace_configured": True,
                "readiness_passed": bool(readiness["ready"]),
                "match_completed": bool(matches),
                "report_available": bool(matches),
            },
            "completed_matches": len(matches),
            "failed_attempts": failed,
            "artifacts": {
                "latest_report": (last_match or {}).get("report"),
                "latest_dashboard": (last_match or {}).get("dashboard"),
            },
            "next_action": next_action,
        }

    @contextmanager
    def _mode_environment(self) -> Iterator[None]:
        from src.simulation.runtime import environment_override

        values = dict(MODE_ENVIRONMENT[self.config.mode])
        evidence = self.evidence()
        if self.config.mode in {"research", "cognitive"}:
            values["MATCH_WM_CHECKPOINT"] = str(
                evidence.get("world_model_checkpoint")
                or values.get("MATCH_WM_CHECKPOINT")
            )
        shot_decision = evidence.get("frozen_shot_head") or {}
        shot_artifact = shot_decision.get("promotion_artifact")
        values["MATCH_WM_SHOT_HEAD"] = (
            str(shot_artifact)
            if shot_decision.get("accepted") and shot_artifact else ""
        )
        checkpoint = values.get("MATCH_WM_CHECKPOINT")
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = self.root / checkpoint_path
            values["MATCH_WM_CHECKPOINT"] = str(checkpoint_path.resolve())
        shot_head = values.get("MATCH_WM_SHOT_HEAD")
        if shot_head:
            shot_path = Path(shot_head)
            if not shot_path.is_absolute():
                shot_path = self.root / shot_path
            values["MATCH_WM_SHOT_HEAD"] = str(shot_path.resolve())
        with environment_override(values):
            yield

    def run_match(self, home: str, away: str, *, fast: bool = False) -> dict[str, Any]:
        with FileLease(self.session_lease_path, timeout=30.0):
            return self._run_match_locked(home, away, fast=fast)

    def _run_match_locked(self, home: str, away: str, *, fast: bool = False) -> dict[str, Any]:
        readiness = self.readiness()
        if not readiness["ready"]:
            raise RuntimeError("studio mode is not ready: " + ", ".join(readiness["blockers"]))
        session = self._session()
        for prior in session.setdefault("runs", []):
            if prior.get("state") in ACTIVE_RUN_STATES:
                prior.update({
                    "state": "interrupted",
                    "finished_at": _now(),
                    "reason": "session_lease_recovered",
                })
        match_index = int(session.get(
            "next_match_index", len(session.get("matches") or []) + 1,
        ))
        session["next_match_index"] = match_index + 1
        seed = self.config.seed + match_index - 1
        match_id = f"{match_index:04d}-{_slug(home)}-vs-{_slug(away)}"
        run_record = {
            "match_id": match_id,
            "state": "running",
            "started_at": _now(),
            "fixture": {"home": home, "away": away, "seed": seed, "fast": fast},
            "mode": self.config.mode,
        }
        session["runs"].append(run_record)
        session["updated_at"] = _now()
        _atomic_json(self.session_path, session)
        try:
            return self._execute_reserved_match(
                home, away, fast=fast, session=session,
                match_index=match_index, seed=seed, match_id=match_id,
                run_record=run_record,
            )
        except BaseException as exc:
            latest = self._session()
            for recorded in latest.get("runs") or []:
                if recorded.get("match_id") == match_id:
                    recorded.update({
                        "state": "failed",
                        "finished_at": _now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    })
                    break
            latest["updated_at"] = _now()
            _atomic_json(self.session_path, latest)
            raise

    def _execute_reserved_match(
        self, home: str, away: str, *, fast: bool,
        session: dict[str, Any], match_index: int, seed: int,
        match_id: str, run_record: dict[str, Any],
    ) -> dict[str, Any]:
        from src import app

        gateway = None
        calls_before = 0
        with self._mode_environment():
            if self.config.mode == "cognitive":
                from src.simulation.llm_gateway import get_shared_llm_gateway
                gateway = get_shared_llm_gateway()
                calls_before = int(gateway.call_count)
            summary = app.run_micro_match(
                home, away, base_dir=self.root, seed=seed, fast=fast,
            )
        calls_after = int(gateway.call_count) if gateway is not None else 0
        provider_calls = max(0, calls_after - calls_before)
        raw = asdict(summary) if is_dataclass(summary) else dict(summary)
        run_record.update({
            "state": "finalizing",
            "simulation_completed_at": _now(),
            "successful_provider_calls": provider_calls,
        })
        session["updated_at"] = _now()
        _atomic_json(self.session_path, session)
        observed_world_model = raw.get("world_model_runtime") or {}
        evidence_snapshot = self.evidence()
        shot_decision = evidence_snapshot.get("frozen_shot_head") or {}
        if self.config.mode == "stable":
            shot_source = "stable_simulator"
        elif shot_decision.get("accepted") and shot_decision.get("promotion_artifact"):
            shot_source = "frozen_backbone_shot_head"
        else:
            shot_source = evidence_snapshot.get("shot_fallback") or "joint_world_model_head"
        shot_source = observed_world_model.get("shot_probability_source") or shot_source
        ball_log_path = _artifact_path(self.root, raw.get("ball_log_path"))
        ball_log_reference = (
            ball_log_path.relative_to(self.root).as_posix()
            if ball_log_path is not None else ""
        )
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
                    "configured": self.config.mode in {"research", "cognitive"},
                    "enabled": bool(observed_world_model.get("loaded")),
                    "runtime": observed_world_model,
                    "shot_probability_source": shot_source,
                    "shot_head_authorized": bool(shot_decision.get("accepted")),
                    "online_calibration": raw.get("world_model_online_calibration") or {},
                    "decision_adoption": raw.get("world_model_decision_adoption") or {},
                },
                "cognition": {
                    "enabled": self.config.mode == "cognitive",
                    "provider": {
                        "model": str(gateway.config.model) if gateway is not None else None,
                        "base_url": str(gateway.config.base_url) if gateway is not None else None,
                        "successful_calls": provider_calls,
                        "real_provider_evidence": provider_calls > 0,
                    },
                    "triggers": raw.get("cognitive_triggers") or [],
                    "plans": raw.get("cognitive_plans") or [],
                    "tier_usage": raw.get("cognitive_tier_usage") or {},
                },
            },
            "timeline": raw.get("timeline_snippet") or [],
            "artifacts": {"ball_log": ball_log_reference},
            "evidence_snapshot": evidence_snapshot,
            "raw_summary": raw,
        }
        integrity_blockers = []
        if (
            self.config.mode in {"research", "cognitive"}
            and not observed_world_model.get("loaded")
        ):
            integrity_blockers.append("required_world_model_not_observed")
        expected_runtime_signature = "sha256:" + str(
            evidence_snapshot.get("world_model_checkpoint_sha256") or ""
        )
        if (
            self.config.mode in {"research", "cognitive"}
            and observed_world_model.get("loaded")
            and observed_world_model.get("checkpoint_signature")
            != expected_runtime_signature
        ):
            integrity_blockers.append("world_model_runtime_identity_mismatch")
        if self.config.mode == "cognitive" and provider_calls <= 0:
            integrity_blockers.append("no_successful_provider_call")
        report["integrity"] = {
            "accepted": not integrity_blockers,
            "state": "accepted" if not integrity_blockers else "degraded",
            "blockers": integrity_blockers,
        }
        report_path = self.output_root / "matches" / f"{match_id}.json"
        if self.config.mode == "cognitive":
            cognitive_path = (
                self.root / "data/persistence/cognitive_log"
                / f"studio_{match_id}.json"
            )
            _atomic_json(cognitive_path, {
                "schema_version": PRODUCT_SCHEMA_VERSION,
                "match_id": match_id,
                "fixture": report["fixture"],
                "provider": report["layers"]["cognition"]["provider"],
                "cognitive_triggers": report["layers"]["cognition"]["triggers"],
                "cognitive_plans": report["layers"]["cognition"]["plans"],
                "cognitive_tier_usage": report["layers"]["cognition"]["tier_usage"],
                "world_model_online_calibration": report["layers"]["world_model"]["online_calibration"],
                "world_model_decision_adoption": report["layers"]["world_model"]["decision_adoption"],
            })
            report["artifacts"]["cognitive_log"] = cognitive_path.relative_to(self.root).as_posix()
        _atomic_json(report_path, report)
        from src.product.reporting import write_match_html
        html_path = write_match_html(report_path.with_suffix(".html"), report)
        session.setdefault("matches", []).append({
            "match_id": match_id, "home": home, "away": away,
            "score": report["result"]["score"],
            "report": report_path.relative_to(self.root).as_posix(),
            "dashboard": html_path.relative_to(self.root).as_posix(),
            "integrity": report["integrity"]["state"],
        })
        run_record.update({
            "state": "completed",
            "finished_at": _now(),
            "integrity": report["integrity"]["state"],
            "report": report_path.relative_to(self.root).as_posix(),
        })
        session["updated_at"] = _now()
        _atomic_json(self.session_path, session)
        return {
            **report, "report_path": str(report_path),
            "dashboard_path": str(html_path),
        }
