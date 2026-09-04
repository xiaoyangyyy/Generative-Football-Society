"""Persistent studio sessions that compose the simulator and research layers."""

from __future__ import annotations

import json
import os
import re
import tempfile
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.infrastructure import (
    FileLease,
    file_sha256,
    portable_text_hash_matches,
    verify_artifact_manifest,
)
from src.product.match_plan import MatchPlan, NATIVE_TACTIC
from src.product.manager_intelligence import (
    build_postmatch_debrief,
    build_prematch_intelligence,
)
from src.product.decision_ledger import build_manager_decision_ledger
from src.product.manager_future import (
    build_manager_future_context,
    validate_manager_future_context,
)
from src.product.manager_future_review import (
    MAX_REVIEWS_PER_FIXTURE,
    build_manager_future_review,
    validate_manager_future_review,
)
from src.product.world_model_fork_set import (
    project_fork_set_scenario_evidence,
)
from src.match_engine.world_model.mirrored_policy_evaluation import (
    study_preflight_identity,
    study_execution_identity,
    validate_m2_candidate_receipt,
    validate_m2_preflight_receipt,
)
from src.product.m2_research_control import build_m2_research_control
from src.product.decision_advice import (
    build_manager_advice_adoption,
    build_manager_advice_comparison,
    build_manager_decision_advice,
    validate_manager_advice_adoption,
    validate_manager_decision_advice,
)
from src.product.world_state_evidence import (
    build_fixture_world_state_transition,
    capture_world_state,
    validate_fixture_world_state_transition,
)
from src.simulation.lineup import (
    automatic_lineup,
    build_squad_catalog,
    validate_and_freeze_lineup,
)
from src.product.season import (
    ClubResourcePlan,
    ManagerDecision,
    SeasonPlan,
    archive_completed_season,
    archived_opponent_preparation,
    club_resource_effects,
    manager_board_review,
    manager_career_profile,
    manager_season_profile,
    matchday_command_center,
    new_season_state,
    next_matchday,
    opponent_preparation,
    season_standings,
    validate_manager_career_transition,
    validate_season_state,
)
from src.product.replay import (
    build_match_replay,
    build_world_model_action_links,
)
from src.simulation.runtime import environment_snapshot
from src.simulation.squad_registry import (
    append_recruitment_transaction,
    load_effective_roster,
    generate_recruitment_market,
    RecruitmentPlan,
    recruitment_market_for_workspace,
    recruitment_market_id,
    replay_squad_registry_through,
    squad_player_quality,
    validate_squad_registry,
)
from src.data_engine.roster_loader import load_roster_json, roster_path_for_team
from src.product.club_finance import (
    append_recruitment_charge,
    append_season_settlement,
    build_season_finance_settlement,
    club_finance_view,
    ensure_finance_club,
    evidence_identity,
    player_wage_tier,
    recruitment_allowance,
    seasonal_wage_expense,
    validate_finance_registry,
)
from src.product.league_ecosystem import (
    ai_club_recruitment_decision,
    append_league_transition,
    league_ecosystem_view,
    validate_league_ecosystem,
)
from src.product.club_strategy import (
    build_season_club_strategies,
    resolve_fixture_club_strategy,
    strategy_identity,
)
from src.product.player_development import (
    append_development_transaction,
    build_development_transaction,
    collect_season_participation,
    development_view,
    validate_development_registry,
    validate_participation_evidence,
    verify_participation_report_files,
)
from src.simulation.player_lifecycle import (
    append_lifecycle_transaction,
    build_lifecycle_preview,
    build_lifecycle_transaction,
    lifecycle_view,
    RetentionPlan,
    retirement_age_for_player,
    validate_lifecycle_registry,
)
from src.simulation.player_market import (
    FreeAgentPlan,
    ai_free_agent_decision,
    execute_free_agent_signing,
    finalize_market_transition,
    market_preview_from_window,
    market_view,
    market_player_quality,
    open_market_window,
    pool_entry_from_player,
    validate_market_registry,
)
from src.simulation.scouting import (
    append_scouting_report,
    reports_for_market,
    scouting_identity,
    scouting_observation,
    scouting_view,
    validate_scouting_registry,
)
from src.product.scouting_outcomes import (
    append_scouting_outcome,
    build_scouting_outcome,
    resolve_signing_observation,
    scouting_outcome_view,
    validate_scouting_outcome_registry,
)
from src.product.sporting_director import (
    build_sporting_brief,
    validate_sporting_brief,
    validate_sporting_choices,
)
from src.product.sporting_reviews import (
    append_sporting_review,
    build_sporting_review,
    sporting_review_view,
    sporting_review_feedback,
    validate_sporting_review_registry,
)
from src.product.club_timeline import (
    build_timeline_resolution,
    club_timeline_view,
    compatible_choice,
    derive_club_situation,
)
from src.product.season_commitments import (
    build_commitment_contract_from_sources,
    commitment_progress_from_evidence,
)
from src.product.player_promises import (
    PlayerPromisePlan,
    build_player_promise_contract,
    player_promise_progress,
    settle_player_promise_outcomes,
)


PRODUCT_SCHEMA_VERSION = 1
PRODUCT_MODES = ("stable", "research", "cognitive")
ACTIVE_RUN_STATES = {"running", "finalizing"}
MODE_ENVIRONMENT = {
    "stable": {
        "MATCH_AUTHORITATIVE_TICK_CLOCK": "1",
        "MATCH_WORLD_MODEL": "0",
        "MATCH_WM_PLAN": "0",
        "MATCH_WORLD_MODEL_REQUIRED": "0",
        "MATCH_COGNITIVE": "0",
        "MATCH_BALL_LOG": "1",
        "MATCH_BALL_LOG_MAX": "800",
        "MATCH_BALL_LOG_TXT": "0",
    },
    "research": {
        "MATCH_AUTHORITATIVE_TICK_CLOCK": "1",
        "MATCH_WORLD_MODEL": "1",
        "MATCH_WM_PLAN": "1",
        "MATCH_WORLD_MODEL_REQUIRED": "1",
        "MATCH_COGNITIVE": "0",
        "MATCH_BALL_LOG": "1",
        "MATCH_BALL_LOG_MAX": "800",
        "MATCH_BALL_LOG_TXT": "0",
        "MATCH_WM_CHECKPOINT": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
    },
    "cognitive": {
        "MATCH_AUTHORITATIVE_TICK_CLOCK": "1",
        "MATCH_WORLD_MODEL": "1",
        "MATCH_WM_PLAN": "1",
        "MATCH_WORLD_MODEL_REQUIRED": "1",
        "MATCH_COGNITIVE": "1",
        "MATCH_COGNITIVE_SYNC": "1",
        "MATCH_COGNITIVE_REQUIRE_LLM": "1",
        "MATCH_COGNITIVE_MAX_PER_TIER": "2,0,0,0,0",
        "MATCH_WM_LLM_TWO_STAGE_DELIBERATION": "0",
        "MATCH_WM_CHECKPOINT": "data/world_model/latent_wm_rollout_calibrated_candidate.pt",
        "MATCH_BALL_LOG": "1",
        "MATCH_BALL_LOG_MAX": "800",
        "MATCH_BALL_LOG_TXT": "0",
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


def _action_adoption_digest(value: Any) -> dict[str, Any]:
    """Keep the latest user-facing influence metrics without persisting trace bulk."""
    if not isinstance(value, Mapping) or not value:
        return {}
    integer_fields = (
        "opportunities",
        "influenced_opportunities",
        "attribution_eligible_opportunities",
        "counterfactual_action_changes",
    )
    float_fields = (
        "expected_counterfactual_action_changes",
        "counterfactual_change_rate",
        "expected_counterfactual_change_rate",
        "mean_recommended_probability_shift",
        "mean_primary_signal_probability_shift",
    )
    digest: dict[str, Any] = {
        "reason": value.get("reason"),
        "probability_policy_version": str(
            value.get("probability_policy_version") or ""
        ),
        "expected_change_estimator": str(
            value.get("expected_change_estimator") or ""
        ),
    }
    for field in integer_fields:
        digest[field] = int(value.get(field) or 0)
    for field in float_fields:
        digest[field] = float(value.get(field) or 0.0)
    raw_breakdown = value.get("action_signal_breakdown") or {}
    breakdown = {}
    breakdown_fields = (
        "signal_opportunities", "positive_guidance", "negative_guidance",
        "realized_as_actual_action", "locally_changed_to_action",
        "mean_probability_delta", "mean_absolute_probability_shift",
        "mean_applied_authority",
    )
    if isinstance(raw_breakdown, Mapping):
        for action in ("pass", "shot", "cross", "hold"):
            raw = raw_breakdown.get(action)
            if not isinstance(raw, Mapping):
                continue
            breakdown[action] = {
                field: (
                    int(raw.get(field) or 0)
                    if field in {
                        "signal_opportunities", "positive_guidance",
                        "negative_guidance", "realized_as_actual_action",
                        "locally_changed_to_action",
                    }
                    else float(raw.get(field) or 0.0)
                )
                for field in breakdown_fields
            }
    digest["action_signal_breakdown"] = breakdown
    raw_reference = value.get("reference_action_breakdown") or {}
    if isinstance(raw_reference, Mapping):
        digest["reference_action_breakdown"] = {
            "action": str(raw_reference.get("action") or "hold"),
            "role": str(
                raw_reference.get("role") or "counterfactual_baseline_only"
            ),
            "direct_signal_opportunities": int(
                raw_reference.get("direct_signal_opportunities") or 0
            ),
            "redistribution_opportunities": int(
                raw_reference.get("redistribution_opportunities") or 0
            ),
            "realized_actions": int(
                raw_reference.get("realized_actions") or 0
            ),
            "counterfactual_changes": int(
                raw_reference.get("counterfactual_changes") or 0
            ),
            "mean_probability_gain": float(
                raw_reference.get("mean_probability_gain") or 0.0
            ),
        }
    return digest


def _formal_evidence_identity(
    root: Path,
    protocol_relative: str,
    protocol: Mapping[str, Any],
    progress: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when a sealed result does not describe current code."""
    if not protocol or not decision:
        return {"verified": False, "reason": "formal_result_unavailable"}
    candidate = protocol.get("candidate") or {}
    integrity = protocol.get("integrity") or {}
    code_files = integrity.get("code_identity_files") or []
    checkpoint_relative = candidate.get("checkpoint")
    checkpoint_expected = candidate.get("checkpoint_sha256")
    if (
        not isinstance(code_files, list) or not code_files
        or not checkpoint_relative or not checkpoint_expected
    ):
        return {"verified": False, "reason": "identity_contract_missing"}
    try:
        resolved_root = root.resolve()
        protocol_path = (root / protocol_relative).resolve()
        checkpoint_path = (root / str(checkpoint_relative)).resolve()
        for path in (protocol_path, checkpoint_path):
            if not path.is_file() or resolved_root not in path.parents:
                raise ValueError("identity artifact unavailable or outside root")
        checkpoint_sha = file_sha256(checkpoint_path)
        if checkpoint_sha != str(checkpoint_expected):
            return {"verified": False, "reason": "checkpoint_identity_mismatch"}
        code_sha = {}
        for relative in code_files:
            path = (root / str(relative)).resolve()
            if not path.is_file() or resolved_root not in path.parents:
                raise ValueError("code identity artifact unavailable or outside root")
            code_sha[str(relative)] = file_sha256(path)
        expected = {
            "protocol_sha256": file_sha256(protocol_path),
            "checkpoint_sha256": checkpoint_sha,
            "code_sha256": code_sha,
        }
    except (OSError, TypeError, ValueError):
        return {"verified": False, "reason": "identity_replay_unavailable"}
    if progress.get("execution_identity") != expected:
        return {"verified": False, "reason": "progress_code_identity_stale"}
    if decision.get("execution_identity") != expected:
        return {"verified": False, "reason": "decision_code_identity_stale"}
    return {"verified": True, "reason": "current_code_identity_verified"}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}-",
        suffix=".tmp",
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
        cls,
        root: str | Path,
        config: StudioConfig,
        *,
        replace: bool = False,
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
            "next_season_index": 1,
            "season_history": [],
            "archived_seasons_total": 0,
            "squad_registry": {"schema_version": 1, "clubs": {}},
            "finance_registry": {"schema_version": 1, "clubs": {}},
            "league_ecosystem": {"schema_version": 1, "transitions": []},
            "player_development": {"schema_version": 1, "clubs": {}},
            "player_lifecycle": {"schema_version": 1, "clubs": {}},
            "player_market": {"schema_version": 1, "transitions": []},
            "scouting_registry": {"schema_version": 1, "reports": []},
            "scouting_outcomes": {"schema_version": 1, "outcomes": []},
            "sporting_reviews": {"schema_version": 1, "reviews": []},
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
        return cls(
            root,
            StudioConfig(
                name=str(session["name"]),
                mode=str(session["mode"]),
                seed=int(session["seed"]),
            ),
        )

    def _session(self) -> dict[str, Any]:
        session = json.loads(self.session_path.read_text(encoding="utf-8"))
        if not isinstance(session, dict) or session.get("schema_version") != 1:
            raise ValueError("invalid product session")
        validate_development_registry(session.get("player_development"))
        validate_lifecycle_registry(session.get("player_lifecycle"))
        validate_market_registry(session.get("player_market"))
        validate_scouting_registry(session.get("scouting_registry"))
        validate_scouting_outcome_registry(session.get("scouting_outcomes"))
        validate_sporting_review_registry(session.get("sporting_reviews"))
        validate_squad_registry(
            self.root,
            session.get("squad_registry"),
            development_registry=session.get("player_development"),
            lifecycle_registry=session.get("player_lifecycle"),
            market_registry=session.get("player_market"),
        )
        validate_finance_registry(session.get("finance_registry"))
        validate_league_ecosystem(session.get("league_ecosystem"))
        season = session.get("season")
        if isinstance(season, Mapping):
            validate_season_state(season)
            plan = SeasonPlan.from_payload(season["plan"])
            if plan.manager_recruitment is not None:
                club = (
                    (session.get("squad_registry") or {})
                    .get("clubs", {})
                    .get(str(plan.manager_team), {})
                )
                matches = (
                    [
                        transaction
                        for transaction in club.get("transactions", [])
                        if transaction.get("season_id") == season.get("season_id")
                    ]
                    if isinstance(club, Mapping)
                    else []
                )
                if (
                    len(matches) != 1
                    or season.get("recruitment_transaction") != matches[0]
                ):
                    raise ValueError(
                        "active season recruitment registry identity mismatch"
                    )
        self._season_history_view(session)
        archives = {
            str(archive.get("season_id") or ""): archive
            for archive in session.get("season_history") or []
            if isinstance(archive, Mapping)
        }
        finance_clubs = (session.get("finance_registry") or {}).get("clubs") or {}
        squad_clubs = (session.get("squad_registry") or {}).get("clubs") or {}
        development_clubs = (session.get("player_development") or {}).get("clubs") or {}
        lifecycle_clubs = (session.get("player_lifecycle") or {}).get("clubs") or {}
        for scouting_report in (session.get("scouting_registry") or {}).get(
            "reports"
        ) or []:
            target_id = str(scouting_report.get("target_season_id") or "")
            target_index = int(target_id.split("-")[-1])
            prior_market = {
                "schema_version": 1,
                "transitions": [
                    copy.deepcopy(item)
                    for item in (session.get("player_market") or {}).get(
                        "transitions", []
                    )
                    if int(str(item["target_season_id"]).split("-")[-1]) < target_index
                ],
            }
            window = open_market_window(
                prior_market,
                target_season_id=target_id,
            )
            team = str(scouting_report.get("team") or "")
            preview = market_preview_from_window(window, team=team)
            player_id = str(scouting_report.get("player_id") or "")
            entry = (window.get("available") or {}).get(player_id)
            if (
                entry is None
                or scouting_report.get("market_id") != preview["market_id"]
                or scouting_report.get("candidate_entry_identity")
                != scouting_identity(entry)
                or scouting_report.get("observation")
                != scouting_observation(
                    market_id=preview["market_id"],
                    team=team,
                    player_id=player_id,
                    true_quality=market_player_quality(entry["player"]),
                    level="scouted",
                )
            ):
                raise ValueError("scouting report source replay mismatch")
        for lifecycle_team, club in lifecycle_clubs.items():
            base_roster = load_roster_json(
                roster_path_for_team(
                    str(self.root),
                    lifecycle_team,
                )
            )
            if base_roster is None:
                raise ValueError("lifecycle source roster is unavailable")
            transactions = club.get("transactions") or []
            for transaction in transactions:
                source_id = str(transaction.get("source_season_id") or "")
                target_id = str(transaction.get("target_season_id") or "")
                if source_id not in archives:
                    continue
                target_index = int(target_id.split("-")[-1])
                prior_lifecycle = {
                    "schema_version": 1,
                    "clubs": {
                        lifecycle_team: {
                            "transactions": [
                                copy.deepcopy(item)
                                for item in transactions
                                if int(str(item["target_season_id"]).split("-")[-1])
                                < target_index
                            ]
                        }
                    },
                }
                development_transactions = (
                    development_clubs.get(lifecycle_team) or {}
                ).get("transactions", [])
                prior_development = {
                    "schema_version": 1,
                    "clubs": {
                        lifecycle_team: {
                            "transactions": [
                                copy.deepcopy(item)
                                for item in development_transactions
                                if int(str(item["target_season_id"]).split("-")[-1])
                                < target_index
                            ]
                        }
                    },
                }
                source_roster = replay_squad_registry_through(
                    base_roster,
                    team=lifecycle_team,
                    registry=session.get("squad_registry"),
                    season_id=source_id,
                    development_registry=session.get("player_development"),
                    lifecycle_registry=session.get("player_lifecycle"),
                    market_registry=session.get("player_market"),
                )
                target_roster = replay_squad_registry_through(
                    base_roster,
                    team=lifecycle_team,
                    registry=session.get("squad_registry"),
                    season_id=target_id,
                    development_registry=prior_development,
                    lifecycle_registry=prior_lifecycle,
                    market_registry=session.get("player_market"),
                )
                raw_plan = transaction.get("plan")
                retention = (
                    RetentionPlan.from_payload(raw_plan)
                    if raw_plan is not None
                    else None
                )
                expected, _result = build_lifecycle_transaction(
                    source_roster,
                    target_roster,
                    team=lifecycle_team,
                    target_season_id=target_id,
                    plan=retention,
                    control=str(transaction.get("control") or ""),
                )
                if transaction != expected:
                    raise ValueError("player lifecycle source replay mismatch")
        for development_team, club in development_clubs.items():
            base_roster = load_roster_json(
                roster_path_for_team(
                    str(self.root),
                    development_team,
                )
            )
            if base_roster is None:
                raise ValueError("development source roster is unavailable")
            transactions = club.get("transactions") or []
            for transaction in transactions:
                source_id = str(transaction.get("source_season_id") or "")
                target_id = str(transaction.get("target_season_id") or "")
                source_archive = archives.get(source_id)
                if source_archive is None:
                    continue
                evidence_by_team = source_archive.get(
                    "player_development_evidence",
                )
                if (
                    not isinstance(evidence_by_team, Mapping)
                    or development_team not in evidence_by_team
                ):
                    raise ValueError("development archive evidence is unavailable")
                participation = evidence_by_team[development_team]
                verify_participation_report_files(self.root, participation)
                source_roster = replay_squad_registry_through(
                    base_roster,
                    team=development_team,
                    registry=session.get("squad_registry"),
                    season_id=source_id,
                    development_registry=session.get("player_development"),
                    lifecycle_registry=session.get("player_lifecycle"),
                    market_registry=session.get("player_market"),
                )
                target_index = int(target_id.split("-")[-1])
                prior_development = {
                    "schema_version": 1,
                    "clubs": {
                        development_team: {
                            "transactions": [
                                copy.deepcopy(item)
                                for item in transactions
                                if int(str(item["target_season_id"]).split("-")[-1])
                                < target_index
                            ],
                        },
                    },
                }
                target_roster = replay_squad_registry_through(
                    base_roster,
                    team=development_team,
                    registry=session.get("squad_registry"),
                    season_id=target_id,
                    development_registry=prior_development,
                    lifecycle_registry=session.get("player_lifecycle"),
                    market_registry=session.get("player_market"),
                )
                expected, _result = build_development_transaction(
                    source_archive,
                    source_roster,
                    target_roster,
                    participation,
                    team=development_team,
                    target_season_id=target_id,
                )
                if transaction != expected:
                    raise ValueError("player development source replay mismatch")
        for team, club in finance_clubs.items():
            for entry in club.get("entries") or []:
                evidence = entry.get("evidence") or {}
                entry_season_id = str(entry.get("season_id") or "")
                if entry.get("type") == "recruitment_charge":
                    squad_entry = squad_clubs.get(team) or {}
                    matches = [
                        transaction
                        for transaction in squad_entry.get("transactions") or []
                        if transaction.get("season_id") == entry_season_id
                    ]
                    if (
                        len(matches) != 1
                        or evidence.get("spent") != matches[0].get("spent")
                        or evidence.get("squad_transaction_identity")
                        != evidence_identity(matches[0])
                    ):
                        raise ValueError("recruitment finance evidence mismatch")
                elif entry_season_id in archives:
                    base_roster = load_roster_json(
                        roster_path_for_team(
                            str(self.root),
                            team,
                        )
                    )
                    historical_roster = (
                        replay_squad_registry_through(
                            base_roster,
                            team=team,
                            registry=session.get("squad_registry"),
                            season_id=entry_season_id,
                            development_registry=session.get("player_development"),
                            lifecycle_registry=session.get("player_lifecycle"),
                            market_registry=session.get("player_market"),
                        )
                        if base_roster is not None
                        else None
                    )
                    expected = build_season_finance_settlement(
                        archives[entry_season_id],
                        historical_roster,
                        team=team,
                    )
                    if evidence != expected:
                        raise ValueError("season finance evidence mismatch")
        season_sources = dict(archives)
        if isinstance(season, Mapping):
            season_sources[str(season.get("season_id") or "")] = season
        for promise_season_id, promise_state in season_sources.items():
            contract = promise_state.get("player_role_promises")
            if not isinstance(contract, Mapping):
                continue
            promise_plan = SeasonPlan.from_payload(promise_state.get("plan") or {})
            team = str(promise_plan.manager_team or "")
            base_roster = (
                load_roster_json(
                    roster_path_for_team(
                        str(self.root),
                        team,
                    )
                )
                if team
                else None
            )
            historical_roster = (
                replay_squad_registry_through(
                    base_roster,
                    team=team,
                    registry=session.get("squad_registry"),
                    season_id=promise_season_id,
                    development_registry=session.get("player_development"),
                    lifecycle_registry=session.get("player_lifecycle"),
                    market_registry=session.get("player_market"),
                )
                if base_roster is not None
                else None
            )
            frozen_plan = (
                PlayerPromisePlan.from_payload(
                    {
                        "schema_version": 1,
                        "promises": [
                            {
                                "player_id": row.get("player_id"),
                                "role": row.get("promised_role"),
                            }
                            for row in contract.get("players") or []
                        ],
                    }
                )
                if contract.get("control") == "manager"
                else None
            )
            expected_contract = build_player_promise_contract(
                season_id=promise_season_id,
                team=team,
                total_managed_matches=(len(promise_plan.teams) - 1) * promise_plan.legs,
                roster=historical_roster,
                plan=frozen_plan,
                control=str(contract.get("control") or ""),
            )
            if contract != expected_contract:
                raise ValueError("player promise contract source replay mismatch")
        for target_id, target_state in season_sources.items():
            target_plan = SeasonPlan.from_payload(target_state.get("plan") or {})
            if target_plan.manager_sporting_directive is None:
                continue
            target_index = int(str(target_id).split("-")[-1])
            source_id = f"season-{target_index - 1:04d}"
            source_archive = archives.get(source_id)
            if source_archive is None:
                continue
            expected_brief = self._sporting_brief_for_transition(
                session,
                source_archive,
                team=str(target_plan.manager_team),
                target_season_id=target_id,
            )
            if target_state.get("sporting_brief") != expected_brief:
                raise ValueError("sporting brief source replay mismatch")
        for lifecycle_team, club in lifecycle_clubs.items():
            for transaction in club.get("transactions") or []:
                target_state = season_sources.get(
                    str(transaction.get("target_season_id") or "")
                )
                if target_state is None:
                    continue
                target_plan = SeasonPlan.from_payload(target_state["plan"])
                manager_controlled = target_plan.manager_team == lifecycle_team
                expected_plan = (
                    target_plan.manager_retention.as_dict()
                    if manager_controlled and target_plan.manager_retention is not None
                    else None
                )
                if (
                    transaction.get("control")
                    != ("manager" if manager_controlled else "ai")
                    or transaction.get("plan") != expected_plan
                ):
                    raise ValueError("player lifecycle season plan mismatch")
        for market_transition in (session.get("player_market") or {}).get(
            "transitions"
        ) or []:
            source_id = str(market_transition.get("source_season_id") or "")
            target_id = str(market_transition.get("target_season_id") or "")
            target_state = season_sources.get(str(target_id))
            if target_state is None:
                continue
            target_plan = SeasonPlan.from_payload(target_state["plan"])
            manager_signings = [
                signing
                for signing in market_transition.get("signings") or []
                if signing.get("control") == "manager"
            ]
            expected_manager_plan = (
                target_plan.manager_free_agent.as_dict()
                if target_plan.manager_free_agent is not None
                else None
            )
            if len(manager_signings) != (1 if expected_manager_plan else 0) or (
                expected_manager_plan is not None
                and (
                    manager_signings[0].get("team") != target_plan.manager_team
                    or manager_signings[0].get("plan") != expected_manager_plan
                )
            ):
                raise ValueError("global market season plan mismatch")
            source_archive = archives.get(source_id)
            if source_archive is None:
                continue
            source_plan = SeasonPlan.from_payload(source_archive["plan"])
            continuing = sorted(set(source_plan.teams) & set(target_plan.teams))
            target_index = int(target_id.split("-")[-1])
            prior_market = {
                "schema_version": 1,
                "transitions": [
                    copy.deepcopy(item)
                    for item in (session.get("player_market") or {}).get(
                        "transitions", []
                    )
                    if int(str(item["target_season_id"]).split("-")[-1]) < target_index
                ],
            }
            replay_window = open_market_window(
                prior_market,
                target_season_id=target_id,
            )
            replay_rosters = {}
            for market_team in continuing:
                base_roster = load_roster_json(
                    roster_path_for_team(
                        str(self.root),
                        market_team,
                    )
                )
                if base_roster is None:
                    continue
                prior_development = {
                    "schema_version": 1,
                    "clubs": {
                        market_team: {
                            "transactions": [
                                copy.deepcopy(item)
                                for item in (
                                    (development_clubs.get(market_team) or {}).get(
                                        "transactions", []
                                    )
                                )
                                if int(str(item["target_season_id"]).split("-")[-1])
                                < target_index
                            ]
                        }
                    },
                }
                prior_lifecycle = {
                    "schema_version": 1,
                    "clubs": {
                        market_team: {
                            "transactions": [
                                copy.deepcopy(item)
                                for item in (
                                    (lifecycle_clubs.get(market_team) or {}).get(
                                        "transactions", []
                                    )
                                )
                                if int(str(item["target_season_id"]).split("-")[-1])
                                < target_index
                            ]
                        }
                    },
                }
                replay_rosters[market_team] = replay_squad_registry_through(
                    base_roster,
                    team=market_team,
                    registry=session.get("squad_registry"),
                    season_id=target_id,
                    development_registry=prior_development,
                    lifecycle_registry=prior_lifecycle,
                    market_registry=prior_market,
                )
            expected_signings = []
            if target_plan.manager_free_agent is not None:
                manager_team = str(target_plan.manager_team or "")
                replay_window, replay_rosters[manager_team], signing = (
                    execute_free_agent_signing(
                        replay_window,
                        replay_rosters[manager_team],
                        team=manager_team,
                        plan=target_plan.manager_free_agent,
                        control="manager",
                    )
                )
                expected_signings.append(signing)
            expected_ai_decisions = []
            for market_team in (
                team
                for team in continuing
                if team != target_plan.manager_team and team in replay_rosters
            ):
                decision = ai_free_agent_decision(
                    replay_window,
                    replay_rosters[market_team],
                    team=market_team,
                )
                expected_ai_decisions.append(decision)
                if decision["plan"] is not None:
                    replay_window, replay_rosters[market_team], signing = (
                        execute_free_agent_signing(
                            replay_window,
                            replay_rosters[market_team],
                            team=market_team,
                            plan=FreeAgentPlan.from_payload(decision["plan"]),
                            control="ai",
                        )
                    )
                    expected_signings.append(signing)
            if (
                market_transition.get("signings") != expected_signings
                or market_transition.get("ai_decisions") != expected_ai_decisions
            ):
                raise ValueError("global market source replay mismatch")
            expected_entries = [
                pool_entry_from_player(
                    signing["outgoing"],
                    origin_team=str(signing["team"]),
                    entered_season_id=target_id,
                    retirement_age=retirement_age_for_player(
                        str(signing["team"]),
                        str(signing["outgoing"]["player_id"]),
                    ),
                    reason="squad_replacement",
                )
                for signing in expected_signings
            ]
            for lifecycle_team in continuing:
                matches = [
                    item
                    for item in (
                        (lifecycle_clubs.get(lifecycle_team) or {}).get(
                            "transactions", []
                        )
                    )
                    if item.get("target_season_id") == target_id
                ]
                if len(matches) != 1:
                    continue
                for exit_row in matches[0]["exits"]:
                    if exit_row["reason"] == "contract_released":
                        expected_entries.append(
                            pool_entry_from_player(
                                exit_row["player"],
                                origin_team=lifecycle_team,
                                entered_season_id=target_id,
                                retirement_age=int(exit_row["retirement_age"]),
                                reason="contract_released",
                            )
                        )
            if market_transition.get("entries") != expected_entries:
                raise ValueError("global market entry source mismatch")
        for outcome in (session.get("scouting_outcomes") or {}).get("outcomes") or []:
            season_id = str(outcome.get("season_id") or "")
            source_archive = archives.get(season_id)
            if source_archive is None:
                continue
            transitions = [
                item
                for item in (session.get("player_market") or {}).get("transitions", [])
                if item.get("target_season_id") == season_id
            ]
            if len(transitions) != 1:
                raise ValueError("scouting outcome market source is unavailable")
            market_transition = transitions[0]
            signings = [
                item
                for item in market_transition.get("signings") or []
                if evidence_identity(item) == outcome.get("signing_identity")
            ]
            if len(signings) != 1:
                raise ValueError("scouting outcome signing source mismatch")
            signing = signings[0]
            team = str(signing.get("team") or "")
            participation = (
                source_archive.get("player_development_evidence") or {}
            ).get(team)
            finance_entries = (finance_clubs.get(team) or {}).get("entries") or []
            settlements = [
                entry.get("evidence")
                for entry in finance_entries
                if entry.get("type") == "season_settlement"
                and entry.get("season_id") == season_id
            ]
            if not isinstance(participation, Mapping) or len(settlements) != 1:
                raise ValueError("scouting outcome season evidence is unavailable")
            observation, observation_source = resolve_signing_observation(
                signing,
                market_transition,
                session.get("scouting_registry"),
            )
            expected = build_scouting_outcome(
                signing,
                observation=observation,
                observation_source=observation_source,
                archive=source_archive,
                participation=participation,
                settlement=settlements[0],
            )
            if outcome != expected:
                raise ValueError("scouting outcome source replay mismatch")
        for review in (session.get("sporting_reviews") or {}).get("reviews") or []:
            season_id = str(review.get("season_id") or "")
            source_archive = archives.get(season_id)
            if source_archive is None:
                continue
            source_plan = SeasonPlan.from_payload(source_archive["plan"])
            team = str(source_plan.manager_team or "")
            participation = (
                source_archive.get("player_development_evidence") or {}
            ).get(team)
            settlements = [
                entry.get("evidence")
                for entry in (finance_clubs.get(team) or {}).get("entries") or []
                if entry.get("type") == "season_settlement"
                and entry.get("season_id") == season_id
            ]
            base_roster = load_roster_json(
                roster_path_for_team(
                    str(self.root),
                    team,
                )
            )
            completed_roster = (
                replay_squad_registry_through(
                    base_roster,
                    team=team,
                    registry=session.get("squad_registry"),
                    season_id=season_id,
                    development_registry=session.get("player_development"),
                    lifecycle_registry=session.get("player_lifecycle"),
                    market_registry=session.get("player_market"),
                )
                if base_roster is not None
                else None
            )
            signings = [
                signing
                for transition in (session.get("player_market") or {}).get(
                    "transitions", []
                )
                if transition.get("target_season_id") == season_id
                for signing in transition.get("signings") or []
                if signing.get("team") == team
            ]
            if (
                not isinstance(participation, Mapping)
                or len(settlements) != 1
                or completed_roster is None
            ):
                raise ValueError("sporting review source evidence is unavailable")
            expected_review = build_sporting_review(
                source_archive,
                completed_roster,
                participation=participation,
                settlement=settlements[0],
                free_agent_signings=signings,
            )
            if review != expected_review:
                raise ValueError("sporting review source replay mismatch")
        ecosystem = session.get("league_ecosystem") or {}
        for event in ecosystem.get("transitions") or []:
            source_id = str(event.get("from_season_id") or "")
            target_id = str(event.get("to_season_id") or "")
            source_archive = archives.get(source_id)
            target_state = season_sources.get(target_id)
            if source_archive is None or target_state is None:
                continue
            source_plan = SeasonPlan.from_payload(source_archive["plan"])
            target_plan = SeasonPlan.from_payload(target_state["plan"])
            participating = sorted(set(source_plan.teams) & set(target_plan.teams))
            controlled_team = (
                target_plan.manager_team
                if target_plan.manager_team in participating
                else None
            )
            expected_clubs = []
            for ai_team in (team for team in participating if team != controlled_team):
                finance_entries = (finance_clubs.get(ai_team) or {}).get(
                    "entries"
                ) or []
                settlement_matches = [
                    entry
                    for entry in finance_entries
                    if entry.get("type") == "season_settlement"
                    and entry.get("season_id") == source_id
                ]
                if len(settlement_matches) != 1:
                    raise ValueError("league ecosystem settlement evidence mismatch")
                settlement_entry = settlement_matches[0]
                base_roster = load_roster_json(
                    roster_path_for_team(
                        str(self.root),
                        ai_team,
                    )
                )
                historical_roster = (
                    replay_squad_registry_through(
                        base_roster,
                        team=ai_team,
                        registry=session.get("squad_registry"),
                        season_id=source_id,
                        development_registry=session.get("player_development"),
                        lifecycle_registry=session.get("player_lifecycle"),
                        market_registry=session.get("player_market"),
                    )
                    if base_roster is not None
                    else None
                )
                decision = ai_club_recruitment_decision(
                    source_archive,
                    historical_roster,
                    team=ai_team,
                    next_season_index=int(target_id.split("-")[-1]),
                    allowance=recruitment_allowance(
                        int(settlement_entry["balance_after"]),
                    ),
                )
                transactions = [
                    transaction
                    for transaction in (squad_clubs.get(ai_team) or {}).get(
                        "transactions", []
                    )
                    if transaction.get("season_id") == target_id
                ]
                if decision["plan"] is None:
                    if transactions:
                        raise ValueError("unexpected AI recruitment transaction")
                    transaction_identity = None
                else:
                    if (
                        len(transactions) != 1
                        or transactions[0].get("plan") != decision["plan"]
                    ):
                        raise ValueError("AI recruitment policy mismatch")
                    transaction_identity = evidence_identity(transactions[0])
                expected_clubs.append(
                    {
                        "team": ai_team,
                        "settlement_identity": evidence_identity(
                            settlement_entry["evidence"],
                        ),
                        "decision": decision,
                        "squad_transaction_identity": transaction_identity,
                    }
                )
            expected_event = {
                "schema_version": 1,
                "from_season_id": source_id,
                "to_season_id": target_id,
                "archive_identity": evidence_identity(source_archive),
                "manager_controlled_team": controlled_team,
                "participating_teams": participating,
                "ai_clubs": expected_clubs,
                "claim_boundary": (
                    "deterministic bounded AI-club evolution from simulated "
                    "standings, game-credit finance, and fictional markets"
                ),
            }
            if dict(event) != expected_event:
                raise ValueError("league ecosystem source replay mismatch")
        strategy_states = list(archives.values())
        if isinstance(season, Mapping):
            strategy_states.append(season)
        ecosystem_transitions = list(ecosystem.get("transitions") or [])
        for strategy_state in strategy_states:
            snapshot = strategy_state.get("club_strategies")
            if snapshot is None:
                continue
            target_id = str(strategy_state.get("season_id") or "")
            target_plan = SeasonPlan.from_payload(strategy_state["plan"])
            source_id = snapshot.get("source_season_id")
            source_archive = None
            previous_snapshot = None
            transition = None
            if source_id is not None:
                source_archive = archives.get(str(source_id))
                if source_archive is None:
                    continue
                previous_snapshot = source_archive.get("club_strategies")
                transition_matches = [
                    item
                    for item in ecosystem_transitions
                    if item.get("from_season_id") == source_id
                    and item.get("to_season_id") == target_id
                ]
                if len(transition_matches) != 1:
                    raise ValueError("club strategy ecosystem source mismatch")
                transition = transition_matches[0]
            strategy_rosters = {}
            recruitment_moves = {}
            for strategy_team in sorted(target_plan.teams):
                base_roster = load_roster_json(
                    roster_path_for_team(
                        str(self.root),
                        strategy_team,
                    )
                )
                strategy_rosters[strategy_team] = (
                    replay_squad_registry_through(
                        base_roster,
                        team=strategy_team,
                        registry=session.get("squad_registry"),
                        season_id=target_id,
                        development_registry=session.get("player_development"),
                        lifecycle_registry=session.get("player_lifecycle"),
                        market_registry=session.get("player_market"),
                    )
                    if base_roster is not None
                    else None
                )
                target_transactions = [
                    transaction
                    for transaction in (squad_clubs.get(strategy_team) or {}).get(
                        "transactions", []
                    )
                    if transaction.get("season_id") == target_id
                ]
                recruitment_moves[strategy_team] = sum(
                    len(transaction.get("moves") or [])
                    for transaction in target_transactions
                )
            expected_snapshot = build_season_club_strategies(
                season_id=target_id,
                teams=target_plan.teams,
                manager_team=target_plan.manager_team,
                rosters=strategy_rosters,
                source_archive=source_archive,
                previous_snapshot=previous_snapshot,
                ecosystem_transition=transition,
                recruitment_moves=recruitment_moves,
            )
            if snapshot != expected_snapshot:
                raise ValueError("club strategy source replay mismatch")
            fixtures = strategy_state.get("fixtures")
            if not isinstance(fixtures, list):
                continue
            fixtures_by_id = {
                str(fixture.get("fixture_id") or ""): fixture
                for fixture in fixtures
                if isinstance(fixture, Mapping)
            }
            for match_record in session.get("matches") or []:
                match_plan = (
                    match_record.get("match_plan")
                    if isinstance(match_record, Mapping)
                    else None
                )
                competition = (
                    match_plan.get("competition")
                    if isinstance(match_plan, Mapping)
                    else None
                )
                if (
                    not isinstance(competition, Mapping)
                    or competition.get("season_id") != target_id
                ):
                    continue
                fixture = fixtures_by_id.get(str(competition.get("fixture_id") or ""))
                if fixture is None:
                    raise ValueError("club strategy match fixture mismatch")
                expected_fixture_strategy = resolve_fixture_club_strategy(
                    snapshot,
                    home=str(fixture["home"]),
                    away=str(fixture["away"]),
                    manager_decision=fixture.get("manager_decision"),
                    opponent_preparation=fixture.get("opponent_preparation"),
                )
                if (
                    match_plan.get("club_strategy") != expected_fixture_strategy
                    or match_plan.get("home_tactic")
                    != expected_fixture_strategy["home_tactic"]
                    or match_plan.get("away_tactic")
                    != expected_fixture_strategy["away_tactic"]
                ):
                    raise ValueError("completed match club strategy mismatch")
        return session

    def evidence(self) -> dict[str, Any]:
        def read(relative: str) -> dict:
            path = self.root / relative
            return (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )

        release = read("data/releases/current.json")
        staged = read("data/evaluation/staged_completion_v1.json")
        phase5 = read("data/evaluation/phase5_research_layer_validation_v1.json")
        action_protocol = read("data/evaluation/action_adoption_protocol_v1.json")
        cross_protocol = read(
            "data/evaluation/cross_action_validation_protocol_v1.json"
        )
        cross_verification = read(
            "data/evaluation/cross_action_validation_verification_v1.json"
        )
        cross_identity_verified = False
        if cross_protocol and cross_verification:
            try:
                cross_protocol_path = (
                    self.root
                    / "data/evaluation/cross_action_validation_protocol_v1.json"
                ).resolve()
                cross_candidate = cross_protocol.get("candidate") or {}
                cross_checkpoint = (
                    self.root / str(cross_candidate.get("checkpoint") or "")
                ).resolve()
                cross_code_sha = {}
                for relative in cross_protocol.get("required_code_paths") or []:
                    path = (self.root / str(relative)).resolve()
                    path.relative_to(self.root.resolve())
                    cross_code_sha[str(relative)] = file_sha256(path)
                expected_cross_identity = {
                    "protocol_sha256": file_sha256(cross_protocol_path),
                    "checkpoint_sha256": file_sha256(cross_checkpoint),
                    "code_sha256": cross_code_sha,
                }
                cross_identity_verified = (
                    cross_verification.get("execution_identity")
                    == expected_cross_identity
                )
            except (OSError, TypeError, ValueError):
                cross_identity_verified = False
        manager_advisor_protocol = read(
            "data/evaluation/manager_advisor_protocol_v1.json"
        )
        action_progress = read("data/evaluation/action_adoption_v1/progress.json")
        action_decision = read("data/evaluation/action_adoption_v1/decision.json")
        outcome_protocol = read("data/evaluation/action_outcome_protocol_v1.json")
        outcome_progress = read("data/evaluation/action_outcome_v1/progress.json")
        outcome_decision = read("data/evaluation/action_outcome_v1/decision.json")
        m2_protocol_relative = (
            "data/evaluation/m2_mirrored_policy_protocol_v1.json"
        )
        m2_protocol = read(m2_protocol_relative)
        m2_preflight = read(
            "data/evaluation/m2_training_preflight_v1.json"
        )
        m2_candidate_report = read(
            "data/evaluation/m2_candidate_eligibility_v1.json"
        )
        m2_progress = read(
            "data/evaluation/m2_mirrored_policy_progress_v1.json"
        )
        m2_preflight_identity_verified = False
        m2_preflight_identity_reason = (
            "not_recorded" if not m2_preflight else "missing_evidence_identity"
        )
        if m2_protocol and m2_preflight.get("evidence_identity"):
            try:
                validate_m2_preflight_receipt(m2_preflight, m2_protocol)
                manifest_relative = str(
                    (m2_protocol.get("candidate") or {})["dataset_manifest"]
                )
                expected_preflight_identity = study_preflight_identity(
                    self.root,
                    self.root / m2_protocol_relative,
                    m2_protocol,
                    self.root / manifest_relative,
                )
                m2_preflight_identity_verified = bool(
                    m2_preflight.get("schema_version") == 1
                    and m2_preflight.get("protocol_id")
                    == m2_protocol.get("protocol_id")
                    and m2_preflight.get("evidence_identity")
                    == expected_preflight_identity
                )
                m2_preflight_identity_reason = (
                    "current_protocol_manifest_and_training_code_verified"
                    if m2_preflight_identity_verified else
                    "protocol_manifest_or_training_code_identity_stale"
                )
            except (KeyError, OSError, TypeError, ValueError):
                m2_preflight_identity_reason = (
                    "invalid_or_unavailable_preflight_identity"
                )
        m2_candidate_source = (
            m2_progress if m2_progress.get("execution_identity")
            else m2_candidate_report
        )
        m2_stored_identity = (
            m2_candidate_source.get("execution_identity") or {}
        )
        m2_candidate_contract_valid = bool(
            m2_progress.get("execution_identity")
        )
        if m2_candidate_report and not m2_progress.get("execution_identity"):
            try:
                validate_m2_candidate_receipt(
                    m2_candidate_report, m2_protocol,
                )
                m2_candidate_contract_valid = True
            except (KeyError, TypeError, ValueError):
                m2_candidate_contract_valid = False
        m2_identity_verified = False
        m2_identity_reason = (
            "not_started" if m2_protocol and not m2_candidate_source
            else "missing_execution_identity"
        )
        if m2_protocol and m2_stored_identity:
            try:
                checkpoint_relative = str(
                    m2_stored_identity["checkpoint_path"]
                )
                recomputed_m2_identity = study_execution_identity(
                    self.root,
                    self.root / m2_protocol_relative,
                    m2_protocol,
                    self.root / checkpoint_relative,
                )
                m2_identity_match = (
                    recomputed_m2_identity == m2_stored_identity
                )
                m2_identity_verified = bool(
                    m2_candidate_contract_valid and m2_identity_match
                )
                if m2_identity_verified:
                    m2_identity_reason = (
                        "current_protocol_checkpoint_code_and_input_identity_"
                        "verified"
                    )
                elif not m2_candidate_contract_valid:
                    m2_identity_reason = (
                        "invalid_candidate_qualification_receipt"
                    )
                else:
                    m2_identity_reason = (
                        "protocol_checkpoint_code_or_input_identity_stale"
                    )
            except (KeyError, OSError, TypeError, ValueError):
                m2_identity_reason = "invalid_or_unavailable_execution_identity"
        m2_analysis = (
            m2_progress.get("analysis")
            if isinstance(m2_progress.get("analysis"), Mapping)
            else {}
        )
        m2_result_current = bool(
            m2_identity_verified
            and m2_progress.get("state") == "completed"
            and m2_analysis.get("state") == "completed"
        )
        if m2_analysis and not m2_result_current:
            m2_execution_state = "stale_current_code_or_input_identity"
        elif m2_progress:
            m2_execution_state = m2_progress.get("state", "invalid_progress")
        elif m2_candidate_report:
            m2_execution_state = (
                "candidate_qualified_awaiting_execution"
                if m2_identity_verified
                and m2_candidate_report.get("candidate_eligible") is True
                else "candidate_rejected"
                if m2_identity_verified else
                m2_identity_reason
            )
        else:
            m2_execution_state = m2_protocol.get("state", "absent")
        m2_runs_executed = sum(
            len((row or {}).get("rows") or [])
            for row in (m2_progress.get("arms") or {}).values()
        )
        m2_fixed_run_budget = int(
            (m2_protocol.get("design") or {}).get("runs_total") or 0
        )
        m2_candidate_eligibility = (
            m2_candidate_source.get("candidate_eligibility") or {}
        )
        m2_candidate_eligible = bool(
            m2_identity_verified
            and m2_candidate_eligibility.get("eligible", False)
        )
        m2_preflight_current = bool(
            m2_preflight_identity_verified
            and m2_preflight.get("status")
            in {"ready_for_m2_training", "blocked_before_training"}
        )
        m2_preflight_ready = bool(
            m2_preflight_current
            and m2_preflight.get("ready") is True
            and m2_preflight.get("status") == "ready_for_m2_training"
        )
        m2_research_control = build_m2_research_control(
            protocol_available=bool(m2_protocol),
            protocol_state=m2_protocol.get("state"),
            preflight_available=bool(m2_preflight),
            preflight_current=m2_preflight_current,
            preflight_ready=m2_preflight_ready,
            preflight_status=m2_preflight.get("status"),
            frozen_training_command=m2_preflight.get(
                "frozen_training_command"
            ),
            candidate_evidence_available=bool(m2_candidate_source),
            candidate_evidence_current=m2_identity_verified,
            candidate_eligible=m2_candidate_eligible,
            checkpoint_path=m2_stored_identity.get("checkpoint_path"),
            execution_state=m2_execution_state,
            runs_executed=m2_runs_executed,
            fixed_run_budget=m2_fixed_run_budget,
            result_current=m2_result_current,
            result_status=m2_analysis.get("decision"),
            promotion_supported=bool(
                m2_result_current
                and m2_analysis.get("promotion_supported", False)
            ),
            promotion_gates=m2_analysis.get("promotion_gates"),
        )
        mechanism_identity = _formal_evidence_identity(
            self.root,
            "data/evaluation/action_adoption_protocol_v1.json",
            action_protocol, action_progress, action_decision,
        )
        outcome_identity = _formal_evidence_identity(
            self.root,
            "data/evaluation/action_outcome_protocol_v1.json",
            outcome_protocol, outcome_progress, outcome_decision,
        )
        mechanism_current = bool(mechanism_identity["verified"])
        outcome_current = bool(outcome_identity["verified"])
        if mechanism_current:
            mechanism_execution_state = action_decision.get("status")
        elif action_decision:
            mechanism_execution_state = "stale_current_code_identity"
        else:
            mechanism_execution_state = action_progress.get(
                "state", "ready_not_started" if action_protocol else "absent",
            )
        if outcome_current:
            outcome_execution_state = outcome_decision.get("decision")
        elif outcome_decision:
            outcome_execution_state = "stale_current_code_identity"
        else:
            outcome_execution_state = outcome_progress.get(
                "state", "not_started" if outcome_protocol else "absent",
            )
        candidate = phase5.get("world_model_candidate") or {}
        live_llm = phase5.get("live_llm_evidence") or {}
        frozen_shot_decision = read(
            "data/evaluation/frozen_shot_head_decision.json"
        )
        sealed_shot = (candidate.get("sealed_test") or {}).get("shot") or {}
        frozen_shot_artifact = _artifact_path(
            self.root, frozen_shot_decision.get("promotion_artifact")
        )
        frozen_shot_identity_verified = bool(
            frozen_shot_decision.get("accepted")
            and frozen_shot_artifact is not None
            and frozen_shot_artifact.is_file()
            and frozen_shot_decision.get("promotion_artifact_sha256")
            and file_sha256(frozen_shot_artifact)
            == frozen_shot_decision.get("promotion_artifact_sha256")
        )
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
            "frozen_shot_head": frozen_shot_decision,
            "shot_action_validation": {
                "available": bool(sealed_shot or frozen_shot_decision),
                "status": (
                    "sealed_frozen_head_authorized"
                    if frozen_shot_identity_verified
                    else "physics_xg_fallback_joint_head_not_authorized"
                ),
                "probability_source": (
                    "frozen_backbone_shot_head"
                    if frozen_shot_identity_verified else "physics_xg_prior"
                ),
                "joint_head_authorized": False,
                "frozen_head_authorized": frozen_shot_identity_verified,
                "sealed_samples": int(sealed_shot.get("samples", 0) or 0),
                "sealed_goals": int(sealed_shot.get("goals", 0) or 0),
                "model_brier": sealed_shot.get("model_brier"),
                "physics_xg_prior_brier": sealed_shot.get(
                    "physics_xg_prior_brier"
                ),
                "skill_vs_physics_xg_prior": sealed_shot.get(
                    "skill_vs_physics_xg_prior"
                ),
                "claim_scope": (
                    "sealed_simulator_shot_probability_proper_score_only_"
                    "no_match_outcome_or_real_football_claim"
                ),
            },
            "live_llm_evaluable": live_llm.get("evaluable", False),
            "action_adoption_mechanism": {
                "available": bool(action_protocol),
                "historical_result_available": bool(action_decision),
                "result_identity_verified": mechanism_current,
                "result_applicable_to_current_code": mechanism_current,
                "identity_reason": mechanism_identity["reason"],
                "protocol_id": action_protocol.get("protocol_id"),
                "claim_scope": action_protocol.get("claim_scope"),
                "protocol_state": action_protocol.get("state"),
                "execution_state": mechanism_execution_state,
                "runs_executed": sum(
                    len((row or {}).get("rows") or [])
                    for row in (action_progress.get("arms") or {}).values()
                ),
                "fixed_run_budget": (
                    (action_protocol.get("design") or {}).get("runs_total")
                ),
                "historical_result_status": action_decision.get("status"),
                "result_status": (
                    action_decision.get("status") if mechanism_current else None
                ),
                "passed": action_decision.get("passed") if mechanism_current else None,
                "mechanism": (
                    dict(action_decision.get("mechanism") or {})
                    if mechanism_current else {}
                ),
                "promotion_authorized": bool(
                    mechanism_current
                    and action_decision.get("promotion_authorized", False)
                ),
            },
            "cross_action_validation": {
                "available": bool(cross_protocol and cross_verification),
                "result_identity_verified": cross_identity_verified,
                "protocol_id": cross_protocol.get("protocol_id"),
                "protocol_state": cross_protocol.get("state"),
                "status": (
                    cross_verification.get("status")
                    if cross_identity_verified else "stale_current_code_identity"
                ),
                "code_ready": bool(
                    cross_identity_verified
                    and cross_verification.get("code_ready", False)
                ),
                "checkpoint_identity_verified": bool(
                    cross_verification.get("checkpoint_identity_verified", False)
                ),
                "cross_planning_authorized": bool(
                    cross_identity_verified
                    and cross_verification.get("cross_planning_authorized", False)
                ),
                "runtime_cross_quality": float(
                    cross_verification.get("runtime_cross_quality", 0.0) or 0.0
                    if cross_identity_verified else 0.0
                ),
                "checkpoint_reason": (
                    cross_verification.get("checkpoint_cross_validation") or {}
                ).get("reason"),
                "minimum_samples": (
                    cross_protocol.get("validation") or {}
                ).get("minimum_samples"),
                "minimum_groups": (
                    cross_protocol.get("validation") or {}
                ).get("minimum_groups"),
                "minimum_skill_vs_persistence": (
                    cross_protocol.get("validation") or {}
                ).get("minimum_skill_vs_persistence"),
                "claim_scope": cross_protocol.get("claim_scope"),
                "training_executed": bool(
                    cross_verification.get("training_executed", False)
                ),
                "matches_executed": int(
                    cross_verification.get("matches_executed", 0) or 0
                ),
                "provider_calls_made": bool(
                    cross_verification.get("provider_calls_made", False)
                ),
            },
            "action_outcome_study": {
                "available": bool(outcome_protocol),
                "historical_result_available": bool(outcome_decision),
                "result_identity_verified": outcome_current,
                "result_applicable_to_current_code": outcome_current,
                "identity_reason": outcome_identity["reason"],
                "protocol_id": outcome_protocol.get("protocol_id"),
                "claim_scope": (
                    "full_match_simulator_outcome_not_real_football_causality"
                ),
                "execution_state": outcome_execution_state,
                "runs_executed": sum(
                    len((row or {}).get("rows") or [])
                    for row in (outcome_progress.get("arms") or {}).values()
                ),
                "fixed_run_budget": (
                    (outcome_protocol.get("design") or {}).get("runs_total")
                ),
                "historical_result_status": outcome_decision.get("decision"),
                "promotion_supported": bool(
                    outcome_current
                    and outcome_decision.get("promotion_supported", False)
                ),
                "result_status": (
                    outcome_decision.get("decision") if outcome_current else None
                ),
                "pairs_total": (
                    outcome_decision.get("pairs_total") if outcome_current else None
                ),
                "primary": (
                    dict(outcome_decision.get("primary") or {})
                    if outcome_current else {}
                ),
                "minimum_meaningful_delta_loss": (
                    outcome_decision.get("minimum_meaningful_delta_loss")
                    if outcome_current else None
                ),
                "behavior": (
                    dict(outcome_decision.get("behavior") or {})
                    if outcome_current else {}
                ),
                "promotion_gates": (
                    dict(outcome_decision.get("promotion_gates") or {})
                    if outcome_current else {}
                ),
            },
            "outcome_aligned_m2_study": {
                "available": bool(m2_protocol),
                "protocol_id": m2_protocol.get("protocol_id"),
                "claim_scope": m2_protocol.get("claim_scope"),
                "protocol_state": m2_protocol.get("state"),
                "training_contract": dict(m2_protocol.get("training") or {}),
                "required_sealed_validation": list(
                    (m2_protocol.get("candidate") or {}).get(
                        "required_sealed_validation"
                    ) or []
                ),
                "execution_state": m2_execution_state,
                "checkpoint_bound": bool(m2_stored_identity.get("checkpoint_path")),
                "candidate_eligible": m2_candidate_eligible,
                "training_preflight": {
                    "available": bool(m2_preflight),
                    "result_identity_verified": (
                        m2_preflight_identity_verified
                    ),
                    "result_applicable_to_current_code": m2_preflight_current,
                    "identity_reason": m2_preflight_identity_reason,
                    "status": (
                        m2_preflight.get("status")
                        if m2_preflight_current else None
                    ),
                    "ready": m2_preflight_ready,
                    "checks": (
                        dict(m2_preflight.get("checks") or {})
                        if m2_preflight_current else {}
                    ),
                    "training_executed": False,
                    "claim_scope": (
                        "zero_training_readiness_only_no_model_or_outcome_evidence"
                    ),
                },
                "candidate_qualification": {
                    "available": bool(m2_candidate_source),
                    "source": (
                        "formal_progress"
                        if m2_progress.get("execution_identity")
                        else "qualification_report"
                        if m2_candidate_report else None
                    ),
                    "result_identity_verified": m2_identity_verified,
                    "eligible": m2_candidate_eligible,
                    "checkpoint_path": (
                        m2_stored_identity.get("checkpoint_path")
                        if m2_identity_verified else None
                    ),
                    "details": (
                        dict(m2_candidate_eligibility)
                        if m2_identity_verified else {}
                    ),
                    "claim_scope": (
                        "checkpoint_qualification_only_no_policy_effect_or_"
                        "outcome_claim"
                    ),
                },
                "historical_result_available": bool(m2_analysis),
                "result_identity_verified": m2_identity_verified,
                "result_applicable_to_current_code": m2_result_current,
                "identity_reason": m2_identity_reason,
                "runs_executed": m2_runs_executed,
                "fixed_run_budget": m2_fixed_run_budget,
                "result_status": (
                    m2_analysis.get("decision") if m2_result_current else None
                ),
                "promotion_supported": bool(
                    m2_result_current
                    and m2_analysis.get("promotion_supported", False)
                ),
                "primary": (
                    dict(m2_analysis.get("primary_controlled_micro_xg_effect") or {})
                    if m2_result_current else {}
                ),
                "mechanism": (
                    dict(m2_analysis.get("mechanism_realized_policy_utility") or {})
                    if m2_result_current else {}
                ),
                "mechanism_coverage": (
                    dict(m2_analysis.get("mechanism_attributable_coverage") or {})
                    if m2_result_current else {}
                ),
                "expected_counterfactual_changes": (
                    dict(
                        m2_analysis.get(
                            "mechanism_expected_counterfactual_changes"
                        ) or {}
                    ) if m2_result_current else {}
                ),
                "external_validity": (
                    dict(
                        m2_analysis.get(
                            "external_continuous_calibration_noninferiority"
                        ) or {}
                    ) if m2_result_current else {}
                ),
                "secondary_goal_difference": (
                    dict(m2_analysis.get("secondary_goal_difference_effect") or {})
                    if m2_result_current else {}
                ),
                "promotion_gates": (
                    dict(m2_analysis.get("promotion_gates") or {})
                    if m2_result_current else {}
                ),
                "runtime_row_identity": (
                    dict(m2_analysis.get("runtime_row_identity") or {})
                    if m2_result_current else {}
                ),
                "experimental_unit_identity": (
                    dict(m2_analysis.get("experimental_unit_identity") or {})
                    if m2_result_current else {}
                ),
                "research_control": m2_research_control,
            },
            "manager_advisor_adoption": {
                "available": bool(manager_advisor_protocol),
                "protocol_id": manager_advisor_protocol.get("protocol_id"),
                "claim_scope": manager_advisor_protocol.get("claim_scope"),
                "protocol_state": manager_advisor_protocol.get("state"),
                "fixed_information_windows": list(
                    (manager_advisor_protocol.get("analysis") or {}).get(
                        "fixed_information_windows"
                    )
                    or []
                ),
                "results_available": (
                    manager_advisor_protocol.get("execution") or {}
                ).get("results_available", False),
                "causal_effect_authorized": (
                    manager_advisor_protocol.get("analysis") or {}
                ).get("causal_effect_authorized", False),
                "promotion_authorized": (
                    manager_advisor_protocol.get("decision_rules") or {}
                ).get("product_or_academic_promotion_authorized", False),
            },
            "production_promotion_ready": staged.get(
                "production_promotion_ready", False
            ),
        }

    def readiness(self) -> dict[str, Any]:
        evidence = self.evidence()
        checkpoint = _artifact_path(
            self.root,
            (
                evidence.get("world_model_checkpoint")
                or "data/world_model/latent_wm_rollout_calibrated_candidate.pt"
            ),
        )
        release_manifest = _artifact_path(
            self.root,
            evidence.get("stable_release_manifest"),
        )
        expected_release_sha256 = str(
            evidence.get("stable_release_manifest_sha256") or ""
        )
        release_identity_verified = bool(
            release_manifest is not None
            and release_manifest.is_file()
            and expected_release_sha256
            and portable_text_hash_matches(
                release_manifest,
                expected_release_sha256,
            )
        )
        release_artifact_verification: dict[str, Any] = {
            "ok": False,
            "artifacts": 0,
            "failures": [{"reason": "release_manifest_identity_unverified"}],
        }
        if release_identity_verified and release_manifest is not None:
            try:
                release_payload = json.loads(
                    release_manifest.read_text(encoding="utf-8-sig")
                )
                release_artifact_verification = verify_artifact_manifest(
                    self.root,
                    release_payload,
                )
            except (OSError, ValueError, TypeError) as exc:
                release_artifact_verification = {
                    "ok": False,
                    "artifacts": 0,
                    "failures": [
                        {
                            "reason": "invalid_release_manifest",
                            "error_type": type(exc).__name__,
                        }
                    ],
                }
        expected_checkpoint_sha256 = str(
            evidence.get("world_model_checkpoint_sha256") or ""
        )
        actual_checkpoint_sha256 = (
            file_sha256(checkpoint)
            if checkpoint is not None and checkpoint.is_file()
            else None
        )
        shot_decision = evidence.get("frozen_shot_head") or {}
        shot_artifact_value = shot_decision.get("promotion_artifact")
        shot_artifact = _artifact_path(self.root, shot_artifact_value)
        shot_identity_verified = not shot_decision.get("accepted") or bool(
            shot_artifact is not None
            and shot_artifact.is_file()
            and shot_decision.get("promotion_artifact_sha256")
            and file_sha256(shot_artifact)
            == shot_decision.get("promotion_artifact_sha256")
        )
        from src.simulation.llm_gateway import (
            LLMGatewayConfig,
            llm_credentials_available,
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
            "research_checkpoint_accepted": bool(
                evidence.get("world_model_candidate_accepted")
            ),
            "research_checkpoint_identity_verified": bool(
                expected_checkpoint_sha256
                and actual_checkpoint_sha256 == expected_checkpoint_sha256
            ),
            "shot_head_identity_verified": shot_identity_verified,
            "llm_credentials_available": llm_credentials_available(llm_values),
            "llm_config_valid": llm_config_error is None,
        }
        if self.config.mode == "stable":
            ready = all(
                checks[name]
                for name in (
                    "stable_release_available",
                    "stable_release_identity_verified",
                    "stable_release_artifacts_verified",
                )
            )
            blockers: list[str] = (
                []
                if ready
                else [
                    name
                    for name in (
                        "stable_release_available",
                        "stable_release_identity_verified",
                        "stable_release_artifacts_verified",
                    )
                    if not checks[name]
                ]
            )
        elif self.config.mode == "research":
            ready = all(
                checks[name]
                for name in (
                    "research_checkpoint_available",
                    "research_checkpoint_accepted",
                    "research_checkpoint_identity_verified",
                    "shot_head_identity_verified",
                )
            )
            blockers = (
                []
                if ready
                else [
                    name
                    for name in (
                        "research_checkpoint_available",
                        "research_checkpoint_accepted",
                        "research_checkpoint_identity_verified",
                        "shot_head_identity_verified",
                    )
                    if not checks[name]
                ]
            )
        else:
            ready = all(
                checks[name]
                for name in (
                    "research_checkpoint_available",
                    "research_checkpoint_accepted",
                    "research_checkpoint_identity_verified",
                    "shot_head_identity_verified",
                    "llm_credentials_available",
                    "llm_config_valid",
                )
            )
            blockers = [
                name
                for name in (
                    "research_checkpoint_available",
                    "research_checkpoint_accepted",
                    "research_checkpoint_identity_verified",
                    "shot_head_identity_verified",
                    "llm_credentials_available",
                    "llm_config_valid",
                )
                if not checks[name]
            ]
        return {
            "mode": self.config.mode,
            "ready": ready,
            "checks": checks,
            "blockers": blockers,
            "release_artifact_verification": release_artifact_verification,
            "llm_provider": llm_provider,
            "llm_config_error": llm_config_error,
        }

    def status(self) -> dict[str, Any]:
        session = self._session()
        from src.product.control_plane import ProductControlPlane

        lease_held = FileLease.is_held(self.session_lease_path)
        runs = list(session.get("runs") or [])
        session_matches = list(session.get("matches") or [])
        last_match = dict((session_matches or [{}])[-1] or {})
        if last_match and not last_match.get("world_model_action_adoption"):
            report_path = _artifact_path(self.root, last_match.get("report"))
            if report_path is not None and report_path.is_file():
                try:
                    historical_report = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                    adoption = (
                        (historical_report.get("layers") or {}).get("world_model") or {}
                    ).get("action_adoption")
                    digest = _action_adoption_digest(adoption)
                    if digest:
                        last_match["world_model_action_adoption"] = digest
                except (OSError, ValueError, TypeError):
                    pass
        observed_runs = [
            {
                **run,
                "observed_state": ("running" if lease_held else "abandoned")
                if run.get("state") in ACTIVE_RUN_STATES
                else run.get("state"),
            }
            for run in runs
        ]
        match_history = []
        for item in reversed(session_matches[-50:]):
            if not isinstance(item, Mapping):
                continue
            plan = item.get("match_plan") or {}
            score = item.get("score") or {}
            match_history.append(
                {
                    "match_id": str(item.get("match_id") or ""),
                    "home": str(item.get("home") or ""),
                    "away": str(item.get("away") or ""),
                    "seed": item.get("seed"),
                    "fast": item.get("fast"),
                    "score": dict(score) if isinstance(score, Mapping) else {},
                    "integrity": str(item.get("integrity") or "unknown"),
                    "experience": (
                        str(plan.get("experience") or "observational")
                        if isinstance(plan, Mapping)
                        else "unknown"
                    ),
                    "home_tactic": (
                        str(plan.get("home_tactic") or NATIVE_TACTIC)
                        if isinstance(plan, Mapping)
                        else "unknown"
                    ),
                    "away_tactic": (
                        str(plan.get("away_tactic") or NATIVE_TACTIC)
                        if isinstance(plan, Mapping)
                        else "unknown"
                    ),
                    "dashboard": item.get("dashboard"),
                    "comparison_dashboard": item.get("comparison_dashboard"),
                }
            )
        season_history = self._season_history_view(session)
        archived_total = session.get("archived_seasons_total", len(season_history))
        if (
            isinstance(archived_total, bool)
            or not isinstance(archived_total, int)
            or archived_total < len(season_history)
        ):
            raise ValueError("invalid archived season count")
        club_finance = None
        current_manager_team = None
        current_season = session.get("season")
        if isinstance(current_season, Mapping):
            current_plan = SeasonPlan.from_payload(current_season["plan"])
            current_manager_team = current_plan.manager_team
            if current_plan.manager_team is not None:
                projected_settlement = None
                base_roster = load_roster_json(
                    roster_path_for_team(
                        str(self.root),
                        current_plan.manager_team,
                    )
                )
                current_roster = (
                    replay_squad_registry_through(
                        base_roster,
                        team=current_plan.manager_team,
                        registry=session.get("squad_registry"),
                        season_id=str(current_season["season_id"]),
                        development_registry=session.get("player_development"),
                        lifecycle_registry=session.get("player_lifecycle"),
                        market_registry=session.get("player_market"),
                    )
                    if base_roster is not None
                    else None
                )
                if current_season.get("state") == "complete":
                    current_archive = self._archive_with_development_evidence(
                        current_season,
                    )
                    projected_settlement = build_season_finance_settlement(
                        current_archive,
                        current_roster,
                        team=current_plan.manager_team,
                    )
                club_finance = club_finance_view(
                    session.get("finance_registry"),
                    team=current_plan.manager_team,
                    projected_settlement=projected_settlement,
                )
                club_finance["current_wage"] = (
                    seasonal_wage_expense(current_roster)
                    if current_roster is not None
                    else {
                        "roster_identity": None,
                        "player_count": None,
                        "wage_tier_total": None,
                        "seasonal_wage_expense": 6,
                        "tier_counts": None,
                        "source": "team_level_baseline_without_roster",
                    }
                )
        status = {
            "product": session["product"],
            "name": session["name"],
            "mode": session["mode"],
            "seed": session["seed"],
            "matches_played": len(session_matches),
            "match_history": match_history,
            "match_history_limit": 50,
            "match_history_truncated": len(session_matches) > 50,
            "last_match": last_match or None,
            "runs_total": len(runs),
            "failed_runs": sum(run.get("state") == "failed" for run in runs),
            "abandoned_runs": sum(
                run.get("observed_state") == "abandoned" for run in observed_runs
            ),
            "last_run": (observed_runs or [None])[-1],
            "readiness": self.readiness(),
            "evidence": self.evidence(),
            "control_plane": ProductControlPlane(self.root).snapshot(),
            "season": (
                self._season_view(session["season"])
                if session.get("season") is not None
                else None
            ),
            "season_history": season_history,
            "season_history_summary": {
                "total": archived_total,
                "retained": len(season_history),
                "limit": 12,
                "truncated": archived_total > len(season_history),
            },
            "manager_career": manager_career_profile(
                season_history,
                current_season=session.get("season"),
            ),
            "club_finance": club_finance,
            "league_ecosystem": league_ecosystem_view(
                session.get("league_ecosystem"),
            ),
            "player_development": development_view(
                session.get("player_development"),
            ),
            "player_lifecycle": lifecycle_view(
                session.get("player_lifecycle"),
            ),
            "player_market": market_view(session.get("player_market")),
            "scouting": scouting_view(session.get("scouting_registry")),
            "scouting_outcomes": scouting_outcome_view(
                session.get("scouting_outcomes"),
                manager_team=current_manager_team,
            ),
            "sporting_reviews": sporting_review_view(
                session.get("sporting_reviews"),
            ),
        }
        status["workflow"] = self.workflow(status=status)
        return status

    def workflow(self, *, status: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the single guided product journey derived from persisted state."""
        if status is None:
            session = self._session()
            current_season = (
                self._season_view(session["season"])
                if session.get("season") is not None
                else None
            )
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
            current_season = status.get("season")
            readiness = status["readiness"]
            matches = [None] * int(status.get("matches_played", 0))
            last_match = status.get("last_match")
            active = (status.get("last_run") or {}).get(
                "observed_state"
            ) in ACTIVE_RUN_STATES
            failed = int(status.get("failed_runs", 0))

        season_present = isinstance(current_season, Mapping)
        season_complete = bool(
            season_present and current_season.get("state") == "complete"
        )
        season_command = (
            current_season.get("matchday_command_center") if season_present else None
        )
        command_phase = (
            season_command.get("phase") if isinstance(season_command, Mapping) else None
        )
        season_plan = current_season.get("plan") or {} if season_present else {}
        manager_team = (
            season_plan.get("manager_team")
            if isinstance(season_plan, Mapping)
            else None
        )
        next_manager_fixture = (
            current_season.get("next_manager_fixture") if season_present else None
        )
        manager_squad = current_season.get("manager_squad") if season_present else None
        player_promise_setup_required = bool(
            manager_team
            and isinstance(next_manager_fixture, Mapping)
            and current_season.get("player_role_promises") is None
            and isinstance(manager_squad, Mapping)
            and manager_squad.get("available") is True
        )

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
        elif season_complete and manager_team:
            state = "season_planning"
            next_action = {
                "id": "review_sporting_plan",
                "target": f"/api/v1/sporting-plans/{manager_team}",
                "reason": (
                    "the season is complete; review and freeze one evidence-bound "
                    "sporting directive before opening the next season"
                ),
            }
        elif season_complete:
            state = "season_review"
            next_action = {
                "id": "review_season",
                "target": "#season-panel",
                "reason": (
                    "the spectator season is complete; review the final table "
                    "before opening the next season"
                ),
            }
        elif season_present and player_promise_setup_required:
            state = "season_setup"
            next_action = {
                "id": "freeze_player_promises",
                "target": "#player-promise-form",
                "fixture_id": next_manager_fixture.get("fixture_id"),
                "matchday": next_manager_fixture.get("matchday"),
                "opponent": (
                    next_manager_fixture.get("away")
                    if next_manager_fixture.get("home") == manager_team
                    else next_manager_fixture.get("home")
                ),
                "reason": (
                    "freeze one small evidence-bound named-player plan before "
                    "the first managed fixture decision"
                ),
            }
        elif season_present and command_phase == "decision_required":
            fixture = (
                season_command.get("current_fixture")
                or season_command.get("next_fixture")
                or {}
            )
            state = "season_decision"
            next_action = {
                "id": "submit_manager_decision",
                "target": "#manager-decision-form",
                "fixture_id": fixture.get("fixture_id"),
                "matchday": fixture.get("matchday"),
                "opponent": fixture.get("opponent"),
                "reason": (
                    "the current managed fixture requires a frozen lineup, "
                    "tactic and in-match plan"
                ),
            }
        elif season_present and command_phase in {
            "ready_to_advance",
            "ready_to_resume",
            "spectator",
        }:
            resume = command_phase == "ready_to_resume"
            state = "season_matchday_ready"
            next_action = {
                "id": (
                    "resume_season_matchday" if resume else "advance_season_matchday"
                ),
                "target": "#play-matchday",
                "matchday": season_command.get("current_matchday"),
                "reason": (
                    "resume the partially completed persisted matchday"
                    if resume
                    else "all required decisions are frozen; advance the matchday"
                ),
            }
        elif season_present:
            state = "blocked"
            next_action = {
                "id": "inspect_season_state",
                "command": "gfs studio status",
                "reason": f"unsupported persisted season phase: {command_phase}",
            }
        elif not matches:
            state = "ready_to_start_season"
            next_action = {
                "id": "start_season",
                "target": "#season-form",
                "reason": (
                    "readiness passed; create the persistent season that joins "
                    "club planning, matchdays, evidence and career consequences"
                ),
            }
        else:
            state = "review"
            next_action = {
                "id": "open_dashboard",
                "target": (last_match or {}).get("dashboard"),
                "reason": "at least one auditable report is available",
            }

        season_journey = []
        if season_present:
            readiness_ready = bool(readiness["ready"])
            decision_pending = command_phase == "decision_required"
            execution_ready = command_phase in {
                "ready_to_advance",
                "ready_to_resume",
                "spectator",
            }
            season_journey = [
                {
                    "id": "season_setup",
                    "status": (
                        "blocked"
                        if not readiness_ready
                        else "action_required"
                        if player_promise_setup_required
                        else "complete"
                    ),
                },
                {
                    "id": "matchday_decision",
                    "status": (
                        "complete"
                        if season_complete
                        else "blocked"
                        if (not readiness_ready or player_promise_setup_required)
                        else "action_required"
                        if decision_pending
                        else "complete"
                        if execution_ready
                        else "pending"
                    ),
                },
                {
                    "id": "matchday_execution",
                    "status": (
                        "complete"
                        if season_complete
                        else "blocked"
                        if (
                            not readiness_ready
                            or player_promise_setup_required
                            or decision_pending
                        )
                        else "pending"
                        if active
                        else "ready"
                        if execution_ready
                        else "pending"
                    ),
                },
                {
                    "id": "season_review",
                    "status": "available" if season_complete else "pending",
                },
                {
                    "id": "next_season",
                    "status": (
                        "blocked"
                        if not readiness_ready
                        else "action_required"
                        if (season_complete and manager_team is not None)
                        else "available"
                        if season_complete
                        else "pending"
                    ),
                },
            ]

        return {
            "schema_version": 2,
            "state": state,
            "progress": {
                "workspace_configured": True,
                "readiness_passed": bool(readiness["ready"]),
                "match_completed": bool(matches),
                "report_available": bool(matches),
                "season_present": season_present,
                "season_setup_complete": (
                    season_present and not player_promise_setup_required
                ),
                "season_matchday_ready": state == "season_matchday_ready",
                "season_complete": season_complete,
                "sporting_plan_required": state == "season_planning",
            },
            "journey": season_journey,
            "alternative_actions": (
                [
                    {
                        "id": "run_standalone_match",
                        "target": "#match-form",
                        "reason": (
                            "optional isolated observation or tactical laboratory entry"
                        ),
                    }
                ]
                if not season_present
                else []
            ),
            "completed_matches": len(matches),
            "failed_attempts": failed,
            "artifacts": {
                "latest_report": (last_match or {}).get("report"),
                "latest_dashboard": (last_match or {}).get("dashboard"),
            },
            "next_action": next_action,
        }

    def replay_seed(self, home: str, away: str) -> int:
        """Resolve the latest completed same-fixture seed for a paired rematch."""
        session = self._session()
        last_match = (session.get("matches") or [None])[-1]
        if not isinstance(last_match, Mapping):
            raise ValueError("no completed match is available for seed reuse")
        if (
            str(last_match.get("home", "")).casefold() != str(home).casefold()
            or str(last_match.get("away", "")).casefold() != str(away).casefold()
        ):
            raise ValueError("seed reuse requires the same ordered fixture")
        seed = last_match.get("seed")
        if seed is None:
            report_path = _artifact_path(self.root, last_match.get("report"))
            if report_path is not None and report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    seed = (report.get("fixture") or {}).get("seed")
                except (OSError, ValueError, TypeError):
                    seed = None
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2**31 - 1
        ):
            raise ValueError("latest match has no reusable deterministic seed")
        return seed

    @contextmanager
    def _mode_environment(
        self, *, world_model_policy: str = "mode_default",
        world_model_branch_at_sec: float | None = None,
    ) -> Iterator[None]:
        from src.simulation.runtime import environment_override

        values = dict(MODE_ENVIRONMENT[self.config.mode])
        evidence = self.evidence()
        if self.config.mode in {"research", "cognitive"}:
            values["MATCH_WM_CHECKPOINT"] = str(
                evidence.get("world_model_checkpoint")
                or values.get("MATCH_WM_CHECKPOINT")
            )
        if world_model_policy == "predict_only":
            values["MATCH_WM_PLAN"] = "0"
        elif world_model_policy == "action_policy":
            values["MATCH_WM_PLAN"] = "1"
        elif world_model_policy != "mode_default":
            raise ValueError("unsupported world-model policy override")
        if world_model_branch_at_sec is not None:
            branch_value = str(float(world_model_branch_at_sec))
            values["MATCH_WM_BRANCH_AT_SEC"] = branch_value
            values["MATCH_WM_PLAN_START_SEC"] = branch_value
        shot_decision = evidence.get("frozen_shot_head") or {}
        shot_artifact = shot_decision.get("promotion_artifact")
        values["MATCH_WM_SHOT_HEAD"] = (
            str(shot_artifact)
            if shot_decision.get("accepted") and shot_artifact
            else ""
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

    def run_match(
        self,
        home: str,
        away: str,
        *,
        fast: bool = False,
        plan: MatchPlan | None = None,
        seed_override: int | None = None,
    ) -> dict[str, Any]:
        with FileLease(self.session_lease_path, timeout=30.0):
            return self._run_match_locked(
                home,
                away,
                fast=fast,
                plan=plan,
                seed_override=seed_override,
            )

    def _archive_with_development_evidence(
        self,
        season: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Freeze season results together with exact available player minutes."""
        archived = archive_completed_season(season)
        plan = SeasonPlan.from_payload(season["plan"])
        archived["player_development_evidence"] = {
            team: collect_season_participation(self.root, season, team=team)
            for team in sorted(plan.teams)
        }
        contract = archived.get("player_role_promises")
        if isinstance(contract, Mapping) and plan.manager_team is not None:
            progress = archived["manager_profile"]["player_role_promises"]
            archived["player_promise_outcomes"] = settle_player_promise_outcomes(
                contract,
                progress,
                archived["player_development_evidence"][plan.manager_team],
            )
        return archived

    def _sporting_brief_for_transition(
        self,
        session: Mapping[str, Any],
        source_archive: Mapping[str, Any],
        *,
        team: str,
        target_season_id: str,
    ) -> dict[str, Any]:
        source_season_id = str(source_archive.get("season_id") or "")
        target_index = int(str(target_season_id).split("-")[-1])
        source_index = int(source_season_id.split("-")[-1])
        source_plan = SeasonPlan.from_payload(source_archive.get("plan") or {})
        if team not in source_plan.teams or target_index <= source_index:
            raise ValueError(
                "sporting plan requires a continuing club and later season"
            )
        base_roster = load_roster_json(
            roster_path_for_team(
                str(self.root),
                team,
            )
        )
        if base_roster is None:
            raise ValueError("sporting plan roster is unavailable")
        roster = replay_squad_registry_through(
            base_roster,
            team=team,
            registry=session.get("squad_registry"),
            season_id=source_season_id,
            development_registry=session.get("player_development"),
            lifecycle_registry=session.get("player_lifecycle"),
            market_registry=session.get("player_market"),
        )
        recruitment = generate_recruitment_market(
            roster,
            team=team,
            market_id=recruitment_market_id(
                season_index=target_index,
                team=team,
            ),
        )
        outgoing = [
            {
                "player_id": str(player.get("player_id") or ""),
                "name": str(player.get("name") or ""),
                "role": str(player.get("role") or ""),
                "age": player.get("age"),
                "quality": round(squad_player_quality(player), 6),
                "wage_tier": player_wage_tier(player),
                "injury_state_not_transferred": True,
            }
            for player in roster.get("players") or []
            if isinstance(player, Mapping)
        ]
        recruitment["outgoing_players"] = outgoing
        settlement = build_season_finance_settlement(
            source_archive,
            roster,
            team=team,
        )
        finance_entries = (
            ((session.get("finance_registry") or {}).get("clubs") or {})
            .get(team, {})
            .get("entries", [])
        )
        settled = [
            entry
            for entry in finance_entries
            if entry.get("type") == "season_settlement"
            and entry.get("season_id") == source_season_id
        ]
        if settled:
            if len(settled) != 1 or settled[0].get("evidence") != settlement:
                raise ValueError("sporting plan finance source mismatch")
            club_balance = int(settled[0]["balance_after"])
        else:
            prospective, _entry = append_season_settlement(
                session.get("finance_registry"),
                team=team,
                settlement=settlement,
            )
            club_balance = int(
                club_finance_view(
                    prospective,
                    team=team,
                )["balance"]
            )
        recruitment["club_balance"] = club_balance
        recruitment["available_budget"] = recruitment_allowance(club_balance)
        lifecycle = build_lifecycle_preview(
            roster,
            team=team,
            target_season_id=target_season_id,
        )
        prior_market = {
            "schema_version": 1,
            "transitions": [
                copy.deepcopy(item)
                for item in (session.get("player_market") or {}).get("transitions", [])
                if int(str(item["target_season_id"]).split("-")[-1]) < target_index
            ],
        }
        market_window = open_market_window(
            prior_market,
            target_season_id=target_season_id,
        )
        free_market = market_preview_from_window(market_window, team=team)
        reports = reports_for_market(
            session.get("scouting_registry"),
            market_id=free_market["market_id"],
            team=team,
        )
        report_by_player = {report["player_id"]: report for report in reports}
        for candidate in free_market["candidates"]:
            report = report_by_player.get(candidate["player_id"])
            if report is not None:
                candidate["observation"] = copy.deepcopy(report["observation"])
                candidate["evidence_quality"] = "budgeted_scouting_report"
                candidate["scouted"] = True
            else:
                candidate["scouted"] = False
        free_market["scouting_budget"] = {
            "limit": 2,
            "used": len(reports),
            "remaining": 2 - len(reports),
        }
        free_market["outgoing_players"] = [
            {key: row[key] for key in ("player_id", "name", "role", "quality")}
            for row in outgoing
        ]
        prior_outcomes = {
            "schema_version": 1,
            "outcomes": [
                copy.deepcopy(item)
                for item in (session.get("scouting_outcomes") or {}).get("outcomes", [])
                if int(str(item["season_id"]).split("-")[-1]) < source_index
            ],
        }
        previous_strategy_review = None
        if (
            source_plan.manager_sporting_directive is not None
            and source_plan.manager_team == team
        ):
            existing_reviews = [
                row
                for row in (session.get("sporting_reviews") or {}).get("reviews", [])
                if row.get("season_id") == source_season_id and row.get("team") == team
            ]
            if len(existing_reviews) > 1:
                raise ValueError("duplicate sporting strategy review")
            if existing_reviews:
                prior_review = existing_reviews[0]
            else:
                source_signings = [
                    signing
                    for transition in (session.get("player_market") or {}).get(
                        "transitions", []
                    )
                    if transition.get("target_season_id") == source_season_id
                    for signing in transition.get("signings") or []
                    if signing.get("team") == team
                ]
                participation = (
                    source_archive.get("player_development_evidence") or {}
                ).get(team)
                if not isinstance(participation, Mapping):
                    raise ValueError("sporting strategy review evidence is unavailable")
                prior_review = build_sporting_review(
                    source_archive,
                    roster,
                    participation=participation,
                    settlement=settlement,
                    free_agent_signings=source_signings,
                )
            previous_strategy_review = sporting_review_feedback(prior_review)
        return build_sporting_brief(
            team=team,
            source_season_id=source_season_id,
            target_season_id=target_season_id,
            roster=roster,
            recruitment_market=recruitment,
            lifecycle_preview=lifecycle,
            free_agent_market=free_market,
            scouting_outcomes=scouting_outcome_view(
                prior_outcomes,
                manager_team=team,
            ),
            previous_strategy_review=previous_strategy_review,
        )

    def sporting_plan(self, team: str) -> dict[str, Any]:
        """Return one evidence-bound planning surface for the next season."""
        session = self._session()
        season = session.get("season")
        if not isinstance(season, Mapping) or season.get("state") != "complete":
            raise ValueError("sporting planning opens only after season completion")
        cleaned_team = str(team or "").strip()
        plan = SeasonPlan.from_payload(season["plan"])
        if cleaned_team not in plan.teams:
            raise ValueError("sporting planning team is outside the competition")
        archive = self._archive_with_development_evidence(season)
        target_index = int(session.get("next_season_index", 1))
        return self._sporting_brief_for_transition(
            session,
            archive,
            team=cleaned_team,
            target_season_id=f"season-{target_index:04d}",
        )

    def create_season(
        self,
        plan: SeasonPlan,
        *,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Create one deterministic competition inside the Studio session."""
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            existing = session.get("season")
            if existing and not replace:
                raise FileExistsError("studio season already exists")
            rollover_archive = None
            rollover_transition = None
            rollover_settlements: dict[str, dict[str, Any]] = {}
            rollover_rosters: dict[str, dict[str, Any] | None] = {}
            if existing:
                validate_season_state(existing)
                if any(
                    fixture.get("state") != "completed"
                    for fixture in existing.get("fixtures") or []
                ):
                    raise ValueError("only a completed season can be replaced")
                history = self._season_history_view(session)
                archived_total = session.get("archived_seasons_total", len(history))
                if (
                    isinstance(archived_total, bool)
                    or not isinstance(archived_total, int)
                    or archived_total < len(history)
                ):
                    raise ValueError("invalid archived season count")
                archived = self._archive_with_development_evidence(existing)
                rollover_archive = archived
                if any(
                    isinstance(item, Mapping)
                    and item.get("season_id") == archived["season_id"]
                    for item in history
                ):
                    raise ValueError("completed season is already archived")
                previous_plan = SeasonPlan.from_payload(existing["plan"])
                for previous_team in sorted(previous_plan.teams):
                    base_roster = load_roster_json(
                        roster_path_for_team(
                            str(self.root),
                            previous_team,
                        )
                    )
                    historical_roster = (
                        replay_squad_registry_through(
                            base_roster,
                            team=previous_team,
                            registry=session.get("squad_registry"),
                            season_id=str(existing["season_id"]),
                            development_registry=session.get("player_development"),
                            lifecycle_registry=session.get("player_lifecycle"),
                            market_registry=session.get("player_market"),
                        )
                        if base_roster is not None
                        else None
                    )
                    rollover_rosters[previous_team] = historical_roster
                    settlement = build_season_finance_settlement(
                        archived,
                        historical_roster,
                        team=previous_team,
                    )
                    rollover_settlements[previous_team] = settlement
                    finance_registry, _settlement_entry = append_season_settlement(
                        session.get("finance_registry"),
                        team=previous_team,
                        settlement=settlement,
                    )
                    session["finance_registry"] = finance_registry
                completed_market_transitions = [
                    item
                    for item in (session.get("player_market") or {}).get(
                        "transitions", []
                    )
                    if item.get("target_season_id") == archived["season_id"]
                ]
                if len(completed_market_transitions) > 1:
                    raise ValueError("duplicate completed-season market transition")
                if completed_market_transitions:
                    completed_market = completed_market_transitions[0]
                    for signing in completed_market.get("signings") or []:
                        signing_team = str(signing.get("team") or "")
                        observation, observation_source = resolve_signing_observation(
                            signing,
                            completed_market,
                            session.get("scouting_registry"),
                        )
                        outcome = build_scouting_outcome(
                            signing,
                            observation=observation,
                            observation_source=observation_source,
                            archive=archived,
                            participation=archived["player_development_evidence"][
                                signing_team
                            ],
                            settlement=rollover_settlements[signing_team],
                        )
                        session["scouting_outcomes"] = append_scouting_outcome(
                            session.get("scouting_outcomes"),
                            outcome,
                        )
                archived_plan = SeasonPlan.from_payload(archived["plan"])
                if archived_plan.manager_sporting_directive is not None:
                    review_team = str(archived_plan.manager_team)
                    review = build_sporting_review(
                        archived,
                        rollover_rosters[review_team],
                        participation=archived["player_development_evidence"][
                            review_team
                        ],
                        settlement=rollover_settlements[review_team],
                        free_agent_signings=(
                            completed_market_transitions[0].get("signings") or []
                            if completed_market_transitions
                            else []
                        ),
                    )
                    session["sporting_reviews"] = append_sporting_review(
                        session.get("sporting_reviews"),
                        review,
                    )
                history.append(archived)
                session["season_history"] = history[-12:]
                session["archived_seasons_total"] = archived_total + 1
            else:
                history = self._season_history_view(session)
            career = manager_career_profile(history)
            validate_manager_career_transition(plan, career)
            if plan.manager_team is not None:
                session["finance_registry"] = ensure_finance_club(
                    session.get("finance_registry"),
                    plan.manager_team,
                )
            season_index = int(session.get("next_season_index", 1))
            used_ids = {
                str(item.get("season_id"))
                for item in (session.get("season_history") or [])
                if isinstance(item, Mapping)
            }
            if existing:
                used_ids.add(str(existing.get("season_id")))
            while f"season-{season_index:04d}" in used_ids:
                season_index += 1
            season_id = f"season-{season_index:04d}"
            sporting_brief = None
            sporting_evaluation = None
            if plan.manager_sporting_directive is not None:
                if rollover_archive is None or plan.manager_team is None:
                    raise ValueError(
                        "sporting directive is available only for a later managed season"
                    )
                sporting_brief = self._sporting_brief_for_transition(
                    session,
                    rollover_archive,
                    team=plan.manager_team,
                    target_season_id=season_id,
                )
                sporting_evaluation = validate_sporting_choices(
                    plan.manager_sporting_directive,
                    sporting_brief,
                    recruitment=(
                        plan.manager_recruitment.as_dict()
                        if plan.manager_recruitment is not None
                        else None
                    ),
                    retention=(
                        plan.manager_retention.as_dict()
                        if plan.manager_retention is not None
                        else None
                    ),
                    free_agent=(
                        plan.manager_free_agent.as_dict()
                        if plan.manager_free_agent is not None
                        else None
                    ),
                )
            recruitment_transaction = None
            if plan.manager_recruitment is not None:
                if existing is None:
                    raise ValueError(
                        "recruitment is available only when starting a later season"
                    )
                expected_market_id = recruitment_market_id(
                    season_index=season_index,
                    team=str(plan.manager_team),
                )
                if plan.manager_recruitment.market_id != expected_market_id:
                    raise ValueError("recruitment market identity mismatch")
                base_roster = load_roster_json(
                    roster_path_for_team(
                        str(self.root),
                        str(plan.manager_team),
                    )
                )
                if base_roster is None:
                    raise ValueError("recruitment roster is unavailable")
                registry, _result_roster, recruitment_transaction = (
                    append_recruitment_transaction(
                        base_roster,
                        team=str(plan.manager_team),
                        plan=plan.manager_recruitment,
                        season_id=season_id,
                        registry=session.get("squad_registry"),
                        development_registry=session.get("player_development"),
                        lifecycle_registry=session.get("player_lifecycle"),
                        market_registry=session.get("player_market"),
                    )
                )
                session["squad_registry"] = registry
                finance_registry, _finance_entry = append_recruitment_charge(
                    session.get("finance_registry"),
                    team=str(plan.manager_team),
                    season_id=season_id,
                    squad_transaction=recruitment_transaction,
                )
                session["finance_registry"] = finance_registry
            if rollover_archive is not None:
                prior_teams = set(
                    SeasonPlan.from_payload(
                        rollover_archive["plan"],
                    ).teams
                )
                participating_teams = sorted(prior_teams & set(plan.teams))
                controlled_team = (
                    plan.manager_team
                    if plan.manager_team in participating_teams
                    else None
                )
                ai_clubs = []
                for ai_team in (
                    team for team in participating_teams if team != controlled_team
                ):
                    finance = club_finance_view(
                        session.get("finance_registry"),
                        team=ai_team,
                    )
                    decision = ai_club_recruitment_decision(
                        rollover_archive,
                        rollover_rosters.get(ai_team),
                        team=ai_team,
                        next_season_index=season_index,
                        allowance=finance["recruitment_allowance"],
                    )
                    ai_transaction = None
                    if decision["plan"] is not None:
                        base_roster = load_roster_json(
                            roster_path_for_team(
                                str(self.root),
                                ai_team,
                            )
                        )
                        if base_roster is None:
                            raise ValueError("AI recruitment roster is unavailable")
                        registry, _result_roster, ai_transaction = (
                            append_recruitment_transaction(
                                base_roster,
                                team=ai_team,
                                plan=RecruitmentPlan.from_payload(decision["plan"]),
                                season_id=season_id,
                                registry=session.get("squad_registry"),
                                development_registry=session.get("player_development"),
                                lifecycle_registry=session.get("player_lifecycle"),
                                market_registry=session.get("player_market"),
                            )
                        )
                        session["squad_registry"] = registry
                        finance_registry, _finance_entry = append_recruitment_charge(
                            session.get("finance_registry"),
                            team=ai_team,
                            season_id=season_id,
                            squad_transaction=ai_transaction,
                        )
                        session["finance_registry"] = finance_registry
                    ai_clubs.append(
                        {
                            "team": ai_team,
                            "settlement_identity": evidence_identity(
                                rollover_settlements[ai_team],
                            ),
                            "decision": decision,
                            "squad_transaction_identity": (
                                evidence_identity(ai_transaction)
                                if ai_transaction is not None
                                else None
                            ),
                        }
                    )
                transition = {
                    "schema_version": 1,
                    "from_season_id": rollover_archive["season_id"],
                    "to_season_id": season_id,
                    "archive_identity": evidence_identity(rollover_archive),
                    "manager_controlled_team": controlled_team,
                    "participating_teams": participating_teams,
                    "ai_clubs": ai_clubs,
                    "claim_boundary": (
                        "deterministic bounded AI-club evolution from simulated "
                        "standings, game-credit finance, and fictional markets"
                    ),
                }
                session["league_ecosystem"] = append_league_transition(
                    session.get("league_ecosystem"),
                    transition,
                )
                rollover_transition = transition
            if rollover_archive is not None:
                source_plan = SeasonPlan.from_payload(rollover_archive["plan"])
                participation = rollover_archive["player_development_evidence"]
                continuing_teams = sorted(set(source_plan.teams) & set(plan.teams))
                market_window = open_market_window(
                    session.get("player_market"),
                    target_season_id=season_id,
                )
                target_rosters: dict[str, dict[str, Any]] = {}
                for market_team in continuing_teams:
                    base_roster = load_roster_json(
                        roster_path_for_team(
                            str(self.root),
                            market_team,
                        )
                    )
                    if base_roster is None:
                        continue
                    target_rosters[market_team] = replay_squad_registry_through(
                        base_roster,
                        team=market_team,
                        registry=session.get("squad_registry"),
                        season_id=season_id,
                        development_registry=session.get("player_development"),
                        lifecycle_registry=session.get("player_lifecycle"),
                        market_registry=session.get("player_market"),
                    )
                if plan.manager_free_agent is not None:
                    manager_team = str(plan.manager_team or "")
                    if manager_team not in target_rosters:
                        raise ValueError(
                            "free-agent signing requires a continuing managed club"
                        )
                    market_window, target_rosters[manager_team], _manager_signing = (
                        execute_free_agent_signing(
                            market_window,
                            target_rosters[manager_team],
                            team=manager_team,
                            plan=plan.manager_free_agent,
                            control="manager",
                        )
                    )
                ai_market_decisions = []
                for market_team in (
                    team
                    for team in continuing_teams
                    if team != plan.manager_team and team in target_rosters
                ):
                    decision = ai_free_agent_decision(
                        market_window,
                        target_rosters[market_team],
                        team=market_team,
                    )
                    ai_market_decisions.append(decision)
                    if decision["plan"] is not None:
                        market_window, target_rosters[market_team], _ai_signing = (
                            execute_free_agent_signing(
                                market_window,
                                target_rosters[market_team],
                                team=market_team,
                                plan=FreeAgentPlan.from_payload(decision["plan"]),
                                control="ai",
                            )
                        )
                market_window["ai_decisions"] = ai_market_decisions
                lifecycle_results: dict[str, dict[str, Any]] = {}
                market_entries = []
                for signing in market_window.get("signings") or []:
                    outgoing = signing["outgoing"]
                    market_entries.append(
                        pool_entry_from_player(
                            outgoing,
                            origin_team=str(signing["team"]),
                            entered_season_id=season_id,
                            retirement_age=retirement_age_for_player(
                                str(signing["team"]),
                                str(outgoing["player_id"]),
                            ),
                            reason="squad_replacement",
                        )
                    )
                for lifecycle_team in continuing_teams:
                    source_roster = rollover_rosters.get(lifecycle_team)
                    target_roster = target_rosters.get(lifecycle_team)
                    if source_roster is None or target_roster is None:
                        continue
                    retention = (
                        plan.manager_retention
                        if lifecycle_team == plan.manager_team
                        else None
                    )
                    transaction, lifecycle_result = build_lifecycle_transaction(
                        source_roster,
                        target_roster,
                        team=lifecycle_team,
                        target_season_id=season_id,
                        plan=retention,
                        control=(
                            "manager" if lifecycle_team == plan.manager_team else "ai"
                        ),
                    )
                    session["player_lifecycle"] = append_lifecycle_transaction(
                        session.get("player_lifecycle"),
                        team=lifecycle_team,
                        transaction=transaction,
                    )
                    lifecycle_results[lifecycle_team] = lifecycle_result
                    for exit_row in transaction["exits"]:
                        if exit_row["reason"] != "contract_released":
                            continue
                        market_entries.append(
                            pool_entry_from_player(
                                exit_row["player"],
                                origin_team=lifecycle_team,
                                entered_season_id=season_id,
                                retirement_age=int(exit_row["retirement_age"]),
                                reason="contract_released",
                            )
                        )
                session["player_market"], _market_transition = (
                    finalize_market_transition(
                        session.get("player_market"),
                        market_window,
                        entries=market_entries,
                    )
                )
                for development_team in continuing_teams:
                    source_roster = rollover_rosters.get(development_team)
                    target_roster = lifecycle_results.get(development_team)
                    if source_roster is None or target_roster is None:
                        continue
                    transaction, _developed_roster = build_development_transaction(
                        rollover_archive,
                        source_roster,
                        target_roster,
                        participation[development_team],
                        team=development_team,
                        target_season_id=season_id,
                    )
                    session["player_development"] = append_development_transaction(
                        session.get("player_development"),
                        team=development_team,
                        transaction=transaction,
                    )
            session["next_season_index"] = season_index + 1
            strategy_rosters = {}
            recruitment_moves = {}
            squad_clubs = (session.get("squad_registry") or {}).get("clubs") or {}
            for strategy_team in sorted(plan.teams):
                base_roster = load_roster_json(
                    roster_path_for_team(
                        str(self.root),
                        strategy_team,
                    )
                )
                strategy_rosters[strategy_team] = (
                    replay_squad_registry_through(
                        base_roster,
                        team=strategy_team,
                        registry=session.get("squad_registry"),
                        season_id=season_id,
                        development_registry=session.get("player_development"),
                        lifecycle_registry=session.get("player_lifecycle"),
                        market_registry=session.get("player_market"),
                    )
                    if base_roster is not None
                    else None
                )
                target_transactions = [
                    transaction
                    for transaction in (squad_clubs.get(strategy_team) or {}).get(
                        "transactions", []
                    )
                    if transaction.get("season_id") == season_id
                ]
                recruitment_moves[strategy_team] = sum(
                    len(transaction.get("moves") or [])
                    for transaction in target_transactions
                )
            club_strategies = build_season_club_strategies(
                season_id=season_id,
                teams=plan.teams,
                manager_team=plan.manager_team,
                rosters=strategy_rosters,
                source_archive=rollover_archive,
                previous_snapshot=(
                    existing.get("club_strategies")
                    if isinstance(existing, Mapping)
                    else None
                ),
                ecosystem_transition=rollover_transition,
                recruitment_moves=recruitment_moves,
            )
            season = new_season_state(
                plan,
                season_id=season_id,
                seed=self.config.seed + season_index - 1,
                created_at=_now(),
                recruitment_transaction=recruitment_transaction,
                club_strategies=club_strategies,
                sporting_brief=sporting_brief,
                sporting_evaluation=sporting_evaluation,
            )
            session["season"] = season
            session["updated_at"] = _now()
            _atomic_json(self.session_path, session)
            return self._season_view(season)

    def recruitment_market(self, team: str) -> dict[str, Any]:
        """Return the frozen deterministic market for the next season."""
        session = self._session()
        season = session.get("season")
        if not isinstance(season, Mapping):
            raise ValueError("recruitment requires an existing season")
        validate_season_state(season)
        if any(
            fixture.get("state") != "completed"
            for fixture in season.get("fixtures") or []
        ):
            raise ValueError("recruitment opens only after the season completes")
        cleaned_team = str(team or "").strip()
        if not cleaned_team or len(cleaned_team) > 96:
            raise ValueError("invalid recruitment team")
        season_index = int(session.get("next_season_index", 1))
        market = recruitment_market_for_workspace(
            self.root,
            team=cleaned_team,
            season_index=season_index,
            registry=session.get("squad_registry"),
            development_registry=session.get("player_development"),
            lifecycle_registry=session.get("player_lifecycle"),
            market_registry=session.get("player_market"),
        )
        roster = load_effective_roster(self.root, cleaned_team)
        if roster is None:
            raise ValueError("recruitment roster is unavailable")
        market["outgoing_players"] = [
            {
                "player_id": str(player.get("player_id") or ""),
                "name": str(player.get("name") or ""),
                "role": str(player.get("role") or ""),
                "quality": round(squad_player_quality(player), 6),
                "injury_state_not_transferred": True,
            }
            for player in roster.get("players") or []
            if isinstance(player, Mapping)
        ]
        market["season_index"] = season_index
        prospective_finance_registry = session.get("finance_registry")
        completed_plan = SeasonPlan.from_payload(season["plan"])
        settlement_preview = None
        balance_before_settlement = club_finance_view(
            prospective_finance_registry,
            team=cleaned_team,
        )["balance"]
        if cleaned_team in completed_plan.teams:
            completed_archive = self._archive_with_development_evidence(season)
            base_roster = load_roster_json(
                roster_path_for_team(
                    str(self.root),
                    cleaned_team,
                )
            )
            completed_roster = (
                replay_squad_registry_through(
                    base_roster,
                    team=cleaned_team,
                    registry=session.get("squad_registry"),
                    season_id=str(season["season_id"]),
                    development_registry=session.get("player_development"),
                    lifecycle_registry=session.get("player_lifecycle"),
                    market_registry=session.get("player_market"),
                )
                if base_roster is not None
                else None
            )
            settlement_preview = build_season_finance_settlement(
                completed_archive,
                completed_roster,
                team=cleaned_team,
            )
            prospective_finance_registry, _ = append_season_settlement(
                prospective_finance_registry,
                team=cleaned_team,
                settlement=settlement_preview,
            )
        finance = club_finance_view(
            prospective_finance_registry,
            team=cleaned_team,
        )
        market["club_balance"] = finance["balance"]
        market["balance_before_settlement"] = balance_before_settlement
        market["settlement_preview"] = settlement_preview
        market["available_budget"] = finance["recruitment_allowance"]
        market["fixed_window_cap"] = finance["recruitment_window_cap"]
        return market

    def lifecycle_preview(self, team: str) -> dict[str, Any]:
        """Return the frozen retention/retirement preview for the next season."""
        session = self._session()
        season = session.get("season")
        if not isinstance(season, Mapping) or season.get("state") != "complete":
            raise ValueError("lifecycle planning opens only after the season completes")
        cleaned_team = str(team or "").strip()
        plan = SeasonPlan.from_payload(season["plan"])
        if cleaned_team not in plan.teams:
            raise ValueError("lifecycle team is outside the completed season")
        base_roster = load_roster_json(
            roster_path_for_team(
                str(self.root),
                cleaned_team,
            )
        )
        if base_roster is None:
            raise ValueError("lifecycle roster is unavailable")
        source_roster = replay_squad_registry_through(
            base_roster,
            team=cleaned_team,
            registry=session.get("squad_registry"),
            season_id=str(season["season_id"]),
            development_registry=session.get("player_development"),
            lifecycle_registry=session.get("player_lifecycle"),
            market_registry=session.get("player_market"),
        )
        target_index = int(session.get("next_season_index", 1))
        return build_lifecycle_preview(
            source_roster,
            team=cleaned_team,
            target_season_id=f"season-{target_index:04d}",
        )

    def free_agent_market(self, team: str) -> dict[str, Any]:
        """Return the identity-preserving global market for the next season."""
        session = self._session()
        season = session.get("season")
        if not isinstance(season, Mapping) or season.get("state") != "complete":
            raise ValueError("free-agent market opens only after the season completes")
        cleaned_team = str(team or "").strip()
        plan = SeasonPlan.from_payload(season["plan"])
        if cleaned_team not in plan.teams:
            raise ValueError("free-agent team is outside the completed season")
        target_index = int(session.get("next_season_index", 1))
        window = open_market_window(
            session.get("player_market"),
            target_season_id=f"season-{target_index:04d}",
        )
        preview = market_preview_from_window(window, team=cleaned_team)
        reports = reports_for_market(
            session.get("scouting_registry"),
            market_id=preview["market_id"],
            team=cleaned_team,
        )
        reports_by_player = {report["player_id"]: report for report in reports}
        for candidate in preview["candidates"]:
            report = reports_by_player.get(candidate["player_id"])
            if report is not None:
                candidate["observation"] = copy.deepcopy(report["observation"])
                candidate["evidence_quality"] = "budgeted_scouting_report"
                candidate["scouted"] = True
            else:
                candidate["scouted"] = False
        preview["scouting_budget"] = {
            "limit": 2,
            "used": len(reports),
            "remaining": 2 - len(reports),
        }
        roster = load_effective_roster(self.root, cleaned_team)
        if roster is None:
            raise ValueError("free-agent signing roster is unavailable")
        preview["outgoing_players"] = [
            {
                "player_id": str(player.get("player_id") or ""),
                "name": str(player.get("name") or ""),
                "role": str(player.get("role") or ""),
                "quality": round(squad_player_quality(player), 6),
            }
            for player in roster.get("players") or []
            if isinstance(player, Mapping)
        ]
        return preview

    def scout_free_agent(self, team: str, player_id: str) -> dict[str, Any]:
        """Consume one persistent scouting report for the current global window."""
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            season = session.get("season")
            if not isinstance(season, Mapping) or season.get("state") != "complete":
                raise ValueError("scouting opens only after the season completes")
            cleaned_team = str(team or "").strip()
            season_plan = SeasonPlan.from_payload(season["plan"])
            if cleaned_team not in season_plan.teams:
                raise ValueError("scouting team is outside the completed season")
            target_index = int(session.get("next_season_index", 1))
            target_id = f"season-{target_index:04d}"
            window = open_market_window(
                session.get("player_market"),
                target_season_id=target_id,
            )
            preview = market_preview_from_window(window, team=cleaned_team)
            cleaned_player = str(player_id or "").strip()
            candidate_ids = {
                candidate["player_id"] for candidate in preview["candidates"]
            }
            entry = (window.get("available") or {}).get(cleaned_player)
            if cleaned_player not in candidate_ids or entry is None:
                raise ValueError("scouting candidate is outside the current market")
            registry, _report = append_scouting_report(
                session.get("scouting_registry"),
                market_id=preview["market_id"],
                team=cleaned_team,
                entry=entry,
                true_quality=market_player_quality(entry["player"]),
            )
            session["scouting_registry"] = registry
            session["updated_at"] = _now()
            _atomic_json(self.session_path, session)
            return self.free_agent_market(cleaned_team)

    def _season_history_view(self, session: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw = session.get("season_history") or []
        if not isinstance(raw, list) or len(raw) > 12:
            raise ValueError("invalid season history")
        history = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, Mapping) or item.get("schema_version") != 1:
                raise ValueError("invalid season archive")
            season_id = str(item.get("season_id") or "")
            plan = SeasonPlan.from_payload(item.get("plan") or {})
            sporting_brief = item.get("sporting_brief")
            sporting_evaluation = item.get("sporting_evaluation")
            if plan.manager_sporting_directive is None:
                if sporting_brief is not None or sporting_evaluation is not None:
                    raise ValueError("invalid archived sporting evidence")
            else:
                if not isinstance(sporting_brief, Mapping) or not isinstance(
                    sporting_evaluation, Mapping
                ):
                    raise ValueError("archived sporting evidence is unavailable")
                validate_sporting_brief(sporting_brief)
                expected_sporting = validate_sporting_choices(
                    plan.manager_sporting_directive,
                    sporting_brief,
                    recruitment=(
                        plan.manager_recruitment.as_dict()
                        if plan.manager_recruitment is not None
                        else None
                    ),
                    retention=(
                        plan.manager_retention.as_dict()
                        if plan.manager_retention is not None
                        else None
                    ),
                    free_agent=(
                        plan.manager_free_agent.as_dict()
                        if plan.manager_free_agent is not None
                        else None
                    ),
                )
                if (
                    sporting_brief.get("target_season_id") != season_id
                    or sporting_brief.get("team") != plan.manager_team
                    or dict(sporting_evaluation) != expected_sporting
                ):
                    raise ValueError("invalid archived sporting evidence")
            participation_by_team = item.get("player_development_evidence")
            if participation_by_team is not None:
                if not isinstance(participation_by_team, Mapping) or list(
                    participation_by_team
                ) != sorted(plan.teams):
                    raise ValueError("invalid archived development evidence")
                for evidence_team, evidence in participation_by_team.items():
                    validate_participation_evidence(evidence)
                    if (
                        evidence.get("team") != evidence_team
                        or evidence.get("season_id") != season_id
                    ):
                        raise ValueError(
                            "archived development evidence identity mismatch"
                        )
            archived_recruitment = item.get("recruitment_transaction")
            if plan.manager_recruitment is None:
                if archived_recruitment is not None:
                    raise ValueError("invalid archived recruitment identity")
            else:
                club = (
                    (session.get("squad_registry") or {})
                    .get("clubs", {})
                    .get(str(plan.manager_team), {})
                )
                matches = (
                    [
                        transaction
                        for transaction in club.get("transactions", [])
                        if transaction.get("season_id") == season_id
                    ]
                    if isinstance(club, Mapping)
                    else []
                )
                if len(matches) != 1 or archived_recruitment != matches[0]:
                    raise ValueError("invalid archived recruitment identity")
            standings = item.get("final_standings")
            profile = item.get("manager_profile")
            expected_commitment_contract = build_commitment_contract_from_sources(
                season_id=season_id,
                plan=item.get("plan") or {},
                club_strategies=item.get("club_strategies"),
            )
            expected_fixture_count = (
                len(plan.teams) * (len(plan.teams) - 1) // 2 * plan.legs
            )
            valid_rows = (
                isinstance(standings, list)
                and len(standings) == len(plan.teams)
                and all(
                    isinstance(row, Mapping)
                    and isinstance(row.get("position"), int)
                    and not isinstance(row.get("position"), bool)
                    and isinstance(row.get("points"), int)
                    and not isinstance(row.get("points"), bool)
                    and row.get("points") >= 0
                    for row in standings
                )
            )
            if (
                not season_id
                or season_id in seen
                or item.get("fixture_count") != expected_fixture_count
                or not valid_rows
                or {row.get("team") for row in standings if isinstance(row, Mapping)}
                != set(plan.teams)
                or {row.get("position") for row in standings}
                != set(range(1, len(plan.teams) + 1))
                or not isinstance(profile, Mapping)
                or item.get("season_commitments") != expected_commitment_contract
                or (
                    plan.manager_team is not None
                    and profile.get("season_id") != season_id
                )
                or (plan.manager_team is None and profile.get("available") is not False)
            ):
                raise ValueError("invalid season archive identity")
            if plan.manager_team is not None:
                objective = profile.get("objective")
                journal = profile.get("journal")
                record = profile.get("record")
                expected_matches = (len(plan.teams) - 1) * plan.legs
                manager_row = next(
                    row for row in standings if row.get("team") == plan.manager_team
                )
                expected_target_position = (
                    1
                    if plan.manager_objective == "champion"
                    else (len(plan.teams) + 1) // 2
                    if plan.manager_objective == "top_half"
                    else None
                )
                objective_met = (
                    manager_row["position"] <= expected_target_position
                    if expected_target_position is not None
                    else manager_row["points"] >= int(plan.manager_points_target or 0)
                )
                maximum_matchday = (
                    len(plan.teams) - 1 if len(plan.teams) % 2 == 0 else len(plan.teams)
                ) * plan.legs
                if (
                    profile.get("available") is not True
                    or profile.get("team") != plan.manager_team
                    or not isinstance(objective, Mapping)
                    or objective.get("kind") != plan.manager_objective
                    or objective.get("status") not in {"achieved", "missed"}
                    or not isinstance(record, Mapping)
                    or not isinstance(journal, list)
                    or len(journal) != expected_matches
                    or objective.get("league_size", len(plan.teams)) != len(plan.teams)
                    or objective.get("target_position") != expected_target_position
                    or objective.get("target_points") != plan.manager_points_target
                    or objective.get("current_position") != manager_row["position"]
                    or objective.get("current_points") != manager_row["points"]
                    or objective.get("remaining_matches") != 0
                    or objective.get("maximum_points") != manager_row["points"]
                    or objective.get("current_snapshot_met") is not objective_met
                    or objective.get("status")
                    != ("achieved" if objective_met else "missed")
                    or record.get("points") != manager_row["points"]
                ):
                    raise ValueError("invalid archived manager profile")
                journal_ids = set()
                prior_journal_entries: list[Mapping[str, Any]] = []
                for entry in journal:
                    if (
                        not isinstance(entry, Mapping)
                        or not isinstance(entry.get("fixture_id"), str)
                        or not entry.get("fixture_id")
                        or entry.get("fixture_id") in journal_ids
                        or entry.get("outcome") not in {"win", "draw", "loss"}
                        or entry.get("opponent") not in plan.teams
                        or not isinstance(entry.get("tactic"), str)
                        or len(entry.get("tactic")) > 64
                        or not isinstance(entry.get("rotation"), str)
                        or len(entry.get("rotation")) > 32
                        or not isinstance(entry.get("matchday"), int)
                        or isinstance(entry.get("matchday"), bool)
                        or not 1 <= entry.get("matchday") <= maximum_matchday
                    ):
                        raise ValueError("invalid archived manager journal")
                    stored_preparation = entry.get("opponent_preparation")
                    if stored_preparation is not None and stored_preparation != (
                        archived_opponent_preparation(
                            plan.manager_team,
                            entry,
                            prior_journal_entries,
                        )
                    ):
                        raise ValueError(
                            "archived opponent preparation does not match journal evidence"
                        )
                    world_transition = entry.get("world_state_transition")
                    if world_transition is not None:
                        home = (
                            plan.manager_team
                            if entry.get("venue") == "home"
                            else entry.get("opponent")
                        )
                        away = (
                            entry.get("opponent")
                            if entry.get("venue") == "home"
                            else plan.manager_team
                        )
                        validate_fixture_world_state_transition(
                            world_transition,
                            season_id=season_id,
                            fixture_id=entry["fixture_id"],
                            match_id=str(entry.get("match_id") or ""),
                            home=str(home),
                            away=str(away),
                        )
                    advice = entry.get("manager_decision_advice")
                    adoption = entry.get("manager_advice_adoption")
                    if advice is not None:
                        validate_manager_decision_advice(advice)
                        archive_seed = item.get("seed")
                        fixture_order = entry.get("order")
                        if (
                            isinstance(archive_seed, bool)
                            or not isinstance(archive_seed, int)
                            or not 0 <= archive_seed <= 2**31 - 1
                            or isinstance(fixture_order, bool)
                            or not isinstance(fixture_order, int)
                            or not 1 <= fixture_order <= (len(plan.teams) + 1) // 2
                            or advice.get("season_id") != season_id
                            or advice.get("fixture_id") != entry["fixture_id"]
                            or advice.get("home") != home
                            or advice.get("away") != away
                            or advice.get("manager_team") != plan.manager_team
                            or advice.get("match_seed")
                            != (
                                archive_seed
                                + int(entry["matchday"]) * 100
                                + fixture_order
                            )
                        ):
                            raise ValueError(
                                "archived world-model manager advice binding mismatch"
                            )
                        if adoption is not None:
                            validate_manager_advice_adoption(
                                adoption,
                                advice=advice,
                                selected_tactic=str(entry.get("tactic") or ""),
                            )
                    elif adoption is not None:
                        raise ValueError(
                            "archived manager advice adoption lacks advice evidence"
                        )
                    future_reviews = entry.get("manager_future_reviews") or []
                    if not isinstance(future_reviews, list):
                        raise ValueError(
                            "archived manager future reviews are invalid"
                        )
                    for review in future_reviews:
                        validate_manager_future_review(
                            review,
                            season_id=season_id,
                            fixture_id=entry["fixture_id"],
                            matchday=entry["matchday"],
                            manager_team=plan.manager_team,
                        )
                    journal_ids.add(entry["fixture_id"])
                    prior_journal_entries.append(entry)
                expected_commitment_progress = commitment_progress_from_evidence(
                    expected_commitment_contract,
                    journal=journal,
                    objective=objective,
                    final=True,
                )
                if profile.get("commitments") != expected_commitment_progress:
                    raise ValueError(
                        "archived season commitment source replay mismatch"
                    )
                player_contract = item.get("player_role_promises")
                if isinstance(player_contract, Mapping):
                    replay_fixtures = [
                        {
                            "fixture_id": entry["fixture_id"],
                            "matchday": entry["matchday"],
                            "home": plan.manager_team,
                            "away": entry["opponent"],
                            "state": "completed",
                            "manager_decision": {"lineup": entry.get("lineup")},
                            "player_promise_availability": entry.get(
                                "player_promise_availability"
                            ),
                        }
                        for entry in journal
                    ]
                    expected_player_progress = player_promise_progress(
                        player_contract,
                        fixtures=replay_fixtures,
                        final=True,
                    )
                    if profile.get("player_role_promises") != expected_player_progress:
                        raise ValueError(
                            "archived player promise source replay mismatch"
                        )
                    participation = (
                        participation_by_team.get(plan.manager_team)
                        if isinstance(participation_by_team, Mapping)
                        else None
                    )
                    expected_outcome = (
                        settle_player_promise_outcomes(
                            player_contract,
                            expected_player_progress,
                            participation,
                        )
                        if isinstance(participation, Mapping)
                        else None
                    )
                    if item.get("player_promise_outcomes") != expected_outcome:
                        raise ValueError(
                            "archived player promise outcome replay mismatch"
                        )
                elif (
                    profile.get("player_role_promises")
                    != {
                        "schema_version": 1,
                        "available": False,
                        "reason": "not_frozen",
                        "entries": [],
                    }
                    or item.get("player_promise_outcomes") is not None
                ):
                    raise ValueError("invalid archived player promise boundary")
                expected_review = manager_board_review(
                    profile,
                    league_size=len(plan.teams),
                )
                if (
                    item.get("board_review") is not None
                    and item.get("board_review") != expected_review
                ):
                    raise ValueError(
                        "archived board review does not match season evidence"
                    )
            elif item.get("board_review") is not None:
                raise ValueError("spectator season cannot have a board review")
            seen.add(season_id)
            normalized = dict(item)
            normalized["board_review"] = (
                expected_review if plan.manager_team is not None else None
            )
            history.append(normalized)
        return history

    def season_status(self) -> dict[str, Any] | None:
        session = self._session()
        season = session.get("season")
        if season is None:
            return None
        validate_season_state(season)
        return self._season_view(season)

    def set_player_role_promises(
        self,
        promise_plan: PlayerPromisePlan,
    ) -> dict[str, Any]:
        """Freeze one named-player plan before the first managed decision."""
        if not isinstance(promise_plan, PlayerPromisePlan):
            raise ValueError("player promises require a PlayerPromisePlan")
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            season = session.get("season")
            if season is None:
                raise ValueError("studio season has not been created")
            validate_season_state(season)
            plan = SeasonPlan.from_payload(season["plan"])
            if plan.manager_team is None:
                raise ValueError("season has no focus team")
            managed = [
                row
                for row in season["fixtures"]
                if plan.manager_team in {row["home"], row["away"]}
            ]
            if season.get("player_role_promises") is not None:
                raise ValueError("player role promises are already frozen")
            if any(
                row.get("manager_decision") is not None
                or row.get("state") != "scheduled"
                or int(row.get("attempts", 0)) != 0
                for row in managed
            ):
                raise ValueError(
                    "player role promises must be frozen before the first decision"
                )
            roster = load_effective_roster(self.root, plan.manager_team)
            contract = build_player_promise_contract(
                season_id=str(season["season_id"]),
                team=plan.manager_team,
                total_managed_matches=len(managed),
                roster=roster,
                plan=promise_plan,
                control="manager",
            )
            season["player_role_promises"] = contract
            season["revision"] = int(season.get("revision", 0)) + 1
            season["updated_at"] = _now()
            session["updated_at"] = _now()
            validate_season_state(season)
            _atomic_json(self.session_path, session)
            return self._season_view(season)

    def set_manager_decision(
        self,
        decision: ManagerDecision,
        *,
        fixture_id: str | None = None,
        expected_revision: int | None = None,
        advice_adoption: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Set or replace the next unstarted managed fixture decision."""
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            season = session.get("season")
            if season is None:
                raise ValueError("studio season has not been created")
            if expected_revision is not None and (
                isinstance(expected_revision, bool)
                or not isinstance(expected_revision, int)
                or expected_revision < 0
            ):
                raise ValueError("manager decision preview revision is invalid")
            if (
                expected_revision is not None
                and int(season.get("revision", 0)) != expected_revision
            ):
                raise ValueError(
                    "manager decision preview is stale; refresh before submitting"
                )
            prepared = self._prepare_manager_decision(
                season,
                decision,
                fixture_id=fixture_id,
            )
            target = prepared["target"]
            target.pop("manager_advice_adoption", None)
            if advice_adoption is not None:
                if (
                    not isinstance(advice_adoption, Mapping)
                    or set(advice_adoption)
                    != {
                        "schema_version",
                        "advice_identity",
                        "intent",
                    }
                    or advice_adoption.get("schema_version") != 1
                ):
                    raise ValueError("manager advice adoption payload is invalid")
                advice = target.get("manager_decision_advice")
                if not isinstance(advice, Mapping):
                    raise ValueError(
                        "manager decision has no world-model advice to adopt"
                    )
                validate_manager_decision_advice(advice)
                if advice_adoption.get("advice_identity") != advice["advice_identity"]:
                    raise ValueError("manager advice identity mismatch")
                if advice.get("issued_revision") != int(season.get("revision", 0)):
                    raise ValueError("manager advice is stale; request fresh advice")
                target["manager_advice_adoption"] = build_manager_advice_adoption(
                    advice,
                    selected_tactic=decision.tactic,
                    intent=str(advice_adoption.get("intent") or ""),
                )
            season["revision"] = int(season.get("revision", 0)) + 1
            season["updated_at"] = _now()
            session["updated_at"] = _now()
            validate_season_state(season)
            _atomic_json(self.session_path, session)
            return self._season_view(season)

    def preview_manager_decision(
        self,
        decision: ManagerDecision,
        *,
        fixture_id: str | None = None,
    ) -> dict[str, Any]:
        """Normalize a decision and expose deterministic effects without writing state."""
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            source = session.get("season")
            if source is None:
                raise ValueError("studio season has not been created")
            base_revision = int(source.get("revision", 0))
            season = copy.deepcopy(source)
            prepared = self._prepare_manager_decision(
                season,
                decision,
                fixture_id=fixture_id,
            )
            target = prepared["target"]
            frozen = target["manager_decision"]
            lineup = frozen.get("lineup")
            plan = frozen.get("in_match_plan")
            rotation_tradeoff = {
                "strongest": {"status_penalty": 0.0, "rotation_level": 0.0},
                "balanced": {"status_penalty": 1.25, "rotation_level": 0.5},
                "rotate": {"status_penalty": 3.0, "rotation_level": 1.0},
            }[frozen["rotation"]]

            commitment_projection = None
            contract = season.get("season_commitments")
            if isinstance(contract, Mapping):
                journal = []
                for row in season["fixtures"]:
                    if row.get("state") != "completed" or contract["team"] not in {
                        row.get("home"),
                        row.get("away"),
                    }:
                        continue
                    recorded = row.get("manager_decision")
                    journal.append(
                        {
                            "fixture_id": row.get("fixture_id"),
                            "matchday": row.get("matchday"),
                            "tactic": recorded.get("tactic", "unrecorded")
                            if isinstance(recorded, Mapping)
                            else "unrecorded",
                            "rotation": recorded.get("rotation", "unrecorded")
                            if isinstance(recorded, Mapping)
                            else "unrecorded",
                        }
                    )
                journal.append(
                    {
                        "fixture_id": target["fixture_id"],
                        "matchday": target["matchday"],
                        "tactic": frozen["tactic"],
                        "rotation": frozen["rotation"],
                    }
                )
                objective = manager_season_profile(season)["objective"]
                commitment_projection = commitment_progress_from_evidence(
                    contract,
                    journal=journal,
                    objective=objective,
                    final=False,
                )

            player_projection = None
            player_contract = season.get("player_role_promises")
            if isinstance(player_contract, Mapping):
                projected_fixtures = copy.deepcopy(season["fixtures"])
                projected_target = next(
                    row
                    for row in projected_fixtures
                    if row["fixture_id"] == target["fixture_id"]
                )
                projected_target["state"] = "completed"
                player_projection = player_promise_progress(
                    player_contract,
                    fixtures=projected_fixtures,
                    final=False,
                )

            timeline = prepared.get("timeline_resolution")
            advice_preview = None
            advice = target.get("manager_decision_advice")
            if isinstance(advice, Mapping):
                validate_manager_decision_advice(advice)
                advice_preview = {
                    **copy.deepcopy(dict(advice)),
                    "current": advice.get("issued_revision") == base_revision,
                    "selected_tactic": frozen["tactic"],
                    "selected_alignment": (
                        frozen["tactic"] == advice.get("recommended_tactic")
                    ),
                    "comparison": build_manager_advice_comparison(
                        advice,
                        selected_tactic=frozen["tactic"],
                    ),
                }
            return {
                "schema_version": 1,
                "season_id": str(season["season_id"]),
                "base_revision": base_revision,
                "fixture": {
                    "fixture_id": target["fixture_id"],
                    "matchday": target["matchday"],
                    "home": target["home"],
                    "away": target["away"],
                },
                "normalized_decision": copy.deepcopy(frozen),
                "lineup": {
                    "available": isinstance(lineup, Mapping),
                    "source": lineup.get("source")
                    if isinstance(lineup, Mapping)
                    else None,
                    "starters": len(lineup.get("starters") or [])
                    if isinstance(lineup, Mapping)
                    else 0,
                    "bench": len(lineup.get("bench") or [])
                    if isinstance(lineup, Mapping)
                    else 0,
                },
                "engine_tradeoff": {
                    **rotation_tradeoff,
                    "boundary": "fixed engine mechanics, not a match-outcome estimate",
                },
                "in_match_plan": {
                    "controls_substitutions": bool(
                        isinstance(plan, Mapping) and plan.get("controls_substitutions")
                    ),
                    "rule_count": len(plan.get("instructions") or [])
                    if isinstance(plan, Mapping)
                    else 0,
                },
                "club_situation": {
                    "available": timeline is not None,
                    "resolution": copy.deepcopy(timeline),
                },
                "opponent_preparation": copy.deepcopy(
                    target.get("opponent_preparation")
                ),
                "season_commitment_projection": commitment_projection,
                "player_promise_projection": player_projection,
                "world_model_advice": advice_preview,
                "claim_boundary": (
                    "deterministic pre-submit preview of controlled mechanics only; "
                    "it writes no season state and makes no score, win-probability, "
                    "causal or real-world performance claim"
                ),
            }

    def manager_future_set_context(
        self, *, fixture_id: str | None = None,
    ) -> dict[str, Any]:
        """Return one read-only future-set context from authoritative season state."""
        if self.config.mode != "research":
            raise ValueError("manager future sets require research mode")
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            season = session.get("season")
            if not isinstance(season, Mapping):
                raise ValueError("studio season has not been created")
            return build_manager_future_context(season, fixture_id=fixture_id)

    def validate_manager_future_set_context(
        self, context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recheck a queued future set without persisting a second state source."""
        if self.config.mode != "research":
            raise ValueError("manager future sets require research mode")
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            season = session.get("season")
            if not isinstance(season, Mapping):
                raise ValueError("studio season has not been created")
            return validate_manager_future_context(context, season)

    def review_manager_future_set(
        self,
        *,
        task_id: str,
        task_request: Mapping[str, Any],
        task_result: Mapping[str, Any],
        intent: str,
        fixture_id: str,
        expected_revision: int,
        final_decision: ManagerDecision | None = None,
    ) -> dict[str, Any]:
        """Record an explicit evidence review and atomically freeze its decision."""
        if self.config.mode != "research":
            raise ValueError("manager future reviews require research mode")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("manager future review revision is invalid")
        if intent == "keep_after_review":
            if final_decision is not None:
                raise ValueError(
                    "keep-after-review cannot supply a replacement decision"
                )
        elif intent == "revise_after_review":
            if not isinstance(final_decision, ManagerDecision):
                raise ValueError(
                    "revise-after-review requires a replacement decision"
                )
        else:
            raise ValueError("manager future review intent is invalid")
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            season = session.get("season")
            if not isinstance(season, dict):
                raise ValueError("studio season has not been created")
            target = next(
                (
                    row for row in season["fixtures"]
                    if row["fixture_id"] == fixture_id
                ),
                None,
            )
            if not isinstance(target, dict):
                raise ValueError("manager future review fixture is unavailable")
            reviews = target.get("manager_future_reviews") or []
            if not isinstance(reviews, list):
                raise ValueError("manager future review history is invalid")
            existing = next(
                (
                    row for row in reviews
                    if isinstance(row, Mapping)
                    and row.get("task_id") == task_id
                ),
                None,
            )
            if existing is not None:
                current_payload = target.get("manager_decision")
                if not isinstance(current_payload, Mapping):
                    raise ValueError(
                        "manager future review requires a frozen decision"
                    )
                if intent == "keep_after_review":
                    repeated_final = current_payload
                else:
                    preview_season = copy.deepcopy(season)
                    preview_target = next(
                        row for row in preview_season["fixtures"]
                        if row["fixture_id"] == fixture_id
                    )
                    preview_target.pop("manager_advice_adoption", None)
                    repeated_final = self._prepare_manager_decision(
                        preview_season,
                        final_decision,
                        fixture_id=fixture_id,
                    )["target"]["manager_decision"]
                repeated = build_manager_future_review(
                    task_id=task_id,
                    request=task_request,
                    result=task_result,
                    final_decision=repeated_final,
                    intent=intent,
                )
                if repeated != existing:
                    raise ValueError(
                        "manager future review task has a conflicting replay"
                    )
                return self._season_view(season)
            if int(season.get("revision", 0)) != expected_revision:
                raise ValueError(
                    "manager future review is stale; refresh before recording"
                )
            context = validate_manager_future_context(
                task_request.get("manager_context"), season,
            )
            if context["fixture"]["fixture_id"] != fixture_id:
                raise ValueError("manager future review fixture identity mismatch")
            expected_artifact = (
                self.output_root / "fork_sets" / task_id / "result.json"
            ).resolve()
            expected_dashboard = expected_artifact.with_name("index.html")
            raw_artifact = Path(str(
                task_result.get("fork_set_result") or ""
            ))
            raw_dashboard = Path(str(
                task_result.get("fork_set_dashboard") or ""
            ))
            artifact_path = (
                raw_artifact
                if raw_artifact.is_absolute()
                else self.root / raw_artifact
            ).resolve()
            dashboard_path = (
                raw_dashboard
                if raw_dashboard.is_absolute()
                else self.root / raw_dashboard
            ).resolve()
            if (
                artifact_path != expected_artifact
                or dashboard_path != expected_dashboard
                or not artifact_path.is_file()
                or not dashboard_path.is_file()
            ):
                raise ValueError(
                    "manager future review artifacts are unavailable"
                )
            try:
                artifact = json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError(
                    "manager future review result artifact is invalid"
                ) from exc
            fixture = context["fixture"]
            if (
                not isinstance(artifact, Mapping)
                or artifact.get("schema_version") != 1
                or artifact.get("set_id") != task_id
                or artifact.get("status") != "complete"
                or artifact.get("fixture") != {
                    "home": fixture["home"], "away": fixture["away"],
                }
                or artifact.get("source_context") != context
                or artifact.get("plan") != task_request.get("plan")
                or artifact.get("aggregate") != task_result.get("aggregate")
                or (
                    "scenario_evidence" in task_result
                    and project_fork_set_scenario_evidence(artifact)
                    != task_result.get("scenario_evidence")
                )
                or artifact.get("claim_authority")
                != task_result.get("claim_authority")
            ):
                raise ValueError(
                    "manager future review artifact identity mismatch"
                )
            if (
                len(reviews) >= MAX_REVIEWS_PER_FIXTURE
                or any(
                    row.get("context_identity") == context["context_identity"]
                    for row in reviews
                    if isinstance(row, Mapping)
                )
            ):
                raise ValueError(
                    "manager future review is duplicated or at capacity"
                )
            current_payload = target.get("manager_decision")
            if not isinstance(current_payload, Mapping):
                raise ValueError(
                    "manager future review requires a frozen decision"
                )
            if intent == "keep_after_review":
                selected = ManagerDecision.from_payload(current_payload)
            else:
                selected = final_decision
            target.pop("manager_advice_adoption", None)
            prepared = self._prepare_manager_decision(
                season, selected, fixture_id=fixture_id,
            )
            target = prepared["target"]
            receipt = build_manager_future_review(
                task_id=task_id,
                request=task_request,
                result=task_result,
                final_decision=target["manager_decision"],
                intent=intent,
            )
            target.setdefault("manager_future_reviews", []).append(receipt)
            season["revision"] = int(season.get("revision", 0)) + 1
            season["updated_at"] = _now()
            session["updated_at"] = _now()
            validate_season_state(season)
            _atomic_json(self.session_path, session)
            return self._season_view(season)

    def request_manager_decision_advice(
        self,
        *,
        fixture_id: str | None = None,
    ) -> dict[str, Any]:
        """Run inference outside the session lease, then commit if inputs still match."""
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            season = session.get("season")
            if season is None:
                raise ValueError("studio season has not been created")
            validate_season_state(season)
            plan = SeasonPlan.from_payload(season["plan"])
            if plan.manager_team is None:
                raise ValueError("season has no focus team")
            candidates = [
                row
                for row in season["fixtures"]
                if plan.manager_team in {row["home"], row["away"]}
                and row["state"] != "completed"
            ]
            if not candidates:
                raise ValueError("focus team has no pending fixture")
            target = candidates[0]
            if fixture_id is not None and fixture_id != target["fixture_id"]:
                raise ValueError("only the next focus-team fixture can receive advice")
            if target["state"] != "scheduled" or int(target.get("attempts", 0)) != 0:
                raise ValueError(
                    "world-model advice is frozen after fixture execution starts"
                )
            mode = str(session["mode"])
            unavailable = {
                "schema_version": 1,
                "available": False,
                "season_id": str(season["season_id"]),
                "fixture_id": str(target["fixture_id"]),
                "mode": mode,
                "reason": "stable_mode_has_no_world_model_advisor",
                "claim_boundary": (
                    "stable mode does not load or imitate the research world model"
                    if mode == "stable"
                    else "research advisor unavailable; no advice or adoption was persisted"
                ),
            }
            if mode == "stable":
                return unavailable
            match_seed = (
                int(season["seed"])
                + int(target["matchday"]) * 100
                + int(target["order"])
            )
            request_state = {
                "season_id": str(season["season_id"]),
                "base_revision": int(season.get("revision", 0)),
                "mode": mode,
                "manager_team": str(plan.manager_team),
                "match_seed": match_seed,
                "fixture": copy.deepcopy(dict(target)),
                "season": copy.deepcopy(dict(season)),
                "existing_advice": copy.deepcopy(target.get("manager_decision_advice")),
            }

        readiness = self.readiness()
        if not readiness.get("ready"):
            unavailable["reason"] = "research_readiness_gate_closed"
            unavailable["blockers"] = list(readiness.get("blockers") or [])
            return unavailable
        evidence = self.evidence()
        checkpoint_artifact = str(evidence.get("world_model_checkpoint") or "")
        checkpoint_sha256 = str(evidence.get("world_model_checkpoint_sha256") or "")
        teams = (
            str(request_state["fixture"]["home"]),
            str(request_state["fixture"]["away"]),
        )
        snapshots = capture_world_state(self.root, teams)
        existing_advice = request_state["existing_advice"]
        existing_matches_inputs = False
        if isinstance(existing_advice, Mapping):
            validate_manager_decision_advice(existing_advice)
            existing_sources = existing_advice["representative_state"][
                "carryover_sources"
            ]
            existing_matches_inputs = bool(
                existing_advice.get("season_id") == request_state["season_id"]
                and existing_advice.get("fixture_id")
                == request_state["fixture"]["fixture_id"]
                and existing_advice.get("home") == teams[0]
                and existing_advice.get("away") == teams[1]
                and existing_advice.get("manager_team") == request_state["manager_team"]
                and existing_advice.get("mode") == request_state["mode"]
                and existing_advice.get("match_seed") == request_state["match_seed"]
                and existing_advice.get("issued_revision")
                == request_state["base_revision"]
                and existing_advice["checkpoint"]["artifact"] == checkpoint_artifact
                and existing_advice["checkpoint"]["sha256"] == checkpoint_sha256
                and all(
                    existing_sources[team]["source_identity"]
                    == snapshots[team]["source_identity"]
                    for team in teams
                )
            )
        if existing_matches_inputs:
            with FileLease(self.session_lease_path, timeout=30.0):
                current_session = self._session()
                current_season = current_session.get("season")
                if isinstance(current_season, Mapping):
                    validate_season_state(current_season)
                    current_plan = SeasonPlan.from_payload(current_season["plan"])
                    current_candidates = (
                        [
                            row
                            for row in current_season["fixtures"]
                            if current_plan.manager_team in {row["home"], row["away"]}
                            and row["state"] != "completed"
                        ]
                        if current_plan.manager_team is not None
                        else []
                    )
                    current_target = (
                        current_candidates[0] if current_candidates else None
                    )
                    expected_fixture = request_state["fixture"]
                    fixture_fields = (
                        "fixture_id",
                        "matchday",
                        "order",
                        "home",
                        "away",
                        "state",
                        "attempts",
                    )
                    session_matches = bool(
                        current_session.get("mode") == request_state["mode"]
                        and str(current_season.get("season_id") or "")
                        == request_state["season_id"]
                        and int(current_season.get("revision", -1))
                        == request_state["base_revision"]
                        and current_plan.manager_team == request_state["manager_team"]
                        and isinstance(current_target, Mapping)
                        and all(
                            current_target.get(field) == expected_fixture.get(field)
                            for field in fixture_fields
                        )
                        and current_target.get("manager_decision_advice")
                        == existing_advice
                    )
                    current_evidence = self.evidence()
                    checkpoint_path = _artifact_path(self.root, checkpoint_artifact)
                    checkpoint_matches = bool(
                        current_evidence.get("world_model_checkpoint")
                        == checkpoint_artifact
                        and current_evidence.get("world_model_checkpoint_sha256")
                        == checkpoint_sha256
                        and checkpoint_path is not None
                        and checkpoint_path.is_file()
                        and file_sha256(checkpoint_path) == checkpoint_sha256
                    )
                    current_snapshots = capture_world_state(self.root, teams)
                    state_matches = all(
                        current_snapshots[team]["source_identity"]
                        == snapshots[team]["source_identity"]
                        for team in teams
                    )
                    if session_matches and checkpoint_matches and state_matches:
                        return {
                            **copy.deepcopy(existing_advice),
                            "available": True,
                            "reused": True,
                        }
            return {
                **unavailable,
                "reason": "advice_inputs_changed_during_generation",
                "retryable": True,
            }
        packet = self._generate_manager_decision_advice_packet(
            season=request_state["season"],
            fixture=request_state["fixture"],
            manager_team=request_state["manager_team"],
            match_seed=request_state["match_seed"],
        )
        if not isinstance(packet, Mapping) or packet.get("available") is not True:
            unavailable["reason"] = str(
                (packet or {}).get("reason", "world_model_advice_unavailable")
            )[:160]
            return unavailable
        advice = build_manager_decision_advice(
            packet=packet,
            season_id=request_state["season_id"],
            fixture_id=str(request_state["fixture"]["fixture_id"]),
            home=teams[0],
            away=teams[1],
            manager_team=request_state["manager_team"],
            mode=request_state["mode"],
            match_seed=request_state["match_seed"],
            base_revision=request_state["base_revision"],
            checkpoint_artifact=checkpoint_artifact,
            checkpoint_sha256=checkpoint_sha256,
            state_snapshots=snapshots,
        )

        with FileLease(self.session_lease_path, timeout=30.0):
            current_session = self._session()
            current_season = current_session.get("season")
            stale = {
                **unavailable,
                "reason": "advice_inputs_changed_during_generation",
                "retryable": True,
            }
            if not isinstance(current_season, Mapping):
                return stale
            validate_season_state(current_season)
            current_plan = SeasonPlan.from_payload(current_season["plan"])
            current_candidates = (
                [
                    row
                    for row in current_season["fixtures"]
                    if current_plan.manager_team in {row["home"], row["away"]}
                    and row["state"] != "completed"
                ]
                if current_plan.manager_team is not None
                else []
            )
            current_target = current_candidates[0] if current_candidates else None
            expected_fixture = request_state["fixture"]
            fixture_fields = (
                "fixture_id",
                "matchday",
                "order",
                "home",
                "away",
                "state",
                "attempts",
            )
            subject_matches = bool(
                current_session.get("mode") == request_state["mode"]
                and str(current_season.get("season_id") or "")
                == request_state["season_id"]
                and current_plan.manager_team == request_state["manager_team"]
                and isinstance(current_target, Mapping)
                and all(
                    current_target.get(field) == expected_fixture.get(field)
                    for field in fixture_fields
                )
            )
            if not subject_matches:
                return stale
            current_evidence = self.evidence()
            checkpoint_path = _artifact_path(self.root, checkpoint_artifact)
            checkpoint_matches = bool(
                current_evidence.get("world_model_checkpoint") == checkpoint_artifact
                and current_evidence.get("world_model_checkpoint_sha256")
                == checkpoint_sha256
                and checkpoint_path is not None
                and checkpoint_path.is_file()
                and file_sha256(checkpoint_path) == checkpoint_sha256
            )
            current_snapshots = capture_world_state(self.root, teams)
            state_matches = all(
                current_snapshots[team]["source_identity"]
                == snapshots[team]["source_identity"]
                for team in teams
            )
            if not checkpoint_matches or not state_matches:
                return stale
            current_advice = current_target.get("manager_decision_advice")
            if (
                int(current_season.get("revision", -1)) == advice["issued_revision"]
                and current_advice == advice
            ):
                return {
                    **copy.deepcopy(current_advice),
                    "available": True,
                    "reused": True,
                }
            if (
                int(current_season.get("revision", -1))
                != request_state["base_revision"]
            ):
                return stale
            current_target["manager_decision_advice"] = advice
            current_target.pop("manager_advice_adoption", None)
            current_season["revision"] = request_state["base_revision"] + 1
            current_season["updated_at"] = _now()
            current_session["updated_at"] = _now()
            validate_season_state(current_season)
            _atomic_json(self.session_path, current_session)
            return {**copy.deepcopy(advice), "available": True}

    def _generate_manager_decision_advice_packet(
        self,
        *,
        season: Mapping[str, Any],
        fixture: Mapping[str, Any],
        manager_team: str,
        match_seed: int,
    ) -> dict[str, Any]:
        """Use the match engine's existing bounded tactical-policy evaluator."""
        from src.match_engine.macro_bridge import build_match_affective_state
        from src.match_engine.tactical_profile import (
            apply_locked_tactical_preset,
            build_tactical_vector_for_agent,
        )
        from src.match_engine.world_model.decision_support import (
            build_prematch_tactical_packet,
        )
        from src.match_engine.world_model.inference import WorldModelRuntime
        from src.product.match_plan import PLAYABLE_TACTICS
        from src.simulation.fusion_audit import fusion_audit_path, load_fusion_audits
        from src.simulation.counterfactual_evidence import (
            counterfactual_evidence_path,
            load_counterfactual_evidence,
        )
        from src.simulation.fusion_reliability import (
            enrich_decision_packet_with_history,
        )
        from src.simulation.match_pipeline import prepare_match_agents
        from src.simulation.random_control import named_rng
        from src.simulation.world_cup_runner import build_world_and_tournament

        home, away = str(fixture["home"]), str(fixture["away"])
        opponent = away if manager_team == home else home
        with self._mode_environment():
            world, _, _ = build_world_and_tournament(
                str(self.root),
                require_tactics=False,
                load_coaches=True,
                initialization_seed=match_seed,
            )
            home_agent, away_agent = world.agents[home], world.agents[away]
            prepare_match_agents(home_agent, away_agent, str(self.root))
            preparation = opponent_preparation(
                season,
                str(fixture["fixture_id"]),
            )
            strategy_snapshot = season.get("club_strategies")
            resolved_strategy = None
            if isinstance(strategy_snapshot, Mapping):
                resolved_strategy = resolve_fixture_club_strategy(
                    strategy_snapshot,
                    home=home,
                    away=away,
                    manager_decision=None,
                    opponent_preparation=preparation,
                )
            for side, agent in (("home", home_agent), ("away", away_agent)):
                tactic = (
                    str(resolved_strategy.get(f"{side}_tactic") or "team_identity")
                    if isinstance(resolved_strategy, Mapping)
                    else "team_identity"
                )
                if agent.team_name == opponent:
                    prepared_tactic = str(
                        preparation.get("selected_tactic") or "team_identity"
                    )
                    if prepared_tactic != "team_identity":
                        tactic = prepared_tactic
                if tactic != "team_identity":
                    apply_locked_tactical_preset(
                        agent,
                        tactic,
                        source="manager_world_model_advice_context",
                    )
            manager_agent = home_agent if manager_team == home else away_agent
            native_vector = build_tactical_vector_for_agent(manager_agent)
            state = build_match_affective_state(
                home_agent,
                away_agent,
                rng=named_rng(
                    match_seed,
                    "manager_world_model_advice",
                    season["season_id"],
                    fixture["fixture_id"],
                ),
            )
            team_state = state.team(manager_team)
            carrier = next(
                (player for player in team_state.players if player.role == "CM"),
                team_state.players[0],
            )
            state.ball.position = carrier.position.copy()
            state.ball.velocity[:] = 0.0
            state.ball.possessor_id = carrier.player_id
            state.ball.possession_team_id = team_state.team_id
            runtime = WorldModelRuntime.load_default(
                str(self.root),
                required=True,
            )
            packet = build_prematch_tactical_packet(
                runtime,
                state,
                manager_team,
                candidate_presets=tuple(PLAYABLE_TACTICS),
                native_tactical_vector=native_vector,
            )
            if packet.get("available"):
                packet = enrich_decision_packet_with_history(
                    packet,
                    team=manager_team,
                    opponent=opponent,
                    audit_records=load_fusion_audits(fusion_audit_path(str(self.root))),
                    counterfactual_records=load_counterfactual_evidence(
                        counterfactual_evidence_path(str(self.root))
                    ),
                )
            return packet

    def _prepare_manager_decision(
        self,
        season: dict[str, Any],
        decision: ManagerDecision,
        *,
        fixture_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply authoritative decision normalization to an in-memory season."""
        validate_season_state(season)
        plan = SeasonPlan.from_payload(season["plan"])
        if plan.manager_team is None:
            raise ValueError("season has no focus team")
        if decision.team != plan.manager_team:
            raise ValueError("manager decision team identity mismatch")
        candidates = [
            item
            for item in season["fixtures"]
            if plan.manager_team in {item["home"], item["away"]}
            and item["state"] != "completed"
        ]
        if not candidates:
            raise ValueError("focus team has no pending fixture")
        if fixture_id is not None and fixture_id != candidates[0]["fixture_id"]:
            raise ValueError("only the next focus-team fixture can receive a decision")
        target = (
            next(
                (item for item in candidates if item["fixture_id"] == fixture_id), None
            )
            if fixture_id is not None
            else candidates[0]
        )
        if target is None:
            raise ValueError("manager fixture identity mismatch")
        if target["state"] != "scheduled" or int(target.get("attempts", 0)) != 0:
            raise ValueError(
                "manager decision is frozen after fixture execution starts"
            )
        squad = build_squad_catalog(self.root, plan.manager_team)
        if season.get("player_role_promises") is None:
            season["player_role_promises"] = build_player_promise_contract(
                season_id=str(season["season_id"]),
                team=plan.manager_team,
                total_managed_matches=len(
                    [
                        row
                        for row in season["fixtures"]
                        if plan.manager_team in {row["home"], row["away"]}
                    ]
                ),
                roster=load_effective_roster(self.root, plan.manager_team),
                plan=None,
                control="deterministic_compatibility",
            )
        promised_ids = [
            row["player_id"] for row in season["player_role_promises"]["players"]
        ]
        squad_players = {
            str(row.get("player_id") or ""): row
            for row in squad.get("players") or []
            if isinstance(row, Mapping)
        }
        if any(player_id not in squad_players for player_id in promised_ids):
            raise ValueError("promised player is missing from the matchday squad")
        target["player_promise_availability"] = {
            player_id: bool(squad_players[player_id].get("selectable"))
            for player_id in promised_ids
        }
        frozen_lineup = decision.lineup
        if squad.get("available"):
            frozen_lineup = (
                automatic_lineup(squad, decision.rotation)
                if frozen_lineup is None
                else validate_and_freeze_lineup(squad, frozen_lineup)
            )
        elif frozen_lineup is not None:
            raise ValueError("manager squad is unavailable for a manual lineup")
        elif squad.get("reason") not in {
            "roster_unavailable",
            "roster_missing_goalkeeper",
        }:
            raise ValueError("manager squad cannot form a legal lineup")
        frozen_in_match_plan = decision.in_match_plan
        if (
            frozen_in_match_plan is not None
            and frozen_in_match_plan.controls_substitutions
        ):
            if frozen_lineup is None:
                raise ValueError(
                    "planned substitutions require an available frozen lineup"
                )
            frozen_in_match_plan.validate_lineup(
                starters=frozen_lineup.starters,
                bench=frozen_lineup.bench,
            )
        situation = derive_club_situation(season, target["fixture_id"])
        event_choice = decision.club_event_choice
        timeline_resolution = None
        if situation is None:
            if event_choice is not None:
                raise ValueError(
                    "manager decision supplied a choice without a club situation"
                )
        else:
            control = "manager"
            if event_choice is None:
                event_choice = compatible_choice(
                    situation,
                    tactic=decision.tactic,
                    rotation=decision.rotation,
                )
                control = "deterministic_compatibility"
            timeline_resolution = build_timeline_resolution(
                situation,
                event_choice,
                tactic=decision.tactic,
                rotation=decision.rotation,
                control=control,
            )
        frozen_decision = ManagerDecision(
            team=decision.team,
            tactic=decision.tactic,
            rotation=decision.rotation,
            lineup=frozen_lineup,
            in_match_plan=frozen_in_match_plan,
            club_event_choice=event_choice,
        )
        target["manager_decision"] = frozen_decision.as_dict()
        if timeline_resolution is not None:
            timeline = season.setdefault(
                "club_timeline",
                {"schema_version": 1, "events": []},
            )
            events = timeline["events"]
            events[:] = [
                row
                for row in events
                if row["event"]["fixture_id"] != target["fixture_id"]
            ]
            events.append(timeline_resolution)
        target["opponent_preparation"] = opponent_preparation(
            season,
            target["fixture_id"],
        )
        return {"target": target, "timeline_resolution": timeline_resolution}

    def _season_view(self, season: Mapping[str, Any]) -> dict[str, Any]:
        fixtures = season.get("fixtures") or []
        completed = sum(
            1 for fixture in fixtures if fixture.get("state") == "completed"
        )
        plan = SeasonPlan.from_payload(season["plan"])
        next_manager_fixture = next(
            (
                dict(fixture)
                for fixture in fixtures
                if plan.manager_team is not None
                and plan.manager_team in {fixture.get("home"), fixture.get("away")}
                and fixture.get("state") != "completed"
            ),
            None,
        )
        command_center = matchday_command_center(season)
        timeline = club_timeline_view(season)
        manager_profile = manager_season_profile(season)
        command_center["club_situation"] = timeline["current_situation"]
        command_center["season_commitments"] = manager_profile.get("commitments")
        command_center["player_role_promises"] = manager_profile.get(
            "player_role_promises"
        )
        manager_squad = (
            build_squad_catalog(self.root, plan.manager_team)
            if plan.manager_team is not None and next_manager_fixture is not None
            else None
        )
        command_fixture = command_center.get("current_fixture") or command_center.get(
            "next_fixture"
        )
        opponent = (
            command_fixture.get("opponent")
            if isinstance(command_fixture, Mapping)
            else None
        )
        strategy_briefing = {
            "schema_version": 1,
            "available": False,
            "reason": "strategy_snapshot_unavailable",
        }
        strategy_snapshot = season.get("club_strategies")
        if (
            isinstance(strategy_snapshot, Mapping)
            and plan.manager_team is not None
            and isinstance(next_manager_fixture, Mapping)
        ):
            strategy_profiles = {
                profile["team"]: profile
                for profile in strategy_snapshot.get("profiles") or []
                if isinstance(profile, Mapping) and isinstance(profile.get("team"), str)
            }
            strategy_opponent = (
                next_manager_fixture["away"]
                if next_manager_fixture["home"] == plan.manager_team
                else next_manager_fixture["home"]
            )
            manager_strategy = strategy_profiles.get(plan.manager_team)
            opponent_strategy = strategy_profiles.get(strategy_opponent)
            if manager_strategy is not None and opponent_strategy is not None:
                opponent_side = (
                    "away"
                    if next_manager_fixture["away"] == strategy_opponent
                    else "home"
                )
                identity_tactic = opponent_strategy[f"{opponent_side}_tactic"]
                applied_preview = None
                decision_payload = next_manager_fixture.get("manager_decision")
                preparation_payload = next_manager_fixture.get("opponent_preparation")
                if isinstance(decision_payload, Mapping) and isinstance(
                    preparation_payload, Mapping
                ):
                    applied_preview = resolve_fixture_club_strategy(
                        strategy_snapshot,
                        home=next_manager_fixture["home"],
                        away=next_manager_fixture["away"],
                        manager_decision=decision_payload,
                        opponent_preparation=preparation_payload,
                    )
                strategy_briefing = {
                    "schema_version": 1,
                    "available": True,
                    "snapshot_identity": strategy_identity(strategy_snapshot),
                    "manager": dict(manager_strategy),
                    "opponent": dict(opponent_strategy),
                    "opponent_identity_tactic": identity_tactic,
                    "applied_preview": applied_preview,
                    "claim_boundary": (
                        "frozen simulated season identity and fixture tactic; "
                        "not a win prediction or real coaching recommendation"
                    ),
                }
        command_center["club_strategy"] = strategy_briefing
        opponent_squad = (
            build_squad_catalog(self.root, str(opponent)) if opponent else None
        )
        command_center["prematch_intelligence"] = build_prematch_intelligence(
            command_center,
            manager_squad,
            opponent_squad,
        )
        last_result = command_center.get("last_result")
        if isinstance(last_result, Mapping) and plan.manager_team is not None:
            report_path = _artifact_path(self.root, last_result.get("report"))
            postmatch_debrief = {
                "schema_version": 1,
                "available": False,
                "reason": "report_reference_unavailable",
            }
            if report_path is not None and report_path.suffix.lower() == ".json":
                try:
                    if (
                        not report_path.is_file()
                        or report_path.stat().st_size > 32 * 1024 * 1024
                    ):
                        raise OSError("manager report is unavailable or oversized")
                    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
                    postmatch_debrief = build_postmatch_debrief(
                        report_payload,
                        manager_team=plan.manager_team,
                        expected_match_id=str(last_result.get("match_id") or ""),
                    )
                except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
                    postmatch_debrief = {
                        "schema_version": 1,
                        "available": False,
                        "reason": "report_unreadable_or_invalid",
                    }
            command_center["postmatch_debrief"] = postmatch_debrief
        decision_ledger = build_manager_decision_ledger(
            season,
            execution_by_fixture=self._manager_execution_evidence(
                season,
                manager_team=plan.manager_team,
            )
            if plan.manager_team is not None
            else {},
        )
        command_center["decision_ledger"] = decision_ledger
        return {
            **dict(season),
            "standings": season_standings(season),
            "next_matchday": next_matchday(season),
            "matchday_command_center": command_center,
            "manager_profile": manager_profile,
            "progress": {"completed": completed, "total": len(fixtures)},
            "next_manager_fixture": next_manager_fixture,
            "manager_squad": manager_squad,
            "club_timeline_view": timeline,
            "season_commitment_view": manager_profile.get("commitments"),
            "player_promise_view": manager_profile.get("player_role_promises"),
            "manager_decision_ledger": decision_ledger,
        }

    def _manager_execution_evidence(
        self,
        season: Mapping[str, Any],
        *,
        manager_team: str,
        live_window: int = 8,
    ) -> dict[str, dict[str, Any]]:
        """Load a bounded recent window through the strict postmatch parser."""
        completed = [
            row
            for row in season.get("fixtures") or []
            if row.get("state") == "completed"
            and manager_team in {row.get("home"), row.get("away")}
            and isinstance(row.get("manager_decision"), Mapping)
        ][-live_window:]
        result: dict[str, dict[str, Any]] = {}
        for fixture in completed:
            fixture_id = str(fixture.get("fixture_id") or "")
            report_path = _artifact_path(self.root, fixture.get("report"))
            if report_path is None or report_path.suffix.lower() != ".json":
                result[fixture_id] = {
                    "schema_version": 1,
                    "available": False,
                    "reason": "report_reference_unavailable",
                }
                continue
            try:
                if (
                    not report_path.is_file()
                    or report_path.stat().st_size > 32 * 1024 * 1024
                ):
                    raise OSError("manager report is unavailable or oversized")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                result[fixture_id] = build_postmatch_debrief(
                    report,
                    manager_team=manager_team,
                    expected_match_id=str(fixture.get("match_id") or ""),
                )
            except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
                result[fixture_id] = {
                    "schema_version": 1,
                    "available": False,
                    "reason": "report_unreadable_or_invalid",
                }
        return result

    def play_next_matchday(
        self,
        *,
        expected_season_id: str | None = None,
        expected_matchday: int | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Complete or resume the earliest unfinished matchday exactly once."""
        with FileLease(self.session_lease_path, timeout=30.0):
            session = self._session()
            season = session.get("season")
            if season is None:
                raise ValueError("studio season has not been created")
            validate_season_state(season)
            if (
                expected_season_id is not None
                and season.get("season_id") != expected_season_id
            ):
                raise ValueError("season task identity conflict")
            if (
                expected_revision is not None
                and int(season.get("revision", 0)) != expected_revision
            ):
                raise ValueError("season decision revision conflict")
            matchday = next_matchday(season)
            if expected_matchday is not None and matchday != expected_matchday:
                if expected_matchday in (season.get("recovered_matchdays") or []):
                    return self._season_view(season)
                raise ValueError("season matchday task identity conflict")
            if matchday is None:
                return self._season_view(season)
            season_id = str(season["season_id"])
            season_plan = SeasonPlan.from_payload(season["plan"])
            resource_plan = season_plan.manager_resources
            resource_effects = (
                club_resource_effects(resource_plan)
                if resource_plan is not None
                else None
            )
            fixture_ids = [
                fixture["fixture_id"]
                for fixture in season["fixtures"]
                if fixture["matchday"] == matchday
            ]
            managed_fixtures = [
                fixture
                for fixture in season["fixtures"]
                if fixture["matchday"] == matchday
                and season_plan.manager_team is not None
                and season_plan.manager_team in {fixture["home"], fixture["away"]}
            ]
            if any(
                fixture.get("manager_decision") is None for fixture in managed_fixtures
            ):
                raise ValueError("manager decision is required before this matchday")
            for fixture_id in fixture_ids:
                session = self._session()
                season = session.get("season")
                validate_season_state(season)
                fixture = next(
                    item
                    for item in season["fixtures"]
                    if item["fixture_id"] == fixture_id
                )
                if fixture["state"] == "completed":
                    continue
                competition = {
                    "season_id": season_id,
                    "fixture_id": fixture_id,
                    "matchday": matchday,
                }
                decision_payload = fixture.get("manager_decision")
                decision = (
                    ManagerDecision.from_payload(decision_payload)
                    if decision_payload is not None
                    else None
                )
                preparation = None
                club_support = None
                club_strategy = None
                home_fatigue_load_factor = away_fatigue_load_factor = 1.0
                home_tactic = away_tactic = "team_identity"
                home_rotation = away_rotation = None
                home_lineup = away_lineup = None
                home_in_match_plan = away_in_match_plan = None
                match_plan = MatchPlan()
                if decision is not None:
                    preparation = opponent_preparation(season, fixture_id)
                    fixture["opponent_preparation"] = preparation
                    opponent_tactic = str(preparation["selected_tactic"])
                    if resource_plan is None or resource_effects is None:
                        raise ValueError("managed season lacks frozen club resources")
                    club_support = {
                        "schema_version": 1,
                        "team": season_plan.manager_team,
                        "plan": resource_plan.as_dict(),
                        "effects": dict(resource_effects),
                        "claim_boundary": (
                            "bounded carryover support only; no current-match status bonus"
                        ),
                    }
                    if decision.team == fixture["home"]:
                        home_tactic, home_rotation = decision.tactic, decision.rotation
                        home_lineup = (
                            decision.lineup.as_dict() if decision.lineup else None
                        )
                        home_in_match_plan = (
                            decision.in_match_plan.as_dict()
                            if decision.in_match_plan
                            else None
                        )
                        away_tactic = opponent_tactic
                        home_fatigue_load_factor = float(
                            resource_effects["fatigue_load_factor"]
                        )
                    else:
                        away_tactic, away_rotation = decision.tactic, decision.rotation
                        away_lineup = (
                            decision.lineup.as_dict() if decision.lineup else None
                        )
                        away_in_match_plan = (
                            decision.in_match_plan.as_dict()
                            if decision.in_match_plan
                            else None
                        )
                        home_tactic = opponent_tactic
                        away_fatigue_load_factor = float(
                            resource_effects["fatigue_load_factor"]
                        )
                    match_plan = MatchPlan(
                        experience="season_manager",
                        home_tactic=home_tactic,
                        away_tactic=away_tactic,
                    )
                strategy_snapshot = season.get("club_strategies")
                if strategy_snapshot is not None:
                    club_strategy = resolve_fixture_club_strategy(
                        strategy_snapshot,
                        home=str(fixture["home"]),
                        away=str(fixture["away"]),
                        manager_decision=(
                            decision.as_dict() if decision is not None else None
                        ),
                        opponent_preparation=preparation,
                    )
                    home_tactic = str(club_strategy["home_tactic"])
                    away_tactic = str(club_strategy["away_tactic"])
                    match_plan = MatchPlan(
                        experience="season_manager",
                        home_tactic=home_tactic,
                        away_tactic=away_tactic,
                    )
                existing = self._find_season_match(session, competition)
                if existing is not None:
                    recorded_preparation = (existing.get("match_plan") or {}).get(
                        "opponent_preparation"
                    )
                    if recorded_preparation != preparation:
                        raise ValueError(
                            "completed season match opponent preparation conflict"
                        )
                    if (existing.get("match_plan") or {}).get(
                        "club_support"
                    ) != club_support:
                        raise ValueError("completed season match club support conflict")
                    if (existing.get("match_plan") or {}).get(
                        "club_strategy"
                    ) != club_strategy:
                        raise ValueError(
                            "completed season match club strategy conflict"
                        )
                    report = self._load_completed_match(existing)
                else:
                    before_world_state = fixture.get("world_state_before")
                    if decision is not None and not isinstance(
                        before_world_state, Mapping
                    ):
                        before_world_state = capture_world_state(
                            self.root,
                            (fixture["home"], fixture["away"]),
                        )
                    running_update = {
                        "state": "running",
                        "attempts": int(fixture.get("attempts", 0)) + 1,
                        "started_at": _now(),
                    }
                    if isinstance(before_world_state, Mapping):
                        running_update["world_state_before"] = before_world_state
                    fixture.update(running_update)
                    season["updated_at"] = _now()
                    session["updated_at"] = _now()
                    _atomic_json(self.session_path, session)
                    try:
                        report = self._run_match_locked(
                            fixture["home"],
                            fixture["away"],
                            fast=season_plan.fast,
                            plan=match_plan,
                            seed_override=int(season["seed"])
                            + matchday * 100
                            + int(fixture["order"]),
                            continuity=True,
                            continuity_id=f"{season_id}:{fixture_id}",
                            competition=competition,
                            manager_decision=(decision.as_dict() if decision else None),
                            opponent_preparation=preparation,
                            club_support=club_support,
                            club_strategy=club_strategy,
                            home_fatigue_load_factor=home_fatigue_load_factor,
                            away_fatigue_load_factor=away_fatigue_load_factor,
                            home_rotation=home_rotation,
                            away_rotation=away_rotation,
                            home_lineup=home_lineup,
                            away_lineup=away_lineup,
                            home_in_match_plan=home_in_match_plan,
                            away_in_match_plan=away_in_match_plan,
                        )
                    except BaseException:
                        failed_session = self._session()
                        failed_season = failed_session.get("season")
                        if isinstance(failed_season, Mapping):
                            failed_fixture = next(
                                (
                                    item
                                    for item in failed_season.get("fixtures") or []
                                    if item.get("fixture_id") == fixture_id
                                ),
                                None,
                            )
                            if failed_fixture is not None:
                                failed_fixture["state"] = "failed"
                                failed_fixture["failed_at"] = _now()
                                failed_season["updated_at"] = _now()
                                failed_session["updated_at"] = _now()
                                _atomic_json(self.session_path, failed_session)
                        raise
                session = self._session()
                season = session["season"]
                fixture = next(
                    item
                    for item in season["fixtures"]
                    if item["fixture_id"] == fixture_id
                )
                score = report.get("result", {}).get("score") or {}
                before_world_state = fixture.get("world_state_before")
                world_state_transition = None
                if isinstance(before_world_state, Mapping):
                    after_match_world_state = capture_world_state(
                        self.root,
                        (fixture["home"], fixture["away"]),
                    )
                    world_state_transition = build_fixture_world_state_transition(
                        season_id=season_id,
                        fixture_id=fixture_id,
                        match_id=str(report["match_id"]),
                        home=fixture["home"],
                        away=fixture["away"],
                        before_match=before_world_state,
                        after_match=after_match_world_state,
                    )
                fixture.update(
                    {
                        "state": "completed",
                        "completed_at": _now(),
                        "match_id": report["match_id"],
                        "report": Path(report["report_path"])
                        .resolve()
                        .relative_to(self.root)
                        .as_posix(),
                        "dashboard": Path(report["dashboard_path"])
                        .resolve()
                        .relative_to(self.root)
                        .as_posix(),
                        "score": {
                            "home": int(score["home"]),
                            "away": int(score["away"]),
                        },
                        "world_state_transition": world_state_transition,
                    }
                )
                fixture.pop("world_state_before", None)
                season["updated_at"] = _now()
                if next_matchday(season) is None:
                    season["state"] = "complete"
                session["updated_at"] = _now()
                _atomic_json(self.session_path, session)
            session = self._session()
            season = session["season"]
            completed_day = all(
                fixture["state"] == "completed"
                for fixture in season["fixtures"]
                if fixture["matchday"] == matchday
            )
            recovered = season.setdefault("recovered_matchdays", [])
            if completed_day and matchday not in recovered:
                from src.simulation.cross_match_state import recover_persisted_teams

                recovery_id = f"{season_id}:recovery:md{matchday:02d}"
                if season_plan.manager_team is None or resource_effects is None:
                    recover_persisted_teams(
                        str(self.root),
                        list(season_plan.teams),
                        rest_units=1.0,
                        transaction_id=recovery_id,
                    )
                else:
                    recover_persisted_teams(
                        str(self.root),
                        [
                            team
                            for team in season_plan.teams
                            if team != season_plan.manager_team
                        ],
                        rest_units=float(resource_effects["baseline_rest_units"]),
                        transaction_id=recovery_id,
                    )
                    recover_persisted_teams(
                        str(self.root),
                        [season_plan.manager_team],
                        rest_units=float(resource_effects["manager_rest_units"]),
                        medical_recovery_credit=float(
                            resource_effects["medical_recovery_credit_per_matchday"]
                        ),
                        transaction_id=recovery_id,
                    )
                for fixture in season["fixtures"]:
                    transition = fixture.get("world_state_transition")
                    if (
                        fixture.get("matchday") != matchday
                        or not isinstance(transition, Mapping)
                        or transition.get("recovery_complete") is True
                    ):
                        continue
                    after_recovery = capture_world_state(
                        self.root,
                        (fixture["home"], fixture["away"]),
                    )
                    phases = transition["phases"]
                    fixture["world_state_transition"] = (
                        build_fixture_world_state_transition(
                            season_id=season_id,
                            fixture_id=fixture["fixture_id"],
                            match_id=fixture["match_id"],
                            home=fixture["home"],
                            away=fixture["away"],
                            before_match=phases["before_match"],
                            after_match=phases["after_match"],
                            after_recovery=after_recovery,
                        )
                    )
                recovered.append(matchday)
                season["updated_at"] = _now()
                session["updated_at"] = _now()
                _atomic_json(self.session_path, session)
            return self._season_view(self._session()["season"])

    @staticmethod
    def _find_season_match(
        session: Mapping[str, Any],
        competition: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        matches = []
        for record in session.get("matches") or []:
            metadata = (record.get("match_plan") or {}).get("competition") or {}
            if all(
                metadata.get(key) == competition.get(key)
                for key in ("season_id", "fixture_id")
            ):
                matches.append(record)
        if len(matches) > 1:
            raise ValueError("season fixture has duplicate completed matches")
        return matches[0] if matches else None

    def run_paired_matches(
        self,
        home: str,
        away: str,
        *,
        fast: bool,
        baseline_plan: MatchPlan,
        treatment_plan: MatchPlan,
        seed: int,
        transaction_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one isolated baseline/treatment pair without interleaving."""
        experience = baseline_plan.experience
        if (
            experience not in {"tactical_lab", "world_model_lab"}
            or treatment_plan.experience != experience
            or baseline_plan.score_path != "physics_official"
            or treatment_plan.score_path != "physics_official"
        ):
            raise ValueError(
                "paired matches require matching tactical_lab or world_model_lab "
                "physics plans"
            )
        if baseline_plan.reuse_last_seed:
            raise ValueError("baseline plan cannot reuse a prior seed")
        if not treatment_plan.reuse_last_seed:
            raise ValueError("treatment plan must declare shared-seed reuse")
        changed_sides = [
            side
            for side in ("home", "away")
            if getattr(baseline_plan, f"{side}_tactic")
            != getattr(treatment_plan, f"{side}_tactic")
        ]
        if experience == "tactical_lab":
            if len(changed_sides) != 1:
                raise ValueError(
                    "paired matches require exactly one changed tactical side"
                )
            if (
                baseline_plan.world_model_policy != "mode_default"
                or treatment_plan.world_model_policy != "mode_default"
            ):
                raise ValueError("tactical pairs cannot alter world-model policy")
        elif (
            changed_sides
            or baseline_plan.world_model_policy != "predict_only"
            or treatment_plan.world_model_policy != "action_policy"
            or baseline_plan.world_model_branch_at_sec
            != treatment_plan.world_model_branch_at_sec
        ):
            raise ValueError(
                "world-model pairs require fixed tactics and exactly "
                "predict_only to action_policy"
            )
        if transaction_id is not None and (
            not isinstance(transaction_id, str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,63}", transaction_id)
        ):
            raise ValueError("paired transaction_id must be a stable identifier")
        with FileLease(self.session_lease_path, timeout=30.0):
            baseline_record = None
            treatment_record = None
            if transaction_id is not None:
                session = self._session()
                for item in session.get("matches") or []:
                    if not isinstance(item, Mapping):
                        continue
                    persisted_plan = item.get("match_plan") or {}
                    if (
                        not isinstance(persisted_plan, Mapping)
                        or persisted_plan.get("pair_transaction_id") != transaction_id
                    ):
                        continue
                    role = persisted_plan.get("pair_role")
                    if role == "baseline":
                        if baseline_record is not None:
                            raise ValueError(
                                "paired transaction has duplicate baseline"
                            )
                        baseline_record = item
                    elif role == "treatment":
                        if treatment_record is not None:
                            raise ValueError(
                                "paired transaction has duplicate treatment"
                            )
                        treatment_record = item
                    else:
                        raise ValueError("paired transaction contains invalid role")
                for record, role, expected_plan in (
                    (baseline_record, "baseline", baseline_plan),
                    (treatment_record, "treatment", treatment_plan),
                ):
                    if record is None:
                        continue
                    persisted_plan = record.get("match_plan") or {}
                    expected = expected_plan.as_dict()
                    if (
                        record.get("home") != home
                        or record.get("away") != away
                        or record.get("seed") != seed
                        or record.get("fast") is not fast
                        or any(
                            persisted_plan.get(key) != expected.get(key)
                            for key in (
                                "experience",
                                "home_tactic",
                                "away_tactic",
                                "reuse_last_seed",
                                "world_model_policy",
                                "world_model_branch_at_sec",
                                "score_path",
                            )
                        )
                        or persisted_plan.get("pair_role") != role
                    ):
                        raise ValueError("paired transaction identity conflict")
                if treatment_record is not None and baseline_record is None:
                    raise ValueError("paired transaction treatment has no baseline")
            baseline = (
                self._load_completed_match(baseline_record)
                if baseline_record is not None
                else self._run_match_locked(
                    home,
                    away,
                    fast=fast,
                    plan=baseline_plan,
                    seed_override=seed,
                    pair_transaction_id=transaction_id,
                    pair_role="baseline" if transaction_id else None,
                )
            )
            treatment = (
                self._load_completed_match(treatment_record)
                if treatment_record is not None
                else self._run_match_locked(
                    home,
                    away,
                    fast=fast,
                    plan=treatment_plan,
                    seed_override=seed,
                    pair_transaction_id=transaction_id,
                    pair_role="treatment" if transaction_id else None,
                    paired_baseline_match_id_override=baseline["match_id"],
                )
            )
        return baseline, treatment

    def _load_completed_match(self, record: Mapping[str, Any]) -> dict[str, Any]:
        report_path = _artifact_path(self.root, record.get("report"))
        dashboard_path = _artifact_path(self.root, record.get("dashboard"))
        if (
            report_path is None
            or dashboard_path is None
            or report_path.suffix.lower() != ".json"
            or dashboard_path.suffix.lower() != ".html"
            or not report_path.is_file()
            or not dashboard_path.is_file()
        ):
            raise ValueError("completed paired match artifacts are unavailable")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping) or report.get("match_id") != record.get(
            "match_id"
        ):
            raise ValueError("completed paired match artifact identity mismatch")
        comparison_path = _artifact_path(self.root, record.get("comparison"))
        comparison_dashboard_path = _artifact_path(
            self.root, record.get("comparison_dashboard")
        )
        if (comparison_path is None) != (comparison_dashboard_path is None):
            raise ValueError("completed paired comparison artifacts are inconsistent")
        if comparison_path is not None and (
            not comparison_path.is_file()
            or not comparison_dashboard_path.is_file()
            or not comparison_path.name.endswith(".comparison.json")
            or comparison_dashboard_path.suffix.lower() != ".html"
        ):
            raise ValueError("completed paired comparison artifacts are unavailable")
        return {
            **dict(report),
            "report_path": str(report_path),
            "dashboard_path": str(dashboard_path),
            "comparison_path": (
                str(comparison_path) if comparison_path is not None else None
            ),
            "comparison_dashboard_path": (
                str(comparison_dashboard_path)
                if comparison_dashboard_path is not None
                else None
            ),
        }

    def _run_match_locked(
        self,
        home: str,
        away: str,
        *,
        fast: bool = False,
        plan: MatchPlan | None = None,
        seed_override: int | None = None,
        pair_transaction_id: str | None = None,
        pair_role: str | None = None,
        paired_baseline_match_id_override: str | None = None,
        continuity: bool = False,
        continuity_id: str | None = None,
        competition: Mapping[str, Any] | None = None,
        manager_decision: Mapping[str, Any] | None = None,
        opponent_preparation: Mapping[str, Any] | None = None,
        club_support: Mapping[str, Any] | None = None,
        club_strategy: Mapping[str, Any] | None = None,
        home_fatigue_load_factor: float = 1.0,
        away_fatigue_load_factor: float = 1.0,
        home_rotation: str | None = None,
        away_rotation: str | None = None,
        home_lineup: Mapping[str, Any] | None = None,
        away_lineup: Mapping[str, Any] | None = None,
        home_in_match_plan: Mapping[str, Any] | None = None,
        away_in_match_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        readiness = self.readiness()
        if not readiness["ready"]:
            raise RuntimeError(
                "studio mode is not ready: " + ", ".join(readiness["blockers"])
            )
        plan = plan or MatchPlan()
        plan.validate_for_mode(
            self.config.mode,
            context="season" if competition is not None else "standalone",
        )
        if club_strategy is not None:
            home_strategy = club_strategy.get("home")
            away_strategy = club_strategy.get("away")
            if (
                not isinstance(home_strategy, Mapping)
                or not isinstance(away_strategy, Mapping)
                or not continuity
                or competition is None
                or club_strategy.get("season_id") != competition.get("season_id")
                or home_strategy.get("team") != home
                or away_strategy.get("team") != away
                or club_strategy.get("home_tactic") != plan.home_tactic
                or club_strategy.get("away_tactic") != plan.away_tactic
            ):
                raise ValueError("club strategy execution evidence mismatch")
        if club_support is None:
            if home_fatigue_load_factor != 1.0 or away_fatigue_load_factor != 1.0:
                raise ValueError("fatigue load factor requires frozen club support")
        else:
            if (
                not continuity
                or competition is None
                or manager_decision is None
                or not isinstance(club_support, Mapping)
            ):
                raise ValueError("club support requires a managed season fixture")
            support_plan = ClubResourcePlan.from_payload(club_support.get("plan") or {})
            expected_effects = club_resource_effects(support_plan)
            support_team = str(club_support.get("team") or "")
            decision_team = str(manager_decision.get("team") or "")
            if (
                support_team != decision_team
                or support_team not in {home, away}
                or club_support.get("effects") != expected_effects
                or club_support.get("plan") != support_plan.as_dict()
            ):
                raise ValueError("club support identity or effects mismatch")
            expected_home_factor = (
                float(expected_effects["fatigue_load_factor"])
                if support_team == home
                else 1.0
            )
            expected_away_factor = (
                float(expected_effects["fatigue_load_factor"])
                if support_team == away
                else 1.0
            )
            if (
                home_fatigue_load_factor != expected_home_factor
                or away_fatigue_load_factor != expected_away_factor
            ):
                raise ValueError("club support fatigue factor mismatch")
        if seed_override is not None and (
            isinstance(seed_override, bool)
            or not isinstance(seed_override, int)
            or not 0 <= seed_override <= 2**31 - 1
        ):
            raise ValueError("seed_override must be a 32-bit non-negative integer")
        session = self._session()
        paired_baseline_match_id = None
        if plan.reuse_last_seed:
            if paired_baseline_match_id_override:
                baseline_record = next(
                    (
                        item
                        for item in session.get("matches") or []
                        if isinstance(item, Mapping)
                        and item.get("match_id") == paired_baseline_match_id_override
                    ),
                    None,
                )
                if baseline_record is None:
                    raise ValueError("explicit paired baseline is unavailable")
                if (
                    str(baseline_record.get("home", "")).casefold()
                    != str(home).casefold()
                    or str(baseline_record.get("away", "")).casefold()
                    != str(away).casefold()
                ):
                    raise ValueError("explicit paired baseline fixture mismatch")
                expected_seed = baseline_record.get("seed")
            else:
                expected_seed = self.replay_seed(home, away)
                baseline_record = (session.get("matches") or [{}])[-1]
            baseline_plan = baseline_record.get("match_plan") or {}
            if (
                baseline_plan.get("experience") != plan.experience
                or baseline_plan.get("experience") not in {
                    "tactical_lab", "world_model_lab",
                }
                or baseline_plan.get("score_path") != "physics_official"
            ):
                raise ValueError(
                    "paired baseline must use the same paired physics experience"
                )
            if baseline_record.get("fast") is not fast:
                raise ValueError("paired baseline must use the same fast configuration")
            if seed_override is not None and seed_override != expected_seed:
                raise ValueError("seed_override does not match the paired baseline")
            seed_override = expected_seed
            paired_baseline_match_id = (
                str(baseline_record.get("match_id") or "") or None
            )
        plan_record = {
            **plan.as_dict(),
            "paired_baseline_match_id": paired_baseline_match_id,
            "pair_transaction_id": pair_transaction_id,
            "pair_role": pair_role,
            "competition": dict(competition) if competition is not None else None,
            "manager_decision": (
                dict(manager_decision) if manager_decision is not None else None
            ),
            "opponent_preparation": (
                dict(opponent_preparation) if opponent_preparation is not None else None
            ),
            "club_support": (dict(club_support) if club_support is not None else None),
            "club_strategy": (
                dict(club_strategy) if club_strategy is not None else None
            ),
        }
        for prior in session.setdefault("runs", []):
            if prior.get("state") in ACTIVE_RUN_STATES:
                prior.update(
                    {
                        "state": "interrupted",
                        "finished_at": _now(),
                        "reason": "session_lease_recovered",
                    }
                )
        match_index = int(
            session.get(
                "next_match_index",
                len(session.get("matches") or []) + 1,
            )
        )
        session["next_match_index"] = match_index + 1
        seed = (
            int(seed_override)
            if seed_override is not None
            else self.config.seed + match_index - 1
        )
        match_id = f"{match_index:04d}-{_slug(home)}-vs-{_slug(away)}"
        run_record = {
            "match_id": match_id,
            "state": "running",
            "started_at": _now(),
            "fixture": {
                "home": home,
                "away": away,
                "seed": seed,
                "fast": fast,
                "plan": plan_record,
            },
            "mode": self.config.mode,
        }
        session["runs"].append(run_record)
        session["updated_at"] = _now()
        _atomic_json(self.session_path, session)
        try:
            return self._execute_reserved_match(
                home,
                away,
                fast=fast,
                session=session,
                match_index=match_index,
                seed=seed,
                match_id=match_id,
                run_record=run_record,
                plan=plan,
                plan_record=plan_record,
                continuity=continuity,
                continuity_id=continuity_id,
                home_rotation=home_rotation,
                away_rotation=away_rotation,
                home_lineup=home_lineup,
                away_lineup=away_lineup,
                home_in_match_plan=home_in_match_plan,
                away_in_match_plan=away_in_match_plan,
                home_fatigue_load_factor=home_fatigue_load_factor,
                away_fatigue_load_factor=away_fatigue_load_factor,
            )
        except BaseException as exc:
            latest = self._session()
            for recorded in latest.get("runs") or []:
                if recorded.get("match_id") == match_id:
                    recorded.update(
                        {
                            "state": "failed",
                            "finished_at": _now(),
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        }
                    )
                    break
            latest["updated_at"] = _now()
            _atomic_json(self.session_path, latest)
            raise

    def _execute_reserved_match(
        self,
        home: str,
        away: str,
        *,
        fast: bool,
        session: dict[str, Any],
        match_index: int,
        seed: int,
        match_id: str,
        run_record: dict[str, Any],
        plan: MatchPlan,
        plan_record: dict[str, Any],
        continuity: bool = False,
        continuity_id: str | None = None,
        home_rotation: str | None = None,
        away_rotation: str | None = None,
        home_lineup: Mapping[str, Any] | None = None,
        away_lineup: Mapping[str, Any] | None = None,
        home_in_match_plan: Mapping[str, Any] | None = None,
        away_in_match_plan: Mapping[str, Any] | None = None,
        home_fatigue_load_factor: float = 1.0,
        away_fatigue_load_factor: float = 1.0,
    ) -> dict[str, Any]:
        from src import app
        from src.match_engine.micro_config import MicroMatchConfig

        gateway = None
        calls_before = 0
        telemetry_cursor = None
        telemetry_cursor_error = None
        provider_scope = nullcontext()
        with self._mode_environment(
            world_model_policy=plan.world_model_policy,
            world_model_branch_at_sec=plan.world_model_branch_at_sec,
        ):
            if self.config.mode == "cognitive":
                from src.simulation.llm_gateway import get_shared_llm_gateway

                gateway = get_shared_llm_gateway()
                calls_before = int(gateway.call_count)
                scope_factory = getattr(gateway, "request_scope", None)
                if callable(scope_factory):
                    provider_scope = scope_factory(match_id)
                else:
                    telemetry_cursor_error = "request_scope_unavailable"
            with provider_scope:
                cursor_reader = getattr(gateway, "telemetry_cursor", None)
                if gateway is not None:
                    if callable(cursor_reader):
                        try:
                            telemetry_cursor = cursor_reader()
                        except Exception as exc:
                            telemetry_cursor_error = type(exc).__name__
                    else:
                        telemetry_cursor_error = "telemetry_cursor_unavailable"
                micro_config = (
                    MicroMatchConfig.fast_demo() if fast else MicroMatchConfig()
                )
                micro_config.use_micro_goals = (
                    plan.score_path == "physics_official"
                )
                summary = app.run_micro_match(
                    home,
                    away,
                    base_dir=self.root,
                    seed=seed,
                    config=micro_config,
                    home_tactic=(
                        None if plan.home_tactic == NATIVE_TACTIC
                        else plan.home_tactic
                    ),
                    away_tactic=(
                        None if plan.away_tactic == NATIVE_TACTIC
                        else plan.away_tactic
                    ),
                    stage_name=match_id,
                    continuity=continuity,
                    continuity_id=continuity_id,
                    home_rotation=home_rotation,
                    away_rotation=away_rotation,
                    home_lineup=home_lineup,
                    away_lineup=away_lineup,
                    home_in_match_plan=home_in_match_plan,
                    away_in_match_plan=away_in_match_plan,
                    home_fatigue_load_factor=home_fatigue_load_factor,
                    away_fatigue_load_factor=away_fatigue_load_factor,
                )
        calls_after = int(gateway.call_count) if gateway is not None else 0
        global_provider_call_delta = max(0, calls_after - calls_before)
        provider_calls = global_provider_call_delta
        provider_transport: dict[str, Any] = {
            "schema_version": 1,
            "available": False,
            "reason": telemetry_cursor_error or "cognitive_mode_disabled",
            "contains_prompts_or_credentials": False,
        }
        if gateway is not None and telemetry_cursor is not None:
            delta_reader = getattr(gateway, "telemetry_since", None)
            if callable(delta_reader):
                try:
                    provider_transport = dict(delta_reader(telemetry_cursor))
                    provider_transport["available"] = True
                    provider_transport["reason"] = None
                    provider_calls = int(
                        provider_transport.get(
                            "aggregate_successful_calls",
                            global_provider_call_delta,
                        )
                    )
                except Exception as exc:
                    provider_transport["reason"] = (
                        "telemetry_projection_failed:" + type(exc).__name__
                    )
            else:
                provider_transport["reason"] = "telemetry_delta_unavailable"
        provider_transport["global_successful_call_delta"] = (
            global_provider_call_delta
        )
        provider_summary: dict[str, Any] = {}
        if gateway is not None:
            summary_reader = getattr(gateway.config, "public_summary", None)
            if callable(summary_reader):
                provider_summary = dict(summary_reader())
            else:
                provider_summary = {
                    "provider": getattr(gateway.config, "provider", None),
                    "model": getattr(gateway.config, "model", None),
                    "base_url": getattr(gateway.config, "base_url", None),
                }
        raw = asdict(summary) if is_dataclass(summary) else dict(summary)
        run_record.update(
            {
                "state": "finalizing",
                "simulation_completed_at": _now(),
                "successful_provider_calls": provider_calls,
            }
        )
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
            shot_source = (
                evidence_snapshot.get("shot_fallback") or "joint_world_model_head"
            )
        shot_source = observed_world_model.get("shot_probability_source") or shot_source
        ball_log_path = _artifact_path(self.root, raw.get("ball_log_path"))
        ball_log_reference = (
            ball_log_path.relative_to(self.root).as_posix()
            if ball_log_path is not None
            else ""
        )
        action_adoption = raw.get("world_model_action_adoption") or {}
        adoption_records = (
            action_adoption.get("records") or []
            if isinstance(action_adoption, Mapping)
            else []
        )
        priority_opportunity_ids = {
            str(record.get("opportunity_id"))
            for record in adoption_records
            if isinstance(record, Mapping) and record.get("opportunity_id")
        }
        replay = build_match_replay(
            ball_log_path,
            root=self.root,
            home=home,
            away=away,
            priority_opportunity_ids=priority_opportunity_ids,
        )
        replay["world_model_action_links"] = build_world_model_action_links(
            replay,
            action_adoption,
        )
        replay["manager_annotations"] = [
            {
                "team": str((runtime.get("plan") or {}).get("team") or side),
                "minute": int(outcome.get("scheduled_minute", 0)),
                "condition": str(outcome.get("condition") or "always"),
                "score_for": int(outcome.get("score_for", 0)),
                "score_against": int(outcome.get("score_against", 0)),
                "status": str(outcome.get("status") or "unknown"),
                "tactic": outcome.get("requested_tactic"),
                "substitution": outcome.get("requested_substitution"),
                "reason": str(outcome.get("reason") or ""),
            }
            for side, runtime in (raw.get("in_match_management") or {}).items()
            if isinstance(runtime, Mapping)
            for outcome in list(runtime.get("outcomes") or [])[:5]
            if isinstance(outcome, Mapping)
        ][:10]
        report = {
            "schema_version": PRODUCT_SCHEMA_VERSION,
            "match_id": match_id,
            "created_at": _now(),
            "studio": {"name": self.config.name, "mode": self.config.mode},
            "fixture": {
                "home": home,
                "away": away,
                "seed": seed,
                "fast": fast,
            },
            "match_plan": plan_record,
            "result": {
                "score": {
                    "home": raw.get("goals_micro_home"),
                    "away": raw.get("goals_micro_away"),
                },
                "xg": {
                    "home": raw.get("micro_xg_home"),
                    "away": raw.get("micro_xg_away"),
                },
                "possession": {
                    "home": raw.get("possession_home"),
                    "away": 1.0 - float(raw.get("possession_home", 0.5)),
                },
                "passes": {
                    "home": raw.get("passes_home"),
                    "away": raw.get("passes_away"),
                },
                "shots": {"home": raw.get("shots_home"), "away": raw.get("shots_away")},
            },
            "layers": {
                "psychology": {
                    "crowd_field": raw.get("final_psi"),
                    "coach_stress": {
                        "home": raw.get("home_coach_stress"),
                        "away": raw.get("away_coach_stress"),
                    },
                    "tactical_drift": {
                        "home": raw.get("tactical_drift_home"),
                        "away": raw.get("tactical_drift_away"),
                    },
                },
                "world_model": {
                    "configured": self.config.mode in {"research", "cognitive"},
                    "enabled": bool(observed_world_model.get("loaded")),
                    "runtime": observed_world_model,
                    "shot_probability_source": shot_source,
                    "shot_head_authorized": bool(shot_decision.get("accepted")),
                    "online_calibration": raw.get("world_model_online_calibration")
                    or {},
                    "decision_adoption": raw.get("world_model_decision_adoption") or {},
                    "action_adoption": action_adoption,
                    "branch_anchor": raw.get("world_model_branch_anchor") or {},
                    "action_adoption_mechanism": evidence_snapshot.get(
                        "action_adoption_mechanism"
                    )
                    or {},
                },
                "cognition": {
                    "enabled": self.config.mode == "cognitive",
                    "provider": {
                        "adapter": provider_summary.get("provider"),
                        "model": provider_summary.get("model"),
                        "base_url": provider_summary.get("base_url"),
                        "successful_calls": provider_calls,
                        "real_provider_evidence": provider_calls > 0,
                        "transport": provider_transport,
                    },
                    "triggers": raw.get("cognitive_triggers") or [],
                    "plans": raw.get("cognitive_plans") or [],
                    "tier_usage": raw.get("cognitive_tier_usage") or {},
                },
                "management": {
                    "decision": plan_record.get("manager_decision"),
                    "club_support": plan_record.get("club_support"),
                    "effects": raw.get("manager_effects") or {},
                    "in_match": raw.get("in_match_management") or {},
                    "tactical_execution": raw.get("tactical_execution") or {},
                    "claim_boundary": (
                        "gameplay effects only; no causal or real-world claim"
                    ),
                },
            },
            "timeline": raw.get("timeline_snippet") or [],
            "simulation_clock": raw.get("simulation_clock") or {},
            "replay": replay,
            "artifacts": {"ball_log": ball_log_reference},
            "evidence_snapshot": evidence_snapshot,
            "raw_summary": raw,
        }
        integrity_blockers = []
        if self.config.mode in {
            "research",
            "cognitive",
        } and not observed_world_model.get("loaded"):
            integrity_blockers.append("required_world_model_not_observed")
        if (
            plan.world_model_branch_at_sec is not None
            and not bool((raw.get("world_model_branch_anchor") or {}).get("available"))
        ):
            integrity_blockers.append("world_model_branch_anchor_missing")
        if not bool((raw.get("simulation_clock") or {}).get(
            "authoritative_tick_clock"
        )):
            integrity_blockers.append("authoritative_product_clock_not_observed")
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
        if self.config.mode == "cognitive":
            if not provider_transport.get("available"):
                integrity_blockers.append("provider_transport_telemetry_unavailable")
            else:
                if provider_transport.get("truncated") is not False:
                    integrity_blockers.append("provider_transport_telemetry_truncated")
                if provider_transport.get("scope_id") != match_id:
                    integrity_blockers.append("provider_transport_scope_mismatch")
                if (
                    int(provider_transport.get("successful_calls", -1))
                    != provider_calls
                ):
                    integrity_blockers.append(
                        "provider_transport_call_count_mismatch"
                    )
                if (
                    provider_transport.get("contains_prompts_or_credentials")
                    is not False
                ):
                    integrity_blockers.append(
                        "provider_transport_secret_boundary_unverified"
                    )
        if plan.score_path == "physics_official" and (
            int(raw.get("goals_micro_home", -1))
            != int(raw.get("goals_physics_home", -2))
            or int(raw.get("goals_micro_away", -1))
            != int(raw.get("goals_physics_away", -2))
        ):
            integrity_blockers.append("physics_score_path_not_observed")
        report["integrity"] = {
            "accepted": not integrity_blockers,
            "state": "accepted" if not integrity_blockers else "degraded",
            "blockers": integrity_blockers,
        }
        report_path = self.output_root / "matches" / f"{match_id}.json"
        comparison_path = None
        comparison_html_path = None
        if plan_record.get("paired_baseline_match_id"):
            from src.product.comparison import (
                build_paired_comparison,
                write_paired_comparison_html,
            )

            baseline_id = str(plan_record["paired_baseline_match_id"])
            baseline_record = next(
                (
                    item
                    for item in (session.get("matches") or [])
                    if item.get("match_id") == baseline_id
                ),
                None,
            )
            baseline_report_path = _artifact_path(
                self.root,
                (baseline_record or {}).get("report"),
            )
            if baseline_report_path is None or not baseline_report_path.is_file():
                raise ValueError("paired baseline report is unavailable")
            baseline_report = json.loads(
                baseline_report_path.read_text(encoding="utf-8")
            )
            comparison = build_paired_comparison(baseline_report, report)
            comparison["created_at"] = _now()
            comparison["reports"] = {
                "baseline": baseline_report_path.relative_to(self.root).as_posix(),
                "treatment": report_path.relative_to(self.root).as_posix(),
            }
            comparison_path = (
                self.output_root
                / "matches"
                / (f"{match_id}-paired-vs-{baseline_id}.comparison.json")
            )
            comparison_html_path = comparison_path.with_suffix(".html")
            comparison["dashboard"] = comparison_html_path.relative_to(
                self.root
            ).as_posix()
            _atomic_json(comparison_path, comparison)
            write_paired_comparison_html(comparison_html_path, comparison)
            report["artifacts"].update(
                {
                    "paired_comparison": comparison_path.relative_to(
                        self.root
                    ).as_posix(),
                    "paired_comparison_dashboard": comparison_html_path.relative_to(
                        self.root
                    ).as_posix(),
                }
            )
        if self.config.mode == "cognitive":
            cognitive_path = (
                self.root / "data/persistence/cognitive_log" / f"studio_{match_id}.json"
            )
            _atomic_json(
                cognitive_path,
                {
                    "schema_version": PRODUCT_SCHEMA_VERSION,
                    "match_id": match_id,
                    "fixture": report["fixture"],
                    "provider": report["layers"]["cognition"]["provider"],
                    "cognitive_triggers": report["layers"]["cognition"]["triggers"],
                    "cognitive_plans": report["layers"]["cognition"]["plans"],
                    "cognitive_tier_usage": report["layers"]["cognition"]["tier_usage"],
                    "world_model_online_calibration": report["layers"]["world_model"][
                        "online_calibration"
                    ],
                    "world_model_decision_adoption": report["layers"]["world_model"][
                        "decision_adoption"
                    ],
                    "world_model_action_adoption": report["layers"]["world_model"][
                        "action_adoption"
                    ],
                },
            )
            report["artifacts"]["cognitive_log"] = cognitive_path.relative_to(
                self.root
            ).as_posix()
        _atomic_json(report_path, report)
        from src.product.reporting import write_match_html

        html_path = write_match_html(report_path.with_suffix(".html"), report)
        session.setdefault("matches", []).append(
            {
                "match_id": match_id,
                "home": home,
                "away": away,
                "seed": seed,
                "fast": fast,
                "score": report["result"]["score"],
                "match_plan": plan_record,
                "report": report_path.relative_to(self.root).as_posix(),
                "dashboard": html_path.relative_to(self.root).as_posix(),
                "integrity": report["integrity"]["state"],
                "world_model_action_adoption": _action_adoption_digest(
                    report["layers"]["world_model"]["action_adoption"]
                ),
                "comparison": (
                    comparison_path.relative_to(self.root).as_posix()
                    if comparison_path is not None
                    else None
                ),
                "comparison_dashboard": (
                    comparison_html_path.relative_to(self.root).as_posix()
                    if comparison_html_path is not None
                    else None
                ),
            }
        )
        run_record.update(
            {
                "state": "completed",
                "finished_at": _now(),
                "integrity": report["integrity"]["state"],
                "report": report_path.relative_to(self.root).as_posix(),
                "comparison": (
                    comparison_path.relative_to(self.root).as_posix()
                    if comparison_path is not None
                    else None
                ),
            }
        )
        session["updated_at"] = _now()
        _atomic_json(self.session_path, session)
        return {
            **report,
            "report_path": str(report_path),
            "dashboard_path": str(html_path),
            "comparison_path": (
                str(comparison_path) if comparison_path is not None else None
            ),
            "comparison_dashboard_path": (
                str(comparison_html_path) if comparison_html_path is not None else None
            ),
        }
