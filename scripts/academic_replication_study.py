"""Audit, status, execute, or analyze the frozen Stage 4 replication study."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_formal_ablation_results import METRICS  # noqa: E402
from scripts.run_formal_experiment import (  # noqa: E402
    _loss as formal_loss,
    execution_identity as confirmatory_identity,
    load_protocol as load_confirmatory_protocol,
    stratified_paired_interval,
)
from src.infrastructure import FileLease, file_sha256  # noqa: E402
from src.match_engine.calibration.ablation import (  # noqa: E402
    AblationSpec, PIPELINE_PRESETS, ablation_context,
)
from src.match_engine.calibration.benchmark_core import (  # noqa: E402
    run_micro_benchmark_rows,
)

PROTOCOL_PATH = (
    ROOT / "data/evaluation/academic_action_replication_protocol_v2.json"
)
IDSSE_MANIFEST = ROOT / "data/external/sportec/derived/manifest.json"
CONFIRMATORY_TEAMS = {
    "Mexico", "South_Korea", "South Korea", "Brazil", "Germany", "France",
    "England", "Argentina", "Netherlands", "Spain", "Morocco", "Portugal",
    "Uruguay",
}
ARM_IDS = ("M0", "M1_predict_only", "M1")
BRANCH_BY_DECISION = {
    "promotion_candidate_pending_release_review": "mechanism_confirmation",
    "no_meaningful_difference_keep_research_only": "adoption_path_diagnosis",
    "inconclusive_keep_research_only": "variance_diagnosis",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _idsse_ranges(manifest: dict[str, Any]) -> dict[str, list[float]]:
    matches = manifest.get("matches") or []
    return {
        "passes_per_team_match": [
            min(float(row["passes"]) / 2 for row in matches),
            max(float(row["passes"]) / 2 for row in matches),
        ],
        "shots_per_team_match": [
            min(float(row["shots"]) / 2 for row in matches),
            max(float(row["shots"]) / 2 for row in matches),
        ],
    }


def _roster_team_ids() -> set[str]:
    return {
        str(_read_json(path).get("team_id") or "")
        for path in (ROOT / "data/rosters").glob("*.json")
    }


def validate_protocol(protocol: dict[str, Any]) -> dict[str, bool]:
    prerequisite = protocol.get("prerequisite") or {}
    branches = protocol.get("branches") or {}
    arms = protocol.get("arms") or []
    design = protocol.get("design") or {}
    analysis = protocol.get("analysis") or {}
    external = protocol.get("external_validity") or {}
    integrity = protocol.get("integrity") or {}
    outputs = protocol.get("outputs") or {}
    current = protocol.get("current_execution") or {}
    fixtures = design.get("fixtures") or []
    fixture_teams = [str(team) for pair in fixtures for team in pair]
    idsse = _read_json(IDSSE_MANIFEST)
    observed = _idsse_ranges(idsse)
    return {
        "schema_state_and_prerequisite_are_frozen": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id")
            == "gfs-action-policy-external-replication-v2"
            and protocol.get("state")
            == "registered_post_result_before_replication"
            and prerequisite.get("confirmatory_protocol")
            == "data/evaluation/action_outcome_protocol_v1.json"
            and prerequisite.get("confirmatory_decision")
            == "data/evaluation/action_outcome_v1/decision.json"
            and prerequisite.get("accepted_decisions") == list(BRANCH_BY_DECISION)
            and prerequisite.get("observed_decision_at_registration")
            == "inconclusive_keep_research_only"
            and prerequisite.get("branch_is_selected_only_from_confirmatory_decision")
            is True
        ),
        "result_contingent_branches_are_exact": (
            set(branches) == set(BRANCH_BY_DECISION)
            and all(
                (branches.get(decision) or {}).get("branch_id") == branch
                for decision, branch in BRANCH_BY_DECISION.items()
            )
            and {
                (branches.get(decision) or {}).get("required_conclusion")
                for decision in BRANCH_BY_DECISION
            } == {
                "replicated_improvement_and_planning_contribution",
                "replicated_equivalence_and_planning_nonadoption",
                "bounded_inconclusive_replication",
            }
        ),
        "arms_are_existing_baseline_prediction_ablation_and_candidate": (
            [row.get("arm_id") for row in arms] == list(ARM_IDS)
            and [row.get("role") for row in arms] == [
                "physics_competitive_baseline",
                "same_checkpoint_prediction_only_negative_control",
                "mechanism_confirmed_action_policy_candidate",
            ]
            and [(row.get("world_model"), row.get("planning")) for row in arms]
            == [(False, False), (True, False), (True, True)]
        ),
        "fixed_disjoint_72_run_design": (
            len(fixtures) == 6
            and all(isinstance(pair, list) and len(pair) == 2 for pair in fixtures)
            and len(fixture_teams) == len(set(fixture_teams)) == 12
            and not (set(fixture_teams) & CONFIRMATORY_TEAMS)
            and set(fixture_teams).issubset(_roster_team_ids())
            and design.get("fixtures_disjoint_from_confirmatory_teams") is True
            and design.get("sample_indices") == [0, 1, 2, 3]
            and design.get("seed_start") == 361000
            and design.get("match_seconds") == 5400.0
            and design.get("units_per_arm") == 24
            and design.get("runs_total") == 72
            and design.get("arm_order") == list(ARM_IDS)
        ),
        "execution_is_explicit_and_offline": (
            design.get("provider_calls_authorized") is False
            and design.get("training_authorized") is False
            and design.get("explicit_execution_token")
            == "I_AUTHORIZE_GFS_ACTION_REPLICATION_V2"
        ),
        "analysis_thresholds_and_multiplicity_are_frozen": (
            analysis.get("method") == "fixture_stratified_paired_bootstrap"
            and analysis.get("bootstrap_draws") == 10000
            and analysis.get("bootstrap_seed") == 361000
            and analysis.get("confidence_level") == 0.95
            and analysis.get("confirmatory_meaningful_delta_loss") == 0.1
            and analysis.get("planning_contribution_delta_loss") == 0.05
            and analysis.get("planning_nonadoption_equivalence_margin") == 0.02
            and analysis.get("inconclusive_maximum_interval_width") == 0.2
            and analysis.get("behavior_change_tolerance") == 1e-7
            and analysis.get("minimum_changed_pair_fraction") == 0.1
            and analysis.get("secondary_metrics_role") == "descriptive_only"
            and isinstance(analysis.get("multiplicity"), str)
            and bool(analysis["multiplicity"])
        ),
        "licensed_idsse_targets_match_manifest_exactly": (
            idsse.get("dataset") == "IDSSE"
            and idsse.get("license") == "CC BY 4.0"
            and len(idsse.get("matches") or []) == external.get("matches") == 7
            and external.get("source") == "sportec_idsse"
            and external.get("license") == "CC-BY-4.0"
            and external.get("manifest")
            == "data/external/sportec/derived/manifest.json"
            and external.get("passes_per_team_match_observed_range")
            == observed["passes_per_team_match"]
            and external.get("shots_per_team_match_observed_range")
            == observed["shots_per_team_match"]
            and external.get("both_full_candidate_means_must_be_inside_observed_ranges")
            is True
            and external.get("no_provider_pooling") is True
        ),
        "identity_negative_results_and_deviations_are_fail_closed": (
            integrity.get("interim_analysis") is False
            and integrity.get("optional_stopping") is False
            and integrity.get("post_hoc_replication_outcome_selection") is False
            and integrity.get("post_hoc_fixture_or_metric_selection") is False
            and integrity.get(
                "resume_requires_identical_protocol_decision_checkpoint_and_code_identity"
            ) is True
            and integrity.get("failed_and_negative_results_are_preserved") is True
            and integrity.get("material_deviations_force_inconclusive") is True
            and set(integrity.get("code_identity_files") or []) == {
                "src/match_engine/action_engine.py",
                "src/match_engine/passing_engine.py",
                "src/match_engine/world_model/action_adoption.py",
                "src/match_engine/world_model/inference.py",
                "src/match_engine/world_model/planner.py",
                "src/match_engine/calibration/ablation.py",
                "src/match_engine/calibration/benchmark_core.py",
                "src/match_engine/calibration/contract.py",
                "src/match_engine/calibration/objective.py",
                "scripts/run_formal_experiment.py",
                "scripts/academic_replication_study.py",
            }
            and set(integrity.get("data_identity_files") or []) == {
                "data/rosters/Austria.json",
                "data/rosters/Switzerland.json",
                "data/rosters/Canada.json",
                "data/rosters/United_States.json",
                "data/rosters/Algeria.json",
                "data/rosters/Ghana.json",
                "data/rosters/Australia.json",
                "data/rosters/Japan.json",
                "data/rosters/Norway.json",
                "data/rosters/Sweden.json",
                "data/rosters/Ecuador.json",
                "data/rosters/Colombia.json",
                "data/coaches/wc2026_coaches.json",
                "data/tactics_master.json",
                "data/calibration/observable_contract.json",
                "data/calibration/statsbomb_match_baselines.json",
                "data/calibration/joint_baselines.json",
                "data/calibration/param_registry.json",
                "data/external/sportec/derived/manifest.json",
            }
        ),
        "outputs_are_confined_and_execution_remains_zero": (
            set(outputs) == {"progress", "decision"}
            and all(
                isinstance(value, str)
                and value.startswith(
                    "data/evaluation/academic_action_replication_v2/"
                )
                and ".." not in Path(value).parts
                for value in outputs.values()
            )
            and current.get("confirmatory_decision_available") is True
            and current.get("branch_selected") == "variance_diagnosis"
            and current.get("runs_executed") == 0
            and current.get("training_executed") is False
            and current.get("provider_calls_made") is False
            and current.get("results_available") is False
        ),
    }


def protocol_report(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol)
    decision_path = ROOT / protocol["prerequisite"]["confirmatory_decision"]
    return {
        "schema_version": 1,
        "verification": "gfs_action_policy_external_replication_v2_registration",
        "generated_at": _now(),
        "status": (
            "registered_waiting_for_confirmatory_decision"
            if all(checks.values()) and not decision_path.is_file()
            else "registered_ready_for_fixed_replication"
            if all(checks.values()) else "failed"
        ),
        "passed": all(checks.values()),
        "ready_to_start": all(checks.values()) and decision_path.is_file(),
        "confirmatory_decision_available": decision_path.is_file(),
        "runs_executed": 0,
        "checks": checks,
        "artifact_sha256": {
            "data/evaluation/academic_action_replication_protocol_v2.json": (
                file_sha256(protocol_path)
            ),
            "docs/ACADEMIC_REPLICATION_STUDY.md": file_sha256(ROOT / "docs/ACADEMIC_REPLICATION_STUDY.md"),
            "scripts/academic_replication_study.py": file_sha256(Path(__file__)),
            "data/external/sportec/derived/manifest.json": file_sha256(IDSSE_MANIFEST),
        },
        "matches_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
    }


def resolve_branch(confirmatory_decision: dict[str, Any]) -> str:
    if confirmatory_decision.get("schema_version") != 2:
        raise ValueError("confirmatory decision schema must be 2")
    if confirmatory_decision.get("protocol_id") != (
        "gfs-action-policy-full-match-outcome-v1"
    ):
        raise ValueError("confirmatory decision protocol mismatch")
    decision = str(confirmatory_decision.get("decision") or "")
    try:
        return BRANCH_BY_DECISION[decision]
    except KeyError as exc:
        raise ValueError("confirmatory decision is not an accepted frozen branch") from exc


def _confirmatory_decision(
    protocol: dict[str, Any], root: Path,
) -> tuple[Path, dict[str, Any], str]:
    path = root / protocol["prerequisite"]["confirmatory_decision"]
    if not path.is_file():
        raise FileNotFoundError("confirmatory decision is required before replication")
    decision = _read_json(path)
    branch = resolve_branch(decision)
    formal_path = root / protocol["prerequisite"]["confirmatory_protocol"]
    formal_protocol = load_confirmatory_protocol(formal_path)
    expected = confirmatory_identity(root, formal_path, formal_protocol)
    if decision.get("execution_identity") != expected:
        raise ValueError("confirmatory decision identity is stale or invalid")
    if decision.get("pairs_total") != 30:
        raise ValueError("confirmatory decision does not contain the fixed 30 pairs")
    return path, decision, branch


def execution_identity(
    protocol_path: Path, protocol: dict[str, Any], decision_path: Path,
    decision: dict[str, Any], root: Path = ROOT,
) -> dict[str, Any]:
    if not all(validate_protocol(protocol).values()):
        raise ValueError("academic replication protocol is invalid")
    resolve_branch(decision)
    formal_path = root / protocol["prerequisite"]["confirmatory_protocol"]
    formal = load_confirmatory_protocol(formal_path)
    checkpoint = root / formal["candidate"]["checkpoint"]
    expected_checkpoint = formal["candidate"]["checkpoint_sha256"]
    if not checkpoint.is_file() or file_sha256(checkpoint) != expected_checkpoint:
        raise ValueError("sealed M1 checkpoint identity mismatch")
    code = {}
    for relative in protocol["integrity"]["code_identity_files"]:
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"code identity path escapes root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"code identity file missing: {relative}")
        code[relative] = file_sha256(path)
    data = {}
    for relative in protocol["integrity"]["data_identity_files"]:
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"data identity path escapes root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"data identity file missing: {relative}")
        data[relative] = file_sha256(path)
    return {
        "protocol_sha256": file_sha256(protocol_path),
        "confirmatory_decision_sha256": file_sha256(decision_path),
        "checkpoint_sha256": expected_checkpoint,
        "code_sha256": code,
        "data_sha256": data,
    }


def status(root: Path = ROOT) -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    audit = protocol_report()
    progress_path = root / protocol["outputs"]["progress"]
    if not (root / protocol["prerequisite"]["confirmatory_decision"]).is_file():
        return {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "state": "blocked_waiting_for_confirmatory_decision",
            "branch": None,
            "completed_runs": {arm: 0 for arm in ARM_IDS},
            "remaining_runs": 72,
            "protocol_audit": audit,
            "next_action": "complete_frozen_confirmatory_experiment",
            "matches_executed": 0,
            "training_executed": False,
            "provider_calls_made": False,
        }
    try:
        decision_path, decision, branch = _confirmatory_decision(protocol, root)
    except ValueError as exc:
        if "confirmatory decision identity is stale or invalid" not in str(exc):
            raise
        decision_path = root / protocol["prerequisite"]["confirmatory_decision"]
        decision = _read_json(decision_path)
        branch = resolve_branch(decision)
        progress = _read_json(progress_path) if progress_path.is_file() else {}
        completed = {
            arm: len(((progress.get("arms") or {}).get(arm) or {}).get("rows") or [])
            for arm in ARM_IDS
        }
        return {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "state": "blocked_prerequisite_identity_drift",
            "branch": branch,
            "completed_runs": completed,
            "remaining_runs": 72 - sum(completed.values()),
            "identity_matches_progress": False,
            "historical_execution_complete": sum(completed.values()) == 72,
            "historical_results_are_current_evidence": False,
            "protocol_audit": audit,
            "next_action": "preregister_new_candidate_protocol",
            "matches_executed": sum(completed.values()),
            "training_executed": False,
            "provider_calls_made": False,
        }
    identity = execution_identity(
        PROTOCOL_PATH, protocol, decision_path, decision, root,
    )
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    completed = {
        arm: len(((progress.get("arms") or {}).get(arm) or {}).get("rows") or [])
        for arm in ARM_IDS
    }
    identity_matches = not progress or progress.get("execution_identity") == identity
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "state": (
            progress.get("state", "ready_not_started")
            if identity_matches else "blocked_identity_drift"
        ),
        "branch": branch,
        "completed_runs": completed,
        "remaining_runs": 72 - sum(completed.values()),
        "identity_matches_progress": identity_matches,
        "protocol_audit": audit,
        "next_action": (
            "inspect_identity_drift" if not identity_matches
            else "python scripts/academic_replication_study.py --analyze"
            if progress.get("state") == "completed"
            else "explicitly_authorize_fixed_replication_execution"
        ),
        "matches_executed": sum(completed.values()),
        "training_executed": False,
        "provider_calls_made": False,
    }


def _arm_spec(arm: str) -> AblationSpec:
    if arm == "M0":
        return PIPELINE_PRESETS["M0"]
    if arm not in {"M1_predict_only", "M1"}:
        raise ValueError(f"unknown replication arm: {arm}")
    return AblationSpec(
        name="M1",
        description=(
            "Sealed M1 prediction with planning disabled."
            if arm == "M1_predict_only"
            else "Mechanism-confirmed M1 world-model action policy."
        ),
        env={
            **PIPELINE_PRESETS["M1"].env,
            "MATCH_WM_PLAN": "0" if arm == "M1_predict_only" else "1",
        },
    )


@contextmanager
def _world_model_environment(checkpoint: Path) -> Iterator[None]:
    keys = ("MATCH_WM_PLAN", "MATCH_WM_CHECKPOINT", "MATCH_WORLD_MODEL_REQUIRED")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["MATCH_WM_CHECKPOINT"] = str(checkpoint)
        os.environ["MATCH_WORLD_MODEL_REQUIRED"] = "1"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute(authorization: str, root: Path = ROOT) -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    if not all(validate_protocol(protocol).values()):
        raise RuntimeError("academic replication protocol is invalid or has drifted")
    if authorization != protocol["design"]["explicit_execution_token"]:
        raise PermissionError("exact academic replication authorization token is required")
    decision_path, decision, branch = _confirmatory_decision(protocol, root)
    identity = execution_identity(
        PROTOCOL_PATH, protocol, decision_path, decision, root,
    )
    progress_path = root / protocol["outputs"]["progress"]
    lock_path = progress_path.with_suffix(".lock")
    with FileLease(lock_path, timeout=1.0):
        if progress_path.is_file():
            state = _read_json(progress_path)
            if (
                state.get("execution_identity") != identity
                or state.get("branch") != branch
            ):
                raise ValueError(
                    "cannot resume after protocol, branch, decision, "
                    "checkpoint, or code drift"
                )
            if state.get("state") == "completed":
                return state
            state.update({"state": "running", "updated_at": _now()})
            state.pop("last_error", None)
        else:
            state = {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "state": "running",
                "branch": branch,
                "started_at": _now(),
                "execution_identity": identity,
                "material_deviations": [],
                "arms": {arm: {"rows": []} for arm in ARM_IDS},
            }
        _atomic_json(progress_path, state)
        fixtures = [tuple(pair) for pair in protocol["design"]["fixtures"]]
        formal_path = root / protocol["prerequisite"]["confirmatory_protocol"]
        formal = load_confirmatory_protocol(formal_path)
        checkpoint = root / formal["candidate"]["checkpoint"]
        try:
            for arm in ARM_IDS:
                arm_state = state["arms"][arm]
                completed = {
                    (str(row["fixture"]), int(row["sample_index"]))
                    for row in arm_state["rows"]
                }

                def save(row: dict[str, Any]) -> None:
                    arm_state["rows"].append(row)
                    state["updated_at"] = _now()
                    _atomic_json(progress_path, state)

                spec = _arm_spec(arm)
                with _world_model_environment(checkpoint), ablation_context(spec) as cfg:
                    run_micro_benchmark_rows(
                        root=str(root), fixtures=fixtures, samples=4,
                        seed_start=361000, match_seconds=5400.0,
                        spec=spec, cfg=cfg, completed_keys=completed,
                        row_callback=save,
                    )
                arm_state["state"] = "completed"
            if any(len(state["arms"][arm]["rows"]) != 24 for arm in ARM_IDS):
                raise RuntimeError("fixed 72-run replication budget was not completed")
            if execution_identity(
                PROTOCOL_PATH, protocol, decision_path, decision, root,
            ) != identity:
                raise RuntimeError("replication identity drifted during execution")
            state.update({"state": "completed", "completed_at": _now()})
            _atomic_json(progress_path, state)
            return state
        except BaseException as exc:
            state.update({
                "state": "failed", "updated_at": _now(),
                "last_error": {
                    "type": type(exc).__name__, "message": str(exc)[:500],
                },
            })
            _atomic_json(progress_path, state)
            raise


def _keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (str(row.get("fixture") or ""), int(row.get("sample_index", -1)))
        if key in result:
            raise ValueError(f"duplicate replication experimental unit: {key}")
        if any(metric not in row for metric in METRICS):
            raise ValueError(f"replication row lacks frozen metrics: {key}")
        result[key] = row
    return result


def _default_comparison(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
    *, draws: int, seed: int,
) -> dict[str, Any]:
    interval = stratified_paired_interval(
        baseline, candidate, draws=draws, seed=seed,
    )
    _, evaluation = formal_loss(candidate)
    return {
        **interval,
        "candidate_all_external_gates": bool(evaluation["all_pass"]),
    }


def analyze(
    protocol: dict[str, Any], state: dict[str, Any],
    confirmatory_decision: dict[str, Any], *, expected_identity: dict[str, Any],
    comparison_fn: Callable[..., dict[str, Any]] = _default_comparison,
) -> dict[str, Any]:
    if not all(validate_protocol(protocol).values()):
        raise ValueError("academic replication protocol is invalid")
    branch = resolve_branch(confirmatory_decision)
    if state.get("state") != "completed" or state.get("branch") != branch:
        raise ValueError("replication must be complete under the selected branch")
    if state.get("execution_identity") != expected_identity:
        raise ValueError("replication execution identity mismatch")
    deviations = state.get("material_deviations")
    if not isinstance(deviations, list):
        raise ValueError("material deviations must be an explicit list")
    arms = state.get("arms") or {}
    rows = {arm: (arms.get(arm) or {}).get("rows") or [] for arm in ARM_IDS}
    if any(len(rows[arm]) != 24 for arm in ARM_IDS):
        raise ValueError("replication does not contain exactly 24 units per arm")
    keyed = {arm: _keyed(rows[arm]) for arm in ARM_IDS}
    if not (
        set(keyed["M0"])
        == set(keyed["M1_predict_only"])
        == set(keyed["M1"])
    ):
        raise ValueError("replication arms are not exactly paired")
    settings = protocol["analysis"]
    replication = comparison_fn(
        rows["M0"], rows["M1"],
        draws=int(settings["bootstrap_draws"]),
        seed=int(settings["bootstrap_seed"]),
    )
    planning = comparison_fn(
        rows["M1_predict_only"], rows["M1"],
        draws=int(settings["bootstrap_draws"]),
        seed=int(settings["bootstrap_seed"]) + 1,
    )
    tolerance = float(settings["behavior_change_tolerance"])
    changed = sum(
        any(
            abs(
                float(keyed["M1"][key][metric])
                - float(keyed["M1_predict_only"][key][metric])
            ) > tolerance
            for metric in METRICS
        )
        for key in keyed["M1"]
    )
    changed_fraction = changed / 24
    meaningful = float(settings["confirmatory_meaningful_delta_loss"])
    contribution = float(settings["planning_contribution_delta_loss"])
    nonadoption = float(settings["planning_nonadoption_equivalence_margin"])
    replicated_promotion = (
        float(replication["ci95_high"]) < -meaningful
        and replication.get("candidate_all_external_gates") is True
    )
    replicated_equivalence = (
        float(replication["ci95_low"]) > -meaningful
        and float(replication["ci95_high"]) < meaningful
    )
    replicated_harm = float(replication["ci95_low"]) > meaningful
    planning_contribution = (
        float(planning["ci95_high"]) < -contribution
        and planning.get("candidate_all_external_gates") is True
        and changed_fraction >= float(settings["minimum_changed_pair_fraction"])
    )
    planning_nonadoption = (
        float(planning["ci95_low"]) > -nonadoption
        and float(planning["ci95_high"]) < nonadoption
        and changed_fraction < float(settings["minimum_changed_pair_fraction"])
    )
    bounded_inconclusive = (
        not replicated_promotion
        and not replicated_equivalence
        and not replicated_harm
        and float(replication["ci95_high"]) - float(replication["ci95_low"])
        <= float(settings["inconclusive_maximum_interval_width"])
    )
    branch_conclusion = (
        replicated_promotion and planning_contribution
        if branch == "mechanism_confirmation"
        else replicated_equivalence and planning_nonadoption
        if branch == "adoption_path_diagnosis"
        else bounded_inconclusive
    )
    external = protocol["external_validity"]
    passes = mean(float(row["passes_per_team_match"]) for row in rows["M1"])
    shots = mean(float(row["shots_per_team_match"]) for row in rows["M1"])
    pass_range = external["passes_per_team_match_observed_range"]
    shot_range = external["shots_per_team_match_observed_range"]
    external_gates = {
        "passes_inside_idsse_observed_range": pass_range[0] <= passes <= pass_range[1],
        "shots_inside_idsse_observed_range": shot_range[0] <= shots <= shot_range[1],
    }
    gates = {
        "fixed_complete_paired_budget": True,
        "execution_identity_verified": True,
        "no_material_deviations": len(deviations) == 0,
        "branch_conclusion_matches_confirmatory_decision": branch_conclusion,
        "full_candidate_passes_internal_external_gates": (
            replication.get("candidate_all_external_gates") is True
            and planning.get("candidate_all_external_gates") is True
        ),
        "licensed_idsse_external_scale_passes": all(external_gates.values()),
    }
    passed = all(gates.values())
    conclusion = (
        protocol["branches"][str(confirmatory_decision["decision"])][
            "required_conclusion"
        ]
        if branch_conclusion else "branch_conclusion_not_supported"
    )
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "generated_at": _now(),
        "status": (
            "passed_academic_replication"
            if passed else "failed_academic_replication"
        ),
        "passed": passed,
        "branch": branch,
        "confirmatory_decision": confirmatory_decision["decision"],
        "conclusion": conclusion,
        "runs_executed": 72,
        "pairs_per_arm": 24,
        "comparisons": {
            "M1_minus_M0": replication,
            "M1_minus_M1_predict_only": planning,
        },
        "mechanism": {
            "changed_pairs": changed,
            "changed_pair_fraction": changed_fraction,
            "replicated_promotion": replicated_promotion,
            "replicated_equivalence": replicated_equivalence,
            "replicated_harm": replicated_harm,
            "planning_contribution": planning_contribution,
            "planning_nonadoption": planning_nonadoption,
            "bounded_inconclusive": bounded_inconclusive,
        },
        "external_validity": {
            "source": "sportec_idsse",
            "full_candidate_mean_passes_per_team_match": passes,
            "full_candidate_mean_shots_per_team_match": shots,
            "checks": external_gates,
        },
        "material_deviations": deviations,
        "gates": gates,
        "execution_identity": expected_identity,
        "training_executed": False,
        "provider_calls_made": False,
        "limitations": [
            "IDSSE external validity uses seven observed matches and literal ranges, not a population-level tolerance interval.",
            "M0 and prediction-only M1 are in-system baselines; this study does not claim comparison with every published football simulator.",
            "A branch failure or negative external-scale result must remain a negative result and cannot be relabelled as replication success.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    actions.add_argument("--protocol-audit", action="store_true")
    parser.add_argument("--authorization", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    protocol = _read_json(PROTOCOL_PATH)
    if args.protocol_audit:
        if args.authorization:
            parser.error("--authorization requires --execute")
        result = protocol_report()
    elif args.execute:
        state = execute(args.authorization)
        result = {
            "state": state["state"], "branch": state["branch"],
            "next_action": "--analyze",
        }
    elif args.analyze:
        decision_path, confirmatory, _branch = _confirmatory_decision(
            protocol, ROOT,
        )
        identity = execution_identity(
            PROTOCOL_PATH, protocol, decision_path, confirmatory,
        )
        progress = _read_json(ROOT / protocol["outputs"]["progress"])
        result = analyze(
            protocol, progress, confirmatory, expected_identity=identity,
        )
        _atomic_json(ROOT / protocol["outputs"]["decision"], result)
    else:
        if args.authorization:
            parser.error("--authorization requires --execute")
        result = status()
    if args.out:
        _atomic_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
