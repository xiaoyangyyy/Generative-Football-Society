#!/usr/bin/env python3
"""Bounded preregistered mechanism study for direct world-model action adoption."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_formal_ablation_results import METRICS  # noqa: E402
from src.infrastructure import (  # noqa: E402
    FileLease,
    code_identity_manifest,
    file_sha256,
)
from src.match_engine.calibration.ablation import (  # noqa: E402
    AblationSpec,
    PIPELINE_PRESETS,
    ablation_context,
)
from src.match_engine.calibration.benchmark_core import (  # noqa: E402
    run_micro_benchmark_rows,
)


PROTOCOL_PATH = ROOT / "data/evaluation/action_adoption_protocol_v1.json"
AUTHORIZATION = "I_AUTHORIZE_GFS_ACTION_ADOPTION_MECHANISM_V1"
ARM_IDS = ("M1_predict_only", "M1_action_policy")
SMOKE_ARM_IDS = (
    "M0_no_advisor",
    "C1_rule_fallback",
    "M1_predict_only",
    "M1_action_policy",
)
SMOKE_OUTPUT = (
    ROOT / "data/evaluation/action_adoption_v1/smoke_verification.json"
)
PROGRESS_STATES = frozenset({"running", "failed", "completed"})
ACTION_ANALYSIS_METRICS = (
    "wm_action_opportunities",
    "wm_action_influenced_opportunities",
    "wm_action_attribution_eligible_opportunities",
    "wm_action_expected_counterfactual_changes",
    "wm_action_counterfactual_changes",
)


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


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expected_schedule(protocol: dict[str, Any]) -> list[tuple[str, int]]:
    design = protocol["design"]
    return [
        (f"{home}_vs_{away}", int(sample_index))
        for home, away in design["fixtures"]
        for sample_index in design["sample_indices"]
    ]


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_progress_rows(
    rows: Any,
    *,
    expected_schedule: list[tuple[str, int]],
) -> dict[str, bool]:
    if not isinstance(rows, list):
        return {
            "rows_are_a_list": False,
            "rows_are_objects": False,
            "rows_follow_exact_schedule_prefix": False,
            "analysis_metrics_are_finite": False,
            "action_counts_are_bounded": False,
        }
    keys: list[tuple[str, int] | None] = []
    rows_are_objects = True
    metrics_are_finite = True
    action_counts_are_bounded = True
    required_metrics = (*METRICS, *ACTION_ANALYSIS_METRICS)
    for row in rows:
        if not isinstance(row, dict):
            rows_are_objects = False
            keys.append(None)
            metrics_are_finite = False
            action_counts_are_bounded = False
            continue
        fixture = row.get("fixture")
        sample_index = row.get("sample_index")
        keys.append(
            (fixture, sample_index)
            if isinstance(fixture, str)
            and isinstance(sample_index, int)
            and not isinstance(sample_index, bool)
            else None
        )
        if not all(_finite_number(row.get(name)) for name in required_metrics):
            metrics_are_finite = False
            action_counts_are_bounded = False
            continue
        opportunities = float(row["wm_action_opportunities"])
        influenced = float(row["wm_action_influenced_opportunities"])
        eligible = float(row["wm_action_attribution_eligible_opportunities"])
        expected = float(row["wm_action_expected_counterfactual_changes"])
        realized = float(row["wm_action_counterfactual_changes"])
        if not (
            opportunities >= 0.0
            and 0.0 <= influenced <= opportunities
            and 0.0 <= eligible <= influenced
            and 0.0 <= expected <= eligible
            and 0.0 <= realized <= eligible
        ):
            action_counts_are_bounded = False
    return {
        "rows_are_a_list": True,
        "rows_are_objects": rows_are_objects,
        "rows_follow_exact_schedule_prefix": (
            len(keys) <= len(expected_schedule)
            and keys == expected_schedule[:len(keys)]
        ),
        "analysis_metrics_are_finite": metrics_are_finite,
        "action_counts_are_bounded": action_counts_are_bounded,
    }


def validate_progress(
    protocol: dict[str, Any],
    state: dict[str, Any],
    *,
    expected_identity: dict[str, Any] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate a resumable fixed-budget ledger without executing a match."""
    expected_schedule = _expected_schedule(protocol)
    runs_per_arm = int(protocol["design"]["runs_per_arm"])
    runs_total = int(protocol["design"]["runs_total"])
    arms = state.get("arms")
    arms_are_exact = (
        isinstance(arms, dict) and set(arms) == set(ARM_IDS)
    )
    row_checks: dict[str, dict[str, bool]] = {}
    completed_runs: dict[str, int] = {}
    arm_states: dict[str, Any] = {}
    for arm in ARM_IDS:
        arm_payload = arms.get(arm) if arms_are_exact else None
        rows = arm_payload.get("rows") if isinstance(arm_payload, dict) else None
        row_checks[arm] = _validate_progress_rows(
            rows, expected_schedule=expected_schedule,
        )
        completed_runs[arm] = len(rows) if isinstance(rows, list) else 0
        arm_states[arm] = (
            arm_payload.get("state") if isinstance(arm_payload, dict) else None
        )
    row_checks_pass = all(
        all(checks.values()) for checks in row_checks.values()
    )
    arm_completion_is_consistent = all(
        arm_states[arm] in {None, "completed"}
        and (
            arm_states[arm] != "completed"
            or completed_runs[arm] == runs_per_arm
        )
        for arm in ARM_IDS
    )
    first_arm, second_arm = ARM_IDS
    arm_order_is_resumable = not (
        completed_runs[first_arm] < runs_per_arm
        and (
            completed_runs[second_arm] > 0
            or arm_states[second_arm] == "completed"
        )
    ) and not (
        completed_runs[second_arm] > 0
        and arm_states[first_arm] != "completed"
    )
    fixed_budget_complete = (
        all(completed_runs[arm] == runs_per_arm for arm in ARM_IDS)
        and all(arm_states[arm] == "completed" for arm in ARM_IDS)
    )
    overall_completion_is_consistent = (
        state.get("state") != "completed" or fixed_budget_complete
    )
    checks = {
        "schema_and_protocol_match": (
            state.get("schema_version") == 1
            and state.get("protocol_id") == protocol["protocol_id"]
        ),
        "state_is_resumable": state.get("state") in PROGRESS_STATES,
        "material_deviations_are_a_list": isinstance(
            state.get("material_deviations"), list,
        ),
        "arms_are_exact": arms_are_exact,
        "rows_follow_frozen_contract": row_checks_pass,
        "arm_completion_is_consistent": arm_completion_is_consistent,
        "arm_order_is_resumable": arm_order_is_resumable,
        "overall_completion_is_consistent": overall_completion_is_consistent,
        "required_complete_budget": (
            not require_complete
            or (state.get("state") == "completed" and fixed_budget_complete)
        ),
        "recorded_rows_do_not_exceed_budget": (
            sum(completed_runs.values()) <= runs_total
        ),
    }
    identity_matches = (
        expected_identity is None
        or state.get("execution_identity") == expected_identity
    )
    structural_valid = all(checks.values())
    return {
        "structural_valid": structural_valid,
        "identity_matches": identity_matches,
        "valid": structural_valid and identity_matches,
        "checks": checks,
        "row_checks": row_checks,
        "completed_runs": completed_runs,
        "fixed_run_budget": runs_total,
        "remaining_runs": max(0, runs_total - sum(completed_runs.values())),
    }


def _require_valid_progress(
    report: dict[str, Any],
    *,
    require_identity: bool = True,
) -> None:
    if not report["structural_valid"]:
        failures = [
            name for name, passed in report["checks"].items() if not passed
        ]
        row_failures = [
            f"{arm}.{name}"
            for arm, checks in report["row_checks"].items()
            for name, passed in checks.items()
            if not passed
        ]
        raise ValueError(
            "action-adoption progress is invalid: "
            + ", ".join([*failures, *row_failures])
        )
    if require_identity and not report["identity_matches"]:
        raise ValueError("action-adoption progress identity drift")


def validate_protocol(protocol: dict[str, Any]) -> dict[str, bool]:
    arms = protocol.get("arms") or []
    design = protocol.get("design") or {}
    analysis = protocol.get("analysis") or {}
    integrity = protocol.get("integrity") or {}
    outputs = protocol.get("outputs") or {}
    fixtures = design.get("fixtures") or []
    return {
        "schema_scope_and_state_are_frozen": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id")
            == "gfs-world-model-action-adoption-mechanism-v1"
            and protocol.get("state") == "preregistered_not_executed"
            and protocol.get("claim_scope")
            == "mechanism_confirmation_only_no_product_or_academic_promotion"
        ),
        "arms_are_exact_negative_control_and_action_policy": (
            [row.get("arm_id") for row in arms] == list(ARM_IDS)
            and [row.get("planning") for row in arms] == [False, True]
            and all(row.get("world_model") is True for row in arms)
        ),
        "fixed_bounded_matched_design": (
            len(fixtures) == 6
            and all(isinstance(pair, list) and len(pair) == 2 for pair in fixtures)
            and design.get("sample_indices") == [0, 1]
            and design.get("seed_start") == 418260
            and design.get("match_seconds") == 900.0
            and design.get("pairs_total") == 12
            and design.get("runs_per_arm") == 12
            and design.get("runs_total") == 24
            and design.get("arm_order") == list(ARM_IDS)
        ),
        "mechanism_thresholds_are_fixed": (
            analysis.get("primary_estimand")
            == "shared_uniform_counterfactual_action_change_rate"
            and analysis.get("minimum_action_opportunities") == 1000
            and analysis.get("minimum_influenced_opportunity_fraction") == 0.90
            and analysis.get("minimum_attribution_eligible_fraction") == 0.95
            and analysis.get("minimum_expected_counterfactual_changes") == 3.0
            and analysis.get("minimum_realized_counterfactual_changes") == 1
            and analysis.get("minimum_changed_match_pair_fraction") == 0.08
            and analysis.get("maximum_probability_blend_weight") == 0.35
            and analysis.get("promotion_forbidden") is True
        ),
        "integrity_and_outputs_are_confined": (
            integrity.get("interim_analysis") is False
            and integrity.get("optional_stopping") is False
            and integrity.get("post_hoc_threshold_changes") is False
            and integrity.get("resume_requires_identical_protocol_checkpoint_and_code_identity")
            is True
            and len(integrity.get("code_identity_files") or []) == 8
            and "src/match_engine/passing_engine.py"
            in (integrity.get("code_identity_files") or [])
            and all(
                str(value).startswith("data/evaluation/action_adoption_v1/")
                for value in outputs.values()
            )
        ),
        "execution_is_currently_absent": (
            protocol.get("current_execution") == {
                "authorized": False,
                "runs_executed": 0,
                "training_executed": False,
                "provider_calls_made": False,
                "results_available": False,
            }
        ),
    }


def execution_identity(
    protocol_path: Path = PROTOCOL_PATH,
    protocol: dict[str, Any] | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    protocol = protocol or _read_json(protocol_path)
    checks = validate_protocol(protocol)
    if not all(checks.values()):
        raise ValueError("action-adoption protocol is invalid")
    checkpoint = root / protocol["candidate"]["checkpoint"]
    expected_checkpoint = protocol["candidate"]["checkpoint_sha256"]
    if not checkpoint.is_file() or file_sha256(checkpoint) != expected_checkpoint:
        raise ValueError("sealed action-adoption checkpoint identity mismatch")
    integrity = protocol["integrity"]
    code = code_identity_manifest(
        root,
        integrity["code_identity_files"],
        mode=str(integrity.get("code_identity_mode") or "explicit_files_v1"),
    )
    return {
        "protocol_sha256": file_sha256(protocol_path),
        "checkpoint_sha256": expected_checkpoint,
        "code_sha256": code,
    }


def protocol_report(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol)
    return {
        "schema_version": 1,
        "verification": "gfs_action_adoption_mechanism_preregistration",
        "passed": all(checks.values()),
        "ready_to_start": all(checks.values()),
        "checks": checks,
        "runs_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
        "artifact_sha256": {
            str(protocol_path.relative_to(ROOT)).replace("\\", "/"):
            file_sha256(protocol_path),
            "scripts/action_adoption_study.py": file_sha256(Path(__file__)),
        },
    }


def status(root: Path = ROOT) -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    identity = execution_identity(PROTOCOL_PATH, protocol, root)
    progress_path = root / protocol["outputs"]["progress"]
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    validation = (
        validate_progress(protocol, progress, expected_identity=identity)
        if progress else None
    )
    completed = (
        validation["completed_runs"]
        if validation else {arm: 0 for arm in ARM_IDS}
    )
    progress_valid = validation is None or validation["structural_valid"]
    identity_matches = validation is None or validation["identity_matches"]
    current_state = (
        "ready_not_started"
        if validation is None
        else "blocked_invalid_progress"
        if not progress_valid
        else "blocked_identity_drift"
        if not identity_matches
        else str(progress.get("state"))
    )
    next_action = (
        "inspect_invalid_progress"
        if not progress_valid
        else "inspect_identity_drift"
        if not identity_matches
        else "python scripts/action_adoption_study.py --analyze"
        if progress.get("state") == "completed"
        else "explicitly_authorize_bounded_mechanism_execution"
    )
    runs_total = int(protocol["design"]["runs_total"])
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "state": current_state,
        "completed_runs": completed,
        "fixed_run_budget": runs_total,
        "remaining_runs": max(0, runs_total - sum(completed.values())),
        "progress_present": validation is not None,
        "progress_valid": progress_valid,
        "progress_checks": validation["checks"] if validation else {},
        "row_checks": validation["row_checks"] if validation else {},
        "identity_matches_progress": identity_matches,
        "next_action": next_action,
        "matches_executed": sum(completed.values()),
        "training_executed": False,
        "provider_calls_made": False,
    }


def _arm_spec(arm: str) -> AblationSpec:
    if arm not in ARM_IDS:
        raise ValueError(f"unknown action-adoption arm: {arm}")
    return AblationSpec(
        name="M1",
        description=(
            "Same checkpoint prediction-only negative control."
            if arm == "M1_predict_only"
            else "Validated direct world-model action policy."
        ),
        env={
            **PIPELINE_PRESETS["M1"].env,
            "MATCH_WM_PLAN": "0" if arm == "M1_predict_only" else "1",
        },
    )


def _smoke_arm_spec(arm: str) -> AblationSpec:
    if arm == "M0_no_advisor":
        return PIPELINE_PRESETS["M0"]
    if arm == "C1_rule_fallback":
        return PIPELINE_PRESETS["C1"]
    return _arm_spec(arm)


@contextmanager
def _world_model_environment(checkpoint: Path, *, planning: bool) -> Iterator[None]:
    keys = ("MATCH_WM_PLAN", "MATCH_WM_CHECKPOINT", "MATCH_WORLD_MODEL_REQUIRED")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["MATCH_WM_PLAN"] = "1" if planning else "0"
        os.environ["MATCH_WM_CHECKPOINT"] = str(checkpoint)
        os.environ["MATCH_WORLD_MODEL_REQUIRED"] = "1"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def smoke(
    *, root: Path = ROOT, output_path: Path | None = None,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Run four one-unit diagnostics without touching formal study progress."""
    protocol = _read_json(PROTOCOL_PATH)
    identity = execution_identity(PROTOCOL_PATH, protocol, root)
    checkpoint = root / protocol["candidate"]["checkpoint"]
    run_rows = runner or run_micro_benchmark_rows
    fixture = [tuple(protocol["design"]["fixtures"][1])]
    seed_start = int(protocol["design"]["seed_start"])
    match_seconds = 180.0
    arms: dict[str, dict[str, Any]] = {}
    for arm in SMOKE_ARM_IDS:
        spec = _smoke_arm_spec(arm)
        with ablation_context(spec) as cfg:
            kwargs = {
                "root": str(root),
                "fixtures": fixture,
                "samples": 1,
                "seed_start": seed_start,
                "match_seconds": match_seconds,
                "spec": spec,
                "cfg": cfg,
            }
            if arm.startswith("M1_"):
                with _world_model_environment(
                    checkpoint, planning=arm == "M1_action_policy",
                ):
                    rows = run_rows(**kwargs)
            else:
                rows = run_rows(**kwargs)
        if len(rows) != 1:
            raise RuntimeError(f"smoke arm {arm} did not produce exactly one row")
        row = dict(rows[0])
        arms[arm] = {
            "pipeline": spec.name,
            "role": (
                "no_advisor_physics_baseline"
                if arm == "M0_no_advisor"
                else "deterministic_rule_fallback_baseline"
                if arm == "C1_rule_fallback"
                else "same_checkpoint_prediction_without_action_adoption"
                if arm == "M1_predict_only"
                else "same_checkpoint_direct_action_adoption"
            ),
            "row": row,
        }
    candidate = arms["M1_action_policy"]["row"]
    prediction = arms["M1_predict_only"]["row"]
    gates = {
        "four_arms_completed_once": set(arms) == set(SMOKE_ARM_IDS),
        "non_world_model_arms_have_zero_action_opportunities": all(
            float(arms[arm]["row"].get("wm_action_opportunities", 0.0)) == 0.0
            for arm in ("M0_no_advisor", "C1_rule_fallback")
        ),
        "prediction_only_has_zero_influenced_opportunities": (
            float(prediction.get("wm_action_influenced_opportunities", 0.0))
            == 0.0
        ),
        "action_policy_reaches_sampling_opportunities": (
            float(candidate.get("wm_action_opportunities", 0.0)) > 0.0
        ),
        "action_policy_changes_sampling_probabilities": (
            float(candidate.get("wm_action_influenced_opportunities", 0.0))
            > 0.0
        ),
        "action_policy_records_counterfactual_attribution": (
            float(candidate.get(
                "wm_action_attribution_eligible_opportunities", 0.0,
            )) > 0.0
        ),
    }
    payload = {
        "schema_version": 1,
        "verification": "gfs_action_adoption_four_arm_smoke",
        "state": "smoke_completed",
        "passed": all(gates.values()),
        "diagnostic_only": True,
        "formal_progress_modified": False,
        "fixture": list(fixture[0]),
        "sample_indices": [0],
        "seed_start": seed_start,
        "match_seconds": match_seconds,
        "runs_executed": len(arms),
        "arms": arms,
        "gates": gates,
        "execution_identity": identity,
        "training_executed": False,
        "provider_calls_made": False,
        "claim_boundary": (
            "code-path smoke only; not a fixed-budget mechanism, outcome, "
            "product-promotion or academic-promotion result"
        ),
    }
    if output_path is not None:
        _atomic_json(output_path, payload)
    return payload


def execute(
    *, authorization: str, root: Path = ROOT,
) -> dict[str, Any]:
    if authorization != AUTHORIZATION:
        raise PermissionError("explicit action-adoption mechanism authorization is required")
    protocol = _read_json(PROTOCOL_PATH)
    identity = execution_identity(PROTOCOL_PATH, protocol, root)
    progress_path = root / protocol["outputs"]["progress"]
    lock_path = progress_path.with_suffix(".lock")
    with FileLease(lock_path):
        state = _read_json(progress_path) if progress_path.is_file() else {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "state": "running",
            "started_at": _now(),
            "execution_identity": identity,
            "material_deviations": [],
            "arms": {arm: {"rows": []} for arm in ARM_IDS},
        }
        initial_validation = validate_progress(
            protocol, state, expected_identity=identity,
        )
        _require_valid_progress(initial_validation)
        if state.get("state") == "completed":
            return state
        state["state"] = "running"
        state["updated_at"] = _now()
        _atomic_json(progress_path, state)
        fixtures = [tuple(pair) for pair in protocol["design"]["fixtures"]]
        checkpoint = root / protocol["candidate"]["checkpoint"]
        expected_schedule = _expected_schedule(protocol)
        try:
            for arm in protocol["design"]["arm_order"]:
                rows = state["arms"][arm]["rows"]
                completed = {(row["fixture"], int(row["sample_index"])) for row in rows}
                spec = _arm_spec(arm)
                with ablation_context(spec) as cfg, _world_model_environment(
                    checkpoint, planning=arm == "M1_action_policy",
                ):
                    def append_row(row: dict[str, Any]) -> None:
                        candidate_rows = [*rows, dict(row)]
                        row_report = _validate_progress_rows(
                            candidate_rows,
                            expected_schedule=expected_schedule,
                        )
                        if not all(row_report.values()):
                            raise RuntimeError(
                                "action-adoption runner emitted an invalid row"
                            )
                        rows.append(dict(row))
                        state["updated_at"] = _now()
                        _atomic_json(progress_path, state)

                    run_micro_benchmark_rows(
                        root=str(root), fixtures=fixtures,
                        samples=len(protocol["design"]["sample_indices"]),
                        seed_start=int(protocol["design"]["seed_start"]),
                        match_seconds=float(protocol["design"]["match_seconds"]),
                        spec=spec, cfg=cfg, completed_keys=completed,
                        row_callback=append_row,
                    )
                state["arms"][arm]["state"] = "completed"
                _require_valid_progress(validate_progress(
                    protocol, state, expected_identity=identity,
                ))
            if execution_identity(PROTOCOL_PATH, protocol, root) != identity:
                raise RuntimeError("action-adoption execution identity drifted")
            state.update({"state": "completed", "completed_at": _now()})
            try:
                _require_valid_progress(validate_progress(
                    protocol,
                    state,
                    expected_identity=identity,
                    require_complete=True,
                ))
            except ValueError as exc:
                raise RuntimeError(
                    "fixed action-adoption run budget was not completed"
                ) from exc
            _atomic_json(progress_path, state)
            return state
        except BaseException as exc:
            state.update({
                "state": "failed", "updated_at": _now(),
                "last_error": {"type": type(exc).__name__, "message": str(exc)[:500]},
            })
            _atomic_json(progress_path, state)
            raise


def _keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["fixture"]), int(row["sample_index"])): row for row in rows}


def analyze(
    protocol: dict[str, Any], state: dict[str, Any],
    *, expected_identity: dict[str, Any],
) -> dict[str, Any]:
    if state.get("state") != "completed":
        raise ValueError("action-adoption study must be complete before analysis")
    validation = validate_progress(
        protocol,
        state,
        expected_identity=expected_identity,
        require_complete=True,
    )
    if not validation["structural_valid"]:
        _require_valid_progress(validation, require_identity=False)
    if not validation["identity_matches"]:
        raise ValueError("completed action-adoption identity is stale")
    rows = {arm: state["arms"][arm]["rows"] for arm in ARM_IDS}
    keyed = {arm: _keyed(rows[arm]) for arm in ARM_IDS}
    candidate = rows["M1_action_policy"]
    negative = rows["M1_predict_only"]
    opportunities = sum(float(row["wm_action_opportunities"]) for row in candidate)
    influenced = sum(float(row["wm_action_influenced_opportunities"]) for row in candidate)
    eligible = sum(
        float(row["wm_action_attribution_eligible_opportunities"])
        for row in candidate
    )
    expected_changes = sum(
        float(row["wm_action_expected_counterfactual_changes"])
        for row in candidate
    )
    realized_changes = sum(
        float(row["wm_action_counterfactual_changes"]) for row in candidate
    )
    changed_pairs = sum(
        any(
            abs(float(keyed["M1_action_policy"][key][metric])
                - float(keyed["M1_predict_only"][key][metric]))
            > float(protocol["analysis"]["behavior_change_tolerance"])
            for metric in METRICS
        )
        for key in keyed["M1_action_policy"]
    )
    settings = protocol["analysis"]
    pairs_per_arm = int(protocol["design"]["runs_per_arm"])
    runs_total = int(protocol["design"]["runs_total"])
    gates = {
        "fixed_complete_paired_budget": True,
        "execution_identity_verified": True,
        "no_material_deviations": not (state.get("material_deviations") or []),
        "minimum_action_opportunities": opportunities
        >= float(settings["minimum_action_opportunities"]),
        "minimum_influenced_opportunity_fraction": influenced / max(1.0, opportunities)
        >= float(settings["minimum_influenced_opportunity_fraction"]),
        "minimum_attribution_eligible_fraction": eligible / max(1.0, opportunities)
        >= float(settings["minimum_attribution_eligible_fraction"]),
        "minimum_expected_counterfactual_changes": expected_changes
        >= float(settings["minimum_expected_counterfactual_changes"]),
        "minimum_realized_counterfactual_changes": realized_changes
        >= float(settings["minimum_realized_counterfactual_changes"]),
        "minimum_changed_match_pair_fraction": changed_pairs / pairs_per_arm
        >= float(settings["minimum_changed_match_pair_fraction"]),
        "negative_control_has_zero_influenced_opportunities": sum(
            float(row["wm_action_influenced_opportunities"]) for row in negative
        ) == 0.0,
        "training_not_executed": True,
        "provider_calls_not_made": True,
        "promotion_forbidden": settings.get("promotion_forbidden") is True,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "mechanism_confirmed" if passed else "mechanism_not_confirmed",
        "passed": passed,
        "claim_scope": protocol["claim_scope"],
        "runs_executed": runs_total,
        "pairs_per_arm": pairs_per_arm,
        "mechanism": {
            "action_opportunities": opportunities,
            "influenced_opportunities": influenced,
            "attribution_eligible_opportunities": eligible,
            "expected_counterfactual_action_changes": expected_changes,
            "realized_counterfactual_action_changes": realized_changes,
            "changed_match_pairs": changed_pairs,
            "changed_match_pair_fraction": changed_pairs / pairs_per_arm,
        },
        "gates": gates,
        "execution_identity": expected_identity,
        "material_deviations": list(state.get("material_deviations") or []),
        "training_executed": False,
        "provider_calls_made": False,
        "promotion_authorized": False,
    }


def export_analysis_artifacts(
    protocol: dict[str, Any], state: dict[str, Any],
    result: dict[str, Any], *, root: Path = ROOT,
) -> dict[str, Any]:
    """Export reviewer-readable rows and a bounded decision summary."""
    _require_valid_progress(validate_progress(
        protocol,
        state,
        expected_identity=result.get("execution_identity"),
        require_complete=True,
    ))
    rows = [
        {"arm": arm, **dict(row)}
        for arm in ARM_IDS
        for row in state["arms"][arm]["rows"]
    ]
    fieldnames = ["arm", *sorted({
        key for row in rows for key in row if key != "arm"
    })]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_path = root / protocol["outputs"]["rows_csv"]
    summary_path = root / protocol["outputs"]["summary_markdown"]
    _atomic_text(csv_path, csv_buffer.getvalue())
    mechanism = result["mechanism"]
    gates = result["gates"]
    summary = "\n".join([
        "# World-model action-adoption mechanism study",
        "",
        f"- Status: {result['status']}",
        f"- Fixed runs: {result['runs_executed']}",
        f"- Matched pairs per arm: {result['pairs_per_arm']}",
        f"- Action opportunities: {mechanism['action_opportunities']}",
        f"- Influenced opportunities: {mechanism['influenced_opportunities']}",
        (
            "- Realized counterfactual action changes: "
            f"{mechanism['realized_counterfactual_action_changes']}"
        ),
        f"- Changed match pairs: {mechanism['changed_match_pairs']}",
        "- Training executed: no",
        "- Provider calls made: no",
        "",
        "## Frozen gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
        *[
            f"| {name} | {'yes' if passed else 'no'} |"
            for name, passed in gates.items()
        ],
        "",
        "## Claim boundary",
        "",
        (
            "This confirms or rejects the identity-bound action-adoption "
            "mechanism only. It does not authorize product or academic "
            "promotion and does not establish causal match-outcome effects."
        ),
        "",
    ])
    _atomic_text(summary_path, summary)
    return {
        "rows_csv": {
            "path": str(csv_path.relative_to(root)).replace("\\", "/"),
            "sha256": file_sha256(csv_path),
            "rows": len(rows),
        },
        "summary_markdown": {
            "path": str(summary_path.relative_to(root)).replace("\\", "/"),
            "sha256": file_sha256(summary_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    parser.add_argument("--authorization", default="")
    parser.add_argument("--smoke-out", type=Path, default=SMOKE_OUTPUT)
    args = parser.parse_args(argv)
    protocol = _read_json(PROTOCOL_PATH)
    if args.execute:
        result = execute(authorization=args.authorization)
    elif args.analyze:
        identity = execution_identity(PROTOCOL_PATH, protocol)
        progress = _read_json(ROOT / protocol["outputs"]["progress"])
        result = analyze(protocol, progress, expected_identity=identity)
        result["exports"] = export_analysis_artifacts(
            protocol, progress, result,
        )
        _atomic_json(ROOT / protocol["outputs"]["decision"], result)
    elif args.smoke:
        if args.authorization:
            parser.error("--authorization is reserved for --execute")
        result = smoke(output_path=args.smoke_out)
    elif args.status:
        result = status()
    else:
        result = protocol_report()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
