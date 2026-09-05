#!/usr/bin/env python3
"""One bounded, preregistered M0-vs-M1 confirmatory experiment workflow."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
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
    PIPELINE_PRESETS,
    ablation_context,
)
from src.match_engine.calibration.benchmark_core import (  # noqa: E402
    run_micro_benchmark_rows,
)
from src.match_engine.calibration.contract import (  # noqa: E402
    evaluate_rows, load_contract, load_statsbomb_baselines,
)
from src.match_engine.calibration.objective import calibration_loss  # noqa: E402

DEFAULT_PROTOCOL = ROOT / "data/evaluation/formal_experiment_protocol_v2.json"
AUTHORIZATION = "I_AUTHORIZE_GFS_FORMAL_EXPERIMENT_V2"
ARM_IDS = ("M0", "M1")
PROGRESS_STATES = frozenset({"running", "failed", "completed"})
ANALYSIS_INPUT_FILES = (
    "data/calibration/observable_contract.json",
    "data/calibration/statsbomb_match_baselines.json",
    "data/calibration/joint_baselines.json",
)
ACTION_ANALYSIS_METRICS = (
    "wm_action_opportunities",
    "wm_action_influenced_opportunities",
    "wm_action_attribution_eligible_opportunities",
    "wm_action_expected_counterfactual_changes",
    "wm_action_counterfactual_changes",
)
OUTCOME_ANALYSIS_METRICS = tuple(dict.fromkeys((
    *METRICS,
    "through_share",
    "goals_to_micro_xg_ratio",
    "possession_home",
    "possession_away",
    "passes_home",
    "passes_away",
    "shots_home",
    "shots_away",
    "goals_home",
    "goals_away",
    *ACTION_ANALYSIS_METRICS,
)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_formal_authorization(value: str) -> None:
    if value != AUTHORIZATION:
        raise PermissionError("explicit formal experiment authorization is required")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write(os.linesep)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_protocol_header(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 2:
        raise ValueError("formal protocol schema_version must be 2")
    if protocol.get("state") != "preregistered_not_executed":
        raise ValueError("formal protocol must remain preregistered and unexecuted")
    if protocol.get("arms") != {"baseline": "M0", "candidate": "M1"}:
        raise ValueError("formal protocol must contain exactly frozen M0 and M1")


def _validated_design(protocol: dict[str, Any]) -> dict[str, int]:
    design = protocol.get("design") or {}
    fixtures = design.get("fixtures") or []
    samples = design.get("sample_indices") or []
    if len(fixtures) != 6 or any(len(pair) != 2 for pair in fixtures):
        raise ValueError("formal protocol requires exactly six two-team fixtures")
    if len({tuple(pair) for pair in fixtures}) != len(fixtures):
        raise ValueError("formal fixtures must be unique")
    if samples != list(range(len(samples))) or len(samples) < 3:
        raise ValueError("sample_indices must be contiguous from zero with at least three seeds")
    pairs = len(fixtures) * len(samples)
    if design.get("pairs_total") != pairs:
        raise ValueError("pairs_total does not match fixtures x samples")
    if design.get("runs_per_arm") != pairs or design.get("runs_total") != pairs * 2:
        raise ValueError("formal run budget is inconsistent")
    if design["runs_total"] > 60:
        raise ValueError("formal run budget exceeds the preregistered compute ceiling")
    if design.get("arm_order") != ["M0", "M1"]:
        raise ValueError("formal arm order must remain exactly M0 then M1")
    return {"fixtures": len(fixtures), "pairs": pairs, "runs": pairs * 2}


def _validate_analysis(protocol: dict[str, Any]) -> None:
    analysis = protocol.get("analysis") or {}
    if analysis.get("primary_method") != "fixture_stratified_paired_bootstrap":
        raise ValueError("primary analysis must preserve matched fixture strata")
    if not 0 < float(analysis.get("minimum_meaningful_delta_loss", 0)) < 1:
        raise ValueError("minimum meaningful effect must be fixed and positive")
    if analysis.get("secondary_role") != "descriptive_only_no_promotion_claim":
        raise ValueError("secondary outcomes cannot drive promotion")
    if tuple(analysis.get("secondary_metrics") or ()) != METRICS:
        raise ValueError("secondary metric registry must remain frozen")
    if float(analysis.get("confidence_level", 0)) != 0.95:
        raise ValueError("confirmatory confidence level must remain 0.95")
    if (
        int(analysis.get("bootstrap_draws", 0)) != 10000
        or int(analysis.get("bootstrap_seed", -1)) != 260806
    ):
        raise ValueError("confirmatory bootstrap budget and seed must remain frozen")


def _validate_integrity(protocol: dict[str, Any]) -> None:
    integrity = protocol.get("integrity") or {}
    if integrity.get("interim_analysis") or integrity.get("optional_stopping"):
        raise ValueError("interim analysis and optional stopping are prohibited")
    if integrity.get("post_hoc_fixture_or_metric_selection") is not False:
        raise ValueError("post-hoc fixture or metric selection is prohibited")
    if not integrity.get("resume_requires_identical_protocol_checkpoint_and_code_identity"):
        raise ValueError("resume identity must be fail-closed")


def validate_protocol(protocol: dict[str, Any]) -> dict[str, int]:
    _validate_protocol_header(protocol)
    budget = _validated_design(protocol)
    _validate_analysis(protocol)
    _validate_integrity(protocol)
    return budget


def execution_identity(root: Path, protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    checkpoint = root / protocol["candidate"]["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing sealed checkpoint: {checkpoint}")
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != protocol["candidate"]["checkpoint_sha256"]:
        raise ValueError("sealed checkpoint identity mismatch")
    integrity = protocol["integrity"]
    code = code_identity_manifest(
        root,
        integrity["code_identity_files"],
        mode=str(integrity.get("code_identity_mode") or "explicit_files_v1"),
    )
    return {
        "protocol_sha256": file_sha256(protocol_path),
        "checkpoint_sha256": checkpoint_sha,
        "code_sha256": code,
        "analysis_input_sha256": {
            relative: file_sha256(root / relative)
            for relative in ANALYSIS_INPUT_FILES
        },
    }


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
            "calibration_and_action_metrics_are_finite": False,
            "action_counts_are_bounded": False,
        }
    keys: list[tuple[str, int] | None] = []
    rows_are_objects = True
    metrics_are_finite = True
    action_counts_are_bounded = True
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
        if not all(
            _finite_number(row.get(name)) for name in OUTCOME_ANALYSIS_METRICS
        ):
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
        "calibration_and_action_metrics_are_finite": metrics_are_finite,
        "action_counts_are_bounded": action_counts_are_bounded,
    }


def validate_progress(
    protocol: dict[str, Any],
    state: dict[str, Any],
    *,
    expected_identity: dict[str, Any] | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Validate the fixed full-match ledger without running or analyzing it."""
    budget = validate_protocol(protocol)
    expected_schedule = _expected_schedule(protocol)
    arms = state.get("arms")
    arms_are_exact = isinstance(arms, dict) and set(arms) == set(ARM_IDS)
    row_checks: dict[str, dict[str, bool]] = {}
    completed_runs: dict[str, int] = {}
    arm_states: dict[str, Any] = {}
    arm_rows: dict[str, list[Any]] = {}
    for arm in ARM_IDS:
        arm_payload = arms.get(arm) if arms_are_exact else None
        rows = arm_payload.get("rows") if isinstance(arm_payload, dict) else None
        row_checks[arm] = _validate_progress_rows(
            rows, expected_schedule=expected_schedule,
        )
        arm_rows[arm] = rows if isinstance(rows, list) else []
        completed_runs[arm] = len(arm_rows[arm])
        arm_states[arm] = (
            arm_payload.get("state") if isinstance(arm_payload, dict) else None
        )
    rows_follow_contract = all(
        all(checks.values()) for checks in row_checks.values()
    )
    arm_completion_is_consistent = all(
        arm_states[arm] in {None, "completed"}
        and (
            arm_states[arm] != "completed"
            or completed_runs[arm] == budget["pairs"]
        )
        for arm in ARM_IDS
    )
    baseline, candidate = ARM_IDS
    arm_order_is_resumable = not (
        completed_runs[baseline] < budget["pairs"]
        and (
            completed_runs[candidate] > 0
            or arm_states[candidate] == "completed"
        )
    ) and not (
        completed_runs[candidate] > 0
        and arm_states[baseline] != "completed"
    )
    fixed_budget_complete = (
        all(completed_runs[arm] == budget["pairs"] for arm in ARM_IDS)
        and all(arm_states[arm] == "completed" for arm in ARM_IDS)
    )
    baseline_has_zero_action_activity = all(
        all(float(row[name]) == 0.0 for name in ACTION_ANALYSIS_METRICS)
        for row in arm_rows[baseline]
        if isinstance(row, dict)
        and all(_finite_number(row.get(name)) for name in ACTION_ANALYSIS_METRICS)
    )
    checks = {
        "schema_and_protocol_match": (
            state.get("schema_version") == protocol["schema_version"]
            and state.get("protocol_id") == protocol["protocol_id"]
        ),
        "state_is_resumable": state.get("state") in PROGRESS_STATES,
        "arms_are_exact": arms_are_exact,
        "rows_follow_frozen_contract": rows_follow_contract,
        "baseline_has_zero_action_activity": baseline_has_zero_action_activity,
        "arm_completion_is_consistent": arm_completion_is_consistent,
        "arm_order_is_resumable": arm_order_is_resumable,
        "overall_completion_is_consistent": (
            state.get("state") != "completed" or fixed_budget_complete
        ),
        "required_complete_budget": (
            not require_complete
            or (state.get("state") == "completed" and fixed_budget_complete)
        ),
        "recorded_rows_do_not_exceed_budget": (
            sum(completed_runs.values()) <= budget["runs"]
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
        "fixed_run_budget": budget["runs"],
        "remaining_runs": max(
            0, budget["runs"] - sum(completed_runs.values()),
        ),
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
            "formal progress is invalid: "
            + ", ".join([*failures, *row_failures])
        )
    if require_identity and not report["identity_matches"]:
        raise ValueError(
            "cannot resume after protocol, checkpoint, or code drift"
        )


def _keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["fixture"]), int(row["sample_index"]))
        if key in keyed:
            raise ValueError(f"duplicate experimental unit: {key}")
        keyed[key] = row
    return keyed


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _loss(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    evaluation = evaluate_rows(
        rows, contract=load_contract(), baselines=load_statsbomb_baselines(),
    )
    return calibration_loss(evaluation), evaluation


def stratified_paired_interval(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int | str]:
    baseline, candidate = _keyed(baseline_rows), _keyed(candidate_rows)
    if set(baseline) != set(candidate):
        raise ValueError("M0 and M1 experimental units do not match exactly")
    strata: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in sorted(baseline):
        strata[key[0]].append(key)
    rng = random.Random(seed)
    delta_draws: list[float] = []
    for _ in range(draws):
        sampled: list[tuple[str, int]] = []
        for fixture in sorted(strata):
            keys = strata[fixture]
            sampled.extend(rng.choice(keys) for _ in keys)
        base_loss, _ = _loss([baseline[key] for key in sampled])
        candidate_loss, _ = _loss([candidate[key] for key in sampled])
        delta_draws.append(candidate_loss - base_loss)
    base_loss, _ = _loss(list(baseline.values()))
    candidate_loss, _ = _loss(list(candidate.values()))
    return {
        "method": "fixture_stratified_paired_bootstrap",
        "draws": draws,
        "seed": seed,
        "point_delta": candidate_loss - base_loss,
        "ci95_low": _percentile(delta_draws, 0.025),
        "ci95_high": _percentile(delta_draws, 0.975),
    }


def analyze(
    protocol: dict[str, Any],
    state: dict[str, Any],
    *,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    budget = validate_protocol(protocol)
    if state.get("state") != "completed":
        raise ValueError("experiment must be completed before analysis")
    validation = validate_progress(
        protocol,
        state,
        expected_identity=expected_identity,
        require_complete=True,
    )
    if not validation["structural_valid"]:
        _require_valid_progress(validation, require_identity=False)
    if not validation["identity_matches"]:
        raise ValueError("completed experiment identity no longer matches frozen inputs")
    identity_verified = True
    arms = state.get("arms") or {}
    baseline_rows = (arms.get("M0") or {}).get("rows") or []
    candidate_rows = (arms.get("M1") or {}).get("rows") or []
    baseline, candidate = _keyed(baseline_rows), _keyed(candidate_rows)
    settings = protocol["analysis"]
    interval = stratified_paired_interval(
        baseline_rows, candidate_rows,
        draws=int(settings["bootstrap_draws"]),
        seed=int(settings["bootstrap_seed"]),
    )
    _, candidate_evaluation = _loss(candidate_rows)
    tolerance = float(settings["behavior_change_tolerance"])
    changed_keys = {
        key for key in baseline
        if any(
            abs(float(candidate[key][metric]) - float(baseline[key][metric])) > tolerance
            for metric in METRICS
        )
    }
    changed_fraction = len(changed_keys) / len(baseline)
    minimum = float(settings["minimum_meaningful_delta_loss"])
    promotion_gates = {
        "candidate_passes_all_external_validity_gates": bool(candidate_evaluation["all_pass"]),
        "upper_interval_below_negative_minimum_effect": float(interval["ci95_high"]) < -minimum,
        "minimum_behavior_change_met": changed_fraction >= float(
            settings["minimum_changed_pair_fraction"]
        ),
        "execution_identity_verified": identity_verified,
    }
    promotion = all(promotion_gates.values())
    equivalent = (
        float(interval["ci95_low"]) > -minimum
        and float(interval["ci95_high"]) < minimum
        and not promotion
    )
    decision = (
        "promotion_candidate_pending_release_review" if promotion
        else "no_meaningful_difference_keep_research_only" if equivalent
        else "inconclusive_keep_research_only"
    )
    secondary = {
        metric: {
            "mean_paired_delta": mean(
                float(candidate[key][metric]) - float(baseline[key][metric])
                for key in sorted(baseline)
            ),
            "role": "descriptive_only",
        }
        for metric in METRICS
    }
    return {
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "promotion_supported": promotion,
        "pairs_total": budget["pairs"],
        "primary": interval,
        "minimum_meaningful_delta_loss": minimum,
        "behavior": {
            "changed_pairs": len(changed_keys),
            "changed_pair_fraction": changed_fraction,
        },
        "promotion_gates": promotion_gates,
        "secondary_metrics": secondary,
        "execution_identity": state["execution_identity"],
        "analyzed_at": _now(),
    }


def status(root: Path, protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    budget = validate_protocol(protocol)
    identity = execution_identity(root, protocol_path, protocol)
    progress_path = root / protocol["outputs"]["progress"]
    state = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file() else None
    )
    validation = (
        validate_progress(protocol, state, expected_identity=identity)
        if state is not None else None
    )
    completed = (
        validation["completed_runs"]
        if validation else {arm: 0 for arm in ARM_IDS}
    )
    progress_valid = validation is None or validation["structural_valid"]
    identity_matches_progress = (
        validation is None or validation["identity_matches"]
    )
    recorded_state = (state or {}).get("state", "not_started")
    execution_state = (
        "blocked_invalid_progress"
        if not progress_valid
        else "blocked_identity_drift"
        if not identity_matches_progress
        else recorded_state
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_state": protocol["state"],
        "execution_state": execution_state,
        "recorded_execution_state": recorded_state,
        "budget": budget,
        "completed_runs": completed,
        "remaining_runs": max(0, budget["runs"] - sum(completed.values())),
        "progress_present": validation is not None,
        "progress_valid": progress_valid,
        "progress_checks": validation["checks"] if validation else {},
        "row_checks": validation["row_checks"] if validation else {},
        "identity_ready": bool(identity),
        "identity_matches_progress": identity_matches_progress,
        "next_action": (
            "inspect_invalid_progress"
            if not progress_valid else
            "inspect_identity_drift"
            if not identity_matches_progress else
            "python scripts/run_formal_experiment.py --execute"
            if state is None or state.get("state") != "completed"
            else "python scripts/run_formal_experiment.py --analyze"
        ),
    }


def _new_progress(
    protocol: dict[str, Any], identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "state": "running",
        "started_at": _now(),
        "execution_identity": identity,
        "arms": {arm: {"rows": []} for arm in ARM_IDS},
    }


def _load_or_initialize_progress(
    protocol: dict[str, Any],
    identity: dict[str, Any],
    progress_path: Path,
) -> dict[str, Any]:
    state = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else _new_progress(protocol, identity)
    )
    _require_valid_progress(validate_progress(
        protocol, state, expected_identity=identity,
    ))
    if state.get("state") != "completed":
        state["state"] = "running"
        state.pop("last_error", None)
        state["updated_at"] = _now()
        _atomic_json(progress_path, state)
    return state


def _restore_environment(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


@contextmanager
def _formal_arm_context(
    arm: str, checkpoint: Path,
) -> Iterator[tuple[Any, Any]]:
    spec = PIPELINE_PRESETS[arm]
    previous_checkpoint = os.environ.get("MATCH_WM_CHECKPOINT")
    previous_required = os.environ.get("MATCH_WORLD_MODEL_REQUIRED")
    with ablation_context(spec) as cfg:
        if arm == "M1":
            os.environ["MATCH_WM_CHECKPOINT"] = str(checkpoint)
            os.environ["MATCH_WORLD_MODEL_REQUIRED"] = "1"
        try:
            yield spec, cfg
        finally:
            _restore_environment("MATCH_WM_CHECKPOINT", previous_checkpoint)
            _restore_environment("MATCH_WORLD_MODEL_REQUIRED", previous_required)


def _execute_arm(
    *,
    arm: str,
    root: Path,
    protocol: dict[str, Any],
    identity: dict[str, Any],
    state: dict[str, Any],
    progress_path: Path,
    checkpoint: Path,
    expected_schedule: list[tuple[str, int]],
) -> None:
    arm_state = state["arms"][arm]
    completed = {
        (str(row["fixture"]), int(row["sample_index"]))
        for row in arm_state["rows"]
    }

    def save(row: dict[str, Any]) -> None:
        candidate_rows = [*arm_state["rows"], dict(row)]
        row_report = _validate_progress_rows(
            candidate_rows, expected_schedule=expected_schedule,
        )
        if not all(row_report.values()):
            raise RuntimeError("formal runner emitted an invalid progress row")
        arm_state["rows"].append(dict(row))
        state["updated_at"] = _now()
        _atomic_json(progress_path, state)

    fixtures = [tuple(pair) for pair in protocol["design"]["fixtures"]]
    with _formal_arm_context(arm, checkpoint) as (spec, cfg):
        run_micro_benchmark_rows(
            root=str(root),
            fixtures=fixtures,
            samples=len(protocol["design"]["sample_indices"]),
            seed_start=int(protocol["design"]["seed_start"]),
            match_seconds=float(protocol["design"]["match_seconds"]),
            spec=spec,
            cfg=cfg,
            completed_keys=completed,
            row_callback=save,
        )
    arm_state["state"] = "completed"
    _require_valid_progress(validate_progress(
        protocol, state, expected_identity=identity,
    ))
    _atomic_json(progress_path, state)


def _complete_progress(
    *,
    root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    identity: dict[str, Any],
    state: dict[str, Any],
    progress_path: Path,
) -> None:
    if execution_identity(root, protocol_path, protocol) != identity:
        raise RuntimeError("execution identity drifted during the formal run")
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
            "fixed formal sample budget was not completed"
        ) from exc
    _atomic_json(progress_path, state)


def _record_execution_failure(
    state: dict[str, Any], progress_path: Path, exc: BaseException,
) -> None:
    state.update({
        "state": "failed",
        "updated_at": _now(),
        "last_error": {
            "type": type(exc).__name__,
            "message": str(exc)[:500],
        },
    })
    _atomic_json(progress_path, state)


def execute(
    root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    *,
    authorization: str = "",
) -> dict[str, Any]:
    _require_formal_authorization(authorization)
    validate_protocol(protocol)
    identity = execution_identity(root, protocol_path, protocol)
    progress_path = root / protocol["outputs"]["progress"]
    lock_path = progress_path.with_suffix(".lock")
    with FileLease(lock_path, timeout=1.0):
        state = _load_or_initialize_progress(protocol, identity, progress_path)
        if state.get("state") == "completed":
            return state
        try:
            checkpoint = root / protocol["candidate"]["checkpoint"]
            expected_schedule = _expected_schedule(protocol)
            for arm in protocol["design"]["arm_order"]:
                _execute_arm(
                    arm=arm,
                    root=root,
                    protocol=protocol,
                    identity=identity,
                    state=state,
                    progress_path=progress_path,
                    checkpoint=checkpoint,
                    expected_schedule=expected_schedule,
                )
            _complete_progress(
                root=root,
                protocol_path=protocol_path,
                protocol=protocol,
                identity=identity,
                state=state,
                progress_path=progress_path,
            )
            return state
        except BaseException as exc:
            _record_execution_failure(state, progress_path, exc)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded preregistered M0-vs-M1 formal experiment",
    )
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL.relative_to(ROOT)))
    parser.add_argument("--authorization", default="")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true", help="Run/resume the fixed 60-run budget")
    actions.add_argument("--analyze", action="store_true", help="Analyze only a completed fixed-budget run")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    if args.execute:
        payload = execute(
            ROOT,
            protocol_path,
            protocol,
            authorization=args.authorization,
        )
        result = {"state": payload["state"], "next_action": "--analyze"}
    elif args.analyze:
        _require_formal_authorization(args.authorization)
        progress_path = ROOT / protocol["outputs"]["progress"]
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        payload = analyze(
            protocol, state,
            expected_identity=execution_identity(ROOT, protocol_path, protocol),
        )
        decision_path = ROOT / protocol["outputs"]["decision"]
        _atomic_json(decision_path, payload)
        result = {"decision": payload["decision"], "output": str(decision_path)}
    else:
        if args.authorization:
            parser.error("--authorization is reserved for --execute or --analyze")
        result = status(ROOT, protocol_path, protocol)
        result["note"] = "status only; no simulations were run"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
