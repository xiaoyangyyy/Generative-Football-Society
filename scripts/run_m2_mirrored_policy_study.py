#!/usr/bin/env python3
"""Execute or validate the preregistered team-scoped M2 policy study."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import FileLease
from src.match_engine.calibration.ablation import (
    PIPELINE_PRESETS,
    ablation_context,
)
from src.match_engine.calibration.benchmark_core import run_micro_benchmark_rows
from src.match_engine.calibration.contract import (
    evaluate_rows,
    load_contract,
    load_statsbomb_baselines,
)
from src.match_engine.calibration.objective import calibration_loss
from src.match_engine.world_model.inference import WorldModelRuntime
from scripts.validate_world_model import _sealed_evaluation
from src.match_engine.world_model.m2_contract import (
    FROZEN_FIXTURES,
    FROZEN_TRAINING_CONFIGURATION,
)
from src.match_engine.world_model.mirrored_policy_evaluation import (
    fixture_stratified_cluster_interval,
    paired_effect_rows,
    study_execution_identity,
)

DEFAULT_PROTOCOL = (
    ROOT / "data/evaluation/m2_mirrored_policy_protocol_v1.json"
)
DEFAULT_PROGRESS = (
    ROOT / "data/evaluation/m2_mirrored_policy_progress_v1.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-",
    )
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


@contextmanager
def _temporary_environment(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_protocol(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(protocol: dict[str, Any]) -> dict[str, int]:
    if protocol.get("schema_version") != 1:
        raise ValueError("M2 protocol schema_version must be 1")
    if protocol.get("protocol_id") != "m2-mirrored-policy-v1":
        raise ValueError("M2 protocol identity changed")
    if protocol.get("claim_scope") != (
        "simulator_team_scoped_outcome_aligned_policy_not_real_football_causality"
    ):
        raise ValueError("M2 claim scope changed")
    if protocol.get("state") != (
        "preregistered_code_ready_awaiting_sealed_checkpoint"
    ):
        raise ValueError("M2 protocol state is not preregistered")
    if protocol.get("arms") != ["M0", "M2_home", "M2_away"]:
        raise ValueError("M2 protocol must retain all three ordered arms")
    candidate = protocol.get("candidate") or {}
    if candidate.get("dataset_manifest") != (
        "data/world_model/dataset_manifest_formal_v8_candidate.json"
    ):
        raise ValueError("M2 dataset manifest identity changed")
    if candidate.get("trace_dir") != "data/world_model/traces":
        raise ValueError("M2 trace directory identity changed")
    if candidate.get("required_sealed_validation") != [
        "two_step", "policy_utility.pass",
    ]:
        raise ValueError("M2 sealed validation requirements changed")
    if candidate.get("pipeline") != "M2":
        raise ValueError("M2 candidate pipeline changed")
    if candidate.get("checkpoint_binding") != (
        "bound_at_first_execution_then_immutable"
    ):
        raise ValueError("M2 checkpoint binding changed")
    if candidate.get("required_policy_utility_branches") != ["pass"]:
        raise ValueError("M2 required utility branch changed")
    if candidate.get("optional_policy_utility_branches") != ["cross"]:
        raise ValueError("M2 optional utility branch changed")
    if candidate.get("shot_authority") != "independent_frozen_shot_head_gate":
        raise ValueError("M2 shot authority changed")
    training = protocol.get("training") or {}
    expected_training_protocol = {
        **FROZEN_TRAINING_CONFIGURATION,
        "script": "scripts/train_world_model.py",
        "preflight_script": "scripts/preflight_m2_training.py",
        "required_preflight_status": "ready_for_m2_training",
    }
    expected_training_protocol.pop("dataset_manifest")
    if training != expected_training_protocol:
        raise ValueError("M2 training configuration changed")
    design = protocol.get("design") or {}
    fixtures = design.get("fixtures") or []
    samples = int(design.get("samples_per_fixture", 0))
    if fixtures != FROZEN_FIXTURES:
        raise ValueError("M2 fixture order and identities are frozen")
    units = len(fixtures) * samples
    if samples != 20 or design.get("matched_units") != units:
        raise ValueError("M2 matched-unit budget is inconsistent")
    if design.get("runs_total") != units * 3:
        raise ValueError("M2 run budget must include all three arms")
    if design.get("control_scope") != {
        "M0": "none", "M2_home": "home", "M2_away": "away",
    }:
        raise ValueError("M2 control scopes must remain mirrored and one-sided")
    if int(design.get("seed_start", -1)) != 260903:
        raise ValueError("M2 seed start is frozen at 260903")
    if float(design.get("match_seconds", 0.0)) != 5400.0:
        raise ValueError("M2 matches must retain full 5400-second duration")
    analysis = protocol.get("analysis") or {}
    if analysis.get("bootstrap_method") != (
        "fixture_stratified_matched_seed_cluster_bootstrap"
    ):
        raise ValueError("M2 bootstrap must preserve matched mirrored clusters")
    if int(analysis.get("bootstrap_draws", 0)) != 10000:
        raise ValueError("M2 bootstrap budget is frozen at 10000")
    if int(analysis.get("bootstrap_seed", -1)) != 260903:
        raise ValueError("M2 bootstrap seed is frozen at 260903")
    if float(analysis.get("confidence_level", 0.0)) != 0.95:
        raise ValueError("M2 confidence level is frozen at 0.95")
    if analysis.get("primary_efficacy_metric") != "controlled_micro_xg_margin":
        raise ValueError("M2 primary efficacy metric changed")
    if float(analysis.get("minimum_meaningful_delta", 0.0)) != 0.1:
        raise ValueError("M2 meaningful efficacy threshold is frozen at 0.1")
    if float(analysis.get("minimum_counterfactual_change_rate", 0.0)) != 0.02:
        raise ValueError("M2 action-change threshold is frozen at 0.02")
    if float(analysis.get("external_noninferiority_margin", 0.0)) != 1.0:
        raise ValueError("M2 external noninferiority margin is frozen at 1.0")
    if analysis.get("goal_difference_role") != "secondary_descriptive":
        raise ValueError("M2 goal difference must remain secondary")
    if analysis.get("promotion_requires_all_gates") is not True:
        raise ValueError("M2 promotion must require every preregistered gate")
    if int(analysis.get("minimum_attributable_outcomes", 0)) < 384:
        raise ValueError("M2 mechanism endpoint needs at least 384 outcomes")
    unit_fraction = float(
        analysis.get("minimum_attributable_unit_fraction", 0.0)
    )
    if not 0.5 <= unit_fraction <= 1.0:
        raise ValueError("M2 attributable unit fraction is invalid")
    integrity = protocol.get("integrity") or {}
    if integrity.get("interim_analysis") or integrity.get("optional_stopping"):
        raise ValueError("M2 interim analysis and optional stopping are forbidden")
    if integrity.get("post_hoc_fixture_or_metric_selection") is not False:
        raise ValueError("M2 post-hoc metric selection is forbidden")
    power = protocol.get("power_analysis") or {}
    if (
        power.get("script") != "scripts/estimate_m2_policy_power.py"
        or float(power.get("historical_home_xg_margin_paired_sd", 0.0))
        != 0.3673
        or int(power.get(
            "normal_approximation_required_units_for_delta_0_10", 0,
        )) != 106
        or int(power.get("planned_matched_units", 0)) != units
        or power.get("historical_candidate_effect_used_as_expected_effect")
        is not False
    ):
        raise ValueError("M2 prospective power contract changed")
    return {"fixtures": len(fixtures), "units": units, "runs": units * 3}


def execution_identity(
    protocol_path: Path,
    protocol: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    validate_protocol(protocol)
    return study_execution_identity(ROOT, protocol_path, protocol, checkpoint)


def candidate_eligibility(
    checkpoint: Path,
    protocol: dict[str, Any],
    *,
    root: Path = ROOT,
) -> dict:
    runtime = WorldModelRuntime.load(str(checkpoint))
    candidate = protocol["candidate"]
    required = list(candidate["required_policy_utility_branches"])
    gates = {
        action: runtime.policy_utility_authority(action)
        for action in required
    }
    optional = {
        action: runtime.policy_utility_authority(action)
        for action in candidate["optional_policy_utility_branches"]
    }
    two_step = runtime.two_step_planning_gate()
    root = root.resolve()
    manifest_path = (root / str(candidate["dataset_manifest"])).resolve()
    trace_dir = (root / str(candidate["trace_dir"])).resolve()
    try:
        manifest_path.relative_to(root)
        trace_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("M2 sealed inputs must remain inside the project") from exc
    meta = runtime.meta if isinstance(runtime.meta, dict) else {}
    training_configuration = meta.get("training_configuration") or {}
    expected_checkpoint_training = {
        **FROZEN_TRAINING_CONFIGURATION,
        "dataset_manifest": str(candidate["dataset_manifest"]),
    }
    training_configuration_verified = (
        training_configuration == expected_checkpoint_training
        and int(getattr(runtime.model, "transition_member_count", 0)) == 3
    )
    metadata_manifest_value = meta.get("dataset_manifest")
    metadata_manifest = None
    if metadata_manifest_value:
        metadata_manifest = Path(str(metadata_manifest_value))
        if not metadata_manifest.is_absolute():
            metadata_manifest = root / metadata_manifest
        metadata_manifest = metadata_manifest.resolve()
    manifest_identity_verified = bool(
        metadata_manifest == manifest_path and manifest_path.is_file()
    )
    sealed_test_unused = meta.get("sealed_test_used") is False
    development_ready = bool(
        all(gate.get("authorized") for gate in gates.values())
        and two_step.get("active")
        and runtime.pass_quality >= runtime.cfg.min_planner_quality
        and training_configuration_verified
    )
    sealed_validation: dict[str, Any] = {
        "executed": False,
        "reason": "development_or_dataset_contract_not_ready",
    }
    sealed_two_step_active = False
    sealed_pass_gate: dict[str, Any] = {
        "authorized": False,
        "reason": "sealed_validation_not_executed",
    }
    if development_ready and manifest_identity_verified and sealed_test_unused:
        sealed_validation = _sealed_evaluation(
            runtime, trace_dir, manifest_path,
        )
        sealed_validation["manifest"] = str(candidate["dataset_manifest"])
        sealed_validation["executed"] = True
        sealed_two_step_active = bool(
            (sealed_validation.get("two_step") or {}).get("active")
        )
        sealed_pass_gate = (
            ((sealed_validation.get("policy_utility") or {}).get("gates") or {})
            .get("pass") or sealed_pass_gate
        )
    eligible = bool(
        development_ready
        and manifest_identity_verified
        and sealed_test_unused
        and sealed_two_step_active
        and sealed_pass_gate.get("authorized") is True
    )
    return {
        "eligible": eligible,
        "required_policy_utility_gates": gates,
        "optional_policy_utility_gates": optional,
        "two_step_gate": two_step,
        "dataset_manifest": candidate["dataset_manifest"],
        "dataset_manifest_identity_verified": manifest_identity_verified,
        "sealed_test_unused_by_training": sealed_test_unused,
        "training_configuration_verified": training_configuration_verified,
        "training_configuration": training_configuration,
        "sealed_two_step_active": sealed_two_step_active,
        "sealed_pass_policy_utility_gate": sealed_pass_gate,
        "sealed_validation": sealed_validation,
        "pass_planner_quality": runtime.pass_quality,
        "minimum_planner_quality": runtime.cfg.min_planner_quality,
    }


def _keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict]:
    output = {}
    for row in rows:
        key = (str(row["fixture"]), int(row["sample_index"]))
        if key in output:
            raise ValueError(f"duplicate row: {key}")
        output[key] = row
    return output


def _continuous_loss(rows: list[dict[str, Any]]) -> float:
    evaluation = evaluate_rows(
        rows,
        contract=load_contract(),
        baselines=load_statsbomb_baselines(),
    )
    return calibration_loss(evaluation, fail_penalty=0.0)


def verify_runtime_row_identity(
    baseline: list[dict[str, Any]],
    home: list[dict[str, Any]],
    away: list[dict[str, Any]],
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Prove every result row ran its declared model and control scope."""
    expected_signature = "sha256:" + checkpoint_sha256
    if any(
        row.get("wm_runtime_loaded")
        or row.get("wm_checkpoint_signature") is not None
        or row.get("wm_control_scope") != "none"
        or row.get("wm_outcome_aligned_policy")
        for row in baseline
    ):
        raise ValueError("M0 rows contain world-model runtime leakage")
    for scope, rows in (("home", home), ("away", away)):
        if any(
            row.get("wm_runtime_loaded") is not True
            or row.get("wm_checkpoint_signature") != expected_signature
            or row.get("wm_control_scope") != scope
            or row.get("wm_outcome_aligned_policy") is not True
            for row in rows
        ):
            raise ValueError(
                f"M2_{scope} rows failed runtime identity verification"
            )
    return {
        "verified": True,
        "checkpoint_signature": expected_signature,
        "rows_by_scope": {
            "none": len(baseline),
            "home": len(home),
            "away": len(away),
        },
    }


def _continuous_loss_interval(
    baseline_rows: list[dict[str, Any]],
    home_rows: list[dict[str, Any]],
    away_rows: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    baseline, home, away = (
        _keyed(baseline_rows), _keyed(home_rows), _keyed(away_rows)
    )
    if set(baseline) != set(home) or set(baseline) != set(away):
        raise ValueError("continuous-loss arms are not matched")
    strata: dict[str, list[int]] = {}
    for fixture, sample_index in baseline:
        strata.setdefault(fixture, []).append(sample_index)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        baseline_sample, candidate_sample = [], []
        for fixture in sorted(strata):
            seeds = sorted(set(strata[fixture]))
            for _slot in seeds:
                chosen = rng.choice(seeds)
                key = (fixture, chosen)
                baseline_sample.extend([baseline[key], baseline[key]])
                candidate_sample.extend([home[key], away[key]])
        estimates.append(
            _continuous_loss(candidate_sample)
            - _continuous_loss(baseline_sample)
        )
    estimates.sort()

    def percentile(q: float) -> float:
        position = (len(estimates) - 1) * q
        low = int(position)
        high = min(low + 1, len(estimates) - 1)
        fraction = position - low
        return estimates[low] * (1.0 - fraction) + estimates[high] * fraction

    doubled_baseline = [
        row for row in baseline.values() for _ in range(2)
    ]
    pooled_candidate = list(home.values()) + list(away.values())
    return {
        "method": "fixture_stratified_matched_seed_cluster_bootstrap",
        "fail_penalty": 0.0,
        "point_delta": (
            _continuous_loss(pooled_candidate)
            - _continuous_loss(doubled_baseline)
        ),
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
        "draws": draws,
        "seed": seed,
    }


def analyze(
    protocol: dict[str, Any],
    state: dict[str, Any],
    *,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    budget = validate_protocol(protocol)
    arms = state.get("arms") or {}
    if any(len((arms.get(arm) or {}).get("rows") or []) != budget["units"]
           for arm in protocol["arms"]):
        raise ValueError("M2 analysis requires the complete frozen run budget")
    if state.get("execution_identity") != expected_identity:
        raise ValueError("M2 execution identity drifted")
    baseline = arms["M0"]["rows"]
    home = arms["M2_home"]["rows"]
    away = arms["M2_away"]["rows"]
    runtime_identity = verify_runtime_row_identity(
        baseline,
        home,
        away,
        checkpoint_sha256=expected_identity["checkpoint_sha256"],
    )
    effects = paired_effect_rows(baseline, home, away)
    settings = protocol["analysis"]
    draws = int(settings["bootstrap_draws"])
    seed = int(settings["bootstrap_seed"])
    xg = fixture_stratified_cluster_interval(
        effects,
        lambda row: row["effects"]["controlled_micro_xg_margin"],
        draws=draws, seed=seed,
    )
    goals = fixture_stratified_cluster_interval(
        effects,
        lambda row: row["effects"]["controlled_goal_difference"],
        draws=draws, seed=seed + 1,
    )
    mechanism = fixture_stratified_cluster_interval(
        effects,
        lambda row: row["candidate"][
            "wm_attributable_realized_policy_utility_mean"
        ],
        draws=draws, seed=seed + 2,
    )
    expected_changes = fixture_stratified_cluster_interval(
        effects,
        lambda row: row["candidate"][
            "wm_action_expected_counterfactual_change_rate"
        ],
        draws=draws, seed=seed + 3,
    )
    external = _continuous_loss_interval(
        baseline, home, away, draws=draws, seed=seed + 4,
    )
    minimum = float(settings["minimum_meaningful_delta"])
    attributable_counts = [
        float(row["candidate"][
            "wm_attributable_realized_policy_utility_count"
        ])
        for row in effects
    ]
    attributable_total = int(sum(attributable_counts))
    attributable_unit_fraction = float(
        sum(count > 0 for count in attributable_counts)
        / max(1, len(attributable_counts))
    )
    gates = {
        "execution_identity_verified": True,
        "checkpoint_policy_utility_eligible": bool(
            state["candidate_eligibility"]["eligible"]
        ),
        "minimum_attributable_outcomes_observed": (
            attributable_total
            >= int(settings["minimum_attributable_outcomes"])
        ),
        "minimum_attributable_unit_coverage_observed": (
            attributable_unit_fraction
            >= float(settings["minimum_attributable_unit_fraction"])
        ),
        "mechanism_realized_utility_ci_above_zero": (
            float(mechanism["ci95_low"]) > 0.0
        ),
        "minimum_expected_counterfactual_change_rate": (
            float(expected_changes["point_estimate"])
            >= float(settings["minimum_counterfactual_change_rate"])
        ),
        "controlled_micro_xg_effect_is_meaningful": (
            float(xg["point_estimate"]) >= minimum
            and float(xg["ci95_low"]) > 0.0
        ),
        "external_continuous_loss_noninferior": (
            float(external["ci95_high"])
            <= float(settings["external_noninferiority_margin"])
        ),
    }
    return {
        "schema_version": 1,
        "state": "completed",
        "decision": (
            "promotion_supported" if all(gates.values())
            else "research_only_default_off"
        ),
        "promotion_supported": all(gates.values()),
        "promotion_gates": gates,
        "primary_controlled_micro_xg_effect": xg,
        "mechanism_realized_policy_utility": mechanism,
        "mechanism_attributable_coverage": {
            "outcomes": attributable_total,
            "minimum_outcomes": int(
                settings["minimum_attributable_outcomes"]
            ),
            "unit_fraction": attributable_unit_fraction,
            "minimum_unit_fraction": float(
                settings["minimum_attributable_unit_fraction"]
            ),
        },
        "mechanism_expected_counterfactual_changes": expected_changes,
        "external_continuous_calibration_noninferiority": external,
        "secondary_goal_difference_effect": goals,
        "sample_budget": budget,
        "runtime_row_identity": runtime_identity,
        "historical_m1_results_used": False,
        "generated_at": _now(),
    }


def _run_arm(
    arm: str,
    protocol: dict[str, Any],
    checkpoint: Path,
    state: dict[str, Any],
    progress_path: Path,
) -> None:
    design = protocol["design"]
    rows = state["arms"][arm]["rows"]
    completed = {
        (str(row["fixture"]), int(row["sample_index"])) for row in rows
    }

    def persist(row: dict[str, Any]) -> None:
        state["arms"][arm]["rows"].append(row)
        state["updated_at"] = _now()
        _atomic_json(progress_path, state)

    spec = PIPELINE_PRESETS["M0" if arm == "M0" else "M2"]
    scope = design["control_scope"][arm]
    environment = {"MATCH_WM_CONTROL_SCOPE": scope}
    if arm != "M0":
        environment["MATCH_WM_CHECKPOINT"] = str(checkpoint.resolve())
    with ablation_context(spec), _temporary_environment(environment):
        run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=[tuple(pair) for pair in design["fixtures"]],
            samples=int(design["samples_per_fixture"]),
            seed_start=int(design["seed_start"]),
            match_seconds=float(design["match_seconds"]),
            spec=spec,
            completed_keys=completed,
            row_callback=persist,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or execute the preregistered M2 mirrored study",
    )
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--checkpoint")
    parser.add_argument("--progress", default=str(DEFAULT_PROGRESS))
    parser.add_argument(
        "--execute", action="store_true",
        help=(
            "Run the sealed 360-match budget; omitted means code-only "
            "validation."
        ),
    )
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    progress_path = Path(args.progress).resolve()
    protocol = load_protocol(protocol_path)
    budget = validate_protocol(protocol)
    if not args.checkpoint:
        print(json.dumps({
            "code_ready": True,
            "protocol_valid": True,
            "execution_started": False,
            "candidate_checkpoint_bound": False,
            "budget": budget,
            "training_contract": protocol["training"],
            "preflight_command": "python scripts/preflight_m2_training.py",
            "next_action": (
                "train_or_supply_one sealed checkpoint, then validate eligibility"
            ),
        }, indent=2))
        if args.execute:
            raise SystemExit("--execute requires --checkpoint")
        return
    checkpoint = Path(args.checkpoint).resolve()
    identity = execution_identity(protocol_path, protocol, checkpoint)
    eligibility = candidate_eligibility(checkpoint, protocol)
    if not eligibility["eligible"]:
        raise SystemExit(json.dumps({
            "code_ready": True,
            "candidate_eligible": False,
            "candidate_eligibility": eligibility,
            "execution_started": False,
        }, indent=2))
    if not args.execute:
        print(json.dumps({
            "code_ready": True,
            "protocol_valid": True,
            "candidate_eligible": True,
            "execution_started": False,
            "execution_identity": identity,
            "budget": budget,
            "training_configuration_verified": eligibility[
                "training_configuration_verified"
            ],
            "sealed_two_step_active": eligibility["sealed_two_step_active"],
            "sealed_pass_policy_utility_gate": eligibility[
                "sealed_pass_policy_utility_gate"
            ],
        }, indent=2))
        return
    with FileLease(str(progress_path) + ".lock", timeout=0.0):
        if progress_path.is_file():
            state = json.loads(progress_path.read_text(encoding="utf-8"))
            if state.get("execution_identity") != identity:
                raise SystemExit("refusing resume after protocol/checkpoint/code drift")
            if state.get("candidate_eligibility") != eligibility:
                raise SystemExit("refusing resume after candidate eligibility drift")
        else:
            state = {
                "schema_version": 1,
                "state": "running",
                "started_at": _now(),
                "updated_at": _now(),
                "execution_identity": identity,
                "candidate_eligibility": eligibility,
                "arms": {arm: {"rows": []} for arm in protocol["arms"]},
            }
            _atomic_json(progress_path, state)
        for arm in protocol["arms"]:
            _run_arm(
                arm, protocol, checkpoint, state, progress_path,
            )
        result = analyze(protocol, state, expected_identity=identity)
        state["state"] = "completed"
        state["analysis"] = result
        state["updated_at"] = _now()
        _atomic_json(progress_path, state)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
