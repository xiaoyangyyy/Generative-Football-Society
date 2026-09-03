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

from src.infrastructure import FileLease, file_sha256
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
from src.match_engine.world_model.mirrored_policy_evaluation import (
    fixture_stratified_cluster_interval,
    paired_effect_rows,
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
    if protocol.get("state") != (
        "preregistered_code_ready_awaiting_sealed_checkpoint"
    ):
        raise ValueError("M2 protocol state is not preregistered")
    if protocol.get("arms") != ["M0", "M2_home", "M2_away"]:
        raise ValueError("M2 protocol must retain all three ordered arms")
    design = protocol.get("design") or {}
    fixtures = design.get("fixtures") or []
    samples = int(design.get("samples_per_fixture", 0))
    if len(fixtures) != 6 or any(len(pair) != 2 for pair in fixtures):
        raise ValueError("M2 protocol requires six two-team fixtures")
    if len({tuple(pair) for pair in fixtures}) != len(fixtures):
        raise ValueError("M2 fixtures must be unique")
    units = len(fixtures) * samples
    if samples < 3 or design.get("matched_units") != units:
        raise ValueError("M2 matched-unit budget is inconsistent")
    if design.get("runs_total") != units * 3:
        raise ValueError("M2 run budget must include all three arms")
    if design.get("control_scope") != {
        "M0": "none", "M2_home": "home", "M2_away": "away",
    }:
        raise ValueError("M2 control scopes must remain mirrored and one-sided")
    analysis = protocol.get("analysis") or {}
    if analysis.get("bootstrap_method") != (
        "fixture_stratified_matched_seed_cluster_bootstrap"
    ):
        raise ValueError("M2 bootstrap must preserve matched mirrored clusters")
    if int(analysis.get("bootstrap_draws", 0)) != 10000:
        raise ValueError("M2 bootstrap budget is frozen at 10000")
    if analysis.get("promotion_requires_all_gates") is not True:
        raise ValueError("M2 promotion must require every preregistered gate")
    integrity = protocol.get("integrity") or {}
    if integrity.get("interim_analysis") or integrity.get("optional_stopping"):
        raise ValueError("M2 interim analysis and optional stopping are forbidden")
    if integrity.get("post_hoc_fixture_or_metric_selection") is not False:
        raise ValueError("M2 post-hoc metric selection is forbidden")
    return {"fixtures": len(fixtures), "units": units, "runs": units * 3}


def execution_identity(
    protocol_path: Path,
    protocol: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    validate_protocol(protocol)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing candidate checkpoint: {checkpoint}")
    code = {}
    for relative in protocol["integrity"]["code_identity_files"]:
        path = (ROOT / relative).resolve()
        if not path.is_file() or ROOT.resolve() not in path.parents:
            raise ValueError(f"invalid code identity path: {relative}")
        code[relative] = file_sha256(path)
    return {
        "protocol_sha256": file_sha256(protocol_path),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "code_sha256": code,
    }


def candidate_eligibility(checkpoint: Path, protocol: dict[str, Any]) -> dict:
    runtime = WorldModelRuntime.load(str(checkpoint))
    required = list(protocol["candidate"]["required_policy_utility_branches"])
    gates = {
        action: runtime.policy_utility_authority(action)
        for action in required
    }
    optional = {
        action: runtime.policy_utility_authority(action)
        for action in protocol["candidate"]["optional_policy_utility_branches"]
    }
    two_step = runtime.two_step_planning_gate()
    eligible = bool(
        all(gate.get("authorized") for gate in gates.values())
        and two_step.get("active")
        and runtime.pass_quality >= runtime.cfg.min_planner_quality
    )
    return {
        "eligible": eligible,
        "required_policy_utility_gates": gates,
        "optional_policy_utility_gates": optional,
        "two_step_gate": two_step,
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
    gates = {
        "execution_identity_verified": True,
        "checkpoint_policy_utility_eligible": bool(
            state["candidate_eligibility"]["eligible"]
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
        "mechanism_expected_counterfactual_changes": expected_changes,
        "external_continuous_calibration_noninferiority": external,
        "secondary_goal_difference_effect": goals,
        "sample_budget": budget,
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
        help="Run the sealed 90-match budget; omitted means code-only validation.",
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
        }, indent=2))
        return
    with FileLease(str(progress_path) + ".lock", timeout=0.0):
        if progress_path.is_file():
            state = json.loads(progress_path.read_text(encoding="utf-8"))
            if state.get("execution_identity") != identity:
                raise SystemExit("refusing resume after protocol/checkpoint/code drift")
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
