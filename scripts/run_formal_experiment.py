#!/usr/bin/env python3
"""One bounded, preregistered M0-vs-M1 confirmatory experiment workflow."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_formal_ablation_results import METRICS
from src.infrastructure import FileLease, file_sha256
from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.benchmark_core import run_micro_benchmark_rows
from src.match_engine.calibration.contract import (
    evaluate_rows, load_contract, load_statsbomb_baselines,
)
from src.match_engine.calibration.objective import calibration_loss

DEFAULT_PROTOCOL = ROOT / "data/evaluation/formal_experiment_protocol_v2.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def validate_protocol(protocol: dict[str, Any]) -> dict[str, int]:
    if protocol.get("schema_version") != 2:
        raise ValueError("formal protocol schema_version must be 2")
    if protocol.get("state") != "preregistered_not_executed":
        raise ValueError("formal protocol must remain preregistered and unexecuted")
    if protocol.get("arms") != {"baseline": "M0", "candidate": "M1"}:
        raise ValueError("formal protocol must contain exactly frozen M0 and M1")
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
    integrity = protocol.get("integrity") or {}
    if integrity.get("interim_analysis") or integrity.get("optional_stopping"):
        raise ValueError("interim analysis and optional stopping are prohibited")
    if integrity.get("post_hoc_fixture_or_metric_selection") is not False:
        raise ValueError("post-hoc fixture or metric selection is prohibited")
    if not integrity.get("resume_requires_identical_protocol_checkpoint_and_code_identity"):
        raise ValueError("resume identity must be fail-closed")
    return {"fixtures": len(fixtures), "pairs": pairs, "runs": pairs * 2}


def execution_identity(root: Path, protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    checkpoint = root / protocol["candidate"]["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing sealed checkpoint: {checkpoint}")
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != protocol["candidate"]["checkpoint_sha256"]:
        raise ValueError("sealed checkpoint identity mismatch")
    code: dict[str, str] = {}
    for relative in protocol["integrity"]["code_identity_files"]:
        path = (root / relative).resolve()
        if not path.is_file() or root.resolve() not in path.parents:
            raise ValueError(f"invalid code identity path: {relative}")
        code[relative] = file_sha256(path)
    return {
        "protocol_sha256": file_sha256(protocol_path),
        "checkpoint_sha256": checkpoint_sha,
        "code_sha256": code,
    }


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
    identity_verified = state.get("execution_identity") == expected_identity
    if not identity_verified:
        raise ValueError("completed experiment identity no longer matches frozen inputs")
    arms = state.get("arms") or {}
    baseline_rows = (arms.get("M0") or {}).get("rows") or []
    candidate_rows = (arms.get("M1") or {}).get("rows") or []
    if len(baseline_rows) != budget["pairs"] or len(candidate_rows) != budget["pairs"]:
        raise ValueError("completed experiment does not contain the fixed sample budget")
    baseline, candidate = _keyed(baseline_rows), _keyed(candidate_rows)
    if set(baseline) != set(candidate):
        raise ValueError("completed arms are not exactly paired")
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
    completed = {
        arm: len(((state or {}).get("arms", {}).get(arm) or {}).get("rows") or [])
        for arm in ("M0", "M1")
    }
    identity_matches_progress = bool(
        state is None or state.get("execution_identity") == identity
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_state": protocol["state"],
        "execution_state": (state or {}).get("state", "not_started"),
        "budget": budget,
        "completed_runs": completed,
        "remaining_runs": budget["runs"] - sum(completed.values()),
        "identity_ready": bool(identity),
        "identity_matches_progress": identity_matches_progress,
        "next_action": (
            "inspect_identity_drift"
            if not identity_matches_progress else
            "python scripts/run_formal_experiment.py --execute"
            if state is None or state.get("state") != "completed"
            else "python scripts/run_formal_experiment.py --analyze"
        ),
    }


def execute(root: Path, protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    budget = validate_protocol(protocol)
    identity = execution_identity(root, protocol_path, protocol)
    progress_path = root / protocol["outputs"]["progress"]
    lock_path = progress_path.with_suffix(".lock")
    with FileLease(lock_path, timeout=1.0):
        if progress_path.is_file():
            state = json.loads(progress_path.read_text(encoding="utf-8"))
            if state.get("execution_identity") != identity:
                raise ValueError("cannot resume after protocol, checkpoint, or code drift")
            if state.get("state") == "completed":
                return state
            state["state"] = "running"
            state.pop("last_error", None)
            state["updated_at"] = _now()
            _atomic_json(progress_path, state)
        else:
            state = {
                "schema_version": 2,
                "protocol_id": protocol["protocol_id"],
                "state": "running",
                "started_at": _now(),
                "execution_identity": identity,
                "arms": {"M0": {"rows": []}, "M1": {"rows": []}},
            }
            _atomic_json(progress_path, state)
        fixtures = [tuple(pair) for pair in protocol["design"]["fixtures"]]
        checkpoint = root / protocol["candidate"]["checkpoint"]
        try:
            for arm in protocol["design"]["arm_order"]:
                arm_state = state["arms"][arm]
                completed = {
                    (str(row["fixture"]), int(row["sample_index"]))
                    for row in arm_state["rows"]
                }

                def save(row: dict[str, Any]) -> None:
                    arm_state["rows"].append(row)
                    state["updated_at"] = _now()
                    _atomic_json(progress_path, state)

                spec = PIPELINE_PRESETS[arm]
                previous_checkpoint = os.environ.get("MATCH_WM_CHECKPOINT")
                previous_required = os.environ.get("MATCH_WORLD_MODEL_REQUIRED")
                with ablation_context(spec) as cfg:
                    if arm == "M1":
                        os.environ["MATCH_WM_CHECKPOINT"] = str(checkpoint)
                        os.environ["MATCH_WORLD_MODEL_REQUIRED"] = "1"
                    try:
                        run_micro_benchmark_rows(
                            root=str(root), fixtures=fixtures,
                            samples=len(protocol["design"]["sample_indices"]),
                            seed_start=int(protocol["design"]["seed_start"]),
                            match_seconds=float(protocol["design"]["match_seconds"]),
                            spec=spec, cfg=cfg, completed_keys=completed,
                            row_callback=save,
                        )
                    finally:
                        if previous_checkpoint is None:
                            os.environ.pop("MATCH_WM_CHECKPOINT", None)
                        else:
                            os.environ["MATCH_WM_CHECKPOINT"] = previous_checkpoint
                        if previous_required is None:
                            os.environ.pop("MATCH_WORLD_MODEL_REQUIRED", None)
                        else:
                            os.environ["MATCH_WORLD_MODEL_REQUIRED"] = previous_required
                arm_state["state"] = "completed"
            if any(len(state["arms"][arm]["rows"]) != budget["pairs"] for arm in ("M0", "M1")):
                raise RuntimeError("fixed formal sample budget was not completed")
            if execution_identity(root, protocol_path, protocol) != identity:
                raise RuntimeError("execution identity drifted during the formal run")
            state.update({"state": "completed", "completed_at": _now()})
            _atomic_json(progress_path, state)
            return state
        except BaseException as exc:
            state.update({
                "state": "failed", "updated_at": _now(),
                "last_error": {"type": type(exc).__name__, "message": str(exc)[:500]},
            })
            _atomic_json(progress_path, state)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded preregistered M0-vs-M1 formal experiment",
    )
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL.relative_to(ROOT)))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true", help="Run/resume the fixed 60-run budget")
    actions.add_argument("--analyze", action="store_true", help="Analyze only a completed fixed-budget run")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    if args.execute:
        payload = execute(ROOT, protocol_path, protocol)
        result = {"state": payload["state"], "next_action": "--analyze"}
    elif args.analyze:
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
        result = status(ROOT, protocol_path, protocol)
        result["note"] = "status only; no simulations were run"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
