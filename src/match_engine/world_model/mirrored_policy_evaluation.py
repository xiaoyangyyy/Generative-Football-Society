"""Pure analysis helpers for team-scoped mirrored M2 policy experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any, Callable, Mapping

import numpy as np

from src.infrastructure import code_identity_manifest, file_sha256


def validate_m2_preflight_receipt(
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Reject internally inconsistent or over-claiming preflight receipts."""
    training = dict(protocol.get("training") or {})
    for key in ("script", "preflight_script", "required_preflight_status"):
        training.pop(key, None)
    training["dataset_manifest"] = str(
        (protocol.get("candidate") or {}).get("dataset_manifest") or ""
    )
    checks = receipt.get("checks")
    if receipt.get("schema_version") != 1:
        raise ValueError("unsupported M2 preflight receipt schema")
    if receipt.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("M2 preflight receipt protocol mismatch")
    if receipt.get("claim_scope") != (
        "zero_training_readiness_only_no_model_or_outcome_evidence"
    ):
        raise ValueError("M2 preflight receipt claim scope changed")
    if receipt.get("training_executed") is not False:
        raise ValueError("M2 preflight receipt cannot claim training")
    if receipt.get("checkpoint_written") is not False:
        raise ValueError("M2 preflight receipt cannot claim a checkpoint")
    if receipt.get("configuration") != training:
        raise ValueError("M2 preflight receipt configuration mismatch")
    if not isinstance(checks, Mapping) or not checks or not all(
        isinstance(value, bool) for value in checks.values()
    ):
        raise ValueError("M2 preflight receipt checks are invalid")
    ready = all(checks.values())
    expected_status = (
        "ready_for_m2_training" if ready else "blocked_before_training"
    )
    if receipt.get("ready") is not ready or receipt.get("status") != expected_status:
        raise ValueError("M2 preflight receipt readiness is inconsistent")
    sealed = receipt.get("sealed_test")
    if (
        not isinstance(sealed, Mapping)
        or sealed.get("used_for_training_or_tuning") is not False
        or sealed.get("rows_loaded_for_model_selection") != 0
        or sealed.get("support_read_from_frozen_manifest_only") is not True
    ):
        raise ValueError("M2 preflight receipt violates sealed-test isolation")
    if not isinstance(receipt.get("evidence_identity"), Mapping):
        raise ValueError("M2 preflight receipt identity is missing")


def validate_m2_candidate_receipt(
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Reject candidate receipts that conflict with qualification evidence."""
    eligibility = receipt.get("candidate_eligibility")
    if receipt.get("schema_version") != 1:
        raise ValueError("unsupported M2 candidate receipt schema")
    if receipt.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("M2 candidate receipt protocol mismatch")
    if receipt.get("claim_scope") != (
        "checkpoint_qualification_only_no_policy_effect_or_outcome_claim"
    ):
        raise ValueError("M2 candidate receipt claim scope changed")
    if (
        receipt.get("formal_execution_started") is not False
        or receipt.get("formal_result_available") is not False
    ):
        raise ValueError("M2 candidate receipt cannot claim formal execution")
    if not isinstance(eligibility, Mapping):
        raise ValueError("M2 candidate receipt qualification details are missing")
    eligible = eligibility.get("eligible") is True
    expected_status = (
        "eligible_for_m2_execution" if eligible else "candidate_rejected"
    )
    if (
        receipt.get("candidate_eligible") is not eligible
        or receipt.get("status") != expected_status
    ):
        raise ValueError("M2 candidate receipt eligibility is inconsistent")
    if not isinstance(receipt.get("execution_identity"), Mapping):
        raise ValueError("M2 candidate receipt identity is missing")


def study_preflight_identity(
    root: str | Path,
    protocol_path: str | Path,
    protocol: dict[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Bind a recorded zero-training preflight to its frozen inputs.

    The preflight itself verifies every trace against the manifest. This
    lighter identity lets product status requests prove that the recorded
    result still belongs to the current protocol, manifest and training code
    without repeatedly parsing the full trace corpus. Training performs the
    full trace verification again before optimization.
    """
    root_path = Path(root).resolve()
    protocol_file = Path(protocol_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    try:
        protocol_relative = protocol_file.relative_to(root_path).as_posix()
        manifest_relative = manifest_file.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(
            "M2 protocol and manifest must be contained by the project root"
        ) from exc
    if not protocol_file.is_file():
        raise FileNotFoundError(f"missing M2 protocol: {protocol_relative}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"missing M2 manifest: {manifest_relative}")

    declared_manifest = str(
        (protocol.get("candidate") or {}).get("dataset_manifest") or ""
    )
    if declared_manifest != manifest_relative:
        raise ValueError("M2 preflight manifest does not match the protocol")

    raw_paths = list(
        (protocol.get("integrity") or {}).get("code_identity_files") or []
    )
    raw_paths.append(
        "src/match_engine/world_model/mirrored_policy_evaluation.py"
    )
    code = code_identity_manifest(
        root_path,
        raw_paths,
        mode=str(
            (protocol.get("integrity") or {}).get("code_identity_mode")
            or "explicit_files_v1"
        ),
    )
    if not code:
        raise ValueError("M2 preflight identity requires code hashes")
    return {
        "protocol_path": protocol_relative,
        "protocol_sha256": file_sha256(protocol_file),
        "manifest_path": manifest_relative,
        "manifest_sha256": file_sha256(manifest_file),
        "code_sha256": code,
    }


def study_execution_identity(
    root: str | Path,
    protocol_path: str | Path,
    protocol: dict[str, Any],
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Hash every frozen M2 dependency using project-relative identities."""
    root_path = Path(root).resolve()
    protocol_file = Path(protocol_path).resolve()
    checkpoint_file = Path(checkpoint).resolve()
    try:
        protocol_relative = protocol_file.relative_to(root_path).as_posix()
        checkpoint_relative = checkpoint_file.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(
            "M2 protocol and checkpoint must be contained by the project root"
        ) from exc
    if not protocol_file.is_file():
        raise FileNotFoundError(f"missing M2 protocol: {protocol_relative}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"missing candidate checkpoint: {checkpoint_relative}"
        )

    integrity = protocol.get("integrity") or {}
    code = code_identity_manifest(
        root_path,
        integrity.get("code_identity_files") or [],
        mode=str(integrity.get("code_identity_mode") or "explicit_files_v1"),
    )

    inputs: dict[str, str] = {}
    for raw_pattern in integrity.get("input_identity_globs") or []:
        pattern = str(raw_pattern)
        matched = sorted(path for path in root_path.glob(pattern) if path.is_file())
        if not matched:
            raise ValueError(f"input identity glob matched no files: {pattern}")
        for path in matched:
            relative = path.resolve().relative_to(root_path).as_posix()
            inputs[relative] = file_sha256(path)
    if not code or not inputs:
        raise ValueError("M2 execution identity requires code and input hashes")

    return {
        "protocol_path": protocol_relative,
        "protocol_sha256": file_sha256(protocol_file),
        "checkpoint_path": checkpoint_relative,
        "checkpoint_sha256": file_sha256(checkpoint_file),
        "code_sha256": code,
        "input_sha256": inputs,
    }


def controlled_perspective(row: dict[str, Any], side: str) -> dict[str, Any]:
    """Project one match row into the predeclared controlled-team perspective."""
    if side not in {"home", "away"}:
        raise ValueError("controlled side must be home or away")
    opponent = "away" if side == "home" else "home"
    sign = 1.0 if side == "home" else -1.0
    return {
        "fixture": str(row["fixture"]),
        "sample_index": int(row["sample_index"]),
        "controlled_side": side,
        "controlled_micro_xg": float(row[f"micro_xg_{side}"]),
        "opponent_micro_xg": float(row[f"micro_xg_{opponent}"]),
        "controlled_micro_xg_margin": sign * float(
            row["micro_xg_home"] - row["micro_xg_away"]
        ),
        "controlled_goal_difference": sign * float(
            row["goals_home"] - row["goals_away"]
        ),
        "controlled_pass_completion": float(
            row[f"pass_completion_{side}"]
        ),
        "controlled_shots": float(row[f"shots_{side}"]),
        "controlled_shots_on_target": float(
            row[f"shots_on_target_{side}"]
        ),
        "wm_action_counterfactual_change_rate": float(
            row.get("wm_action_counterfactual_change_rate", 0.0)
        ),
        "wm_action_expected_counterfactual_change_rate": float(
            row.get("wm_action_expected_counterfactual_change_rate", 0.0)
        ),
        "wm_realized_policy_utility_count": float(
            row.get("wm_realized_policy_utility_count", 0.0)
        ),
        "wm_realized_policy_utility_mean": float(
            row.get("wm_realized_policy_utility_mean", 0.0)
        ),
        "wm_attributable_realized_policy_utility_count": float(
            row.get("wm_attributable_realized_policy_utility_count", 0.0)
        ),
        "wm_attributable_realized_policy_utility_mean": float(
            row.get("wm_attributable_realized_policy_utility_mean", 0.0)
        ),
    }


def paired_effect_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_home_rows: list[dict[str, Any]],
    candidate_away_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return two mirrored effects per fixture-seed, with strict identity checks."""
    def keyed(rows):
        out = {}
        for row in rows:
            key = (str(row["fixture"]), int(row["sample_index"]))
            if key in out:
                raise ValueError(f"duplicate experimental unit: {key}")
            out[key] = row
        return out

    baseline = keyed(baseline_rows)
    home = keyed(candidate_home_rows)
    away = keyed(candidate_away_rows)
    if not baseline or set(baseline) != set(home) or set(baseline) != set(away):
        raise ValueError("M0, M2_home and M2_away units must match exactly")
    output = []
    for key in sorted(baseline):
        for side, candidate in (
            ("home", home[key]),
            ("away", away[key]),
        ):
            reference = controlled_perspective(baseline[key], side)
            intervention = controlled_perspective(candidate, side)
            effects = {
                metric: intervention[metric] - reference[metric]
                for metric in (
                    "controlled_micro_xg_margin",
                    "controlled_goal_difference",
                    "controlled_pass_completion",
                    "controlled_shots",
                    "controlled_shots_on_target",
                )
            }
            output.append({
                "fixture": key[0],
                "sample_index": key[1],
                "controlled_side": side,
                "baseline": reference,
                "candidate": intervention,
                "effects": effects,
            })
    return output


def fixture_stratified_cluster_interval(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Bootstrap matched seeds within fixture while retaining both mirrored sides."""
    if not rows or draws < 100:
        raise ValueError("cluster interval needs rows and at least 100 draws")
    clusters: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        clusters[str(row["fixture"])][int(row["sample_index"])].append(row)
    if any(
        {item["controlled_side"] for item in cluster} != {"home", "away"}
        for fixture in clusters.values()
        for cluster in fixture.values()
    ):
        raise ValueError("every matched seed cluster must contain both mirrored sides")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = []
        for fixture in sorted(clusters):
            seeds = sorted(clusters[fixture])
            for _seed_slot in seeds:
                chosen = rng.choice(seeds)
                sample.extend(clusters[fixture][chosen])
        estimates.append(float(np.mean([value(row) for row in sample])))
    estimates.sort()

    def percentile(q: float) -> float:
        position = (len(estimates) - 1) * q
        low = int(position)
        high = min(low + 1, len(estimates) - 1)
        fraction = position - low
        return estimates[low] * (1.0 - fraction) + estimates[high] * fraction

    return {
        "method": "fixture_stratified_matched_seed_cluster_bootstrap",
        "draws": draws,
        "seed": seed,
        "point_estimate": float(np.mean([value(row) for row in rows])),
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
    }
