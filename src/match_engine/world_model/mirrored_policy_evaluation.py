"""Pure analysis helpers for team-scoped mirrored M2 policy experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np

from src.infrastructure import file_sha256


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
    code: dict[str, str] = {}
    for raw_relative in integrity.get("code_identity_files") or []:
        relative = str(raw_relative)
        path = (root_path / relative).resolve()
        try:
            canonical = path.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"invalid code identity path: {relative}") from exc
        if not path.is_file():
            raise ValueError(f"invalid code identity path: {relative}")
        code[canonical] = file_sha256(path)

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
