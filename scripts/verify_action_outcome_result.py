#!/usr/bin/env python3
"""Replay and verify the frozen full-match action-policy outcome result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_formal_experiment import (  # noqa: E402
    ROOT,
    analyze,
    execution_identity,
    file_sha256,
    load_protocol,
    validate_protocol,
)

PROTOCOL_PATH = ROOT / "data/evaluation/action_outcome_protocol_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required result artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result artifact must contain an object: {path}")
    return payload


def _unit_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys = {
        (str(row.get("fixture") or ""), int(row.get("sample_index", -1)))
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("formal outcome rows contain duplicate experimental units")
    return keys


def _without_timestamp(payload: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(payload)
    comparable.pop("analyzed_at", None)
    return comparable


def verify(
    *,
    root: Path = ROOT,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    budget = validate_protocol(protocol)
    progress_path = root / protocol["outputs"]["progress"]
    decision_path = root / protocol["outputs"]["decision"]
    progress = _read_json(progress_path)
    decision = _read_json(decision_path)
    identity = execution_identity(root, protocol_path, protocol)

    if progress.get("state") != "completed":
        raise ValueError("formal outcome experiment is not complete")
    if progress.get("execution_identity") != identity:
        raise ValueError(
            "formal outcome progress identity does not match frozen inputs"
        )
    if decision.get("execution_identity") != identity:
        raise ValueError(
            "formal outcome decision identity does not match frozen inputs"
        )

    arms = progress.get("arms") or {}
    if set(arms) != {"M0", "M1"}:
        raise ValueError("formal outcome progress must contain exactly M0 and M1")
    baseline = (arms.get("M0") or {}).get("rows") or []
    candidate = (arms.get("M1") or {}).get("rows") or []
    if len(baseline) != budget["pairs"] or len(candidate) != budget["pairs"]:
        raise ValueError(
            "formal outcome progress does not contain the fixed pair budget"
        )

    expected_units = {
        (f"{home}_vs_{away}", sample)
        for home, away in protocol["design"]["fixtures"]
        for sample in protocol["design"]["sample_indices"]
    }
    if (
        _unit_keys(baseline) != expected_units
        or _unit_keys(candidate) != expected_units
    ):
        raise ValueError("formal outcome arms do not match the preregistered units")

    prerequisite = protocol["prerequisite"]
    mechanism_path = root / prerequisite["mechanism_decision"]
    mechanism = _read_json(mechanism_path)
    if mechanism.get("status") != prerequisite["required_status"]:
        raise ValueError("action-adoption prerequisite status no longer matches")
    if file_sha256(mechanism_path) != prerequisite["mechanism_decision_sha256"]:
        raise ValueError("action-adoption prerequisite identity no longer matches")

    replay = analyze(protocol, progress, expected_identity=identity)
    if _without_timestamp(replay) != _without_timestamp(decision):
        raise ValueError("stored outcome decision does not match deterministic replay")
    if decision.get("decision") != "inconclusive_keep_research_only":
        raise ValueError(
            "formal outcome decision is not the sealed research-only result"
        )
    if decision.get("promotion_supported") is not False:
        raise ValueError("formal outcome result must not authorize promotion")
    if decision.get("behavior") != {
        "changed_pairs": 30,
        "changed_pair_fraction": 1.0,
    }:
        raise ValueError(
            "formal outcome behavior result does not match the sealed result"
        )
    if decision.get("promotion_gates") != {
        "candidate_passes_all_external_validity_gates": False,
        "upper_interval_below_negative_minimum_effect": False,
        "minimum_behavior_change_met": True,
        "execution_identity_verified": True,
    }:
        raise ValueError(
            "formal outcome promotion gates do not match the sealed result"
        )

    primary = decision.get("primary") or {}
    return {
        "verification": "gfs_action_policy_full_match_outcome_v1",
        "passed": True,
        "decision": decision["decision"],
        "promotion_supported": False,
        "runs": budget["runs"],
        "pairs": budget["pairs"],
        "changed_pairs": decision["behavior"]["changed_pairs"],
        "point_delta": primary.get("point_delta"),
        "ci95": [primary.get("ci95_low"), primary.get("ci95_high")],
        "protocol_sha256": identity["protocol_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "progress_sha256": file_sha256(progress_path),
        "decision_sha256": file_sha256(decision_path),
        "replay_matches": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    print(json.dumps(verify(protocol_path=args.protocol), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
