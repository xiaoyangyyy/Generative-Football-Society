#!/usr/bin/env python3
"""Validate and analyze manager world-model advisor adoption evidence.

This script is deliberately read-only unless ``--out`` is supplied. It never
runs a match, trains a model, or calls a provider.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import file_sha256  # noqa: E402
from src.product.decision_ledger import (  # noqa: E402
    validate_manager_decision_ledger,
    world_model_advisor_summary,
)
from src.product.workspace import ProductWorkspace  # noqa: E402


PROTOCOL_PATH = ROOT / "data/evaluation/manager_advisor_protocol_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def validate_protocol(protocol: Mapping[str, Any]) -> dict[str, bool]:
    source = protocol.get("source") or {}
    inclusion = protocol.get("inclusion") or {}
    analysis = protocol.get("analysis") or {}
    rules = protocol.get("decision_rules") or {}
    integrity = protocol.get("integrity") or {}
    execution = protocol.get("execution") or {}
    return {
        "scope_is_adoption_instrumentation_only": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id")
            == "gfs-manager-world-model-advisor-adoption-v1"
            and protocol.get("state") == "preregistered_collection_not_started"
            and protocol.get("claim_scope")
            == "product_adoption_instrumentation_only_no_outcome_effectiveness_or_causal_claim"
        ),
        "identity_validated_ledger_is_the_only_source": (
            source.get("artifact") == "manager_decision_ledger"
            and source.get("required_schema_version") == 1
            and source.get("identity_validation")
            == "entry_summary_and_ledger_semantic_replay"
            and source.get("raw_personal_data_collected") is False
        ),
        "explicit_intent_and_missingness_are_preserved": (
            inclusion.get("requires_world_model_advice") is True
            and inclusion.get("requires_explicit_intent_for_interaction") is True
            and inclusion.get("allowed_intents")
            == ["adopt_recommendation", "reviewed_then_selected"]
            and inclusion.get("unlinked_advice_is_retained_as_missing_interaction_evidence")
            is True
            and inclusion.get("completed_without_direct_execution_evidence_is_retained")
            is True
            and inclusion.get("fixed_window_requires_all_decisions_executed") is True
        ),
        "windows_and_gates_are_frozen": (
            analysis.get("fixed_information_windows") == [12, 24, 48]
            and analysis.get("minimum_advised_decisions") == 12
            and analysis.get("minimum_explicit_interaction_coverage") == 0.95
            and analysis.get("minimum_direct_execution_coverage") == 0.9
            and analysis.get("maximum_unlinked_advice_fraction") == 0.05
            and analysis.get("minimum_distinct_recommended_tactics") == 2
        ),
        "outcomes_and_promotion_are_forbidden": (
            analysis.get("outcome_effect_estimate") is None
            and analysis.get("causal_effect_authorized") is False
            and analysis.get("real_football_generalization_authorized") is False
            and rules.get("product_or_academic_promotion_authorized") is False
        ),
        "integrity_files_are_fixed": (
            integrity.get("interim_threshold_changes") is False
            and integrity.get("optional_stopping") is False
            and integrity.get("post_hoc_exclusions") is False
            and integrity.get("code_identity_files") == [
                "src/product/decision_advice.py",
                "src/product/decision_ledger.py",
                "src/product/season.py",
                "src/product/workspace.py",
                "src/product/web.py",
                "scripts/manager_advisor_study.py",
            ]
        ),
        "execution_is_absent": execution == {
            "matches_executed": 0,
            "training_executed": False,
            "provider_calls_made": False,
            "results_available": False,
        },
    }


def protocol_identity(
    protocol_path: Path = PROTOCOL_PATH, *, root: Path = ROOT,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if not all(validate_protocol(protocol).values()):
        raise ValueError("manager advisor protocol is invalid")
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
    return {
        "protocol_sha256": file_sha256(protocol_path),
        "code_sha256": code,
    }


def protocol_report(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol)
    return {
        "schema_version": 1,
        "verification": "gfs_manager_world_model_advisor_preregistration",
        "passed": all(checks.values()),
        "ready_for_identity_bound_collection": all(checks.values()),
        "checks": checks,
        "matches_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
        "results_available": False,
        "identity": protocol_identity(protocol_path) if all(checks.values()) else None,
    }


def _extract_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") == 1 and "ledger_identity" in payload:
        return copy.deepcopy(dict(payload))
    candidates = (
        payload.get("manager_decision_ledger"),
        (payload.get("season") or {}).get("manager_decision_ledger")
        if isinstance(payload.get("season"), Mapping) else None,
        ((payload.get("studio") or {}).get("season") or {}).get(
            "manager_decision_ledger"
        ) if isinstance(payload.get("studio"), Mapping) else None,
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return copy.deepcopy(dict(candidate))
    raise ValueError("manager decision ledger is unavailable in the supplied artifact")


def analyze_ledger(
    ledger: Mapping[str, Any], protocol: Mapping[str, Any],
    *, identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks = validate_protocol(protocol)
    if not all(checks.values()):
        raise ValueError("manager advisor protocol is invalid")
    validate_manager_decision_ledger(ledger)
    advised = [
        row for row in reversed(ledger["entries"])
        if (row.get("world_model_decision_support") or {}).get("available") is True
    ]
    windows = list(protocol["analysis"]["fixed_information_windows"])
    completed_windows = [window for window in windows if len(advised) >= window]
    next_window = next((window for window in windows if len(advised) < window), None)
    base = {
        "schema_version": 1,
        "verification": "gfs_manager_world_model_advisor_adoption_evidence",
        "protocol_id": protocol["protocol_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger_identity": ledger["ledger_identity"],
        "protocol_identity": copy.deepcopy(dict(identity or {})),
        "advised_decisions_available": len(advised),
        "next_information_window": next_window,
        "matches_executed_by_analyzer": 0,
        "training_executed": False,
        "provider_calls_made": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": protocol["claim_scope"],
    }
    if not completed_windows:
        return {
            **base,
            "state": "insufficient_evidence",
            "information_window": None,
            "gates": None,
            "evidence": ledger["summary"]["world_model_advisor"],
            "passed": False,
        }
    window = max(completed_windows)
    evidence_rows = advised[:window]
    evidence = world_model_advisor_summary(evidence_rows)
    if any(
        row.get("lifecycle_state") == "frozen_awaiting_execution"
        for row in evidence_rows
    ):
        return {
            **base,
            "state": "window_not_closed",
            "information_window": window,
            "gates": None,
            "evidence": evidence,
            "passed": False,
        }
    analysis = protocol["analysis"]
    distinct = len(evidence["recommendation_counts"])
    unlinked_fraction = evidence["unlinked_advice"] / max(
        1, evidence["advised_decisions"],
    )
    gates = {
        "minimum_advised_decisions": (
            evidence["advised_decisions"] >= analysis["minimum_advised_decisions"]
        ),
        "minimum_explicit_interaction_coverage": (
            evidence["explicit_interaction_coverage"]
            >= analysis["minimum_explicit_interaction_coverage"]
        ),
        "minimum_direct_execution_coverage": (
            evidence["direct_execution_coverage"]
            >= analysis["minimum_direct_execution_coverage"]
        ),
        "maximum_unlinked_advice_fraction": (
            unlinked_fraction <= analysis["maximum_unlinked_advice_fraction"]
        ),
        "minimum_distinct_recommended_tactics": (
            distinct >= analysis["minimum_distinct_recommended_tactics"]
        ),
    }
    passed = all(gates.values())
    return {
        **base,
        "state": "instrumentation_confirmed" if passed else "instrumentation_failed",
        "information_window": window,
        "gates": gates,
        "evidence": evidence,
        "passed": passed,
    }


def _ledger_from_workspace(base_dir: Path) -> dict[str, Any]:
    status = ProductWorkspace.load(base_dir).status()
    season = status.get("season")
    if not isinstance(season, Mapping):
        raise ValueError("studio has no current season ledger")
    return _extract_ledger(season)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--base-dir", type=Path)
    source.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.analyze:
        if args.ledger is not None:
            ledger = _extract_ledger(_read_json(args.ledger))
        elif args.base_dir is not None:
            ledger = _ledger_from_workspace(args.base_dir)
        else:
            parser.error("--analyze requires --base-dir or --ledger")
        protocol = _read_json(PROTOCOL_PATH)
        payload = analyze_ledger(
            ledger, protocol, identity=protocol_identity(PROTOCOL_PATH),
        )
    else:
        payload = protocol_report(PROTOCOL_PATH)
    if args.out is not None:
        _atomic_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("passed") or payload.get("state") in {
        "insufficient_evidence", "window_not_closed",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
