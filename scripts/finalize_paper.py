#!/usr/bin/env python3
"""Build and verify an evidence-locked completed-paper result ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "data/evaluation/paper_finalization_protocol_v1.json"
BRANCH_BY_DECISION = {
    "promotion_candidate_pending_release_review": "mechanism_confirmation",
    "no_meaningful_difference_keep_research_only": "adoption_path_diagnosis",
    "inconclusive_keep_research_only": "variance_diagnosis",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confined(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    try:
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return path


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _atomic_write(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def validate_protocol(protocol: dict[str, Any], root: Path = ROOT) -> dict[str, bool]:
    inputs = protocol.get("inputs") or {}
    outputs = protocol.get("outputs") or {}
    authority = protocol.get("authorization") or {}
    contract = protocol.get("manuscript_contract") or {}
    integrity = protocol.get("integrity") or {}
    execution = protocol.get("execution") or {}
    return {
        "schema_state_and_objective_are_frozen": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id")
            == "gfs-evidence-locked-paper-finalization-v1"
            and protocol.get("state") == "registered_waiting_for_complete_evidence"
            and isinstance(protocol.get("objective"), str)
        ),
        "inputs_are_exact_and_confined": (
            inputs
            == {
                "confirmatory_decision": "data/evaluation/formal_confirmatory_v2/decision.json",
                "academic_replication_decision": "data/evaluation/academic_replication_v1/decision.json",
                "independent_reproduction_verification": "data/evaluation/independent_reproduction_verification_v1.json",
                "preexecution_paper_package": "data/evaluation/paper_package_verification_v1.json",
            }
            and all(_confined(root, value) is not None for value in inputs.values())
        ),
        "outputs_are_exact_and_confined": (
            set(outputs) == {"result_ledger", "final_manuscript", "verification"}
            and all(_confined(root, value) is not None for value in outputs.values())
        ),
        "ledger_requires_explicit_zero_compute_authority": (
            authority.get("ledger_build_requires_explicit_token") is True
            and authority.get("token") == "I_AUTHORIZE_GFS_PAPER_RESULT_LEDGER_V1"
            and authority.get("simulation_authorized") is False
            and authority.get("training_authorized") is False
            and authority.get("provider_calls_authorized") is False
        ),
        "manuscript_markers_and_sections_are_complete": (
            all(isinstance(contract.get(key), str) and contract.get(key)
                for key in (
                    "stage_marker", "confirmatory_marker", "replication_marker",
                    "independent_marker", "ledger_begin_marker", "ledger_end_marker",
                    "ledger_sha_marker_prefix",
                ))
            and len(contract.get("required_sections") or []) == 6
            and len(contract.get("forbidden_pending_phrases") or []) == 4
        ),
        "integrity_is_fail_closed": (
            integrity
            == {
                "all_source_hashes_must_match": True,
                "embedded_ledger_must_equal_generated_ledger": True,
                "negative_failed_or_inconclusive_results_must_be_preserved": True,
                "manual_numeric_overrides_allowed": False,
            }
        ),
        "execution_remains_zero": (
            execution
            == {
                "ledger_built": False,
                "final_manuscript_available": False,
                "verification_available": False,
                "matches_executed": 0,
                "training_executed": False,
                "provider_calls_made": False,
            }
        ),
    }


def _evidence(root: Path, protocol: dict[str, Any]) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    values: dict[str, dict[str, Any]] = {}
    for name, relative in protocol["inputs"].items():
        path = _confined(root, relative)
        if path is not None:
            paths[name] = path
            if path.is_file():
                try:
                    values[name] = _read_json(path)
                except (OSError, ValueError):
                    pass
    return paths, values


def evidence_checks(root: Path, protocol: dict[str, Any]) -> dict[str, bool]:
    paths, values = _evidence(root, protocol)
    confirmatory = values.get("confirmatory_decision") or {}
    replication = values.get("academic_replication_decision") or {}
    independent = values.get("independent_reproduction_verification") or {}
    paper = values.get("preexecution_paper_package") or {}
    decision = str(confirmatory.get("decision") or "")
    return {
        "confirmatory_decision_is_complete": (
            confirmatory.get("schema_version") == 2
            and confirmatory.get("protocol_id")
            == "gfs-m0-vs-sealed-m1-confirmatory-v2"
            and confirmatory.get("pairs_total") == 30
            and decision in BRANCH_BY_DECISION
            and isinstance(confirmatory.get("execution_identity"), dict)
        ),
        "academic_replication_is_complete_and_branch_locked": (
            replication.get("schema_version") == 1
            and replication.get("protocol_id")
            == "gfs-result-contingent-mechanism-replication-v1"
            and replication.get("runs_executed") == 72
            and replication.get("pairs_per_arm") == 24
            and replication.get("confirmatory_decision") == decision
            and replication.get("branch") == BRANCH_BY_DECISION.get(decision)
            and replication.get("training_executed") is False
            and replication.get("provider_calls_made") is False
        ),
        "full_study_independent_review_is_complete": (
            independent.get("passed") is True
            and independent.get("independent_review_available") is True
            and independent.get("scope") == "full_study_132_runs"
            and independent.get("stages_verified")
            == ["confirmatory", "academic_replication"]
            and set(independent.get("computed_comparison") or {})
            == {"confirmatory", "academic_replication"}
            and independent.get("reviewer_conclusion")
            in {"reproduced", "not_reproduced", "inconclusive"}
        ),
        "preexecution_paper_package_is_current": (
            paper.get("passed") is True
            and paper.get("status") == "passed_preexecution_package"
            and paths.get("preexecution_paper_package") is not None
        ),
    }


def build_ledger(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    checks = evidence_checks(root, protocol)
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"paper evidence is incomplete: {failed}")
    paths, values = _evidence(root, protocol)
    confirmatory = values["confirmatory_decision"]
    replication = values["academic_replication_decision"]
    independent = values["independent_reproduction_verification"]
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "source_sha256": {
            protocol["inputs"][name]: _sha256(paths[name])
            for name in sorted(protocol["inputs"])
        },
        "confirmatory": {
            key: confirmatory[key]
            for key in (
                "decision", "promotion_supported", "pairs_total", "primary",
                "minimum_meaningful_delta_loss", "behavior", "promotion_gates",
                "secondary_metrics", "execution_identity",
            )
        },
        "academic_replication": {
            key: replication[key]
            for key in (
                "status", "passed", "branch", "confirmatory_decision",
                "conclusion", "runs_executed", "pairs_per_arm", "comparisons",
                "mechanism", "external_validity", "material_deviations", "gates",
                "execution_identity",
            )
        },
        "independent_reproduction": {
            key: independent[key]
            for key in (
                "reviewer_conclusion", "reproduction_matches", "material_deviation",
                "scope", "stages_verified", "computed_comparison",
            )
        },
    }


def protocol_report(root: Path = ROOT, protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol, root)
    evidence = evidence_checks(root, protocol)
    ledger = _confined(root, protocol["outputs"]["result_ledger"])
    manuscript = _confined(root, protocol["outputs"]["final_manuscript"])
    return {
        "schema_version": 1,
        "verification": "gfs_paper_finalization_preregistration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "ready_to_build_result_ledger"
            if all(checks.values()) and all(evidence.values())
            else "registered_waiting_for_complete_evidence"
            if all(checks.values())
            else "failed"
        ),
        "passed": all(checks.values()),
        "ready_to_finalize": all(evidence.values()),
        "result_ledger_available": bool(ledger and ledger.is_file()),
        "final_manuscript_available": bool(manuscript and manuscript.is_file()),
        "checks": checks,
        "evidence_checks": evidence,
        "artifact_sha256": {
            "data/evaluation/paper_finalization_protocol_v1.json": _sha256(protocol_path),
            "docs/PAPER_FINALIZATION.md": _sha256(root / "docs/PAPER_FINALIZATION.md"),
            "scripts/finalize_paper.py": _sha256(Path(__file__)),
        },
        "matches_executed": 0,
        "training_executed": False,
        "provider_calls_made": False,
    }


def verify_final(root: Path = ROOT, protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    ledger_path = _confined(root, protocol["outputs"]["result_ledger"])
    manuscript_path = _confined(root, protocol["outputs"]["final_manuscript"])
    if ledger_path is None or not ledger_path.is_file():
        raise FileNotFoundError("result ledger is unavailable")
    if manuscript_path is None or not manuscript_path.is_file():
        raise FileNotFoundError("final manuscript is unavailable")
    ledger = _read_json(ledger_path)
    expected = build_ledger(root, protocol)
    manuscript = manuscript_path.read_text(encoding="utf-8")
    contract = protocol["manuscript_contract"]
    canonical = _canonical(ledger)
    embedded = (
        f'{contract["ledger_begin_marker"]}\n```json\n{canonical}\n```\n'
        f'{contract["ledger_end_marker"]}'
    )
    digest_marker = contract["ledger_sha_marker_prefix"] + _sha256(ledger_path)
    checks = {
        "protocol_is_valid": all(validate_protocol(protocol, root).values()),
        "ledger_matches_current_sources": ledger == expected,
        "all_completion_markers_are_present": all(
            contract[key] in manuscript
            for key in (
                "stage_marker", "confirmatory_marker", "replication_marker",
                "independent_marker",
            )
        ),
        "required_sections_are_present": all(
            section in manuscript for section in contract["required_sections"]
        ),
        "canonical_ledger_is_embedded_exactly_once": manuscript.count(embedded) == 1,
        "ledger_digest_marker_matches": manuscript.count(digest_marker) == 1,
        "preexecution_claims_are_absent": all(
            phrase not in manuscript
            for phrase in contract["forbidden_pending_phrases"]
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_completed_paper_evidence_lock",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified_completed_manuscript" if passed else "failed",
        "passed": passed,
        "completed_manuscript_available": True,
        "checks": checks,
        "artifact_sha256": {
            ledger_path.resolve().relative_to(root.resolve()).as_posix(): _sha256(ledger_path),
            manuscript_path.resolve().relative_to(root.resolve()).as_posix(): _sha256(manuscript_path),
            **expected["source_sha256"],
        },
        "matches_executed_by_verifier": 0,
        "training_executed_by_verifier": False,
        "provider_calls_made_by_verifier": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--build-ledger", action="store_true")
    actions.add_argument("--verify-final", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    protocol = _read_json(PROTOCOL_PATH)
    if args.build_ledger:
        if args.authorization != protocol["authorization"]["token"]:
            parser.error("exact paper-ledger authorization token is required")
        result = build_ledger(ROOT, protocol)
        target = args.out or ROOT / protocol["outputs"]["result_ledger"]
        _atomic_write(target, result, overwrite=args.overwrite)
    elif args.verify_final:
        result = verify_final()
        if args.out is not None:
            _atomic_write(args.out, result, overwrite=args.overwrite)
    else:
        result = protocol_report()
        if args.out is not None:
            _atomic_write(args.out, result, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
