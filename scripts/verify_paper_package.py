#!/usr/bin/env python3
"""Verify the completed action-policy paper package without simulation."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_research_evidence import build_report as build_research_report  # noqa: E402
from scripts.run_formal_experiment import (  # noqa: E402
    load_protocol,
    validate_protocol,
)
from scripts.verify_action_outcome_result import (  # noqa: E402
    PROTOCOL_PATH,
    verify as verify_action_outcome,
)
from src.infrastructure import file_sha256  # noqa: E402


MANIFEST = ROOT / "data/evaluation/reproduction_manifest_v1.json"
CLAIMS = ROOT / "data/evaluation/paper_claims_v1.json"
ENVIRONMENT = ROOT / "data/evaluation/reproduction_environment_v1.json"
MANUSCRIPT = ROOT / "docs/PAPER_DRAFT.md"
REFERENCES = ROOT / "docs/PAPER_REFERENCES.bib"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _confined_file(relative: str) -> bool:
    candidate = (ROOT / relative).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return False
    return candidate.is_file()


def _claim_checks(registry: dict, manuscript: str) -> dict[str, bool]:
    claims = registry.get("claims") or []
    identifiers = [str(claim.get("claim_id") or "") for claim in claims]
    allowed_status = {
        "verified_existing",
        "verified_current_result",
        "preregistered_not_executed",
        "not_available",
    }
    supported_evidence = all(
        (
            claim.get("status")
            not in {"verified_existing", "verified_current_result"}
            or (
                bool(claim.get("evidence"))
                and all(_confined_file(str(path)) for path in claim["evidence"])
            )
        )
        for claim in claims
    )
    pending_is_bounded = all(
        (
            claim.get("status") not in {
                "preregistered_not_executed",
                "not_available",
            }
            or not any(
                token in str(claim.get("statement") or "").casefold()
                for token in ("significantly", "outperforms", "promotion supported")
            )
        )
        for claim in claims
    )
    markers = re.findall(r"<!-- claim:(CLM-[0-9]{3}) -->", manuscript)
    prohibited = [
        str(phrase).casefold()
        for phrase in registry.get("prohibited_phrases") or []
    ]
    lowered = manuscript.casefold()
    return {
        "claim_registry_schema": registry.get("schema_version") == 1,
        "claim_ids_unique_and_well_formed": (
            len(identifiers) >= 8
            and len(identifiers) == len(set(identifiers))
            and all(re.fullmatch(r"CLM-[0-9]{3}", item) for item in identifiers)
        ),
        "claim_statuses_bounded": all(
            claim.get("status") in allowed_status for claim in claims
        ),
        "supported_claim_evidence_exists": supported_evidence,
        "pending_claims_have_no_result_language": pending_is_bounded,
        "every_claim_marked_exactly_once": (
            sorted(markers) == sorted(identifiers)
            and all(markers.count(identifier) == 1 for identifier in identifiers)
        ),
        "prohibited_claim_phrases_absent": all(
            phrase not in lowered for phrase in prohibited
        ),
    }


def verify_paper_package() -> dict:
    checks: dict[str, bool] = {}
    manifest = _read_json(MANIFEST)
    registry = _read_json(CLAIMS)
    environment = _read_json(ENVIRONMENT)
    protocol = load_protocol(PROTOCOL_PATH)
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    references = REFERENCES.read_text(encoding="utf-8")
    normalized_manuscript = " ".join(manuscript.split())

    artifacts = manifest.get("artifacts") or {}
    checks["manifest_schema_and_stage"] = (
        manifest.get("schema_version") == 1
        and manifest.get("stage") == "completed_action_outcome_results_package"
    )
    checks["manifest_artifacts_exist_and_are_confined"] = (
        len(artifacts) >= 12
        and all(_confined_file(str(path)) for path in artifacts.values())
    )
    checks.update(_claim_checks(registry, manuscript))

    required_sections = (
        "## Abstract",
        "## 1. Introduction",
        "## 2. Related work",
        "## 3. System and methods",
        "## 4. Existing evidence",
        "## 5. Confirmatory results",
        "## 8. Limitations, validity, and ethics",
        "## 9. Reproducibility and independent review",
        "## References",
    )
    checks["manuscript_has_required_sections"] = all(
        section in manuscript for section in required_sections
    )
    checks["manuscript_status_is_explicitly_completed"] = all(
        marker in manuscript
        for marker in (
            "MANUSCRIPT_STAGE: COMPLETED_RESULTS_DRAFT",
            "CONFIRMATORY_RESULT_STATUS: COMPLETE_RESEARCH_ONLY",
            "INDEPENDENT_REPRODUCTION_STATUS: NOT_PERFORMED",
            "**Status: COMPLETE, RESEARCH-ONLY.**",
        )
    )
    checks["confirmatory_result_cells_match_sealed_result"] = all(
        marker in manuscript
        for marker in (
            "| Completed M0 runs | 30/30 |",
            "| Completed M1 runs | 30/30 |",
            "| Point delta in external loss | -0.03189252164278855 |",
            "| 95% paired interval | [-26.061954645805613, 5.137429588662575] |",
            "| Changed-pair fraction | 30/30 (1.0) |",
            "| Decision | `inconclusive_keep_research_only` |",
            "| Promotion supported | no |",
        )
    )

    budget = validate_protocol(protocol)
    outputs = manifest.get("expected_confirmatory_outputs") or {}
    progress_path = ROOT / str(outputs.get("progress") or "")
    decision_path = ROOT / str(outputs.get("decision") or "")
    checks["protocol_is_frozen_and_compute_bounded"] = (
        protocol.get("state") == "preregistered_not_executed"
        and budget == {"fixtures": 6, "pairs": 30, "runs": 60}
    )
    try:
        outcome_verification = verify_action_outcome()
    except (OSError, ValueError, KeyError, TypeError):
        outcome_verification = {"passed": False}
    checks["formal_result_identity_and_replay_pass"] = (
        outcome_verification.get("passed") is True
        and outcome_verification.get("runs") == 60
        and outcome_verification.get("pairs") == 30
        and outcome_verification.get("replay_matches") is True
    )
    decision = _read_json(decision_path) if decision_path.is_file() else {}
    progress = _read_json(progress_path) if progress_path.is_file() else {}
    checks["confirmatory_outputs_are_complete_and_research_only"] = (
        outputs.get("current_status") == "complete_research_only"
        and progress.get("state") == "completed"
        and decision.get("decision") == "inconclusive_keep_research_only"
        and decision.get("promotion_supported") is False
        and decision.get("behavior", {}).get("changed_pairs") == 30
    )

    stored_research = _read_json(ROOT / "data/evaluation/research_evidence_v1.json")
    checks["existing_research_evidence_is_fresh"] = (
        stored_research == build_research_report()
    )
    checks["existing_m1_is_negative_and_not_relabelled"] = (
        _read_json(
            ROOT / "data/evaluation/formal_ablation/M1_calibrated_decision.json"
        ).get("decision") == "research_only_default_off"
        and "These are existing negative results, not the outcome"
        in normalized_manuscript
    )

    commands = manifest.get("commands") or []
    command_ids = [str(row.get("id") or "") for row in commands]
    simulation_commands = [row for row in commands if row.get("runs_simulation")]
    checks["commands_are_unique_and_effect_classified"] = (
        len(commands) == len(set(command_ids))
        and all(row.get("effect") and row.get("command") for row in commands)
    )
    checks["simulation_commands_are_exact_and_explicit"] = (
        len(simulation_commands) == 3
        and {
            row.get("command") for row in simulation_commands
        } == {
            "python scripts/run_formal_experiment.py --protocol data/evaluation/action_outcome_protocol_v1.json --execute",
            "python scripts/run_production_validation.py --execute --workspace . --authorization I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION --deployment-instance-id instance-replace",
            "python scripts/academic_replication_study.py --execute --authorization I_AUTHORIZE_GFS_ACTION_REPLICATION_V2",
        }
        and all(
            row.get("explicit_authority_required") is True
            for row in simulation_commands
        )
    )
    checks["read_only_commands_cannot_run_simulation"] = all(
        row.get("runs_simulation") is False
        for row in commands
        if row.get("effect") == "read_only"
    )

    checks["environment_snapshot_has_verified_reference_matrix"] = (
        environment.get("status") == "supported_profile_matrix_runtime_verified"
        and environment.get("direct_dependency_lock")
        == "data/evaluation/dependency_lock_contract_v1.json"
        and environment.get("target_runtime_lock")
        == "requirements-linux-py312.lock"
        and environment.get("windows_reference_runtime_lock")
        == "requirements-windows-py313.lock"
        and environment.get("direct_packages", {}).get("kloppy") == "3.19.0"
        and len(environment.get("limitations") or []) >= 4
        and manifest.get("environment", {}).get("full_transitive_hash_lock") is True
        and manifest.get("environment", {}).get("cross_platform_lock_matrix")
        is True
        and manifest.get("environment", {}).get("container_images_digest_pinned")
        is True
        and manifest.get("environment", {}).get("container_build_verified")
        is False
    )
    checks["data_distribution_limits_are_explicit"] = (
        manifest.get("data_distribution", {}).get("license_review_complete") is True
        and manifest.get("data_distribution", {}).get("known_source_registry_complete")
        is True
        and manifest.get("data_distribution", {}).get("approved_source_count") == 1
        and manifest.get("data_distribution", {}).get("excluded_source_count") == 4
        and _confined_file(
            str(manifest.get("data_distribution", {}).get("registry") or "")
        )
        and _confined_file(
            str(manifest.get("data_distribution", {}).get("archive_manifest") or "")
        )
        and "Only IDSSE/Sportec-derived artifacts"
        in " ".join(
            (ROOT / "docs/RESEARCH_DATA_CARD.md").read_text(
                encoding="utf-8"
            ).split()
        )
    )
    checks["model_card_keeps_research_layers_default_off"] = (
        "M1 is default-off, research-only"
        in " ".join(
            (ROOT / "docs/RESEARCH_MODEL_CARD.md").read_text(
                encoding="utf-8"
            ).split()
        )
    )
    checks["independent_reproduction_remains_unclaimed"] = (
        manifest.get("independent_reproduction", {}).get("status")
        == "not_performed"
        and manifest.get("independent_reproduction", {}).get("report") is None
    )

    bibliography_keys = set(
        re.findall(r"@[A-Za-z]+\{([^,]+),", references)
    )
    cited_keys = set(re.findall(r"\[@([A-Za-z0-9]+)\]", manuscript))
    checks["all_manuscript_citations_resolve"] = (
        len(cited_keys) >= 5 and cited_keys <= bibliography_keys
    )
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    project_version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    checks["software_citation_has_no_placeholder_and_matches_version"] = (
        "<your-org-or-user>" not in citation
        and f'version: "{project_version}"' in citation
        and 'license: "MIT"' in citation
    )

    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_completed_action_outcome_paper_package",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_completed_results_package" if passed else "failed",
        "passed": passed,
        "package_stage": manifest.get("stage"),
        "claim_count": len(registry.get("claims") or []),
        "artifact_count": len(artifacts),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "claim_registry_sha256": file_sha256(CLAIMS),
        "manuscript_sha256": file_sha256(MANUSCRIPT),
        "reproduction_manifest_sha256": file_sha256(MANIFEST),
        "environment_snapshot_sha256": file_sha256(ENVIRONMENT),
        "confirmatory_result_available": outcome_verification.get("passed") is True,
        "independent_reproduction_available": False,
        "external_calls_made": False,
        "matches_executed": 60 if outcome_verification.get("passed") is True else 0,
        "training_executed": False,
        "formal_experiment_executed": outcome_verification.get("passed") is True,
        "checks": checks,
        "limitations": [
            "the completed simulator-internal result is inconclusive and does not authorize M1 promotion",
            "the Linux/Windows reference matrix and image digests are verified, but an actual Docker build remains incomplete",
            "the licensed archive is limited to IDSSE/Sportec derivatives; four other providers remain excluded",
            "the licensed supplement boundary does not audit or rewrite repository history",
            "no independent reviewer has reproduced the confirmatory experiment",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_paper_package()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
