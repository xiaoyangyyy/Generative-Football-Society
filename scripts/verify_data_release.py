#!/usr/bin/env python3
"""Verify the source-specific licensed external-data release boundary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_data_release_archive import (  # noqa: E402
    ATTRIBUTION,
    MANIFEST,
    REGISTRY,
    expected_manifest,
)
from src.infrastructure import file_sha256  # noqa: E402


EXPECTED_SOURCES = {
    "statsbomb_open_data",
    "metrica_sample_data",
    "skillcorner_open_data",
    "sportec_idsse",
    "transfermarkt_dataset_mirror",
}


def verify_data_release() -> dict:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = expected_manifest()
    sources = {row["source_id"]: row for row in registry.get("sources") or []}
    eligible = {
        identifier
        for identifier, row in sources.items()
        if row.get("archive_eligible") is True
    }
    attribution = ATTRIBUTION.read_text(encoding="utf-8")
    included_paths = [row["path"] for row in stored.get("entries") or []]
    checks = {
        "registry_has_exact_source_coverage": set(sources) == EXPECTED_SOURCES,
        "review_is_complete_and_default_deny": (
            registry.get("policy", {}).get("default") == "exclude_from_distribution"
            and registry.get("policy", {}).get("source_specific_review_complete") is True
            and registry.get("policy", {}).get("blanket_redistribution_approved") is False
        ),
        "only_idsse_is_archive_eligible": eligible == {"sportec_idsse"},
        "idsse_has_authoritative_cc_by_evidence": (
            sources["sportec_idsse"].get("license_status") == "CC-BY-4.0"
            and sources["sportec_idsse"].get("verification_status")
            == "authoritative_publication_and_dataset_verified"
            and sources["sportec_idsse"].get("official_dataset_doi")
            == "https://doi.org/10.6084/m9.figshare.28196177"
        ),
        "every_unapproved_source_is_excluded": all(
            row.get("archive_decision") == "exclude_external_data"
            for identifier, row in sources.items()
            if identifier != "sportec_idsse"
        ),
        "manifest_is_current_and_content_addressed": stored == expected,
        "archive_contains_only_idsse_paths_and_attribution": all(
            path == "docs/IDSSE_ATTRIBUTION.md"
            or "/sportec_" in path
            or path == "data/external/sportec/derived/manifest.json"
            for path in included_paths
        ),
        "attribution_and_change_notice_are_complete": all(
            marker in attribution
            for marker in (
                "Deutsche Fußball Liga (DFL)",
                "Creative Commons Attribution 4.0 International",
                "10.1038/s41597-025-04505-y",
                "10.6084/m9.figshare.28196177",
                "Changes made by GFS",
                "No endorsement",
            )
        ),
        "zero_execution_boundary": (
            stored.get("matches_executed") == 0
            and stored.get("training_executed") is False
            and stored.get("formal_experiment_executed") is False
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_source_specific_external_data_release",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "approved_source_count": len(eligible),
        "excluded_source_count": len(sources) - len(eligible),
        "archive_file_count": stored.get("file_count"),
        "archive_bytes": stored.get("deterministic_archive", {}).get("bytes"),
        "archive_sha256": stored.get("deterministic_archive", {}).get("sha256"),
        "checks": checks,
        "artifact_sha256": {
            "data/evaluation/data_license_registry_v1.json": file_sha256(REGISTRY),
            "data/evaluation/data_release_manifest_v1.json": file_sha256(MANIFEST),
            "docs/IDSSE_ATTRIBUTION.md": file_sha256(ATTRIBUTION),
            "scripts/build_data_release_archive.py": file_sha256(
                ROOT / "scripts/build_data_release_archive.py"
            ),
            "scripts/verify_data_release.py": file_sha256(
                ROOT / "scripts/verify_data_release.py"
            ),
        },
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "limitations": [
            "Only the DFL-authorized IDSSE/Sportec derivative subset is approved.",
            "StatsBomb, Metrica, SkillCorner, and Transfermarkt-derived files remain excluded.",
            "The supplement boundary does not assert that excluded data is absent from repository history.",
            "This engineering review records source evidence and is not legal advice.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_data_release()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
