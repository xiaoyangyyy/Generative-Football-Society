#!/usr/bin/env python3
"""Verify immutable container-image pins against their recorded registry evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import file_sha256  # noqa: E402


CONTRACT = ROOT / "data/evaluation/container_image_contract_v1.json"
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def verify_container_images() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "deploy/compose.yaml").read_text(encoding="utf-8")
    images = contract.get("images") or []
    by_id = {str(row.get("id")): row for row in images}
    python = by_id.get("python_runtime") or {}
    caddy = by_id.get("caddy_gateway") or {}
    python_ref = f"python:{python.get('tag')}@{python.get('digest')}"
    caddy_ref = f"caddy:{caddy.get('tag')}@{caddy.get('digest')}"
    checks = {
        "contract_schema_and_registry": (
            contract.get("schema_version") == 1
            and contract.get("registry") == "https://registry-1.docker.io"
            and contract.get("authentication") == "anonymous_pull_token"
        ),
        "exact_image_set": set(by_id) == {"python_runtime", "caddy_gateway"},
        "all_digests_are_sha256": (
            len(images) == 2
            and all(DIGEST.fullmatch(str(row.get("digest") or "")) for row in images)
        ),
        "all_are_oci_index_observations": all(
            row.get("media_type") == "application/vnd.oci.image.index.v1+json"
            and row.get("registry_head_verified") is True
            and str(row.get("manifest_endpoint") or "").startswith(
                "https://registry-1.docker.io/v2/library/"
            )
            for row in images
        ),
        "python_digest_is_pinned_in_dockerfile": (
            f"ARG PYTHON_IMAGE={python_ref}" in dockerfile
            and "FROM ${PYTHON_IMAGE}" in dockerfile
        ),
        "caddy_digest_is_pinned_in_compose": f"image: {caddy_ref}" in compose,
        "human_readable_tags_are_retained": (
            python.get("tag") == "3.12.11-slim-bookworm"
            and caddy.get("tag") == "2.8.4-alpine"
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_immutable_container_image_contract",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "images": {
            "python_runtime": python_ref,
            "caddy_gateway": caddy_ref,
        },
        "checks": checks,
        "artifact_sha256": {
            "Dockerfile": file_sha256(ROOT / "Dockerfile"),
            "deploy/compose.yaml": file_sha256(ROOT / "deploy/compose.yaml"),
            "data/evaluation/container_image_contract_v1.json": file_sha256(CONTRACT),
        },
        "verification_external_calls_made": False,
        "registry_observation_external_calls_recorded": True,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_container_images()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
