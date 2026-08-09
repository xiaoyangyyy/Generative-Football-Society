#!/usr/bin/env python3
"""Verify the deployment contract statically and with Docker when available."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_container_runtime() -> tuple[bool, dict]:
    path = ROOT / "data/evaluation/container_runtime_verification_v1.json"
    if not path.is_file():
        return False, {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, {}
    expected = report.get("artifact_sha256") or {}
    current = bool(expected) and all(
        (ROOT / relative).is_file()
        and _sha256(ROOT / relative) == digest
        for relative, digest in expected.items()
    )
    verified = (
        report.get("passed") is True
        and report.get("image_built") is True
        and report.get("deployment_started") is True
        and all((report.get("checks") or {}).values())
        and current
    )
    return verified, report


def verify_deployment_contract() -> dict:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "deploy/compose.yaml").read_text(encoding="utf-8")
    caddy = (ROOT / "deploy/Caddyfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    checks = {
        "non_root_application": "USER 10001:10001" in dockerfile,
        "container_healthcheck": "HEALTHCHECK" in dockerfile and "/healthz" in dockerfile,
        "hashed_python_dependency_install": all(value in dockerfile for value in (
            "requirements-linux-py312.lock", "--require-hashes",
            "--no-build-isolation --no-deps .",
        )),
        "immutable_container_image_digests": all(value in text for text, value in (
            (dockerfile, "python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"),
            (compose, "caddy:2.8.4-alpine@sha256:af32e97399febea808609119bb21544d0265c58a02836576e32a2d082c262c17"),
        )),
        "release_evidence_copied": all(value in dockerfile for value in (
            "COPY data ./data", "COPY reports/acceptance ./reports/acceptance",
        )),
        "generated_state_excluded": all(value in dockerignore for value in (
            "outputs", "data/persistence", "data/training/runs", ".env.*",
        )),
        "provider_credentials_not_wired": all(value not in compose for value in (
            "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "API_KEY:",
        )),
        "independent_product_secret_file_required": all(value in compose for value in (
            "GFS_WEB_ACCESS_TOKEN_FILE: /run/secrets/gfs_web_access_token",
            "secrets:", "file: ${GFS_WEB_ACCESS_TOKEN_FILE:?",
        )),
        "product_token_value_not_in_container_environment": (
            "GFS_WEB_ACCESS_TOKEN:" not in compose
        ),
        "remote_policy_explicit": all(value in compose for value in (
            "--allow-remote", "--allowed-host", "GFS_WEB_HOST",
        )),
        "application_not_host_published": (
            'expose:\n      - "8765"' in compose
            and 'gfs:\n    build:' in compose
            and '      - "8765:8765"' not in compose
        ),
        "gateway_is_only_public_edge": all(value in compose for value in (
            '"80:80"', '"443:443"', '"443:443/udp"',
        )),
        "backend_network_internal": "backend:\n    internal: true" in compose,
        "read_only_and_capability_reduced": (
            compose.count("read_only: true") >= 2
            and compose.count("no-new-privileges:true") >= 2
            and compose.count("cap_drop:") >= 2
        ),
        "persistent_product_volumes": all(value in compose for value in (
            "gfs_outputs:/app/outputs", "gfs_persistence:/app/data/persistence",
            "gfs_backups:/app/backups",
        )),
        "trusted_tls_proxy_contract": all(value in caddy for value in (
            "reverse_proxy gfs:8765", "X-Forwarded-Proto https",
            "Strict-Transport-Security", "{$GFS_WEB_HOST}",
        )),
    }
    docker = shutil.which("docker")
    compose_validation = None
    compose_output = "docker_unavailable"
    if docker:
        environment = dict(os.environ)
        environment.update({
            "GFS_WEB_HOST": "studio.example.invalid",
            "GFS_WEB_ACCESS_TOKEN_FILE": str(
                ROOT / "deploy/.verification-secret-does-not-need-to-exist",
            ),
        })
        result = subprocess.run([
            docker, "compose", "-f", str(ROOT / "deploy/compose.yaml"), "config",
        ], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
        compose_validation = result.returncode == 0
        compose_output = "passed" if compose_validation else "failed"
        checks["docker_compose_config"] = compose_validation
    container_verified, container_report = _verified_container_runtime()
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_deployment_contract",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "passed_built_runtime" if passed and container_verified
            else "passed_docker_config" if passed and docker
            else "passed_static_contract_unbuilt" if passed else "failed"
        ),
        "passed": passed,
        "docker_available": bool(docker),
        "docker_compose_validation": compose_output,
        "image_built": container_verified,
        "container_runtime_report_available": bool(container_report),
        "target_python_dependencies_hash_locked": checks[
            "hashed_python_dependency_install"
        ],
        "container_images_digest_pinned": checks[
            "immutable_container_image_digests"
        ],
        "deployment_started": container_verified,
        "external_calls_made": False,
        "checks": checks,
        "limitations": [
            *([] if container_verified else ["the image has not been built and runtime-verified in this evidence set"]),
            "the full Compose stack and Caddy TLS edge have not been started",
            "image digests are locked, but their platform manifests have not been pulled and built on this host",
            "external DNS, certificates, firewall, volume recovery, and penetration tests remain open",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_deployment_contract()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
