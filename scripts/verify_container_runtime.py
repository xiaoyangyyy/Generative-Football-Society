#!/usr/bin/env python3
"""Build, harden, smoke-test, and clean up the pinned GFS container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_USER = "10001:10001"
DIRECT_IMPORTS = "numpy,pandas,openai,dotenv,matplotlib,torch,statsbombpy,kloppy"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _json_output(result: subprocess.CompletedProcess[str]) -> Any:
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except (TypeError, ValueError):
        return None


def _diagnostic(result: subprocess.CompletedProcess[str] | None) -> str | None:
    if result is None:
        return None
    value = (result.stderr or result.stdout or "").strip()
    return value[-4000:] or None


def _source_revision() -> str | None:
    github_sha = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch(r"[0-9a-f]{40}", github_sha):
        return github_sha
    try:
        result = _run(["git", "rev-parse", "HEAD"], timeout=30)
    except subprocess.TimeoutExpired:
        return None
    revision = result.stdout.strip()
    return (
        revision
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", revision)
        else None
    )


def _server_identity(docker: str) -> tuple[bool, dict[str, Any] | None]:
    try:
        result = _run(
            [docker, "version", "--format", "{{json .Server}}"],
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, None
    payload = _json_output(result)
    return result.returncode == 0 and isinstance(payload, dict), payload


def verify_container_runtime(*, execute: bool = False) -> dict:
    docker = shutil.which("docker")
    checks = {
        "docker_cli_available": bool(docker),
        "execution_explicitly_authorized": execute,
        "source_revision_is_commit": False,
        "docker_daemon_reachable": False,
        "image_build_passed": False,
        "image_id_is_content_addressed": False,
        "image_revision_label_matches_source": False,
        "image_declares_non_root_user": False,
        "runtime_uid_gid_are_non_root": False,
        "direct_dependency_import_smoke_passed": False,
        "runtime_security_options_verified": False,
        "health_response_contract_passed": False,
        "docker_health_status_passed": False,
        "ephemeral_container_removed": False,
        "verification_image_removed": False,
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "verification": "gfs_pinned_container_build_and_runtime_smoke",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_executed",
        "passed": False,
        "image_built": False,
        "deployment_started": False,
        "docker_available": bool(docker),
        "source_revision": _source_revision(),
        "execution_context": {
            "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
        },
        "image_reference": None,
        "image_id": None,
        "docker_server": None,
        "checks": checks,
        "artifact_sha256": {
            relative: _sha256(relative)
            for relative in (
                ".dockerignore",
                "Dockerfile",
                "gfs.py",
                "pyproject.toml",
                "requirements.txt",
                "requirements-linux-py312.lock",
                "data/evaluation/container_image_contract_v1.json",
                "scripts/verify_container_runtime.py",
            )
        },
        "provider_calls_made": False,
        "registry_access_attempted": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "limitations": [],
    }
    if not docker:
        report["status"] = "docker_unavailable"
        report["limitations"] = [
            "Docker CLI is unavailable; no image build or container launch occurred."
        ]
        return report
    if not execute:
        report["status"] = "awaiting_explicit_execute"
        report["limitations"] = [
            "Use --execute on a Docker host to build and launch the pinned image."
        ]
        return report

    checks["source_revision_is_commit"] = bool(
        isinstance(report["source_revision"], str)
        and re.fullmatch(r"[0-9a-f]{40}", report["source_revision"])
    )
    checks["docker_daemon_reachable"], server = _server_identity(docker)
    report["docker_server"] = server
    if not checks["docker_daemon_reachable"]:
        report["status"] = "docker_daemon_unreachable"
        report["limitations"] = [
            "Docker CLI exists, but its server is unavailable; no build occurred."
        ]
        return report

    suffix = uuid.uuid4().hex[:12]
    image = f"gfs-studio:verification-{suffix}"
    container = f"gfs-studio-verification-{suffix}"
    report["image_reference"] = image
    image_created = False
    container_created = False
    failure: str | None = None
    diagnostic: str | None = None
    try:
        report["registry_access_attempted"] = True
        built = _run(
            [
                docker,
                "build",
                "--pull",
                "--label",
                (
                    "org.opencontainers.image.revision="
                    f"{report['source_revision'] or 'unknown'}"
                ),
                "--tag",
                image,
                ".",
            ],
            timeout=1800,
        )
        checks["image_build_passed"] = built.returncode == 0
        report["image_built"] = checks["image_build_passed"]
        image_created = checks["image_build_passed"]
        if not image_created:
            failure = "docker_build_failed"
            diagnostic = _diagnostic(built)
        else:
            inspected = _run([docker, "image", "inspect", image], timeout=60)
            images = _json_output(inspected)
            image_info = (
                images[0]
                if isinstance(images, list)
                and len(images) == 1
                and isinstance(images[0], dict)
                else {}
            )
            report["image_id"] = image_info.get("Id")
            declared_user = (image_info.get("Config") or {}).get("User")
            labels = (image_info.get("Config") or {}).get("Labels") or {}
            checks["image_id_is_content_addressed"] = bool(
                isinstance(report["image_id"], str)
                and SHA256.fullmatch(report["image_id"])
            )
            checks["image_declares_non_root_user"] = declared_user == EXPECTED_USER
            checks["image_revision_label_matches_source"] = (
                labels.get("org.opencontainers.image.revision")
                == report["source_revision"]
            )

            runtime_probe = _run(
                [
                    docker,
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    "256",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=64m",
                    "--entrypoint",
                    "python",
                    image,
                    "-c",
                    (
                        "import json,os;"
                        f"import {DIRECT_IMPORTS};"
                        "print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),"
                        "'imports':'ok'}))"
                    ),
                ],
                timeout=300,
            )
            runtime_payload = _json_output(runtime_probe)
            checks["runtime_uid_gid_are_non_root"] = (
                isinstance(runtime_payload, dict)
                and runtime_payload.get("uid") == 10001
                and runtime_payload.get("gid") == 10001
            )
            checks["direct_dependency_import_smoke_passed"] = (
                isinstance(runtime_payload, dict)
                and runtime_payload.get("imports") == "ok"
            )

            started = _run(
                [
                    docker,
                    "run",
                    "--detach",
                    "--name",
                    container,
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    "512",
                    "--health-interval",
                    "2s",
                    "--health-timeout",
                    "3s",
                    "--health-start-period",
                    "1s",
                    "--health-retries",
                    "5",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=256m",
                    "--tmpfs",
                    "/app/outputs:rw,nosuid,size=64m",
                    "--tmpfs",
                    "/app/backups:rw,nosuid,size=64m",
                    "--tmpfs",
                    "/app/data/persistence:rw,nosuid,size=64m",
                    image,
                ],
                timeout=60,
            )
            container_created = started.returncode == 0
            report["deployment_started"] = container_created
            if not container_created:
                failure = "container_start_failed"
                diagnostic = _diagnostic(started)
            else:
                container_inspect = _run(
                    [docker, "container", "inspect", container],
                    timeout=60,
                )
                containers = _json_output(container_inspect)
                container_info = (
                    containers[0]
                    if isinstance(containers, list)
                    and len(containers) == 1
                    and isinstance(containers[0], dict)
                    else {}
                )
                host = container_info.get("HostConfig") or {}
                checks["runtime_security_options_verified"] = (
                    host.get("ReadonlyRootfs") is True
                    and "ALL" in (host.get("CapDrop") or [])
                    and any(
                        str(value).startswith("no-new-privileges")
                        for value in (host.get("SecurityOpt") or [])
                    )
                    and host.get("NetworkMode") == "none"
                    and host.get("PidsLimit") == 512
                )

                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    health = _run(
                        [
                            docker,
                            "exec",
                            container,
                            "python",
                            "-c",
                            (
                                "import json,urllib.request;"
                                "payload=json.load(urllib.request.urlopen("
                                "'http://127.0.0.1:8765/healthz',timeout=2));"
                                "assert payload['schema_version']==1;"
                                "assert payload['status']=='ok';"
                                "assert payload['service']=='gfs-product-web';"
                                "assert payload['external_calls_made'] is False;"
                                "print('container-health-contract-ok')"
                            ),
                        ],
                        timeout=10,
                    )
                    checks["health_response_contract_passed"] = (
                        health.returncode == 0
                        and "container-health-contract-ok" in health.stdout
                    )
                    state = _run(
                        [
                            docker,
                            "container",
                            "inspect",
                            container,
                            "--format",
                            "{{json .State.Health}}",
                        ],
                        timeout=10,
                    )
                    health_state = _json_output(state)
                    checks["docker_health_status_passed"] = (
                        isinstance(health_state, dict)
                        and health_state.get("Status") == "healthy"
                    )
                    if (
                        checks["health_response_contract_passed"]
                        and checks["docker_health_status_passed"]
                    ):
                        break
                    time.sleep(1)
    except subprocess.TimeoutExpired as exc:
        failure = "docker_command_timeout"
        diagnostic = str(exc)
    finally:
        if container_created:
            try:
                removed = _run([docker, "rm", "--force", container], timeout=60)
            except subprocess.TimeoutExpired:
                removed = None
            checks["ephemeral_container_removed"] = (
                removed is not None and removed.returncode == 0
            )
        else:
            checks["ephemeral_container_removed"] = True
        if image_created:
            try:
                removed_image = _run(
                    [docker, "image", "rm", "--force", image],
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                removed_image = None
            checks["verification_image_removed"] = (
                removed_image is not None and removed_image.returncode == 0
            )
        else:
            checks["verification_image_removed"] = True

    report["passed"] = all(checks.values())
    if report["passed"]:
        report["status"] = "passed"
    else:
        report["status"] = failure or "failed"
        report["diagnostic"] = diagnostic
        report["limitations"] = [
            "The container build/runtime gate remains closed until every check passes."
        ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_container_runtime(execute=args.execute)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not args.execute:
        return 0
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
