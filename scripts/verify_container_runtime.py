#!/usr/bin/env python3
"""Build and smoke-test the pinned GFS container when Docker is available."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def verify_container_runtime(*, execute: bool = False) -> dict:
    docker = shutil.which("docker")
    checks = {
        "docker_cli_available": bool(docker),
        "execution_explicitly_authorized": execute,
        "image_build_passed": False,
        "image_runs_as_declared_non_root_user": False,
        "direct_dependency_import_smoke_passed": False,
        "read_only_container_healthcheck_passed": False,
    }
    report = {
        "schema_version": 1,
        "verification": "gfs_pinned_container_build_and_runtime_smoke",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_executed",
        "passed": False,
        "image_built": False,
        "deployment_started": False,
        "docker_available": bool(docker),
        "image_reference": None,
        "image_id": None,
        "checks": checks,
        "artifact_sha256": {
            relative: _sha256(relative)
            for relative in (
                "Dockerfile",
                "requirements-linux-py312.lock",
                "data/evaluation/container_image_contract_v1.json",
                "scripts/verify_container_runtime.py",
            )
        },
        "provider_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
        "container_registry_access_possible": execute,
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

    suffix = uuid.uuid4().hex[:12]
    image = f"gfs-studio:verification-{suffix}"
    container = f"gfs-studio-verification-{suffix}"
    report["image_reference"] = image
    failure: str | None = None
    try:
        built = _run([docker, "build", "--tag", image, "."], timeout=1800)
        checks["image_build_passed"] = built.returncode == 0
        report["image_built"] = checks["image_build_passed"]
        if not checks["image_build_passed"]:
            failure = "docker_build_failed"
            report["diagnostic"] = (built.stderr or built.stdout)[-4000:]
            return report

        inspected = _run([
            docker, "image", "inspect", image, "--format", "{{.Id}}|{{.Config.User}}",
        ], timeout=60)
        identity = inspected.stdout.strip().split("|", 1)
        report["image_id"] = identity[0] if identity else None
        declared_user = identity[1] if len(identity) == 2 else ""
        checks["image_runs_as_declared_non_root_user"] = (
            inspected.returncode == 0 and declared_user == "10001:10001"
        )

        imported = _run([
            docker, "run", "--rm", "--entrypoint", "python", image, "-c",
            (
                "import numpy,pandas,openai,dotenv,matplotlib,torch,statsbombpy,kloppy;"
                "print('runtime-imports-ok')"
            ),
        ], timeout=300)
        checks["direct_dependency_import_smoke_passed"] = (
            imported.returncode == 0 and "runtime-imports-ok" in imported.stdout
        )

        started = _run([
            docker, "run", "--detach", "--name", container,
            "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
            "--tmpfs", "/app/outputs:rw,nosuid,size=64m",
            "--tmpfs", "/app/backups:rw,nosuid,size=64m",
            "--tmpfs", "/app/data/persistence:rw,nosuid,size=64m",
            image,
        ], timeout=60)
        report["deployment_started"] = started.returncode == 0
        if started.returncode == 0:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                health = _run([
                    docker, "exec", container, "python", "-c",
                    (
                        "import urllib.request;"
                        "urllib.request.urlopen('http://127.0.0.1:8765/healthz',"
                        "timeout=2).read()"
                    ),
                ], timeout=10)
                if health.returncode == 0:
                    checks["read_only_container_healthcheck_passed"] = True
                    break
                time.sleep(1)
        else:
            report["diagnostic"] = (started.stderr or started.stdout)[-4000:]
        report["passed"] = all(checks.values())
        report["status"] = "passed" if report["passed"] else "failed"
        return report
    except subprocess.TimeoutExpired as exc:
        failure = "docker_command_timeout"
        report["diagnostic"] = str(exc)
        return report
    finally:
        if docker:
            _run([docker, "rm", "--force", container], timeout=60)
        if failure:
            report["status"] = failure
        if not report["passed"]:
            report["limitations"] = [
                "The container build/runtime gate remains closed until every check passes."
            ]


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
