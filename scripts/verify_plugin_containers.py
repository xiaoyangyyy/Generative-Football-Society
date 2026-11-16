#!/usr/bin/env python3
"""Build and execute the versioned Core/Studio container delivery contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_IMAGE = "gfs-core:7.0.0a1"
STUDIO_IMAGE = "gfs-studio:0.1.0a1"
ARTIFACTS = (
    "gfs-core/Dockerfile",
    "gfs-core/requirements-linux-py312.lock",
    "gfs-core/supply-chain/sbom.cdx.json",
    "gfs-studio/Dockerfile",
    "gfs-studio/requirements-linux-py312.lock",
    "gfs-studio/supply-chain/sbom.cdx.json",
    "deployment/compose.yaml",
    "deployment/manifest.json",
    "scripts/verify_plugin_containers.py",
)


def _run(command: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, encoding="utf-8",
        errors="backslashreplace", timeout=timeout, check=False,
    )


def _json(result: subprocess.CompletedProcess[str]):
    if result.returncode:
        return None
    try:
        return json.loads(result.stdout)
    except ValueError:
        return None


def _diagnostic(result: subprocess.CompletedProcess[str] | None) -> str | None:
    if result is None:
        return None
    return (result.stderr or result.stdout).strip()[-4000:] or None


def verify(*, execute: bool = False) -> dict:
    docker = shutil.which("docker")
    revision = os.environ.get("GITHUB_SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        result = _run(["git", "rev-parse", "HEAD"])
        revision = result.stdout.strip() if result.returncode == 0 else None
    checks = {
        "execution_explicitly_authorized": execute,
        "source_revision_is_commit": bool(re.fullmatch(r"[0-9a-f]{40}", revision or "")),
        "docker_available": bool(docker),
        "core_image_built": False,
        "studio_image_built": False,
        "image_ids_content_addressed": False,
        "image_revision_labels_match": False,
        "non_root_users_declared": False,
        "core_cli_executed": False,
        "studio_health_contract_passed": False,
        "docker_health_status_passed": False,
        "runtime_security_options_verified": False,
        "ephemeral_container_removed": False,
        "verification_images_removed": False,
    }
    report = {
        "schema_version": 1,
        "verification": "gfs_plugin_container_runtime_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": revision,
        "status": "awaiting_explicit_execute" if execute else "not_executed",
        "passed": False,
        "checks": checks,
        "artifact_sha256": {
            path: _sha256(ROOT / path) for path in ARTIFACTS
        },
        "external_provider_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "limitations": [],
    }
    if not execute or not docker:
        report["status"] = "docker_unavailable" if not docker else report["status"]
        return report

    suffix = uuid.uuid4().hex[:12]
    core_image = f"{CORE_IMAGE}-verification-{suffix}"
    studio_image = f"{STUDIO_IMAGE}-verification-{suffix}"
    container = f"gfs-studio-plugin-verification-{suffix}"
    built: list[str] = []
    running = False
    failure = None
    try:
        for image, context in ((core_image, "gfs-core"), (studio_image, "gfs-studio")):
            command = [
                docker, "build", "--pull", "--label",
                f"org.opencontainers.image.revision={revision}",
                "--tag", image,
            ]
            if image == studio_image:
                command.extend(["--build-arg", f"CORE_IMAGE={core_image}"])
            command.append(context)
            result = _run(command, timeout=1800)
            checks["core_image_built" if image == core_image else "studio_image_built"] = (
                result.returncode == 0
            )
            if result.returncode:
                failure = f"{'core' if image == core_image else 'studio'}_build_failed"
                report["diagnostic"] = _diagnostic(result)
                break
            built.append(image)
        if len(built) != 2:
            return _finalize(report, failure)
        images = {}
        for name, image in (("core", core_image), ("studio", studio_image)):
            result = _run([docker, "image", "inspect", image])
            payload = _json(result)
            images[name] = payload[0] if isinstance(payload, list) and payload else {}
        checks["image_ids_content_addressed"] = all(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(images[name].get("Id", "")))
            for name in ("core", "studio")
        )
        checks["image_revision_labels_match"] = all(
            (images[name].get("Config", {}).get("Labels") or {}).get(
                "org.opencontainers.image.revision"
            ) == revision
            for name in ("core", "studio")
        )
        checks["non_root_users_declared"] = all(
            images[name].get("Config", {}).get("User") == "10001:10001"
            for name in ("core", "studio")
        )
        core = _run([
            docker, "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m,mode=1777",
            core_image, "status", "--top", "1", "--json",
        ], timeout=300)
        checks["core_cli_executed"] = core.returncode == 0 and bool(core.stdout.strip())
        if not checks["core_cli_executed"]:
            failure = "core_cli_failed"
            report["diagnostic"] = _diagnostic(core)
            return _finalize(report, failure)
        started = _run([
            docker, "run", "--detach", "--name", container, "--network", "none",
            "--read-only", "--cap-drop", "ALL", "--security-opt",
            "no-new-privileges", "--pids-limit", "256", "--health-interval", "2s",
            "--health-timeout", "3s", "--health-start-period", "1s",
            "--health-retries", "10", "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m,mode=1777", "--tmpfs",
            "/app/data/persistence:rw,nosuid,size=64m,mode=1777", "--tmpfs",
            "/app/backups:rw,nosuid,size=64m,mode=1777", studio_image,
        ])
        running = started.returncode == 0
        if not running:
            failure = "studio_start_failed"
            report["diagnostic"] = _diagnostic(started)
            return _finalize(report, failure)
        inspected = _json(_run([docker, "container", "inspect", container]))
        host = inspected[0].get("HostConfig", {}) if inspected else {}
        checks["runtime_security_options_verified"] = (
            host.get("ReadonlyRootfs") is True
            and host.get("NetworkMode") == "none"
            and "ALL" in (host.get("CapDrop") or [])
            and host.get("PidsLimit") == 256
        )
        deadline = time.monotonic() + 60
        health = None
        while time.monotonic() < deadline:
            health = _run([
                docker, "exec", container, "python", "-c",
                (
                    "import json,urllib.request; payload=json.load(urllib.request.urlopen("
                    "'http://127.0.0.1:8765/health',timeout=3)); "
                    "assert payload=={'status':'ok','transport':'loopback'}; "
                    "print('plugin-health-ok')"
                ),
            ], timeout=10)
            checks["studio_health_contract_passed"] = (
                health.returncode == 0 and "plugin-health-ok" in health.stdout
            )
            state = _json(_run([
                docker, "container", "inspect", container, "--format",
                "{{json .State.Health}}",
            ]))
            checks["docker_health_status_passed"] = (
                isinstance(state, dict) and state.get("Status") == "healthy"
            )
            if checks["studio_health_contract_passed"] and checks["docker_health_status_passed"]:
                break
            time.sleep(1)
        if not all((checks["studio_health_contract_passed"], checks["docker_health_status_passed"])):
            failure = "studio_health_failed"
            report["runtime_diagnostics"] = {
                "health": _diagnostic(health),
                "state": _json(_run([
                    docker, "container", "inspect", container, "--format", "{{json .State}}",
                ])),
                "logs": _diagnostic(_run([docker, "logs", "--tail", "80", container])),
            }
    except subprocess.TimeoutExpired as error:
        failure = "docker_command_timeout"
        report["diagnostic"] = str(error)
    finally:
        checks["ephemeral_container_removed"] = not running or _run([
            docker, "rm", "--force", container,
        ]).returncode == 0
        removed = [
            _run([docker, "image", "rm", "--force", image], timeout=120).returncode == 0
            for image in built
        ]
        checks["verification_images_removed"] = len(removed) == len(built) and all(removed)
    return _finalize(report, failure)


def _finalize(report: dict, failure: str | None) -> dict:
    report["passed"] = all(report["checks"].values())
    report["status"] = "passed" if report["passed"] else failure or "failed"
    if not report["passed"]:
        report["limitations"] = [
            "Every runtime check must pass before deployment evidence is current."
        ]
    return report


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify(execute=args.execute)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if report["passed"] else 1 if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
