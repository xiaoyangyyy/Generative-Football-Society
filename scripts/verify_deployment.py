from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_plugin_containers import ARTIFACTS

EXPECTED_DIGEST = (
    "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
)
CORE_IMAGE = "gfs-core:7.0.0a1"
STUDIO_IMAGE = "gfs-studio:0.1.0a1"
RUNTIME_REPORT = "gfs-studio/evidence/current/container_runtime.json"


def verify(root: Path = ROOT) -> dict:
    errors = []
    paths = (
        "gfs-core/Dockerfile",
        "gfs-studio/Dockerfile",
        "deployment/compose.yaml",
        "deployment/manifest.json",
    )
    for relative in paths:
        if not (root / relative).is_file():
            errors.append(f"missing:{relative}")
    if errors:
        return _result(errors)
    core = (root / "gfs-core" / "Dockerfile").read_text(encoding="utf-8")
    studio = (root / "gfs-studio" / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "deployment" / "compose.yaml").read_text(encoding="utf-8")
    manifest = json.loads(
        (root / "deployment" / "manifest.json").read_text(encoding="utf-8")
    )
    if EXPECTED_DIGEST not in core:
        errors.append("python image digest mismatch")
    if not core.startswith("ARG PYTHON_IMAGE"):
        errors.append("python image must be pinned through ARG")
    if f"ARG CORE_IMAGE={CORE_IMAGE}" not in studio or "FROM ${CORE_IMAGE}" not in studio:
        errors.append("studio must inherit the versioned core image")
    if "USER 10001:10001" not in core or "USER 10001:10001" not in studio:
        errors.append("non-root runtime user missing")
    for name, document in (("core", core), ("studio", studio)):
        if "python -m pip install --require-hashes -r requirements-linux-py312.lock" not in document:
            errors.append(f"{name} dependencies must use the hashed lock")
    if 'CMD ["--base-dir", "/app", "web"]' not in studio:
        errors.append("studio container must provide its required workspace")
    if "/health" not in studio or "payload['status'] == 'ok'" not in studio:
        errors.append("studio health response contract missing")
    if "HEALTHCHECK" in core or "EXPOSE 8765" in core:
        errors.append("core is a library image, not a network service")
    if "\n  core:\n" in compose or "depends_on:" in compose:
        errors.append("compose must not run the core library as a service")
    if "core_persistence" in compose:
        errors.append("core service state must not exist")
    if "read_only: true" not in compose or "cap_drop:" not in compose:
        errors.append("container hardening is incomplete")
    if "127.0.0.1:8765:8765" not in compose or "0.0.0.0" in compose:
        errors.append("deployment must expose loopback only")
    if "mode=1777" not in compose:
        errors.append("tmpfs must be writable by the non-root runtime user")
    if manifest.get("deployment") != "loopback_only":
        errors.append("manifest deployment boundary mismatch")
    if manifest.get("core_image") != CORE_IMAGE:
        errors.append("manifest core image mismatch")
    if manifest.get("studio_image") != STUDIO_IMAGE:
        errors.append("manifest studio image mismatch")
    if manifest.get("runtime_topology") != "studio_process_with_embedded_core_library":
        errors.append("manifest runtime topology mismatch")
    if manifest.get("runtime_verified") is not False:
        errors.append("runtime verification requires separate executed evidence")
    if ":latest" in core + studio + compose + json.dumps(manifest):
        errors.append("mutable latest image references are forbidden")
    return _result(errors, root)


def _runtime_evidence(root: Path) -> dict:
    path = root / RUNTIME_REPORT
    if not path.is_file():
        return {"verified": False, "source_revision": None, "report": None}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"verified": False, "source_revision": None, "report": None}
    artifacts = report.get("artifact_sha256", {})
    checks = report.get("checks", {})
    current = set(artifacts) == set(ARTIFACTS)
    for relative in ARTIFACTS:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            current = False
            continue
        if not candidate.is_file() or hashlib.sha256(
            candidate.read_bytes()
        ).hexdigest() != artifacts.get(relative):
            current = False
    verified = (
        current
        and report.get("schema_version") == 1
        and report.get("verification") == "gfs_plugin_container_runtime_v1"
        and report.get("passed") is True
        and re.fullmatch(r"[0-9a-f]{40}", str(report.get("source_revision", "")))
        is not None
        and bool(checks)
        and all(checks.values())
    )
    return {
        "verified": verified,
        "source_revision": report.get("source_revision"),
        "report": report,
    }


def _result(errors, root: Path = ROOT):
    runtime = _runtime_evidence(root) if not errors else {
        "verified": False, "source_revision": None, "report": None,
    }
    return {
        "schema_version": 2,
        "static_verified": not errors,
        "runtime_verified": runtime["verified"],
        "runtime_evidence": runtime,
        "docker_available": bool(shutil.which("docker")),
        "errors": errors,
        "external_calls_made": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0 if result["static_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
