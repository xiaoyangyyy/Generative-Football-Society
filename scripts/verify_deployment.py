from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DIGEST = "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"


def verify(root: Path = ROOT) -> dict:
    errors = []
    for relative in ("gfs-core/Dockerfile", "gfs-studio/Dockerfile", "deployment/compose.yaml", "deployment/manifest.json"):
        if not (root / relative).is_file():
            errors.append(f"missing:{relative}")
    if errors:
        return _result(errors)
    core = (root / "gfs-core" / "Dockerfile").read_text(encoding="utf-8")
    studio = (root / "gfs-studio" / "Dockerfile").read_text(encoding="utf-8")
    compose = (root / "deployment" / "compose.yaml").read_text(encoding="utf-8")
    manifest = json.loads((root / "deployment" / "manifest.json").read_text(encoding="utf-8"))
    if EXPECTED_DIGEST not in core:
        errors.append("python image digest mismatch")
    if not core.startswith("ARG PYTHON_IMAGE"):
        errors.append("python image must be pinned through ARG")
    if "FROM gfs-core:latest" not in studio:
        errors.append("studio must inherit the core image")
    if "USER 10001:10001" not in core or "USER 10001:10001" not in studio:
        errors.append("non-root runtime user missing")
    if "read_only: true" not in compose or "cap_drop:" not in compose:
        errors.append("container hardening is incomplete")
    if "127.0.0.1:8765:8765" not in compose or "0.0.0.0" in compose:
        errors.append("deployment must expose loopback only")
    if "/health" not in core or "/health" not in studio:
        errors.append("healthcheck endpoint missing")
    if manifest.get("deployment") != "loopback_only":
        errors.append("manifest deployment boundary mismatch")
    if manifest.get("runtime_verified") is not False:
        errors.append("runtime verification cannot be claimed")
    return _result(errors)


def _result(errors):
    return {
        "schema_version": 1,
        "static_verified": not errors,
        "runtime_verified": False,
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
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["static_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
