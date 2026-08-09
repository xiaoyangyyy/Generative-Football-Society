#!/usr/bin/env python3
"""Verify the installed GFS runtime without simulation or provider calls."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import build_parser  # noqa: E402
from src.infrastructure import file_sha256  # noqa: E402
from src.product.control_plane import ProductControlPlane  # noqa: E402


IMPORT_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "openai": "openai",
    "python-dotenv": "dotenv",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "statsbombpy": "statsbombpy",
    "kloppy": "kloppy",
}
EXACT = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)$")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _expected_dependencies() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    for requirement in project["project"]["dependencies"]:
        match = EXACT.fullmatch(requirement)
        if not match:
            raise ValueError(f"dependency is not exact: {requirement}")
        expected[_canonical(match.group(1))] = match.group(2)
    return expected


def verify_local_runtime() -> dict:
    expected = _expected_dependencies()
    packages: dict[str, dict] = {}
    for name, version in expected.items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed = None
        import_name = IMPORT_NAMES[name]
        try:
            importlib.import_module(import_name)
            imported = True
            import_error = None
        except Exception as exc:  # pragma: no cover - environment dependent
            imported = False
            import_error = f"{type(exc).__name__}: {exc}"
        packages[name] = {
            "expected": version,
            "observed": observed,
            "version_matches": observed == version,
            "import_name": import_name,
            "imported": imported,
            "import_error": import_error,
        }

    cli = build_parser().parse_args([
        "--base-dir", str(ROOT), "studio", "status",
    ])
    snapshot = ProductControlPlane(ROOT).snapshot()
    checks = {
        "all_direct_versions_match": all(
            row["version_matches"] for row in packages.values()
        ),
        "all_direct_imports_succeed": all(
            row["imported"] for row in packages.values()
        ),
        "cli_studio_status_route_resolves": callable(getattr(cli, "func", None)),
        "integrated_control_plane_loads": (
            snapshot.get("schema_version") == 1
            and snapshot.get("release", {}).get("schema_version") == 1
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "verification": "gfs_local_runtime_import_and_control_plane_smoke",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "checks": checks,
        "artifact_sha256": {
            "pyproject.toml": file_sha256(ROOT / "pyproject.toml"),
            "requirements.txt": file_sha256(ROOT / "requirements.txt"),
        },
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify_local_runtime()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
