#!/usr/bin/env python3
"""Build or check a deterministic CycloneDX SBOM for a hashed runtime lock."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.supply_chain import build_cyclonedx_sbom  # noqa: E402


LOCK = ROOT / "requirements-linux-py312.lock"
DEFAULT_OUTPUT = ROOT / "data/evaluation/supply_chain_sbom_v1.cdx.json"
DEFAULT_TARGET = "CPython 3.12 / x86_64 manylinux_2_28 / CPU"


def expected_sbom(
    lock: Path = LOCK, *, resolution_target: str = DEFAULT_TARGET,
) -> dict:
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    return build_cyclonedx_sbom(
        lock,
        project_name=str(project["name"]),
        project_version=str(project["version"]),
        resolution_target=resolution_target,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution-target", default=DEFAULT_TARGET)
    args = parser.parse_args(argv)
    expected = expected_sbom(
        args.lock, resolution_target=args.resolution_target,
    )
    rendered = json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(json.dumps({
            "status": "written",
            "path": str(args.out),
            "components": len(expected["components"]),
        }))
        return 0
    if not args.out.is_file():
        print(json.dumps({"status": "missing", "path": str(args.out)}))
        return 1
    current = json.loads(args.out.read_text(encoding="utf-8"))
    matches = current == expected
    print(json.dumps({
        "status": "current" if matches else "stale",
        "path": str(args.out),
        "components": len(expected["components"]),
    }))
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
