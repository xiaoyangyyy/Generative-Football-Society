#!/usr/bin/env python3
"""Build or check the deterministic CycloneDX SBOM for the Linux runtime lock."""

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


def expected_sbom() -> dict:
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    return build_cyclonedx_sbom(
        LOCK,
        project_name=str(project["name"]),
        project_version=str(project["version"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    expected = expected_sbom()
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
