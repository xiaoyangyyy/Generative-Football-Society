#!/usr/bin/env python3
"""Verify immutable artifacts listed in a frozen release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    failures = []
    for artifact in manifest.get("artifacts", []):
        path = ROOT / artifact["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != artifact["sha256"]:
            failures.append({"path": artifact["path"], "expected": artifact["sha256"], "actual": actual})
    report = {"release": manifest.get("release"), "artifacts": len(manifest.get("artifacts", [])), "failures": failures, "ok": not failures}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
