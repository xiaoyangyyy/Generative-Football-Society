#!/usr/bin/env python3
"""Verify immutable artifacts listed in a frozen release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_artifacts(
    manifest: dict, *, root: Path = ROOT,
) -> tuple[list[dict], list[dict]]:
    """Verify bytes, allowing only Git's reversible LF/CRLF text transform."""
    failures = []
    normalized = []
    for artifact in manifest.get("artifacts", []):
        path = root / artifact["path"]
        raw = path.read_bytes() if path.is_file() else None
        actual = hashlib.sha256(raw).hexdigest() if raw is not None else None
        if actual == artifact["sha256"]:
            continue
        accepted_normalization = None
        if raw is not None and b"\x00" not in raw:
            candidates = {
                "lf_to_crlf": raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"),
                "crlf_to_lf": raw.replace(b"\r\n", b"\n"),
            }
            for name, candidate in candidates.items():
                if hashlib.sha256(candidate).hexdigest() == artifact["sha256"]:
                    accepted_normalization = name
                    break
        if accepted_normalization:
            normalized.append({
                "path": artifact["path"],
                "normalization": accepted_normalization,
            })
        else:
            failures.append({
                "path": artifact["path"],
                "expected": artifact["sha256"],
                "actual": actual,
            })
    return failures, normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    failures, normalized = verify_artifacts(manifest)
    report = {
        "release": manifest.get("release"),
        "artifacts": len(manifest.get("artifacts", [])),
        "normalized_text_artifacts": normalized,
        "failures": failures,
        "ok": not failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
