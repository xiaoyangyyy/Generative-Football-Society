#!/usr/bin/env python3
"""Verify the complete v7 hash chain, lifecycle registry, and rollback pointer."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.freeze_v7_release import source_digest
from scripts.verify_release_manifest import verify_artifacts


def portable_artifact_path(value: str) -> Path:
    """Resolve legacy registry paths without trusting the original checkout root."""
    original = Path(value)
    if original.is_file():
        return original
    normalized = value.replace(chr(92), "/")
    marker = "/data/"
    if marker in normalized:
        return ROOT / ("data/" + normalized.split(marker, 1)[1])
    return ROOT / normalized


def hash_matches(path: Path, expected: str) -> bool:
    """Accept exact bytes or only Git's reversible LF/CRLF text conversion."""
    if not path.is_file():
        return False
    raw = path.read_bytes()
    candidates = [raw]
    if b"\x00" not in raw:
        lf = raw.replace(b"\r\n", b"\n")
        candidates.extend((lf, lf.replace(b"\n", b"\r\n")))
    return any(hashlib.sha256(value).hexdigest() == expected for value in candidates)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--artifacts-only",action="store_true"); args=parser.parse_args()
    path=ROOT/"data/releases/v7.0.0.json"; release=json.loads(path.read_text(encoding="utf-8")); failures=[]
    parent=ROOT/release["parent"]["manifest"]
    if not hash_matches(parent,release["parent"]["sha256"]): failures.append("parent")
    artifact_failures, normalized=verify_artifacts(release,root=ROOT)
    failures.extend(item["path"] for item in artifact_failures)
    if not args.artifacts_only and source_digest()!=release["source_snapshot"]: failures.append("source_snapshot")
    current=json.loads((ROOT/"data/releases/current.json").read_text())
    release_hash=current.get("manifest_sha256","")
    if not hash_matches(path,release_hash): failures.append("release_manifest")
    registry=json.loads((ROOT/"data/releases/model_registry.json").read_text())
    for name in ("joint_pass","shot","probabilistic_world"):
        entry=registry["models"][f"{name}:7.0.0"]
        artifact=portable_artifact_path(entry["artifact"])
        if entry["status"]!="deployed" or not hash_matches(artifact,entry["sha256"]) or entry["evidence"].get("release_sha256")!=release_hash: failures.append(f"registry:{name}")
    if current.get("active")!="7.0.0": failures.append("active_pointer")
    rollback=current.get("rollback",{})
    if rollback.get("manifest_sha256")!=release["parent"]["sha256"] or not hash_matches(parent,rollback.get("manifest_sha256","")): failures.append("rollback_pointer")
    result={"release":"7.0.0","mode":"artifacts-only" if args.artifacts_only else "full","artifacts":len(release["artifacts"]),"normalized_text_artifacts":normalized,"source_files":release["source_snapshot"]["files"],"failures":failures,"ok":not failures}; print(json.dumps(result,indent=2)); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())
