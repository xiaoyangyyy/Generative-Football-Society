#!/usr/bin/env python3
"""Verify the complete v7 hash chain, lifecycle registry, and rollback pointer."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.freeze_v7_release import source_digest
from src.data_engine.dataset_registry import file_sha256
from src.model_registry import artifact_sha256
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--artifacts-only",action="store_true"); args=parser.parse_args()
    path=ROOT/"data/releases/v7.0.0.json"; release=json.loads(path.read_text(encoding="utf-8")); failures=[]
    parent=ROOT/release["parent"]["manifest"]
    if file_sha256(parent)!=release["parent"]["sha256"]: failures.append("parent")
    for item in release["artifacts"]:
        artifact=ROOT/item["path"]
        if not artifact.is_file() or artifact.stat().st_size!=item["bytes"] or file_sha256(artifact)!=item["sha256"]: failures.append(item["path"])
    if not args.artifacts_only and source_digest()!=release["source_snapshot"]: failures.append("source_snapshot")
    release_hash=file_sha256(path); registry=json.loads((ROOT/"data/releases/model_registry.json").read_text())
    for name in ("joint_pass","shot","probabilistic_world"):
        entry=registry["models"][f"{name}:7.0.0"]
        if entry["status"]!="deployed" or artifact_sha256(entry["artifact"])!=entry["sha256"] or entry["evidence"].get("release_sha256")!=release_hash: failures.append(f"registry:{name}")
    current=json.loads((ROOT/"data/releases/current.json").read_text())
    if current.get("active")!="7.0.0" or current.get("manifest_sha256")!=release_hash: failures.append("active_pointer")
    if current.get("rollback",{}).get("manifest_sha256")!=file_sha256(parent): failures.append("rollback_pointer")
    result={"release":"7.0.0","mode":"artifacts-only" if args.artifacts_only else "full","artifacts":len(release["artifacts"]),"source_files":release["source_snapshot"]["files"],"failures":failures,"ok":not failures}; print(json.dumps(result,indent=2)); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())
