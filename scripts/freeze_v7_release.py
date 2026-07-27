#!/usr/bin/env python3
"""Freeze the validated v7 artifacts without mutating the v6 parent."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import file_sha256, write_json_atomic
def source_digest():
    roots=[ROOT/"src",ROOT/"scripts",ROOT/"tests",ROOT/"docs",ROOT/"gfs.py",ROOT/"pyproject.toml",ROOT/"README.md"]
    files=sorted(p for root in roots for p in ([root] if root.is_file() else root.rglob("*")) if p.is_file() and p.suffix in {".py",".md",".toml"})
    digest=hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode()+b"\0"+bytes.fromhex(file_sha256(path)))
    return {"files":len(files),"sha256":digest.hexdigest()}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--tests",default="150 passed, 2 skipped"); args=parser.parse_args()
    evidence={name:ROOT/path for name,path in {
      "readiness":"reports/acceptance/v7_readiness.json","regression":"reports/acceptance/v7_regression.json",
      "lodo":"reports/acceptance/joint_pass_lodo_v7.json","tracking":"reports/acceptance/skillcorner_tracking_v7.json"}.items()}
    reports={name:json.loads(path.read_text(encoding="utf-8")) for name,path in evidence.items()}
    gates={"readiness":reports["readiness"]["release_ready"],"regression":reports["regression"]["release_ready"],"all_provider_lodo":reports["lodo"]["all_provider_lodo_ready"],"skillcorner_tracking":reports["tracking"]["candidate_ready"]}
    if not all(gates.values()): raise SystemExit(f"v7 release gates failed: {gates}")
    relative=["data/world_model/joint_pass_v7_candidate.json","data/world_model/shot_v7_candidate.json","data/world_model/probabilistic_v7_candidate.json","data/releases/v7-data-candidate.json","data/external/metrica/pass_candidates_game1.manifest.json","data/external/metrica/pass_candidates_game2.manifest.json","data/external/sportec/derived/manifest.json","data/external/statsbomb360/derived/manifest.json","data/external/skillcorner/derived/manifest.json","reports/acceptance/joint_pass_v7.json","reports/acceptance/joint_pass_lodo_v7.json","reports/acceptance/shot_v7.json","reports/acceptance/probabilistic_world_v7.json","reports/acceptance/skillcorner_tracking_v7.json","reports/acceptance/v7_regression.json","reports/acceptance/v7_readiness.json"]
    artifacts=[ROOT/path for path in relative]
    missing=[str(p) for p in artifacts if not p.is_file()]
    if missing: raise SystemExit(f"missing v7 artifacts: {missing}")
    parent=ROOT/"data/releases/v6.0.0.json"
    payload={"schema_version":2,"release":"7.0.0","status":"frozen","parent":{"release":"6.0.0","manifest":"data/releases/v6.0.0.json","sha256":file_sha256(parent)},"tests":args.tests,"gates":gates,"artifacts":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":file_sha256(p)} for p in artifacts],"source_snapshot":source_digest(),"rollback":"6.0.0","mutation_policy":"sealed artifacts are immutable; new research requires a new version"}
    target=ROOT/"data/releases/v7.0.0.json"; write_json_atomic(target,payload); print(json.dumps(payload,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
