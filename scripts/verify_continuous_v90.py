#!/usr/bin/env python3
"""Verify frozen v9.0 continuous-time evidence and deployment boundary."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    failures=[]; release=json.loads((ROOT/"data/releases/v9.0.0-candidate.json").read_text()); training=json.loads((ROOT/"reports/acceptance/continuous_time_v90_lopo.json").read_text()); router=json.loads((ROOT/"reports/acceptance/continuous_router_v90.json").read_text()); locked=json.loads((ROOT/"data/frame_world/continuous_router_v90.json").read_text()); registry=json.loads((ROOT/"data/releases/model_registry.json").read_text()); current=json.loads((ROOT/"data/releases/current.json").read_text())
    for item in release["artifacts"]:
        path=ROOT/item["path"]
        if not path.is_file() or path.stat().st_size!=item["bytes"] or sha(path)!=item["sha256"]: failures.append(f"release:{item['path']}")
    for item in locked["dependencies"]:
        path=ROOT/item["artifact"]
        if not path.is_file() or sha(path)!=item["sha256"]: failures.append(f"dependency:{item['artifact']}")
    if not training.get("ready") or len(training.get("folds",[]))!=2: failures.append("training_ready")
    for fold in training.get("folds",[]):
        if fold["held_provider"] in fold["train_providers"] or fold["time_gain_s"]<=0 or not all(fold["gates"].values()): failures.append(f"fold:{fold['held_provider']}")
    if not router.get("ready") or not all(router.get("gates",{}).values()): failures.append("router_ready")
    for name in ("continuous_time_lopo_skillcorner","continuous_time_lopo_statsbomb360","continuous_event_router"):
        entry=registry["models"].get(f"{name}:9.0.0-candidate",{})
        if entry.get("status")!="evaluated" or not Path(entry.get("artifact","")).is_file() or sha(Path(entry["artifact"]))!=entry.get("sha256"): failures.append(f"registry:{name}")
    if current.get("active")!="7.0.0" or current.get("rollback",{}).get("release")!="6.0.0": failures.append("deployment_boundary")
    result={"release":"9.0.0-candidate","folds":len(training.get("folds",[])),"dependencies":len(locked.get("dependencies",[])),"failures":failures,"ok":not failures}; print(json.dumps(result,indent=2)); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())
