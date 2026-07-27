#!/usr/bin/env python3
"""Verify frozen v9.2 evidence, calibrated intervals, and deployment boundary."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    failures=[]; release=json.loads((ROOT/"data/releases/v9.2.0-candidate.json").read_text()); quality=json.loads((ROOT/"reports/acceptance/temporal_provider_quality_v92.json").read_text()); point=json.loads((ROOT/"reports/acceptance/continuous_time_v92_lopo.json").read_text()); calibration=json.loads((ROOT/"reports/acceptance/continuous_calibration_v92.json").read_text()); intervals=json.loads((ROOT/"data/frame_world/continuous_intervals_v92.json").read_text()); registry=json.loads((ROOT/"data/releases/model_registry.json").read_text()); current=json.loads((ROOT/"data/releases/current.json").read_text())
    for item in release["artifacts"]:
        path=ROOT/item["path"]
        if not path.is_file() or path.stat().st_size!=item["bytes"] or sha(path)!=item["sha256"]: failures.append(f"artifact:{item['path']}")
    if not quality.get("ready") or len(quality.get("providers",[]))!=3: failures.append("provider_quality")
    if not point.get("ready") or len(point.get("folds",[]))!=3: failures.append("point_lopo")
    if not calibration.get("ready") or len(calibration.get("folds",[]))!=3: failures.append("coverage_lopo")
    for fold in calibration.get("folds",[]):
        if fold["held_provider"] in fold["calibration_providers"] or len(fold["calibration_providers"])!=2 or not all(fold["gates"].values()): failures.append(f"coverage:{fold['held_provider']}")
    for provider,value in intervals.get("providers",{}).items():
        path=ROOT/value["model"]
        if not path.is_file() or sha(path)!=value["model_sha256"] or float(value["radii_s"]["0.9"])<=float(value["radii_s"]["0.5"]): failures.append(f"interval:{provider}")
    for name in [*(f"continuous_time_lopo_{p}" for p in ("metrica","skillcorner","statsbomb360")),"continuous_interval_router"]:
        entry=registry["models"].get(f"{name}:9.2.0-candidate",{})
        if entry.get("status")!="evaluated" or not Path(entry.get("artifact","")).is_file() or sha(Path(entry["artifact"]))!=entry.get("sha256"): failures.append(f"registry:{name}")
    if current.get("active")!="7.0.0" or current.get("rollback",{}).get("release")!="6.0.0": failures.append("deployment_boundary")
    result={"release":"9.2.0-candidate","artifacts":len(release["artifacts"]),"folds":len(point.get("folds",[])),"failures":failures,"ok":not failures}; print(json.dumps(result,indent=2)); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())