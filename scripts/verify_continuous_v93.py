#!/usr/bin/env python3
"""Verify immutable v9.3 operational evidence and deployment isolation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    failures=[]; release=json.loads((ROOT/"data/releases/v9.3.0-candidate.json").read_text()); registry=json.loads((ROOT/"data/releases/model_registry.json").read_text()); current=json.loads((ROOT/"data/releases/current.json").read_text())
    for item in release["artifacts"]:
        path=ROOT/item["path"]
        if not path.is_file() or path.stat().st_size!=item["bytes"] or sha(path)!=item["sha256"]: failures.append(f"artifact:{item['path']}")
    for report in ("continuous_operational_v93.json","operational_calibration_v93.json","conditional_calibration_v93.json","interval_rollout_v93.json"):
        if not json.loads((ROOT/"reports/acceptance"/report).read_text()).get("ready"): failures.append(f"report:{report}")
    for name in ("continuous_time_operational","conditional_interval_router"):
        entry=registry["models"].get(f"{name}:9.3.0-candidate",{}); path=Path(entry.get("artifact",""))
        if entry.get("status")!="evaluated" or not path.is_file() or sha(path)!=entry.get("sha256"): failures.append(f"registry:{name}")
    if current.get("active")!="7.0.0" or current.get("rollback",{}).get("release")!="6.0.0": failures.append("deployment_boundary")
    result={"release":"9.3.0-candidate","artifacts":len(release["artifacts"]),"failures":failures,"ok":not failures}; print(json.dumps(result,indent=2)); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())