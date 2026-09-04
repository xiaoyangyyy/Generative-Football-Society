#!/usr/bin/env python3
"""Freeze v9.3 operational continuous-time candidate without deployment."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
from src.model_registry import ModelRegistry
VERSION="9.3.0-candidate"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def descriptor(path): return {"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)}
def main():
    paths=[ROOT/"data/frame_world/continuous_time_v93_operational.pt",ROOT/"data/frame_world/continuous_intervals_v93.json",ROOT/"reports/acceptance/continuous_operational_v93.json",ROOT/"reports/acceptance/operational_calibration_v93.json",ROOT/"reports/acceptance/conditional_calibration_v93.json",ROOT/"reports/acceptance/interval_rollout_v93.json"]
    for path in paths[2:]:
        if not json.loads(path.read_text()).get("ready"): raise RuntimeError(f"not ready: {path}")
    registry=ModelRegistry(ROOT/"data/releases/model_registry.json"); evidence={"operational_report":"reports/acceptance/continuous_operational_v93.json","conditional_lopo":"reports/acceptance/conditional_calibration_v93.json","interval_rollout":"reports/acceptance/interval_rollout_v93.json","deployed":False}
    for name,path in (("continuous_time_operational",paths[0]),("conditional_interval_router",paths[1])):
        key=f"{name}:{VERSION}"
        if key not in registry.data["models"]: registry.register(name,VERSION,path); registry.promote(name,VERSION,"evaluated",evidence)
    release={"schema_version":2,"release":VERSION,"status":"frozen_operational_research_candidate","parent_research":"9.2.0-candidate","parent_deployed":"7.0.0","rollback":"6.0.0","deployment_policy":"evaluated only; production promotion requires explicit release decision","artifacts":[descriptor(x) for x in paths],"gates":{"operational_model":True,"conditional_lopo":True,"calibration_holdout":True,"real_tracking_rollout":True,"deployment_unchanged":True}}; write_json_atomic(ROOT/f"data/releases/v{VERSION}.json",release); print(json.dumps({"release":VERSION,"artifacts":len(paths),"registered":2},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())