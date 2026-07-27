#!/usr/bin/env python3
"""Freeze evaluated v9.2 models and calibrated interval contract without deployment."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
from src.model_registry import ModelRegistry
VERSION="9.2.0-candidate"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def descriptor(path): return {"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)}
def main():
    quality=ROOT/"reports/acceptance/temporal_provider_quality_v92.json"; point=ROOT/"reports/acceptance/continuous_time_v92_lopo.json"; calibration=ROOT/"reports/acceptance/continuous_calibration_v92.json"; reports=[json.loads(x.read_text()) for x in (quality,point,calibration)]
    if not all(x.get("ready") for x in reports): raise RuntimeError("v9.2 evidence is not ready")
    calibrated=json.loads(calibration.read_text()); models={fold["held_provider"]:ROOT/fold["artifact"] for fold in json.loads(point.read_text())["folds"]}; interval={"version":VERSION,"coverage_levels":[.5,.9],"method":calibrated["method"],"providers":{fold["held_provider"]:{"model":models[fold["held_provider"]].relative_to(ROOT).as_posix(),"model_sha256":sha(models[fold["held_provider"]]),"radii_s":{level:fold["intervals"][level]["radius_s"] for level in ("0.5","0.9")}} for fold in calibrated["folds"]},"deployment":"evaluated_only"}; interval_path=ROOT/"data/frame_world/continuous_intervals_v92.json"; write_json_atomic(interval_path,interval)
    registry=ModelRegistry(ROOT/"data/releases/model_registry.json"); evidence={"point_report":point.relative_to(ROOT).as_posix(),"calibration_report":calibration.relative_to(ROOT).as_posix(),"strict_three_way_lopo":True,"deployed":False}
    entries={**{f"continuous_time_lopo_{p}":path for p,path in models.items()},"continuous_interval_router":interval_path}
    for name,path in entries.items():
        key=f"{name}:{VERSION}"
        if key not in registry.data["models"]: registry.register(name,VERSION,path); registry.promote(name,VERSION,"evaluated",evidence)
    artifacts=[descriptor(x) for x in [*models.values(),interval_path,quality,point,calibration,ROOT/"data/frame_world/v92_temporal_providers/manifest.json"]]; release={"schema_version":2,"release":VERSION,"status":"frozen_research_candidate","parent_research":"9.1.0-candidate","parent_deployed":"7.0.0","rollback":"6.0.0","deployment_policy":"evaluated only; not sealed or deployed","artifacts":artifacts,"gates":{"provider_quality":True,"three_lopo_folds":True,"point_metrics":True,"multi_source_coverage":True,"deployment_unchanged":True}}; write_json_atomic(ROOT/f"data/releases/v{VERSION}.json",release); print(json.dumps({"release":VERSION,"artifacts":len(artifacts),"registered":len(entries)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())