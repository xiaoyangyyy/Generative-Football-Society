#!/usr/bin/env python3
"""Freeze official Metrica Game 3 v9.4 evidence without deployment."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
from src.model_registry import ModelRegistry
VERSION="9.4.0-candidate"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def descriptor(path): return {"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)}
def main():
    quality=ROOT/"reports/acceptance/temporal_provider_quality_v94.json"; point=ROOT/"reports/acceptance/continuous_time_v94_lopo.json"; calibration=ROOT/"reports/acceptance/continuous_calibration_v94.json"; reports=[quality,point,calibration]
    if not all(json.loads(x.read_text()).get("ready") for x in reports): raise RuntimeError("v9.4 evidence not ready")
    models=[ROOT/f"data/frame_world/continuous_time_v94_lopo_{p}.pt" for p in ("metrica","skillcorner","statsbomb360")]; registry=ModelRegistry(ROOT/"data/releases/model_registry.json"); evidence={"quality_report":quality.relative_to(ROOT).as_posix(),"point_report":point.relative_to(ROOT).as_posix(),"calibration_report":calibration.relative_to(ROOT).as_posix(),"official_metrica_game3":True,"deployed":False}
    for provider,path in zip(("metrica","skillcorner","statsbomb360"),models):
        name=f"continuous_time_lopo_{provider}"; key=f"{name}:{VERSION}"
        if key not in registry.data["models"]: registry.register(name,VERSION,path); registry.promote(name,VERSION,"evaluated",evidence)
    artifacts=[*models,*reports,ROOT/"data/frame_world/v94_temporal_providers/manifest.json",ROOT/"data/frame_world/v94_temporal_providers/metrica_game3.npz"]; release={"schema_version":2,"release":VERSION,"status":"frozen_research_candidate","parent_research":"9.3.0-candidate","parent_deployed":"7.0.0","rollback":"6.0.0","deployment_policy":"evaluated only","artifacts":[descriptor(x) for x in artifacts],"gates":{"official_source_integrity":True,"three_metrica_matches":True,"point_lopo":True,"coverage_lopo":True,"deployment_unchanged":True}}; write_json_atomic(ROOT/f"data/releases/v{VERSION}.json",release); print(json.dumps({"release":VERSION,"artifacts":len(artifacts)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())