#!/usr/bin/env python3
"""Calibrate the v9.3 operational model on untouched provider-stratified matches."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.evaluate_conditional_calibration_v93 import KINDS,predict
from scripts.evaluate_continuous_calibration_v92 import radius
from scripts.train_continuous_time_v92_lopo import provider_sets,split_items
from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.frame_world.continuous import load_continuous_time

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/"data/frame_world/v92_temporal_providers/manifest.json").read_text()); _,items=split_items(manifest["matches"]); data=provider_sets(items,10000,9302); model_path=ROOT/"data/frame_world/continuous_time_v93_operational.pt"; model,_,meta=load_continuous_time(model_path); model.eval(); cuts=(float(meta["continue_threshold"]),float(meta["subtype_threshold"])); providers={}; audit={}
    for provider,dataset in data.items():
        prediction,kind=predict(model,dataset,cuts); observed=dataset.values["observed"]; truth=dataset.values["delay"]; radii={}; counts={}
        for level in (.5,.9):
            global_radius=radius(prediction[observed],truth[observed],level); radii.setdefault("global",{})[str(level)]=global_radius
            for name in KINDS:
                mask=observed&(kind==name); radii.setdefault(name,{})[str(level)]=radius(prediction[mask],truth[mask],level) if mask.sum()>=100 else global_radius; counts[name]=int(mask.sum())
        providers[provider]={"model":model_path.relative_to(ROOT).as_posix(),"model_sha256":sha(model_path),"radii_s":radii}; audit[provider]={"observed":int(observed.sum()),"predicted_kind_counts":counts,"radii_s":radii}
    artifact={"version":"9.3.0-candidate","deployment":"operational_candidate","method":"provider-stratified predicted-mark Mondrian conformal with global fallback below 100 samples","providers":providers}; write_json_atomic(ROOT/"data/frame_world/continuous_intervals_v93.json",artifact); report={"version":"9.3.0-candidate","calibration_matches":meta["calibration_matches"],"providers":audit,"lopo_evidence":"reports/acceptance/conditional_calibration_v93.json","ready":True}; write_json_atomic(ROOT/"reports/acceptance/operational_calibration_v93.json",report); print(json.dumps(report,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())