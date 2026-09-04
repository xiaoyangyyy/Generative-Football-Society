#!/usr/bin/env python3
"""Strict LOPO Mondrian conformal audit by predicted next-action kind."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.evaluate_continuous_calibration_v92 import dev_items,interval,radius
from scripts.train_temporal_mark_v89_lopo import Arrays
from src.match_engine.frame_world.continuous import load_continuous_time
KINDS=np.asarray(["pass","shot","terminal"])
@torch.no_grad()
def predict(model,dataset,thresholds):
    times=[]; kinds=[]
    for start in range(0,len(dataset),2048):
        features=torch.from_numpy(dataset.values["features"][start:start+2048]); *_,clog,slog=model(features); continuation=torch.sigmoid(clog)>=thresholds[0]; subtype=torch.sigmoid(slog)>=thresholds[1]; index=torch.where(continuation,torch.where(subtype,torch.ones_like(clog,dtype=torch.long),torch.zeros_like(clog,dtype=torch.long)),torch.full_like(clog,2,dtype=torch.long)); times.append(model.median_time(features).cpu().numpy()); kinds.append(index.cpu().numpy())
    return np.concatenate(times),KINDS[np.concatenate(kinds)]
def main():
    manifest=json.loads((ROOT/"data/frame_world/v92_temporal_providers/manifest.json").read_text()); folds=[]
    for held in ("metrica","skillcorner","statsbomb360"):
        source=[x for x in manifest["matches"] if x["provider"]!=held]; providers=sorted({x["provider"] for x in source}); model,_,meta=load_continuous_time(ROOT/f"data/frame_world/continuous_time_v92_lopo_{held}.pt"); model.eval(); cuts=(float(meta["continue_threshold"]),float(meta["subtype_threshold"])); by_source={}
        for provider in providers:
            data=Arrays(dev_items(source,provider),10000,9202); prediction,kind=predict(model,data,cuts); observed=data.values["observed"]; truth=data.values["delay"]; global_radius=radius(prediction[observed],truth[observed],.9); groups={}
            for name in KINDS:
                mask=observed&(kind==name); groups[name]={"samples":int(mask.sum()),"radius_s":radius(prediction[mask],truth[mask],.9) if mask.sum()>=100 else global_radius,"fallback":bool(mask.sum()<100)}
            by_source[provider]={"global_radius_s":global_radius,"groups":groups}
        final={name:max(by_source[p]["groups"][name]["radius_s"] for p in providers) for name in KINDS}; target=Arrays([x for x in manifest["matches"] if x["provider"]==held],100000,9203); prediction,kind=predict(model,target,cuts); observed=target.values["observed"]; truth=target.values["delay"]; group_results={}; covered=[]
        for name in KINDS:
            mask=observed&(kind==name); result=interval(prediction[mask],truth[mask],final[name]); result["samples"]=int(mask.sum()); result["ready"]=mask.sum()<100 or abs(result["coverage"]-.9)<=.1; group_results[name]=result; covered.extend(((truth[mask]>=np.clip(prediction[mask]-final[name],0,5))&(truth[mask]<=np.clip(prediction[mask]+final[name],0,5))).tolist())
        overall=float(np.mean(covered)); gates={"strict_holdout":held not in providers,"conditional_coverage":all(x["ready"] for x in group_results.values()),"overall_coverage":abs(overall-.9)<=.07,"finite":all(np.isfinite(x) for x in final.values())}; folds.append({"held_provider":held,"source_calibration":by_source,"final_radii_s":final,"target_groups":group_results,"overall_coverage":overall,"gates":gates,"ready":all(gates.values())})
    report={"version":"9.3.0-candidate","method":"predicted-mark Mondrian conformal; worst source-provider radius; minimum group 100","folds":folds,"ready":all(x["ready"] for x in folds)}; (ROOT/"reports/acceptance/conditional_calibration_v93.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())