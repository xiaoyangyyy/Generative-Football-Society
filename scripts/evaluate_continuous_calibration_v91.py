#!/usr/bin/env python3
"""Source-calibrated conformal coverage audit for v9.1."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_temporal_mark_v89_lopo import Arrays
from src.match_engine.frame_world.continuous import load_continuous_time


@torch.no_grad()
def distribution(model,dataset,levels):
    output={f"q{x}":[] for x in levels}; output.update({f"cdf{x}":[] for x in (1,2,3,4)})
    for start in range(0,len(dataset),2048):
        features=torch.from_numpy(dataset.values["features"][start:start+2048])
        for level in levels: output[f"q{level}"].append(model.quantile_time(features,level).cpu().numpy())
        for second in (1,2,3,4): output[f"cdf{second}"].append(model.distribution_cdf(features,float(second)).cpu().numpy())
    return {k:np.concatenate(v) for k,v in output.items()}


def residual_interval(calibration,cal_y,test,test_y,coverage):
    score=np.abs(cal_y-calibration["q0.5"]); rank=min(len(score)-1,int(np.ceil((len(score)+1)*coverage))-1); radius=float(np.sort(score)[rank]); lower=np.clip(test["q0.5"]-radius,0,5); upper=np.clip(test["q0.5"]+radius,.025,5); return {"coverage":float(((test_y>=lower)&(test_y<=upper)).mean()),"mean_width_s":float((upper-lower).mean()),"correction_s":radius}

def main():
    manifest=json.loads((ROOT/"data/frame_world/v91_temporal_providers/manifest.json").read_text()); folds=[]; levels=(.05,.25,.5,.75,.95)
    for held in ("skillcorner","statsbomb360"):
        source=[x for x in manifest["matches"] if x["provider"]!=held]; ids=sorted({x["match_id"] for x in source}); dev_ids=set(ids[-max(1,len(ids)//10):]); dev=Arrays([x for x in source if x["match_id"] in dev_ids],20000,9102); test=Arrays([x for x in manifest["matches"] if x["provider"]==held],100000,9103); model,_,meta=load_continuous_time(ROOT/f"data/frame_world/continuous_time_v91_lopo_{held}.pt"); model.eval(); dp=distribution(model,dev,levels); tp=distribution(model,test,levels); do=dev.values["observed"]; to=test.values["observed"]; dy=dev.values["delay"][do]; ty=test.values["delay"][to]; dc={k:v[do] for k,v in dp.items() if k.startswith("q")}; tc={k:v[to] for k,v in tp.items() if k.startswith("q")};         intervals={str(level):residual_interval(dc,dy,tc,ty,level) for level in (.5,.9)}; weights=[]
        for second in (1,2,3,4):
            actual=(to&(test.values["delay"]<=second)).astype(float); weights.append(float(np.mean((tp[f"cdf{second}"]-actual)**2)))
        gates={"coverage_50":abs(intervals["0.5"]["coverage"]-.5)<=.1,"coverage_90":abs(intervals["0.9"]["coverage"]-.9)<=.07,"ordered_width":intervals["0.9"]["mean_width_s"]>intervals["0.5"]["mean_width_s"],"nondegenerate_width":intervals["0.9"]["mean_width_s"]<4.5,"integrated_brier":float(np.mean(weights))<.25}; folds.append({"held_provider":held,"calibration_provider":meta["train_providers"][0],"samples":{"calibration_observed":int(do.sum()),"test_observed":int(to.sum())},"intervals":intervals,"brier_by_second":{str(i+1):x for i,x in enumerate(weights)},"integrated_brier":float(np.mean(weights)),"gates":gates,"ready":all(gates.values())})
    report={"version":"9.1.0-candidate","method":"source-provider split-conformal absolute-residual intervals around shrinkage-calibrated mixture median","censoring_policy":"all samples enter time-dependent Brier; interval coverage uses observed event times only","folds":folds,"ready":all(x["ready"] for x in folds)}; target=ROOT/"reports/acceptance/continuous_calibration_v91.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())
