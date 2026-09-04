#!/usr/bin/env python3
"""Audit worst-source conformal coverage under strict three-way provider holdout."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_temporal_mark_v89_lopo import Arrays
from src.match_engine.frame_world.continuous import load_continuous_time

def dev_items(items,provider):
    group=sorted([x for x in items if x["provider"]==provider],key=lambda x:x["match_id"]); return group[-max(1,len(group)//10):]
@torch.no_grad()
def predictions(model,dataset):
    median=[]; cdf={s:[] for s in (1,2,3,4)}
    for start in range(0,len(dataset),2048):
        features=torch.from_numpy(dataset.values["features"][start:start+2048]); median.append(model.median_time(features).cpu().numpy())
        for second in cdf: cdf[second].append(model.distribution_cdf(features,float(second)).cpu().numpy())
    return np.concatenate(median),{k:np.concatenate(v) for k,v in cdf.items()}
def radius(prediction,truth,coverage):
    score=np.abs(truth-prediction); rank=min(len(score)-1,int(np.ceil((len(score)+1)*coverage))-1); return float(np.sort(score)[rank])
def interval(prediction,truth,value):
    lower=np.clip(prediction-value,0,5); upper=np.clip(prediction+value,.025,5); return {"coverage":float(((truth>=lower)&(truth<=upper)).mean()),"mean_width_s":float((upper-lower).mean()),"radius_s":value}
def main():
    manifest=json.loads((ROOT/"data/frame_world/v94_temporal_providers/manifest.json").read_text()); folds=[]
    for held in ("metrica","skillcorner","statsbomb360"):
        source=[x for x in manifest["matches"] if x["provider"]!=held]; providers=sorted({x["provider"] for x in source}); model,_,meta=load_continuous_time(ROOT/f"data/frame_world/continuous_time_v94_lopo_{held}.pt"); model.eval(); source_radii={}; source_samples={}
        for provider in providers:
            data=Arrays(dev_items(source,provider),10000,9402); pred,_=predictions(model,data); observed=data.values["observed"]; truth=data.values["delay"][observed]; source_radii[provider]={str(level):radius(pred[observed],truth,level) for level in (.5,.9)}; source_samples[provider]=int(observed.sum())
        target=Arrays([x for x in manifest["matches"] if x["provider"]==held],100000,9403); pred,cdf=predictions(model,target); observed=target.values["observed"]; truth=target.values["delay"][observed]; intervals={str(level):interval(pred[observed],truth,max(source_radii[p][str(level)] for p in providers)) for level in (.5,.9)}; brier={}
        for second,probability in cdf.items(): actual=(target.values["observed"]&(target.values["delay"]<=second)).astype(float); brier[str(second)]=float(np.mean((probability-actual)**2))
        gates={"strict_provider_holdout":held not in providers,"two_calibration_providers":len(providers)==2,"coverage_50":abs(intervals["0.5"]["coverage"]-.5)<=.1,"coverage_90":abs(intervals["0.9"]["coverage"]-.9)<=.07,"ordered_width":intervals["0.9"]["mean_width_s"]>intervals["0.5"]["mean_width_s"],"nondegenerate_width":intervals["0.9"]["mean_width_s"]<4.5,"integrated_brier":float(np.mean(list(brier.values())))<.25}; folds.append({"held_provider":held,"calibration_providers":providers,"calibration_observed":source_samples,"source_radii_s":source_radii,"intervals":intervals,"brier_by_second":brier,"integrated_brier":float(np.mean(list(brier.values()))),"gates":gates,"ready":all(gates.values())})
    report={"version":"9.4.0-candidate","method":"provider-stratified split conformal absolute residuals; worst source-provider radius","target_label_policy":"held labels used only for final scoring","folds":folds,"ready":all(x["ready"] for x in folds)}; target=ROOT/"reports/acceptance/continuous_calibration_v94.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())