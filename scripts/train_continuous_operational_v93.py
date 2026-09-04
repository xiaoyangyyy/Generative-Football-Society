#!/usr/bin/env python3
"""Train one provider-balanced operational model with untouched calibration matches."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_continuous_time_v92_lopo import balanced_loader,combined_loader,evaluate,label_weights,provider_medians,provider_sets,split_items,thresholds
from src.match_engine.frame_world.continuous import ContinuousTimeConfig,ContinuousTimeMarkModel,mixture_survival_loss

def main():
    manifest=json.loads((ROOT/"data/frame_world/v92_temporal_providers/manifest.json").read_text()); train_items,calibration_items=split_items(manifest["matches"]); train=provider_sets(train_items,50000,9301); calibration=provider_sets(calibration_items,10000,9302); loader=balanced_loader(train,9303); dev_loader=combined_loader(calibration); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(9300); model=ContinuousTimeMarkModel(ContinuousTimeConfig()).to(device); cw=torch.tensor([label_weights(train,"continuation")],device=device); sw=torch.tensor([label_weights(train,"subtype")],device=device); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=3e-3); best=None; history=[]
    for epoch in range(6):
        model.train(); losses=[]
        for features,_,observed,continuation,subtype,delay in loader:
            features,observed,continuation,subtype,delay=features.to(device),observed.to(device),continuation.to(device),subtype.to(device),delay.to(device); weights,location,scale,clog,slog=model(features); loss=mixture_survival_loss(weights,location,scale,delay,observed); valid=continuation>=0
            if valid.any(): loss=loss+.2*torch.nn.functional.binary_cross_entropy_with_logits(clog[valid],continuation[valid].float(),pos_weight=cw)
            valid=subtype>=0
            if valid.any(): loss=loss+.05*torch.nn.functional.binary_cross_entropy_with_logits(slog[valid],subtype[valid].float(),pos_weight=sw)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss))
        metrics=evaluate(model,dev_loader,device); score=metrics["time_mae_s"]+.25*(1-metrics["continue_balanced_accuracy"]); history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"dev":metrics}); print(json.dumps(history[-1]),flush=True)
        if best is None or score<best[0]: best=(score,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); medians=provider_medians(train); anchor=float(np.median(list(medians.values()))); model.time_anchor_s=anchor; model.time_shrinkage=.5; cuts=thresholds(model,dev_loader,device); per_provider={}; ready=True
    for provider,data in calibration.items():
        metrics=evaluate(model,combined_loader({provider:data}),device,*cuts); truth=data.values["delay"][data.values["observed"]]; baseline=float(np.abs(truth-anchor).mean()); passed=metrics["time_mae_s"]<baseline and metrics["event_2s_brier"]<.3 and metrics["continue_balanced_accuracy"]>.5 and metrics["pass_shot_balanced_accuracy"]>.5; per_provider[provider]={"metrics":metrics,"anchor_baseline_mae_s":baseline,"ready":passed}; ready &= passed
    artifact=ROOT/"data/frame_world/continuous_time_v93_operational.pt"; payload=model.checkpoint(); payload["meta"]={"train_providers":sorted(train),"calibration_matches":{p:[x["match_id"] for x in calibration_items if x["provider"]==p] for p in calibration},"continue_threshold":cuts[0],"subtype_threshold":cuts[1],"time_anchor_s":anchor,"time_shrinkage":.5,"sampling":"equal provider mass","role":"operational_candidate"}; torch.save(payload,artifact); report={"version":"9.3.0-candidate","artifact":artifact.relative_to(ROOT).as_posix(),"train_samples":{p:len(d) for p,d in train.items()},"calibration_samples":{p:len(d) for p,d in calibration.items()},"per_provider":per_provider,"history":history,"ready":bool(ready)}; (ROOT/"reports/acceptance/continuous_operational_v93.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if ready else 1
if __name__=="__main__": raise SystemExit(main())