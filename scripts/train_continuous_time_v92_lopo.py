#!/usr/bin/env python3
"""Train provider-balanced continuous-time models with strict three-way LOPO."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset,DataLoader,WeightedRandomSampler
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_continuous_time_v91_lopo import evaluate,thresholds
from scripts.train_temporal_mark_v89_lopo import Arrays
from src.match_engine.frame_world.continuous import ContinuousTimeConfig,ContinuousTimeMarkModel,mixture_survival_loss

def split_items(items):
    train=[]; dev=[]
    for provider in sorted({x["provider"] for x in items}):
        group=sorted([x for x in items if x["provider"]==provider],key=lambda x:x["match_id"]); count=max(1,len(group)//10); dev.extend(group[-count:]); train.extend(group[:-count])
    return train,dev
def provider_sets(items,limit,seed):
    return {p:Arrays([x for x in items if x["provider"]==p],limit,seed+i) for i,p in enumerate(sorted({x["provider"] for x in items}))}
def combined_loader(sets,batch=512): return DataLoader(ConcatDataset(list(sets.values())),batch_size=batch)
def balanced_loader(sets,seed):
    datasets=list(sets.values()); concat=ConcatDataset(datasets); weights=np.concatenate([np.full(len(d),1/(len(d)*len(datasets)),np.float64) for d in datasets]); generator=torch.Generator().manual_seed(seed); sampler=WeightedRandomSampler(torch.from_numpy(weights),num_samples=60000,replacement=True,generator=generator); return DataLoader(concat,batch_size=512,sampler=sampler)
def provider_medians(sets): return {p:float(np.median(d.values["delay"][d.values["observed"]])) for p,d in sets.items()}
def provider_equal_mae(sets,value): return float(np.mean([np.abs(d.values["delay"][d.values["observed"]]-value).mean() for d in sets.values()]))
def label_weights(sets,key):
    values=np.concatenate([d.values[key] for d in sets.values()]); valid=values>=0; values=values[valid]; return float((values==0).sum()/max(1,(values==1).sum()))
def fold(held,manifest,device,epochs=6):
    source=[x for x in manifest["matches"] if x["provider"]!=held]; target=[x for x in manifest["matches"] if x["provider"]==held]; train_items,dev_items=split_items(source); train=provider_sets(train_items,50000,9201); dev=provider_sets(dev_items,10000,9202); test=provider_sets(target,100000,9203); train_loader=balanced_loader(train,9204); dev_loader=combined_loader(dev); test_loader=combined_loader(test); torch.manual_seed(9200+("metrica","skillcorner","statsbomb360").index(held)); model=ContinuousTimeMarkModel(ContinuousTimeConfig()).to(device); cw=torch.tensor([label_weights(train,"continuation")],device=device); sw=torch.tensor([label_weights(train,"subtype")],device=device); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=3e-3); best=None; history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for features,_,observed,continuation,subtype,delay in train_loader:
            features,observed,continuation,subtype,delay=features.to(device),observed.to(device),continuation.to(device),subtype.to(device),delay.to(device); weights,location,scale,clog,slog=model(features); loss=mixture_survival_loss(weights,location,scale,delay,observed); valid=continuation>=0
            if valid.any(): loss=loss+.2*torch.nn.functional.binary_cross_entropy_with_logits(clog[valid],continuation[valid].float(),pos_weight=cw)
            valid=subtype>=0
            if valid.any(): loss=loss+.05*torch.nn.functional.binary_cross_entropy_with_logits(slog[valid],subtype[valid].float(),pos_weight=sw)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss))
        metrics=evaluate(model,dev_loader,device); score=metrics["time_mae_s"]+.25*(1-metrics["continue_balanced_accuracy"]); history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"dev":metrics}); print(json.dumps({"held":held,**history[-1]}),flush=True)
        if best is None or score<best[0]: best=(score,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); medians=provider_medians(train); anchor=float(np.median(list(medians.values()))); model.time_anchor_s=anchor; model.time_shrinkage=.5; cut=thresholds(model,dev_loader,device); metrics=evaluate(model,test_loader,device,*cut); baseline=provider_equal_mae(test,anchor); artifact=ROOT/f"data/frame_world/continuous_time_v92_lopo_{held}.pt"; payload=model.checkpoint(); payload["meta"]={"held_provider":held,"train_providers":sorted(train),"continue_threshold":cut[0],"subtype_threshold":cut[1],"feature_contract":"v92_static_mirror_invariant_explicit_receiver_clock","time_anchor_s":anchor,"time_shrinkage":.5,"source_provider_medians_s":medians,"sampling":"equal provider mass per epoch"}; torch.save(payload,artifact); gates={"strict_provider_holdout":held not in train,"two_source_providers":len(train)==2,"continuous_time_gain":metrics["time_mae_s"]<baseline,"continue_gain":metrics["continue_balanced_accuracy"]>.5,"mark_gain":metrics["pass_shot_balanced_accuracy"]>.5,"calibrated":metrics["event_2s_brier"]<.3,"finite":all(np.isfinite(x) for x in metrics.values())}; return {"held_provider":held,"train_providers":sorted(train),"samples":{"train_by_provider":{p:len(d) for p,d in train.items()},"dev_by_provider":{p:len(d) for p,d in dev.items()},"test":sum(map(len,test.values()))},"metrics":metrics,"provider_equal_anchor_baseline_mae_s":baseline,"time_gain_s":baseline-metrics["time_mae_s"],"gates":gates,"ready":all(gates.values()),"artifact":artifact.relative_to(ROOT).as_posix(),"history":history}
def main():
    manifest=json.loads((ROOT/"data/frame_world/v92_temporal_providers/manifest.json").read_text()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); folds=[fold(x,manifest,device) for x in ("metrica","skillcorner","statsbomb360")]; report={"version":"9.2.0-candidate","model":"provider-balanced three-component log-normal mixture survival","policy":"strict three-way leave-one-provider-out; provider-stratified development split","folds":folds,"ready":all(x["ready"] for x in folds)}; target=ROOT/"reports/acceptance/continuous_time_v92_lopo.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())