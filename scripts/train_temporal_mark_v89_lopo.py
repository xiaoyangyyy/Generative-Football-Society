#!/usr/bin/env python3
"""Strict bidirectional leave-one-provider-out training for v8.9 temporal marks."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_temporal_mark_v88 import calibrate_thresholds,evaluate
from src.match_engine.frame_world.temporal import TemporalMarkConfig,TemporalMarkModel,survival_loss


class Arrays(Dataset):
    def __init__(self,items,limit=None,seed=8901):
        values={k:[] for k in ("features","event_bin","observed","continuation","subtype","delay")}
        for item in items:
            with np.load(ROOT/item["file"],allow_pickle=False) as source:
                for key in values: values[key].append(source[key])
        self.values={k:np.concatenate(v) for k,v in values.items()}
        if limit and len(self)>limit:
            rng=np.random.default_rng(seed); index=np.sort(rng.choice(len(self),limit,replace=False)); self.values={k:v[index] for k,v in self.values.items()}
    def __len__(self): return len(self.values["features"])
    def __getitem__(self,index): return tuple(torch.from_numpy(self.values[k][index]) if self.values[k][index].ndim else torch.tensor(self.values[k][index].item()) for k in self.values)


def train_fold(held,manifest,device,epochs):
    source=[x for x in manifest["matches"] if x["provider"]!=held]; held_items=[x for x in manifest["matches"] if x["provider"]==held]; providers=sorted({x["provider"] for x in source}); matches=sorted({x["match_id"] for x in source}); dev_ids=set(matches[-max(1,len(matches)//10):]); train_items=[x for x in source if x["match_id"] not in dev_ids]; dev_items=[x for x in source if x["match_id"] in dev_ids]
    train=Arrays(train_items,120000,8901); dev=Arrays(dev_items,20000,8902); test=Arrays(held_items,100000,8903); loaders=[DataLoader(x,batch_size=512,shuffle=i==0) for i,x in enumerate((train,dev,test))]; cfg=TemporalMarkConfig(direct_subtype=True); torch.manual_seed(8900+(0 if held=="skillcorner" else 1)); model=TemporalMarkModel(cfg).to(device)
    cont=train.values["continuation"]; sub=train.values["subtype"]; cw=torch.tensor([(cont==0).sum()/max(1,(cont==1).sum())],device=device); sw=torch.tensor([(sub==0).sum()/max(1,(sub==1).sum())],device=device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-3); best=None; history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for features,bins,observed,continuation,subtype,_ in loaders[0]:
            features,bins,observed,continuation,subtype=features.to(device),bins.to(device),observed.to(device),continuation.to(device),subtype.to(device); hazard,clog,slog=model(features); loss=survival_loss(hazard,bins,observed); valid=continuation>=0
            if valid.any(): loss=loss+.25*torch.nn.functional.binary_cross_entropy_with_logits(clog[valid],continuation[valid].float(),pos_weight=cw)
            valid=subtype>=0
            if valid.any(): loss=loss+.25*torch.nn.functional.binary_cross_entropy_with_logits(slog[valid],subtype[valid].float(),pos_weight=sw)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss))
        metrics=evaluate(model,loaders[1],device); score=metrics["time_mae_s"]+.5*(1-metrics["continue_balanced_accuracy"])+.25*(1-metrics["pass_shot_balanced_accuracy"]); history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"dev":metrics}); print(json.dumps({"held":held,**history[-1]}),flush=True)
        if best is None or score<best[0]: best=(score,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); thresholds=calibrate_thresholds(model,loaders[1],device); metrics=evaluate(model,loaders[2],device,*thresholds); observed=test.values["observed"]; train_delay=train.values["delay"][train.values["observed"]]; test_delay=test.values["delay"][observed]; continuous_baseline=float(np.mean(np.abs(test_delay-np.median(train_delay)))); choices=np.arange(cfg.bin_seconds,cfg.bins*cfg.bin_seconds+.001,cfg.bin_seconds); baseline_value=float(choices[np.argmin([np.abs(train_delay-x).mean() for x in choices])]); baseline_time=float(np.mean(np.abs(test_delay-baseline_value))); artifact=ROOT/f"data/frame_world/temporal_mark_v89_lopo_{held}.pt"; payload=model.checkpoint(); payload["meta"]={"held_provider":held,"train_providers":providers,"continue_threshold":thresholds[0],"subtype_threshold":thresholds[1],"static_geometry":True}; torch.save(payload,artifact)
    gates={"strict_provider_holdout":held not in providers,"time_beats_train_median":metrics["time_mae_s"]<baseline_time,"continue_above_majority":metrics["continue_balanced_accuracy"]>.5,"pass_shot_above_majority":metrics["pass_shot_balanced_accuracy"]>.5,"calibrated":metrics["event_2s_brier"]<.3,"finite":all(np.isfinite(x) for x in metrics.values())}; return {"held_provider":held,"train_providers":providers,"train_matches":len(train_items),"dev_matches":len(dev_items),"samples":{"train":len(train),"dev":len(dev),"test":len(test)},"thresholds":{"continue":thresholds[0],"pass_shot":thresholds[1]},"metrics":metrics,"quantized_train_baseline_s":baseline_value,"quantized_train_baseline_mae_s":baseline_time,"continuous_train_median_baseline_mae_s":continuous_baseline,"gates":gates,"ready":all(gates.values()),"artifact":artifact.relative_to(ROOT).as_posix(),"history":history}


def main():
    manifest=json.loads((ROOT/"data/frame_world/v89_temporal_providers/manifest.json").read_text()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); folds=[train_fold(x,manifest,device,10) for x in ("skillcorner","statsbomb360")]; report={"version":"8.9.0-candidate","policy":"strict bidirectional leave-one-provider-out; static geometry only","folds":folds,"ready":all(x["ready"] for x in folds)}; target=ROOT/"reports/acceptance/temporal_mark_v89_lopo.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1


if __name__=="__main__": raise SystemExit(main())
