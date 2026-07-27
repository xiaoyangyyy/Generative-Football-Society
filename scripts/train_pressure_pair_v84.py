#!/usr/bin/env python3
"""Train and falsify a match-held-out pressure-onset pair transition."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.match_engine.frame_world.pressure import PressurePairConfig,PressurePairTransition


class PressureEvents(Dataset):
    def __init__(self,items,cfg):
        self.rows=[]
        for item in items:
            with np.load(ROOT/item["pressure_file"],allow_pickle=False) as source: labels={key:source[key] for key in source.files}
            with np.load(ROOT/item["file"],allow_pickle=False) as source: frame={key:source[key] for key in ("positions","velocities","visible")}
            for n in np.flatnonzero(labels["event_geometry_valid"]):
                index=int(labels["event_frames"][n]); actor=int(labels["event_actor_indices"][n]); target=int(labels["event_target_indices"][n]); future=index+cfg.horizon_steps
                if index<1 or future>=len(frame["positions"]) or not frame["visible"][future,actor] or not frame["visible"][future,target]: continue
                slots=np.array([actor,target]); current=frame["positions"][index,slots].astype(np.float32); velocity=frame["velocities"][index,slots].astype(np.float32); acceleration=(frame["velocities"][index,slots]-frame["velocities"][index-1,slots]).astype(np.float32); features=np.r_[labels["event_features"][n],velocity.ravel(),acceleration.ravel()].astype(np.float32); future_pos=frame["positions"][future,slots].astype(np.float32)
                self.rows.append((current,velocity,features,future_pos))
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return tuple(torch.from_numpy(x) for x in self.rows[index])


def items():
    frames=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text())["matches"]; pressure=json.loads((ROOT/"data/frame_world/v84_pressure/manifest.json").read_text())["matches"]; lookup={x["match_id"]:x for x in pressure}
    return [{**x,"pressure_file":lookup[x["match_id"]]["file"]} for x in frames if x["provider"]=="skillcorner"]


def metric(pred,target):
    delta=pred-target; return torch.sqrt((delta[...,0]*105)**2+(delta[...,1]*68)**2)


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); sums={"conditioned":0.,"zero":0.,"shuffled_pair":0.,"n":0}
    for current,velocity,features,target in loader:
        current,velocity,features,target=[x.to(device) for x in (current,velocity,features,target)]; pred=model.predict(current,velocity,features); zero=model.predict(current,velocity,features,False); shuffled=model.predict(current,velocity,torch.roll(features,1,0)); sums["conditioned"]+=float(metric(pred,target).sum()); sums["zero"]+=float(metric(zero,target).sum()); sums["shuffled_pair"]+=float(metric(shuffled,target).sum()); sums["n"]+=target.shape[0]*2
    return {key:sums[key]/max(1,sums["n"]) for key in ("conditioned","zero","shuffled_pair")}|{"entities":sums["n"]}


def main():
    data=sorted(items(),key=lambda x:x["match_id"]); train,dev,test=data[:-2],data[-2:-1],data[-1:]; cfg=PressurePairConfig(); sets=[PressureEvents(x,cfg) for x in (train,dev,test)]; loaders=[DataLoader(x,batch_size=256,shuffle=i==0) for i,x in enumerate(sets)]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(8401); model=PressurePairTransition(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3); best=None; history=[]
    for epoch in range(12):
        model.train(); losses=[]
        for current,velocity,features,target in loaders[0]:
            current,velocity,features,target=[x.to(device) for x in (current,velocity,features,target)]; opt.zero_grad(set_to_none=True); pred=model.predict(current,velocity,features); loss=torch.nn.functional.smooth_l1_loss(pred,target,beta=.002); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss))
        metrics=evaluate(model,loaders[1],device); history.append({"epoch":epoch+1,"loss":sum(losses)/len(losses),"dev":metrics}); print(json.dumps(history[-1]),flush=True)
        if best is None or metrics["conditioned"]<best[0]: best=(metrics["conditioned"],{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); metrics=evaluate(model,loaders[2],device); artifact=ROOT/"data/frame_world/frame_world_v84_pressure_pair.pt"; payload=model.checkpoint(); payload["meta"]={"test_match":test[0]["match_id"]}; torch.save(payload,artifact); gates={"match_disjoint":not({x["match_id"] for x in train}&{test[0]["match_id"]}),"onset_gain":metrics["conditioned"]<metrics["zero"],"pair_specificity":metrics["conditioned"]<metrics["shuffled_pair"],"finite":all(np.isfinite(x) for x in metrics.values())}; report={"version":"8.4.0-candidate","split":{"train":[x["match_id"] for x in train],"dev":[x["match_id"] for x in dev],"test":[x["match_id"] for x in test]},"samples":{"train":len(sets[0]),"dev":len(sets[1]),"test":len(sets[2])},"metrics":metrics,"gates":gates,"ready":all(gates.values()),"artifact":artifact.relative_to(ROOT).as_posix(),"history":history}; target=ROOT/"reports/acceptance/frame_pressure_v84.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1


if __name__=="__main__": raise SystemExit(main())
