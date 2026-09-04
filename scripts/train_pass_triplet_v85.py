#!/usr/bin/env python3
"""Strict provider-LODO training for joint pass triplet transitions."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.match_engine.frame_world.pass_triplet import PassTripletConfig,PassTripletTransition
from src.match_engine.frame_world.schema import BALL_INDEX


def metric_xy(value):
    out=value.copy(); out[...,0]*=105; out[...,1]*=68; return out


def feature_vector(pos,vel,acc,lane):
    ball,receiver,defender=pos; rel_receiver=metric_xy(receiver-ball)/50; rel_defender=metric_xy(defender-ball)/50; rel_pair=metric_xy(defender-receiver)/20; distance=np.linalg.norm(metric_xy(receiver-ball))/50; direction=rel_receiver/max(float(np.linalg.norm(rel_receiver)),1e-6); speed=metric_xy(vel).reshape(-1)/10; acceleration=metric_xy(acc).reshape(-1)/10
    return np.r_[rel_receiver,rel_defender,rel_pair,lane/10,speed,acceleration,direction,distance].astype(np.float32)


class PassTriplets(Dataset):
    def __init__(self,items,cfg):
        self.rows=[]
        for item in items:
            with np.load(ROOT/item["triplet_file"],allow_pickle=False) as source: labels={key:source[key] for key in source.files}
            with np.load(ROOT/item["file"],allow_pickle=False) as source: frame={key:source[key] for key in ("positions","velocities","visible")}
            for n in np.flatnonzero(labels["valid"]):
                index=int(labels["event_frames"][n]); future=index+cfg.horizon_steps; slots=np.array([BALL_INDEX,int(labels["receiver_indices"][n]),int(labels["defender_indices"][n])])
                if index<1 or future>=len(frame["positions"]) or not frame["visible"][future,slots].all(): continue
                current=frame["positions"][index,slots].astype(np.float32); velocity=frame["velocities"][index,slots].astype(np.float32); acceleration=(frame["velocities"][index,slots]-frame["velocities"][index-1,slots]).astype(np.float32); features=feature_vector(current,velocity,acceleration,float(labels["lane_distance_m"][n])); target=frame["positions"][future,slots].astype(np.float32); self.rows.append((current,velocity,features,target))
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return tuple(torch.from_numpy(x) for x in self.rows[index])


def joined_items():
    base=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text())["matches"]; corrected=json.loads((ROOT/"data/frame_world/v85_tracking/manifest.json").read_text())["matches"]; frames={(x["provider"],x["match_id"]):x for x in base}; frames.update({(x["provider"],x["match_id"]):x for x in corrected}); triplets=json.loads((ROOT/"data/frame_world/v85_pass_triplets/manifest.json").read_text())["matches"]
    return [{**frames[(x["provider"],x["match_id"])],"triplet_file":x["file"]} for x in triplets]


def split(items,held):
    test=[x for x in items if x["provider"]==held]; rest=[x for x in items if x["provider"]!=held and x["provider"] in {"metrica","skillcorner"}]; train=[]; dev=[]
    for provider in sorted({x["provider"] for x in rest}): group=sorted((x for x in rest if x["provider"]==provider),key=lambda x:x["match_id"]); train+=group[:-1]; dev+=group[-1:]
    return train,dev,test


def shuffle_features(features,role):
    output=features.clone(); rolled=torch.roll(features,1,0)
    indices=[0,1,4,5,9,10,15,16,19,20,21] if role=="receiver" else [2,3,4,5,6,11,12,17,18]
    output[:,indices]=rolled[:,indices]; return output


def errors(pred,target):
    delta=pred-target; return torch.sqrt((delta[...,0]*105)**2+(delta[...,1]*68)**2)


@torch.no_grad()
def evaluate(model,loader,device,intervention_enabled=True):
    model.eval(); sums={key:np.zeros(3,float) for key in ("conditioned","zero","receiver_shuffled","defender_shuffled")}; count=0
    for current,velocity,features,target in loader:
        current,velocity,features,target=[x.to(device) for x in (current,velocity,features,target)]; values={"conditioned":model.predict(current,velocity,features,intervention_enabled),"zero":model.predict(current,velocity,features,False),"receiver_shuffled":model.predict(current,velocity,shuffle_features(features,"receiver"),intervention_enabled),"defender_shuffled":model.predict(current,velocity,shuffle_features(features,"defender"),intervention_enabled)}
        for key,value in values.items(): sums[key]+=errors(value,target).sum(0).cpu().numpy(); count+=0
        count+=len(target)
    names=("ball","receiver","defender"); return {key:{name:float(sums[key][i]/max(1,count)) for i,name in enumerate(names)}|{"joint":float(sums[key].sum()/max(1,count*3))} for key in sums}|{"events":count}


def train_fold(held,items,args,device):
    train,dev,test=split(items,held); cfg=PassTripletConfig(); sets=[PassTriplets(x,cfg) for x in (train,dev,test)]; loaders=[DataLoader(x,batch_size=256,shuffle=i==0) for i,x in enumerate(sets)]; torch.manual_seed(8500+len(held)); model=PassTripletTransition(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3); best=None; history=[]
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for current,velocity,features,target in loaders[0]:
            current,velocity,features,target=[x.to(device) for x in (current,velocity,features,target)]; opt.zero_grad(set_to_none=True); prediction=model.predict(current,velocity,features); meter=prediction.new_tensor([105.,68.]); residual=(prediction-target)*meter; loss=torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.2); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss))
        metrics=evaluate(model,loaders[1],device); history.append({"epoch":epoch+1,"loss":sum(losses)/len(losses),"dev_joint":metrics["conditioned"]["joint"]}); print(json.dumps({"held":held,**history[-1]}),flush=True)
        if best is None or metrics["conditioned"]["joint"]<best[0]: best=(metrics["conditioned"]["joint"],{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); supported=held in {"metrica","skillcorner"}; metrics=evaluate(model,loaders[2],device,supported); artifact=ROOT/f"data/frame_world/frame_world_v85_pass_triplet_lodo_{held}.pt"; payload=model.checkpoint(); payload["meta"]={"held_provider":held,"intervention_enabled":supported}; torch.save(payload,artifact); conditioned=metrics["conditioned"]; zero=metrics["zero"]; gates={"strict_lodo":held not in {x["provider"] for x in train},"evidence_policy":supported or conditioned==zero,"joint_gain":not supported or conditioned["joint"]<zero["joint"],"ball_gain":not supported or conditioned["ball"]<zero["ball"],"receiver_noninferior":not supported or conditioned["receiver"]<=zero["receiver"]*1.05,"defender_noninferior":not supported or conditioned["defender"]<=zero["defender"]*1.05,"receiver_identity_specific":not supported or conditioned["joint"]<metrics["receiver_shuffled"]["joint"],"defender_identity_specific":not supported or conditioned["joint"]<metrics["defender_shuffled"]["joint"]}; return {"held_provider":held,"train_providers":sorted({x["provider"] for x in train}),"intervention_supported":supported,"samples":{"train":len(sets[0]),"dev":len(sets[1]),"test":len(sets[2])},"metrics":metrics,"gates":gates,"ready":all(gates.values()),"artifact":artifact.relative_to(ROOT).as_posix(),"history":history}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=12); args=parser.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); folds=[train_fold(x,joined_items(),args,device) for x in ("metrica","skillcorner","sportec")]; report={"version":"8.5.0-candidate","policy":"strict provider LODO joint ball/receiver/interceptor transition","folds":folds,"ready":all(x["ready"] for x in folds)}; target=ROOT/"reports/acceptance/frame_pass_triplet_v85.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1


if __name__=="__main__": raise SystemExit(main())
