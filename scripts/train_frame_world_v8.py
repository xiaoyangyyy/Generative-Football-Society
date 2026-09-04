#!/usr/bin/env python3
"""Train and evaluate the v8 frame-level graph-temporal world model."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.match_engine.frame_world.model import FrameWorldConfig,GraphTemporalFrameWorldModel
from src.match_engine.frame_world.schema import BALL_INDEX

class Windows(Dataset):
    def __init__(self,items,cfg,limit,seed):
        self.cfg=cfg; self.data=[]; self.index=[]
        for item in items:
            raw=np.load(ROOT/item["file"],allow_pickle=False); seq={k:raw[k] for k in ("timestamps","positions","velocities","visible","teams","possession")}; self.data.append(seq)
            max_h=max(cfg.horizons); valid=np.flatnonzero((seq["timestamps"][max_h:]-seq["timestamps"][:-max_h])<3.0)+cfg.history-1
            valid=valid[(valid>=cfg.history-1)&(valid+max_h<len(seq["timestamps"]))]
            valid=valid[seq["visible"][valid].sum(axis=1)>=4]
            self.index.extend((len(self.data)-1,int(i)) for i in valid[::5])
        rng=random.Random(seed); rng.shuffle(self.index); self.index=self.index[:limit]
    def __len__(self): return len(self.index)
    def __getitem__(self,n):
        s,i=self.index[n]; d=self.data[s]; start=i-self.cfg.history+1; target=np.array([i+h for h in self.cfg.horizons])
        return tuple(torch.from_numpy(x) for x in (d["positions"][start:i+1].astype(np.float32),d["velocities"][start:i+1].astype(np.float32),d["visible"][start:i+1].astype(bool),d["teams"].astype(np.int64),d["positions"][target].astype(np.float32),d["visible"][target].astype(bool),d["possession"][target].astype(np.int64)))
def split(manifest):
    groups={p:[] for p in manifest["providers"]}
    for item in manifest["matches"]: groups[item["provider"]].append(item)
    train=[];dev=[];test=[]
    for provider,items in groups.items():
        items=sorted(items,key=lambda x:x["match_id"])
        if provider=="metrica": train+=items[:1]; test+=items[1:]
        else: train+=items[:-2]; dev+=items[-2:-1]; test+=items[-1:]
    return train,dev,test
def loss_batch(model,batch,device):
    pos,vel,vis,teams,target,target_vis,poss=[x.to(device) for x in batch]; mean,logvar,poss_logits=model(pos,vel,vis,teams)
    horizons=torch.tensor(model.cfg.horizons,device=device).float()/model.cfg.sample_rate_hz
    baseline=pos[:,-1,None]+vel[:,-1,None]*horizons[None,:,None,None]; prediction=baseline+mean
    mask=target_vis&vis[:,-1,None]; error=prediction-target
    nll=.5*(error.square()*torch.exp(-logvar)+logvar); position=(nll*mask[...,None]).sum()/mask.sum().clamp_min(1)/2
    valid_poss=poss>=0
    possession=torch.nn.functional.cross_entropy(poss_logits[:,None].expand(-1,len(model.cfg.horizons),-1)[valid_poss],poss[valid_poss]+1) if valid_poss.any() else position*0
    return position+.05*possession,prediction,baseline,error,logvar,mask
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); sums={"n":0,"model":0.,"base":0.,"ball_n":0,"ball":0.,"ball_base":0.,"cover50":0.,"cover90":0.}; by_h=[{"n":0,"m":0.,"b":0.} for _ in model.cfg.horizons]
    for batch in loader:
        _,pred,base,error,logvar,mask=loss_batch(model,batch,device); target=batch[4].to(device); dist=lambda x:torch.sqrt((x[...,0]*105)**2+(x[...,1]*68)**2)
        md,bd=dist(pred-target),dist(base-target); standardized=torch.sqrt((error.square()/torch.exp(logvar)).sum(-1))
        sums["n"]+=int(mask.sum()); sums["model"]+=float((md*mask).sum()); sums["base"]+=float((bd*mask).sum()); sums["cover50"]+=float(((standardized<=1.177)*mask).sum()); sums["cover90"]+=float(((standardized<=2.146)*mask).sum())
        bm=mask[:,:,BALL_INDEX]; sums["ball_n"]+=int(bm.sum()); sums["ball"]+=float((md[:,:,BALL_INDEX]*bm).sum()); sums["ball_base"]+=float((bd[:,:,BALL_INDEX]*bm).sum())
        for i in range(len(by_h)):
            by_h[i]["n"]+=int(mask[:,i].sum()); by_h[i]["m"]+=float((md[:,i]*mask[:,i]).sum()); by_h[i]["b"]+=float((bd[:,i]*mask[:,i]).sum())
    return {"ade_m":sums["model"]/sums["n"],"baseline_ade_m":sums["base"]/sums["n"],"ball_ade_m":sums["ball"]/max(1,sums["ball_n"]),"ball_baseline_ade_m":sums["ball_base"]/max(1,sums["ball_n"]),"coverage_50":sums["cover50"]/sums["n"],"coverage_90":sums["cover90"]/sums["n"],"horizons":{str(h/model.cfg.sample_rate_hz):{"ade_m":x["m"]/x["n"],"baseline_ade_m":x["b"]/x["n"]} for h,x in zip(model.cfg.horizons,by_h)}}
@torch.no_grad()
def calibrate_variance(model,loader,device):
    model.eval(); values=[[] for _ in model.cfg.horizons]
    for batch in loader:
        _,_,_,error,logvar,mask=loss_batch(model,batch,device)
        radial=torch.sqrt((error.square()/torch.exp(logvar)).sum(-1))
        for i in range(len(values)): values[i].append(radial[:,i][mask[:,i]].cpu())
    scales=[]
    for chunks in values:
        radial=torch.cat(chunks)
        median=float(torch.quantile(radial,.5))/1.177
        tail=float(torch.quantile(radial,.85))/2.146
        scales.append(max(median,tail,1e-3))
    offsets=2*torch.log(torch.tensor(scales,device=device)); model.variance_log_offset.copy_(offsets)
    return {"fit_split":"dev","target_coverages":[.5,.85],"scales":scales,"log_variance_offsets":offsets.cpu().tolist()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=6); ap.add_argument("--train-windows",type=int,default=100000); args=ap.parse_args()
    torch.manual_seed(20260725); np.random.seed(20260725); manifest=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text()); train,dev,test=split(manifest); cfg=FrameWorldConfig(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets=[Windows(items,cfg,limit,seed) for items,limit,seed in ((train,args.train_windows,1),(dev,20000,2),(test,30000,3))]; loaders=[DataLoader(d,batch_size=256,shuffle=i==0,num_workers=0,pin_memory=device.type=="cuda") for i,d in enumerate(datasets)]
    model=GraphTemporalFrameWorldModel(cfg).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); best=None; history=[]
    for epoch in range(args.epochs):
        model.train(); total=0
        for batch in loaders[0]: optimizer.zero_grad(set_to_none=True); loss,*_=loss_batch(model,batch,device); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); total+=float(loss)
        dev_metrics=evaluate(model,loaders[1],device); history.append({"epoch":epoch+1,"loss":total/len(loaders[0]),"dev":dev_metrics}); print(json.dumps(history[-1]),flush=True)
        if best is None or dev_metrics["ade_m"]<best[0]: best=(dev_metrics["ade_m"],{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); calibration=calibrate_variance(model,loaders[1],device); dev_calibrated=evaluate(model,loaders[1],device); test_metrics=evaluate(model,loaders[2],device)
    provider_test={}
    for item in test:
        dataset=Windows([item],cfg,10000,4); provider_test[item["provider"]]=evaluate(model,DataLoader(dataset,batch_size=256,num_workers=0),device)
    gates={"beats_constant_velocity":test_metrics["ade_m"]<=test_metrics["baseline_ade_m"]*.97,"long_horizon_gain":test_metrics["horizons"]["2.0"]["ade_m"]<test_metrics["horizons"]["2.0"]["baseline_ade_m"],"uncertainty_50":.40<=test_metrics["coverage_50"]<=.65,"uncertainty_90":.80<=test_metrics["coverage_90"]<=.97,"three_providers":len(manifest["providers"])==3,"all_provider_holdouts":all(x["ade_m"]<x["baseline_ade_m"] for x in provider_test.values())}
    report={"version":"8.0.0-candidate","device":str(device),"windows":{"train":len(datasets[0]),"dev":len(datasets[1]),"test":len(datasets[2])},"split":{"train":[x["provider"]+":"+x["match_id"] for x in train],"dev":[x["provider"]+":"+x["match_id"] for x in dev],"test":[x["provider"]+":"+x["match_id"] for x in test]},"calibration":calibration,"dev_calibrated":dev_calibrated,"test":test_metrics,"provider_test":provider_test,"gates":gates,"candidate_ready":all(gates.values()),"history":history}
    output=ROOT/"data/frame_world/frame_world_v8_candidate.pt"; payload=model.checkpoint(); payload["meta"]={"report":"reports/acceptance/frame_world_v8.json","data_manifest":"data/frame_world/v8/manifest.json"}; torch.save(payload,output)
    target=ROOT/"reports/acceptance/frame_world_v8.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["candidate_ready"] else 1
if __name__=="__main__": raise SystemExit(main())
