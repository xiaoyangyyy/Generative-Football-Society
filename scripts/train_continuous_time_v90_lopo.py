#!/usr/bin/env python3
"""Train and evaluate strict provider-LODO continuous-time survival models."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_temporal_mark_v89_lopo import Arrays
from src.match_engine.frame_world.continuous import ContinuousTimeConfig,ContinuousTimeMarkModel,mixture_survival_loss


def ba(y,p):
    y=np.asarray(y,dtype=bool); p=np.asarray(p,dtype=bool); return float(((p[y]).mean()+(~p[~y]).mean())/2)


@torch.no_grad()
def evaluate(model,loader,device,ct=.5,st=.5):
    model.eval(); times=[]; truth=[]; cont=[]; cp=[]; sub=[]; sp=[]; losses=[]; brier=[]; count=0
    for features,_,observed,continuation,subtype,delay in loader:
        features,observed,delay=features.to(device),observed.to(device),delay.to(device); weights,location,scale,clog,slog=model(features); losses.append(float(mixture_survival_loss(weights,location,scale,delay,observed))*len(features)); count+=len(features); prediction=model.median_time(features); mask=observed.cpu().numpy(); times.extend(prediction.cpu().numpy()[mask]); truth.extend(delay.cpu().numpy()[mask]); cv=continuation>=0; cont.extend(continuation[cv].numpy()); cp.extend((torch.sigmoid(clog.cpu()[cv])>=ct).numpy()); sv=subtype>=0; sub.extend(subtype[sv].numpy()); sp.extend((torch.sigmoid(slog.cpu()[sv])>=st).numpy()); z=(np.log(2)-location.cpu().numpy())/scale.cpu().numpy(); event2=(torch.softmax(weights,1).cpu().numpy()*(.5*(1+torch.erf(torch.from_numpy(z)/np.sqrt(2))).numpy())).sum(1); actual=(observed.cpu().numpy()&(delay.cpu().numpy()<=2)).astype(float); brier.extend((event2-actual)**2)
    return {"time_mae_s":float(np.mean(np.abs(np.asarray(times)-np.asarray(truth)))),"survival_nll":sum(losses)/count,"event_2s_brier":float(np.mean(brier)),"continue_balanced_accuracy":ba(cont,cp),"pass_shot_balanced_accuracy":ba(sub,sp),"observed_events":len(times)}


@torch.no_grad()
def thresholds(model,loader,device):
    model.eval(); ys=[[],[]]; ps=[[],[]]
    for features,_,_,continuation,subtype,_ in loader:
        *_,clog,slog=model(features.to(device))
        for i,(y,logit) in enumerate(((continuation,clog.cpu()),(subtype,slog.cpu()))):
            valid=y>=0; ys[i].extend(y[valid].numpy()); ps[i].extend(torch.sigmoid(logit[valid]).numpy())
    output=[]
    for y,p in zip(ys,ps):
        y=np.asarray(y,dtype=bool); p=np.asarray(p); choices=np.linspace(.02,.98,193); output.append(float(max(choices,key=lambda x:ba(y,p>=x))))
    return tuple(output)


def fold(held,manifest,device,epochs):
    source=[x for x in manifest["matches"] if x["provider"]!=held]; target=[x for x in manifest["matches"] if x["provider"]==held]; ids=sorted({x["match_id"] for x in source}); dev_ids=set(ids[-max(1,len(ids)//10):]); train=Arrays([x for x in source if x["match_id"] not in dev_ids],120000,9001); dev=Arrays([x for x in source if x["match_id"] in dev_ids],20000,9002); test=Arrays(target,100000,9003); loaders=[DataLoader(x,batch_size=512,shuffle=i==0) for i,x in enumerate((train,dev,test))]; torch.manual_seed(9000+(held=="statsbomb360")); model=ContinuousTimeMarkModel(ContinuousTimeConfig()).to(device); cont=train.values["continuation"]; sub=train.values["subtype"]; cw=torch.tensor([(cont==0).sum()/max(1,(cont==1).sum())],device=device); sw=torch.tensor([(sub==0).sum()/max(1,(sub==1).sum())],device=device); opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=3e-3); best=None; history=[]
    for epoch in range(epochs):
        model.train(); values=[]
        for features,_,observed,continuation,subtype,delay in loaders[0]:
            features,observed,continuation,subtype,delay=features.to(device),observed.to(device),continuation.to(device),subtype.to(device),delay.to(device); weights,location,scale,clog,slog=model(features); loss=mixture_survival_loss(weights,location,scale,delay,observed); valid=continuation>=0
            if valid.any(): loss=loss+.2*torch.nn.functional.binary_cross_entropy_with_logits(clog[valid],continuation[valid].float(),pos_weight=cw)
            valid=subtype>=0
            if valid.any(): loss=loss+.05*torch.nn.functional.binary_cross_entropy_with_logits(slog[valid],subtype[valid].float(),pos_weight=sw)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); values.append(float(loss))
        metrics=evaluate(model,loaders[1],device); score=metrics["time_mae_s"]+.25*(1-metrics["continue_balanced_accuracy"]); history.append({"epoch":epoch+1,"loss":float(np.mean(values)),"dev":metrics}); print(json.dumps({"held":held,**history[-1]}),flush=True)
        if best is None or score<best[0]: best=(score,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); cut=thresholds(model,loaders[1],device); metrics=evaluate(model,loaders[2],device,*cut); tr=train.values["delay"][train.values["observed"]]; te=test.values["delay"][test.values["observed"]]; baseline=float(np.mean(np.abs(te-np.median(tr)))); artifact=ROOT/f"data/frame_world/continuous_time_v90_lopo_{held}.pt"; payload=model.checkpoint(); payload["meta"]={"held_provider":held,"train_providers":sorted({x["provider"] for x in source}),"continue_threshold":cut[0],"subtype_threshold":cut[1],"feature_contract":"v89_static_mirror_invariant"}; torch.save(payload,artifact); gates={"strict_provider_holdout":held not in {x["provider"] for x in source},"continuous_time_gain":metrics["time_mae_s"]<baseline,"continue_gain":metrics["continue_balanced_accuracy"]>.5,"mark_gain":metrics["pass_shot_balanced_accuracy"]>.5,"calibrated":metrics["event_2s_brier"]<.3,"finite":all(np.isfinite(x) for x in metrics.values())}; return {"held_provider":held,"train_providers":sorted({x["provider"] for x in source}),"samples":{"train":len(train),"dev":len(dev),"test":len(test)},"thresholds":{"continue":cut[0],"pass_shot":cut[1]},"metrics":metrics,"continuous_train_median_baseline_mae_s":baseline,"time_gain_s":baseline-metrics["time_mae_s"],"gates":gates,"ready":all(gates.values()),"artifact":artifact.relative_to(ROOT).as_posix(),"history":history}


def main():
    manifest=json.loads((ROOT/"data/frame_world/v89_temporal_providers/manifest.json").read_text()); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); folds=[fold(x,manifest,device,15) for x in ("skillcorner","statsbomb360")]; report={"version":"9.0.0-candidate","model":"three-component log-normal mixture survival","policy":"strict bidirectional provider LODO; continuous baseline gate","folds":folds,"ready":all(x["ready"] for x in folds)}; target=ROOT/"reports/acceptance/continuous_time_v90_lopo.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())
