#!/usr/bin/env python3
"""Train right-censored event timing and hierarchical next-action marks."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_pass_triplet_v85 import feature_vector
from src.match_engine.frame_world.schema import BALL_INDEX
from src.match_engine.frame_world.temporal import TemporalMarkConfig,TemporalMarkModel,survival_loss


class TemporalEvents(Dataset):
    def __init__(self,items,cfg):
        self.rows=[]
        for item in items:
            chains=[json.loads(line) for line in (ROOT/item["chain_file"]).read_text().splitlines()]
            with np.load(ROOT/item["triplet_file"],allow_pickle=False) as source: labels={key:source[key] for key in source.files}
            with np.load(ROOT/item["file"],allow_pickle=False) as source: frame={key:source[key] for key in ("positions","velocities")}
            for n,chain in enumerate(chains):
                if not labels["valid"][n]: continue
                index=int(labels["event_frames"][n]); slots=np.array([BALL_INDEX,int(labels["receiver_indices"][n]),int(labels["defender_indices"][n])])
                if index<1: continue
                pos=frame["positions"][index,slots].astype(np.float32); vel=frame["velocities"][index,slots].astype(np.float32); acc=(frame["velocities"][index,slots]-frame["velocities"][index-1,slots]).astype(np.float32); features=feature_vector(pos,vel,acc,float(labels["lane_distance_m"][n])); action=chain["next_action"]; observed=action in {"pass","shot","loss","stoppage"} and chain["delay_s"] is not None and chain["delay_s"]<=5; event_bin=min(cfg.bins-1,max(0,int(np.ceil(float(chain["delay_s"])/cfg.bin_seconds))-1)) if observed else cfg.bins-1; continuation=1 if action in {"pass","shot"} else 0 if action in {"loss","stoppage"} else -1; subtype=1 if action=="shot" else 0 if action=="pass" else -1; self.rows.append((features,np.int64(event_bin),np.bool_(observed),np.int64(continuation),np.int64(subtype),np.float32(min(float(chain["delay_s"] or 5),5))))
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        return tuple(torch.from_numpy(x) if isinstance(x,np.ndarray) else torch.tensor(x.item() if isinstance(x,np.generic) else x) for x in self.rows[index])


def data_items():
    frames=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text())["matches"]; triplets=json.loads((ROOT/"data/frame_world/v85_pass_triplets/manifest.json").read_text())["matches"]; chains=json.loads((ROOT/"data/frame_world/v87_reception_chains/manifest.json").read_text())["matches"]; frame={(x["provider"],x["match_id"]):x for x in frames}; tri={(x["provider"],x["match_id"]):x for x in triplets}; return [{**frame[(x["provider"],x["match_id"])],"triplet_file":tri[(x["provider"],x["match_id"])]["file"],"chain_file":x["file"]} for x in chains if x["provider"]=="skillcorner"]


@torch.no_grad()
def evaluate(model,loader,device,continue_threshold=.5,subtype_threshold=.5):
    model.eval(); predicted_time=[]; true_time=[]; continuation=[]; continuation_guess=[]; subtype=[]; subtype_guess=[]; nll=[]; brier=[]
    for features,bins,observed,cont,sub,delay in loader:
        features,bins,observed=features.to(device),bins.to(device),observed.to(device); hazard,cont_logits,sub_logits=model(features); nll.append(float(survival_loss(hazard,bins,observed))*len(features)); predicted_bins,kinds=model.predict_mark(features,continue_threshold,subtype_threshold); mask=observed.cpu().numpy(); predicted_time.extend((predicted_bins.cpu().numpy()[mask]*model.cfg.bin_seconds).tolist()); true_time.extend(delay.numpy()[mask].tolist()); cvalid=cont>=0; continuation.extend(cont[cvalid].numpy()); continuation_guess.extend((kinds.cpu()[cvalid]!=2).numpy()); svalid=sub>=0; subtype.extend(sub[svalid].numpy()); subtype_guess.extend((kinds.cpu()[svalid]==1).numpy()); observed_2s=(observed&(bins<4)).float(); event_2s=1-torch.cumprod(1-torch.sigmoid(hazard),1)[:,3]; brier.extend((event_2s-observed_2s).square().cpu().numpy())
    def ba(y,p): y=np.array(y,dtype=bool); p=np.array(p,dtype=bool); return float((p[y].mean()+(~p[~y]).mean())/2)
    return {"time_mae_s":float(np.mean(np.abs(np.array(predicted_time)-np.array(true_time)))),"survival_nll":sum(nll)/sum(len(x[0]) for x in loader),"event_2s_brier":float(np.mean(brier)),"continue_balanced_accuracy":ba(continuation,continuation_guess),"pass_shot_balanced_accuracy":ba(subtype,subtype_guess),"observed_events":len(true_time)}

@torch.no_grad()
def calibrate_thresholds(model,loader,device):
    model.eval(); continuation=[]; continuation_prob=[]; subtype=[]; subtype_prob=[]
    for features,_,_,cont,sub,_ in loader:
        _,cont_logits,sub_logits=model(features.to(device)); cvalid=cont>=0; continuation.extend(cont[cvalid].numpy()); continuation_prob.extend(torch.sigmoid(cont_logits.cpu()[cvalid]).numpy()); svalid=sub>=0; subtype.extend(sub[svalid].numpy()); subtype_prob.extend(torch.sigmoid(sub_logits.cpu()[svalid]).numpy())
    def choose(y,p):
        y=np.array(y,dtype=bool); p=np.array(p); values=np.linspace(.05,.95,181); return float(max(values,key=lambda t:((p[y]>=t).mean()+(p[~y]<t).mean())/2))
    return choose(continuation,continuation_prob),choose(subtype,subtype_prob)


def main():
    data=sorted(data_items(),key=lambda x:x["match_id"]); train,dev,test=data[:-2],data[-2:-1],data[-1:]; cfg=TemporalMarkConfig(); sets=[TemporalEvents(x,cfg) for x in (train,dev,test)]; loaders=[DataLoader(x,batch_size=256,shuffle=i==0) for i,x in enumerate(sets)]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(8801); model=TemporalMarkModel(cfg).to(device); train_rows=sets[0].rows; cont=np.array([x[3] for x in train_rows]); sub=np.array([x[4] for x in train_rows]); cont_weight=torch.tensor([(cont==0).sum()/max(1,(cont==1).sum())],device=device); sub_weight=torch.tensor([(sub==0).sum()/max(1,(sub==1).sum())],device=device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3); best=None; history=[]
    for epoch in range(20):
        model.train(); losses=[]
        for features,bins,observed,continuation,subtype,_ in loaders[0]:
            features,bins,observed,continuation,subtype=features.to(device),bins.to(device),observed.to(device),continuation.to(device),subtype.to(device); hazard,cont_logits,sub_logits=model(features); loss=survival_loss(hazard,bins,observed); valid=continuation>=0
            if valid.any(): loss=loss+.25*torch.nn.functional.binary_cross_entropy_with_logits(cont_logits[valid],continuation[valid].float(),pos_weight=cont_weight)
            valid=subtype>=0
            if valid.any(): loss=loss+.25*torch.nn.functional.binary_cross_entropy_with_logits(sub_logits[valid],subtype[valid].float(),pos_weight=sub_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss))
        metrics=evaluate(model,loaders[1],device); score=metrics["time_mae_s"]+.5*(1-metrics["continue_balanced_accuracy"])+.25*(1-metrics["pass_shot_balanced_accuracy"]); history.append({"epoch":epoch+1,"loss":sum(losses)/len(losses),"dev":metrics}); print(json.dumps(history[-1]),flush=True)
        if best is None or score<best[0]: best=(score,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); thresholds=calibrate_thresholds(model,loaders[1],device); metrics=evaluate(model,loaders[2],device,*thresholds); artifact=ROOT/"data/frame_world/temporal_mark_v88_skillcorner.pt"; payload=model.checkpoint(); payload["meta"]={"test_match":test[0]["match_id"],"continue_threshold":thresholds[0],"subtype_threshold":thresholds[1]}; torch.save(payload,artifact); gates={"match_disjoint":test[0]["match_id"] not in {x["match_id"] for x in train},"time_mae":metrics["time_mae_s"]<1.5,"continue_ba":metrics["continue_balanced_accuracy"]>.55,"pass_shot_ba":metrics["pass_shot_balanced_accuracy"]>.55,"calibration":metrics["event_2s_brier"]<.25,"finite":all(np.isfinite(x) for x in metrics.values())}; report={"version":"8.8.0-candidate","provider_scope":"skillcorner","bins":cfg.bins,"bin_seconds":cfg.bin_seconds,"thresholds":{"continue":thresholds[0],"pass_shot":thresholds[1]},"split":{"train":[x["match_id"] for x in train],"dev":[x["match_id"] for x in dev],"test":[x["match_id"] for x in test]},"samples":{"train":len(sets[0]),"dev":len(sets[1]),"test":len(sets[2])},"metrics":metrics,"gates":gates,"ready":all(gates.values()),"artifact":artifact.relative_to(ROOT).as_posix(),"history":history}; target=ROOT/"reports/acceptance/temporal_mark_v88.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1


if __name__=="__main__": raise SystemExit(main())
