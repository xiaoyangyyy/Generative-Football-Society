#!/usr/bin/env python3
"""Train strong-label reception and next-action heads with match holdout."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_pass_triplet_v85 import feature_vector
from src.match_engine.frame_world.reception import NEXT_ACTIONS,ReceptionChainModel,ReceptionConfig
from src.match_engine.frame_world.schema import BALL_INDEX


class Chains(Dataset):
    def __init__(self,items,cfg):
        self.rows=[]
        for item in items:
            chains=[json.loads(line) for line in (ROOT/item["chain_file"]).read_text().splitlines()]
            with np.load(ROOT/item["triplet_file"],allow_pickle=False) as source: labels={key:source[key] for key in source.files}
            with np.load(ROOT/item["file"],allow_pickle=False) as source: frame={key:source[key] for key in ("positions","velocities")}
            assert len(chains)==len(labels["event_frames"])
            for n,chain in enumerate(chains):
                if not labels["valid"][n] or chain["outcome"] not in {"complete","turnover"}: continue
                index=int(labels["event_frames"][n]); slots=np.array([BALL_INDEX,int(labels["receiver_indices"][n]),int(labels["defender_indices"][n])]);
                if index<1: continue
                pos=frame["positions"][index,slots].astype(np.float32); vel=frame["velocities"][index,slots].astype(np.float32); acc=(frame["velocities"][index,slots]-frame["velocities"][index-1,slots]).astype(np.float32); features=feature_vector(pos,vel,acc,float(labels["lane_distance_m"][n])); outcome=np.float32(chain["outcome"]=="complete"); raw_action="terminal" if chain["next_action"] in {"loss","stoppage"} else chain["next_action"]; action=NEXT_ACTIONS.index(raw_action) if raw_action in NEXT_ACTIONS else -1; self.rows.append((features,outcome,np.int64(action)))
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return tuple(torch.from_numpy(x) if isinstance(x,np.ndarray) else torch.tensor(x) for x in self.rows[index])


def items():
    frames=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text())["matches"]; triplets=json.loads((ROOT/"data/frame_world/v85_pass_triplets/manifest.json").read_text())["matches"]; chains=json.loads((ROOT/"data/frame_world/v87_reception_chains/manifest.json").read_text())["matches"]; frame={(x["provider"],x["match_id"]):x for x in frames}; tri={(x["provider"],x["match_id"]):x for x in triplets}; return [{**frame[(x["provider"],x["match_id"])],"triplet_file":tri[(x["provider"],x["match_id"])]["file"],"chain_file":x["file"],"supervision":x["supervision"]} for x in chains if x["provider"]=="skillcorner"]


def scores(truth,pred,count):
    values=[]
    for c in range(count):
        tp=np.sum((pred==c)&(truth==c)); fp=np.sum((pred==c)&(truth!=c)); fn=np.sum((pred!=c)&(truth==c)); values.append(2*tp/max(1,2*tp+fp+fn))
    return float(np.mean(values)),values


@torch.no_grad()
def evaluate(model,loader,device,threshold=.5):
    model.eval(); ys=[]; ps=[]; actions=[]; guesses=[]
    for features,outcome,action in loader:
        completion,next_logits=model(features.to(device)); ys.extend(outcome.numpy()); ps.extend(torch.sigmoid(completion).cpu().numpy()); valid=action>=0; actions.extend(action[valid].numpy()); guesses.extend(next_logits[valid.to(device)].argmax(1).cpu().numpy())
    y=np.array(ys,dtype=int); p=np.array(ps); pred=p>=threshold; tpr=float(pred[y==1].mean()); tnr=float((~pred[y==0]).mean()); macro,per_class=scores(np.array(actions),np.array(guesses),len(NEXT_ACTIONS)); return {"balanced_accuracy":(tpr+tnr)/2,"tpr":tpr,"tnr":tnr,"next_action_macro_f1":macro,"next_action_f1":dict(zip(NEXT_ACTIONS,per_class)),"events":len(y),"next_action_events":len(actions)}


def main():
    data=sorted(items(),key=lambda x:x["match_id"]); train,dev,test=data[:-2],data[-2:-1],data[-1:]; cfg=ReceptionConfig(); sets=[Chains(x,cfg) for x in (train,dev,test)]; loaders=[DataLoader(x,batch_size=256,shuffle=i==0) for i,x in enumerate(sets)]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(8701); model=ReceptionChainModel(cfg).to(device); outcomes=np.array([int(row[1]) for row in sets[0].rows]); action_values=np.array([int(row[2]) for row in sets[0].rows]); pos_weight=torch.tensor([(outcomes==0).sum()/max(1,(outcomes==1).sum())],device=device); counts=np.array([max(1,(action_values==i).sum()) for i in range(len(NEXT_ACTIONS))]); action_weight=torch.tensor(len(action_values)/(len(NEXT_ACTIONS)*counts),dtype=torch.float32,device=device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3); best=None; history=[]
    for epoch in range(20):
        model.train(); losses=[]
        for features,outcome,action in loaders[0]:
            features,outcome,action=features.to(device),outcome.to(device),action.to(device); completion,next_logits=model(features); loss=torch.nn.functional.binary_cross_entropy_with_logits(completion,outcome,pos_weight=pos_weight); valid=action>=0
            if valid.any(): loss=loss+.5*torch.nn.functional.cross_entropy(next_logits[valid],action[valid],weight=action_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss))
        metrics=evaluate(model,loaders[1],device); score=metrics["balanced_accuracy"]+.25*metrics["next_action_macro_f1"]; history.append({"epoch":epoch+1,"loss":sum(losses)/len(losses),"dev":metrics}); print(json.dumps(history[-1]),flush=True)
        if best is None or score>best[0]: best=(score,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); metrics=evaluate(model,loaders[2],device); artifact=ROOT/"data/frame_world/reception_chain_v87_skillcorner.pt"; payload=model.checkpoint(); payload["meta"]={"test_match":test[0]["match_id"]}; torch.save(payload,artifact); gates={"match_disjoint":test[0]["match_id"] not in {x["match_id"] for x in train},"completion_ba":metrics["balanced_accuracy"]>.55,"next_action_macro_f1":metrics["next_action_macro_f1"]>.25,"finite":all(np.isfinite(x) for x in (metrics["balanced_accuracy"],metrics["next_action_macro_f1"]))}; report={"version":"8.7.0-candidate","provider_scope":"skillcorner","next_action_contract":["pass","shot","terminal"],"split":{"train":[x["match_id"] for x in train],"dev":[x["match_id"] for x in dev],"test":[x["match_id"] for x in test]},"samples":{"train":len(sets[0]),"dev":len(sets[1]),"test":len(sets[2])},"metrics":metrics,"gates":gates,"ready":all(gates.values()),"artifact":artifact.relative_to(ROOT).as_posix(),"history":history}; target=ROOT/"reports/acceptance/reception_chain_v87.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1


if __name__=="__main__": raise SystemExit(main())
