#!/usr/bin/env python3
"""Match-held-out pressure recognition on SkillCorner strong labels."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_semantic_frame_world_v83 import EventWindows,evaluate,join_manifests,loss_batch
from src.match_engine.frame_world.semantic import SemanticActionFrameWorld,SemanticFrameConfig


def main():
    items=sorted((x for x in join_manifests() if x["provider"]=="skillcorner"),key=lambda x:x["match_id"]); train,dev,test=items[:-2],items[-2:-1],items[-1:]; cfg=SemanticFrameConfig(pressure_enabled=True,actor_roles_enabled=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets=[EventWindows(x,cfg,n,s) for x,n,s in ((train,30000,51),(dev,12000,52),(test,20000,53))]; loaders=[DataLoader(x,batch_size=256,shuffle=i==0,num_workers=0) for i,x in enumerate(sets)]; torch.manual_seed(8351); model=SemanticActionFrameWorld(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3); best=None; history=[]
    for epoch in range(5):
        model.train(); losses=[]
        for batch in loaders[0]: opt.zero_grad(set_to_none=True); loss=loss_batch(model,batch,device); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss))
        metrics=evaluate(model,loaders[1],device); history.append({"epoch":epoch+1,"loss":sum(losses)/len(losses),"dev":metrics}); print(json.dumps(history[-1]),flush=True)
        score=metrics["pressure_local_conditioned"] if metrics["pressure_local_entities"] else float("inf")
        if best is None or score<best[0]: best=(score,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); metrics=evaluate(model,loaders[2],device); artifact=ROOT/"data/frame_world/frame_world_v83_pressure_skillcorner.pt"; payload=model.checkpoint(); payload["meta"]={"test_match":test[0]["match_id"]}; torch.save(payload,artifact)
    gates={"match_disjoint":not ({x["match_id"] for x in train}&{test[0]["match_id"]}),"pressure_f1":metrics["f1"][2]>=.5,"local_transition_gain":metrics["pressure_local_entities"]>=20 and metrics["pressure_local_conditioned"]<metrics["pressure_local_zero"],"finite":all(torch.isfinite(torch.tensor(x)) for x in metrics["f1"])}; report={"version":"8.3.0-candidate","policy":"SkillCorner match-held-out actor-aware strong pressure labels","split":{"train":[x["match_id"] for x in train],"dev":[x["match_id"] for x in dev],"test":[x["match_id"] for x in test]},"metrics":metrics,"gates":gates,"ready":all(gates.values()),"artifact":artifact.relative_to(ROOT).as_posix(),"history":history}; target=ROOT/"reports/acceptance/frame_pressure_v83.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())
