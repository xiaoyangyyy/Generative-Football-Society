#!/usr/bin/env python3
"""Counterfactual and closed-loop acceptance for the v8.8 temporal router."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_temporal_mark_v88 import TemporalEvents,data_items
from src.match_engine.frame_world.pass_triplet import load_pass_triplet
from src.match_engine.frame_world.router import ActionRequest,ActionTransitionRouter,FrameContext
from src.match_engine.frame_world.temporal import TemporalMarkConfig,load_temporal_mark


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def meters(delta): return np.sqrt((delta[...,0]*105)**2+(delta[...,1]*68)**2)


def main():
    temporal_path=ROOT/"data/frame_world/temporal_mark_v88_skillcorner.pt"; pass_path=ROOT/"data/frame_world/frame_world_v85_pass_triplet_lodo_skillcorner.pt"
    model,cfg,meta=load_temporal_mark(temporal_path); model.eval(); pass_model=load_pass_triplet(pass_path)[0].eval(); thresholds=(float(meta["continue_threshold"]),float(meta["subtype_threshold"])); router=ActionTransitionRouter(pass_models={"skillcorner":pass_model},temporal_models={"skillcorner":model},temporal_thresholds={"skillcorner":thresholds})
    item=sorted(data_items(),key=lambda x:x["match_id"])[-1]; dataset=TemporalEvents([item],TemporalMarkConfig()); features=torch.stack([dataset[n][0] for n in range(len(dataset))])
    generator=torch.Generator().manual_seed(8802); permutation=torch.randperm(len(features),generator=generator)
    with torch.no_grad():
        factual_steps,factual_kinds=model.predict_mark(features,*thresholds); shuffled_steps,_=model.predict_mark(features[permutation],*thresholds); _,shuffled_kinds=model.predict_mark(features[permutation],*thresholds)
    time_change=(factual_steps!=shuffled_steps).float(); kind_change=(factual_kinds!=shuffled_kinds).float()
    with np.load(ROOT/item["file"],allow_pickle=False) as source: frame={key:source[key] for key in source.files}
    with np.load(ROOT/item["triplet_file"],allow_pickle=False) as source: labels={key:source[key] for key in source.files}
    samples=[]
    for n in np.flatnonzero(labels["valid"]):
        index=int(labels["event_frames"][n])
        if index<4: continue
        actor=int(labels["actor_indices"][n]); receiver=int(labels["receiver_indices"][n]); defender=int(labels["defender_indices"][n])
        state=FrameContext(frame["positions"][index-4:index+1].astype(np.float32),frame["velocities"][index-4:index+1].astype(np.float32),frame["visible"][index-4:index+1].astype(bool),frame["teams"].astype(np.int8)); action=ActionRequest("pass","skillcorner",actor,receiver,defender)
        driven,plan,_=router.rollout_event_driven(state,action,cfg.bins); persistent,_=router.rollout(state,cfg.bins,{0:[action]})
        deleted_visible=state.visible.copy(); deleted_visible[-1,receiver]=False; deleted=FrameContext(state.positions,state.velocities,deleted_visible,state.teams)
        samples.append({"finite":bool(np.isfinite(driven).all()),"bounded":bool(driven.min()>=-.150001 and driven.max()<=1.150001),"delay_steps":plan.delay_steps if plan else -1,"termination_effect_m":float(meters(driven[-1,[32,receiver,defender]]-persistent[-1,[32,receiver,defender]]).mean()),"identity_deleted":router.plan_next_mark(deleted,action) is None})
        if len(samples)>=64: break
    summary={"test_match":item["match_id"],"feature_samples":len(features),"rollout_samples":len(samples),"time_shuffle_changed_fraction":float(time_change.mean()),"time_shuffle_mean_step_delta":float((factual_steps-shuffled_steps).abs().float().mean()),"action_shuffle_changed_fraction":float(kind_change.mean()),"mean_termination_effect_m":float(np.mean([x["termination_effect_m"] for x in samples])),"delay_range_steps":[min(x["delay_steps"] for x in samples),max(x["delay_steps"] for x in samples)],"finite":all(x["finite"] for x in samples),"bounded":all(x["bounded"] for x in samples),"identity_deletion_blocks":all(x["identity_deleted"] for x in samples)}
    gates={"held_match":str(item["match_id"])==str(meta["test_match"]),"time_condition_used":summary["time_shuffle_changed_fraction"]>.2 and summary["time_shuffle_mean_step_delta"]>.25,"action_condition_used":summary["action_shuffle_changed_fraction"]>.1,"effect_termination":summary["mean_termination_effect_m"]>.01,"identity_gate":summary["identity_deletion_blocks"],"valid_schedule":summary["delay_range_steps"][0]>=1 and summary["delay_range_steps"][1]<=cfg.bins,"finite_bounded":summary["finite"] and summary["bounded"]}
    report={"version":"8.8.0-candidate","provider_scope":"skillcorner","counterfactual_note":"Feature rows are permuted only at the selected decoder, preserving the other factual output for separate timing and mark tests.","summary":summary,"gates":gates,"ready":all(gates.values())}; report_path=ROOT/"reports/acceptance/temporal_router_v88.json"; report_path.write_text(json.dumps(report,indent=2)+"\n")
    dependencies=[temporal_path,pass_path,ROOT/"data/frame_world/reception_chain_v87_skillcorner.pt"]; manifest={"version":"8.8.0-candidate","router":"src/match_engine/frame_world/router.py","provider_scope":"skillcorner","dependencies":[{"artifact":x.relative_to(ROOT).as_posix(),"sha256":sha256(x)} for x in dependencies],"evidence":[report_path.relative_to(ROOT).as_posix(),"reports/acceptance/temporal_mark_v88.json"],"ready":report["ready"]}; (ROOT/"data/frame_world/temporal_router_v88.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1


if __name__=="__main__": raise SystemExit(main())
