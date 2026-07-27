#!/usr/bin/env python3
"""Counterfactual router acceptance for v9.0 continuous schedules."""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_temporal_mark_v88 import data_items
from scripts.train_temporal_mark_v89_lopo import Arrays
from src.match_engine.frame_world.continuous import load_continuous_time
from src.match_engine.frame_world.pass_triplet import load_pass_triplet
from src.match_engine.frame_world.router import ActionRequest,ActionTransitionRouter,FrameContext


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    model_path=ROOT/"data/frame_world/continuous_time_v90_lopo_skillcorner.pt"; pass_path=ROOT/"data/frame_world/frame_world_v85_pass_triplet_lodo_skillcorner.pt"; model,_,meta=load_continuous_time(model_path); model.eval(); pass_model=load_pass_triplet(pass_path)[0].eval(); cuts=(float(meta["continue_threshold"]),float(meta["subtype_threshold"])); router=ActionTransitionRouter(pass_models={"skillcorner":pass_model},temporal_models={"skillcorner":model},temporal_thresholds={"skillcorner":cuts})
    manifest=json.loads((ROOT/"data/frame_world/v89_temporal_providers/manifest.json").read_text()); held=[x for x in manifest["matches"] if x["provider"]=="skillcorner"]; data=Arrays(held); features=torch.from_numpy(data.values["features"]); perm=torch.randperm(len(features),generator=torch.Generator().manual_seed(9004))
    with torch.no_grad(): factual=model.median_time(features); shuffled=model.median_time(features[perm])
    non_grid=((factual/.5-(factual/.5).round()).abs()>1e-3).float(); item=sorted(data_items(),key=lambda x:x["match_id"])[-1]
    with np.load(ROOT/item["file"],allow_pickle=False) as source: frame={k:source[k] for k in source.files}
    with np.load(ROOT/item["triplet_file"],allow_pickle=False) as source: labels={k:source[k] for k in source.files}
    samples=[]
    for n in np.flatnonzero(labels["valid"]):
        index=int(labels["event_frames"][n])
        if index<4: continue
        actor=int(labels["actor_indices"][n]); receiver=int(labels["receiver_indices"][n]); defender=int(labels["defender_indices"][n]); state=FrameContext(frame["positions"][index-4:index+1].astype(np.float32),frame["velocities"][index-4:index+1].astype(np.float32),frame["visible"][index-4:index+1].astype(bool),frame["teams"].astype(np.int8)); action=ActionRequest("pass","skillcorner",actor,receiver,defender); rollout,plan,_=router.rollout_event_driven(state,action,10); visibility=state.visible.copy(); visibility[-1,receiver]=False; deleted=FrameContext(state.positions,state.velocities,visibility,state.teams); samples.append({"finite":bool(np.isfinite(rollout).all()),"bounded":bool(rollout.min()>=-.150001 and rollout.max()<=1.150001),"seconds":plan.delay_seconds,"step_consistent":plan.delay_steps==max(1,math.ceil(plan.delay_seconds/.5)),"identity_deleted":router.plan_next_mark(deleted,action) is None})
        if len(samples)>=64: break
    training=json.loads((ROOT/"reports/acceptance/continuous_time_v90_lopo.json").read_text()); summary={"feature_samples":len(features),"rollout_samples":len(samples),"mean_time_shuffle_delta_s":float((factual-shuffled).abs().mean()),"non_grid_fraction":float(non_grid.mean()),"delay_range_s":[min(x["seconds"] for x in samples),max(x["seconds"] for x in samples)],"finite":all(x["finite"] for x in samples),"bounded":all(x["bounded"] for x in samples),"step_consistent":all(x["step_consistent"] for x in samples),"identity_deletion_blocks":all(x["identity_deleted"] for x in samples)}; gates={"training_lodo_ready":training["ready"],"continuous_gain_both_folds":all(x["gates"]["continuous_time_gain"] for x in training["folds"]),"time_condition_used":summary["mean_time_shuffle_delta_s"]>.01,"not_discrete_grid":summary["non_grid_fraction"]>.9,"execution_boundary":summary["step_consistent"],"identity_gate":summary["identity_deletion_blocks"],"finite_bounded":summary["finite"] and summary["bounded"]}; report={"version":"9.0.0-candidate","provider_scope":"skillcorner tracking rollout; bidirectional LODO metrics in training report","summary":summary,"gates":gates,"ready":all(gates.values())}; target=ROOT/"reports/acceptance/continuous_router_v90.json"; target.write_text(json.dumps(report,indent=2)+"\n"); dependencies=[model_path,ROOT/"data/frame_world/continuous_time_v90_lopo_statsbomb360.pt",pass_path,ROOT/"data/frame_world/v89_temporal_providers/manifest.json",ROOT/"reports/acceptance/continuous_time_v90_lopo.json",target]; locked={"version":"9.0.0-candidate","dependencies":[{"artifact":x.relative_to(ROOT).as_posix(),"sha256":sha(x)} for x in dependencies],"ready":report["ready"]}; (ROOT/"data/frame_world/continuous_router_v90.json").write_text(json.dumps(locked,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())
