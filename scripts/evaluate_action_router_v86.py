#!/usr/bin/env python3
"""Real-checkpoint closed-loop and counterfactual acceptance for v8.6 routing."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.match_engine.frame_world.pass_triplet import load_pass_triplet
from src.match_engine.frame_world.pressure import load_pressure_pair
from src.match_engine.frame_world.router import ActionRequest,ActionTransitionRouter,FrameContext
from src.match_engine.frame_world.semantic import load_semantic_frame_world


def meters(delta): return np.sqrt((delta[...,0]*105)**2+(delta[...,1]*68)**2)
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def tracking_lookup():
    base=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text())["matches"]; corrected=json.loads((ROOT/"data/frame_world/v85_tracking/manifest.json").read_text())["matches"]; result={(x["provider"],x["match_id"]):x["file"] for x in base}; result.update({(x["provider"],x["match_id"]):x["file"] for x in corrected}); return result


def context(frame,index): return FrameContext(frame["positions"][index-4:index+1].astype(np.float32),frame["velocities"][index-4:index+1].astype(np.float32),frame["visible"][index-4:index+1].astype(bool),frame["teams"].astype(np.int8))


def audit(router,state,action,affected):
    factual,_=router.rollout(state,10,{0:[action]}); deleted,_=router.rollout(state,10,{}); delayed,_=router.rollout(state,10,{1:[action]}); endpoint=float(meters(factual[-1,affected]-deleted[-1,affected]).mean()); timing=float(meters(factual[-1,affected]-delayed[-1,affected]).mean())
    step_delta=np.diff(np.concatenate([state.positions[-1,None],factual],0),axis=0); max_speed=float((meters(step_delta)/.5).max()); horizons={str(second):float(meters(factual[second*2-1,affected]-deleted[second*2-1,affected]).mean()) for second in (1,2,3,5)}; transition=router.transition(state,[action]); return {"finite":bool(np.isfinite(factual).all()),"bounded":bool(factual.min()>=-.150001 and factual.max()<=1.150001),"max_speed_mps":max_speed,"deletion_effect_m":endpoint,"time_shift_effect_m":timing,"horizon_effect_m":horizons,"possession_valid":transition.possessor_index==-1 or (0<=transition.possessor_index<32 and transition.possession_team in (0,1))}


def main():
    pass_models={p:load_pass_triplet(ROOT/f"data/frame_world/frame_world_v85_pass_triplet_lodo_{p}.pt")[0].eval() for p in ("metrica","skillcorner")}; pressure=load_pressure_pair(ROOT/"data/frame_world/frame_world_v84_pressure_pair.pt")[0].eval(); shots={p:load_semantic_frame_world(ROOT/f"data/frame_world/frame_world_v83_actor_lodo_{p}.pt")[0].eval() for p in ("metrica","skillcorner")}; router=ActionTransitionRouter(pass_models, {"skillcorner":pressure},shots); lookup=tracking_lookup(); triplets=json.loads((ROOT/"data/frame_world/v85_pass_triplets/manifest.json").read_text())["matches"]; records=[]; route_counts={}
    for item in triplets:
        provider=item["provider"]
        if provider not in {"metrica","skillcorner"}: continue
        frame=np.load(ROOT/lookup[(provider,item["match_id"])],allow_pickle=False); labels=np.load(ROOT/item["file"],allow_pickle=False); taken=0
        for n in np.flatnonzero(labels["valid"]):
            index=int(labels["event_frames"][n]); actor=int(labels["actor_indices"][n]); receiver=int(labels["receiver_indices"][n]); defender=int(labels["defender_indices"][n])
            if index<4: continue
            state=context(frame,index); action=ActionRequest("pass",provider,actor,receiver,defender); audit_result=audit(router,state,action,[32,receiver,defender]); decision=router.transition(state,[action]).decisions[0]; route_counts[decision.route]=route_counts.get(decision.route,0)+1; records.append({"provider":provider,**audit_result}); taken+=1
            if taken>=8: break
    pressure_manifest=json.loads((ROOT/"data/frame_world/v84_pressure/manifest.json").read_text())["matches"]
    for item in pressure_manifest[-2:]:
        frame=np.load(ROOT/lookup[("skillcorner",item["match_id"])],allow_pickle=False); labels=np.load(ROOT/item["file"],allow_pickle=False); taken=0
        for n in np.flatnonzero(labels["event_geometry_valid"]):
            index=int(labels["event_frames"][n]); actor=int(labels["event_actor_indices"][n]); target=int(labels["event_target_indices"][n])
            if index<4: continue
            state=context(frame,index); action=ActionRequest("pressure","skillcorner",actor,target); result=audit(router,state,action,[actor,target]); decision=router.transition(state,[action]).decisions[0]; route_counts[decision.route]=route_counts.get(decision.route,0)+1; records.append({"provider":"skillcorner_pressure",**result}); taken+=1
            if taken>=8: break
    sample_state=context(np.load(ROOT/lookup[("metrica","game1")],allow_pickle=False),10); shot=router.transition(sample_state,[ActionRequest("shot","metrica",actor_index=0,direction=(1,0))]).decisions[0]; unsupported=router.transition(sample_state,[ActionRequest("shot","sportec",actor_index=0,direction=(1,0))]).decisions[0]; route_counts[shot.route]=route_counts.get(shot.route,0)+1; route_counts[unsupported.route]=route_counts.get(unsupported.route,0)+1
    horizon_effect={second:float(np.mean([x["horizon_effect_m"][second] for x in records])) for second in ("1","2","3","5")}; summary={"samples":len(records),"route_counts":route_counts,"finite":all(x["finite"] for x in records),"bounded":all(x["bounded"] for x in records),"possession_valid":all(x["possession_valid"] for x in records),"max_speed_mps":max(x["max_speed_mps"] for x in records),"mean_deletion_effect_m":float(np.mean([x["deletion_effect_m"] for x in records])),"mean_time_shift_effect_m":float(np.mean([x["time_shift_effect_m"] for x in records])),"horizon_effect_m":horizon_effect,"shot_enabled":shot.enabled,"sportec_shot_fallback":not unsupported.enabled}; gates={"routes_exercised":all(route_counts.get(x,0)>0 for x in ("pass_triplet_v85","pressure_pair_v84","semantic_shot_v83","fallback")),"finite":summary["finite"],"bounded":summary["bounded"],"possession_contract":summary["possession_valid"],"plausible_speed":summary["max_speed_mps"]<=45.01,"deletion_counterfactual":all(x>.01 for x in horizon_effect.values()),"time_shift_counterfactual":summary["mean_time_shift_effect_m"]>.01,"shot_evidence_switch":summary["shot_enabled"] and summary["sportec_shot_fallback"]}; report={"version":"8.6.0-candidate","horizon_seconds":[1,2,3,5],"summary":summary,"gates":gates,"ready":all(gates.values())}; target=ROOT/"reports/acceptance/action_router_v86.json"; target.write_text(json.dumps(report,indent=2)+"\n"); dependency_paths=[ROOT/f"data/frame_world/frame_world_v85_pass_triplet_lodo_{p}.pt" for p in ("metrica","skillcorner")]+[ROOT/"data/frame_world/frame_world_v84_pressure_pair.pt"]+[ROOT/f"data/frame_world/frame_world_v83_actor_lodo_{p}.pt" for p in ("metrica","skillcorner")]; manifest={"version":"8.6.0-candidate","router":"src/match_engine/frame_world/router.py","dependencies":[{"artifact":x.relative_to(ROOT).as_posix(),"sha256":sha256(x)} for x in dependency_paths],"evidence":target.relative_to(ROOT).as_posix(),"ready":report["ready"]}; (ROOT/"data/frame_world/action_router_v86.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1


if __name__=="__main__": raise SystemExit(main())
