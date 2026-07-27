#!/usr/bin/env python3
"""Audit calibrated lower/median/upper event timing on real tracking contexts."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.match_engine.frame_world.calibration import load_calibrated_temporal_router
from src.match_engine.frame_world.pass_triplet import load_pass_triplet
from src.match_engine.frame_world.router import ActionRequest,FrameContext
from src.match_engine.frame_world.schema import BALL_INDEX,load_sequence

def contexts(sequence,count=20):
    output=[]
    for index in np.linspace(5,len(sequence.positions)-1,400,dtype=int):
        visible=sequence.visible[index]; home=np.flatnonzero(visible[:16]); away=np.flatnonzero(visible[16:32])+16
        if visible[BALL_INDEX] and len(home)>=2 and len(away)>=1:
            actor,target=int(home[0]),int(home[1]); defender=int(away[np.argmin(np.linalg.norm(sequence.positions[index,away]-sequence.positions[index,target],axis=1))]); output.append((FrameContext(sequence.positions[index-4:index+1],sequence.velocities[index-4:index+1],sequence.visible[index-4:index+1],sequence.teams),ActionRequest("pass","metrica",actor,target,defender)))
            if len(output)>=count: break
    return output
def main():
    pass_model=load_pass_triplet(ROOT/"data/frame_world/frame_world_v85_pass_triplet_lodo_metrica.pt")[0].eval(); router=load_calibrated_temporal_router(ROOT/"data/frame_world/continuous_intervals_v93.json",pass_models={"metrica":pass_model}); records=[]
    for game in ("game1","game2"):
        sequence=load_sequence(ROOT/f"data/frame_world/v85_tracking/metrica_{game}.npz")
        for context,action in contexts(sequence):
            trajectories={}; interval=None
            for bound in ("lower","median","upper"):
                first,interval,_=router.rollout_event_driven_interval(context,action,12,bound); second,_,_=router.rollout_event_driven_interval(context,action,12,bound); assert np.array_equal(first,second); trajectories[bound]=first
            values=np.stack(list(trajectories.values())); spread=float(np.max(np.linalg.norm((values-values[1:2])*np.array([105,68]),axis=-1))); records.append({"match":game,"lower_step":interval.lower_steps,"median_step":interval.delay_steps,"upper_step":interval.upper_steps,"finite":bool(np.isfinite(values).all()),"bounded":bool(values.min()>=-.150001 and values.max()<=1.150001),"max_counterfactual_spread_m":spread,"reachable_bound_m":45*.5*(interval.upper_steps-interval.lower_steps)+1})
    gates={"contexts":len(records)>=20,"ordered_interventions":all(x["lower_step"]<=x["median_step"]<=x["upper_step"] for x in records),"finite_bounded":all(x["finite"] and x["bounded"] for x in records),"deterministic":True,"nontrivial_intervention":sum(x["max_counterfactual_spread_m"]>.01 for x in records)>=len(records)//2,"reachable_spread":all(x["max_counterfactual_spread_m"]<=x["reachable_bound_m"] for x in records)}; report={"version":"9.3.0-candidate","contexts":len(records),"spread_m":{"median":float(np.median([x["max_counterfactual_spread_m"] for x in records])),"max":max(x["max_counterfactual_spread_m"] for x in records),"max_reachable_ratio":max(x["max_counterfactual_spread_m"]/x["reachable_bound_m"] for x in records)},"gates":gates,"ready":all(gates.values()),"records":records}; (ROOT/"reports/acceptance/interval_rollout_v93.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())