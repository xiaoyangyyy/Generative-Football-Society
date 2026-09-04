#!/usr/bin/env python3
"""Build event-level passer/receiver/interceptor triplets from aligned passes."""
from __future__ import annotations
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.build_frame_actions_v82 import entity_mapping
from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.frame_world.schema import BALL_INDEX

OUT=ROOT/"data/frame_world/v85_pass_triplets"


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(points):
    value=np.asarray(points,np.float32).copy(); value[...,0]*=105; value[...,1]*=68; return value


def nearest_interceptor(frame,index,actor,receiver):
    teams=frame["teams"]; actor_team=int(teams[actor]); candidates=np.flatnonzero((teams>=0)&(teams!=actor_team)&frame["visible"][index])
    if not len(candidates): return -1,float("inf")
    start=metric(frame["positions"][index,BALL_INDEX]); end=metric(frame["positions"][index,receiver]); points=metric(frame["positions"][index,candidates]); line=end-start; denom=max(float(line@line),1e-6); alpha=np.clip(((points-start)@line)/denom,0,1); distances=np.linalg.norm(points-(start+alpha[:,None]*line),axis=1); choice=int(np.argmin(distances)); return int(candidates[choice]),float(distances[choice])


def main():
    OUT.mkdir(parents=True,exist_ok=True); frames=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text())["matches"]; corrected=json.loads((ROOT/"data/frame_world/v85_tracking/manifest.json").read_text())["matches"]; correction={(x["provider"],x["match_id"]):x for x in corrected}; frames=[correction.get((x["provider"],x["match_id"]),x) for x in frames]; actions=json.loads((ROOT/"data/frame_world/v82_actions/manifest.json").read_text())["matches"]; lookup={(x["provider"],x["match_id"]):x for x in actions}; reports=[]
    for item in frames:
        action=lookup[(item["provider"],item["match_id"])]; mapping=entity_mapping(item["provider"],item["match_id"]); events=[json.loads(line) for line in (ROOT/action["events_file"]).read_text().splitlines() if line]; passes=[x for x in events if x["action"]=="pass" and x["supervision"]=="strong"]
        with np.load(ROOT/item["file"],allow_pickle=False) as source: frame={key:source[key] for key in ("positions","velocities","visible","teams")}
        event_frames=[]; actors=[]; receivers=[]; defenders=[]; lane_distances=[]; valid=[]
        for event in passes:
            index=int(event["frame_index"]); actor=mapping.get(str(event["actor_id"]),-1); receiver=mapping.get(str(event["target_id"]),-1); ok=actor>=0 and receiver>=0 and frame["visible"][index,actor] and frame["visible"][index,receiver] and frame["visible"][index,BALL_INDEX]
            defender,distance=nearest_interceptor(frame,index,actor,receiver) if ok else (-1,float("inf")); ok=ok and defender>=0
            event_frames.append(index); actors.append(actor); receivers.append(receiver); defenders.append(defender); lane_distances.append(distance if np.isfinite(distance) else -1); valid.append(ok)
        path=OUT/f'{item["provider"]}_{item["match_id"]}.npz'; np.savez_compressed(path,event_frames=np.array(event_frames,np.int32),actor_indices=np.array(actors,np.int8),receiver_indices=np.array(receivers,np.int8),defender_indices=np.array(defenders,np.int8),lane_distance_m=np.array(lane_distances,np.float32),valid=np.array(valid,np.uint8)); reports.append({"provider":item["provider"],"match_id":item["match_id"],"passes":len(passes),"valid_triplets":int(np.sum(valid)),"valid_rate":float(np.mean(valid)) if valid else 0.,"median_lane_distance_m":float(np.median(np.array(lane_distances)[np.array(valid,dtype=bool)])) if any(valid) else None,"file":path.relative_to(ROOT).as_posix(),"sha256":sha256(path)}); del frame,events,passes; gc.collect()
    manifest={"version":"8.5.0-candidate","defender_policy":"visible opponent nearest to ball-receiver segment at aligned pass frame","matches":reports,"totals":{"passes":sum(x["passes"] for x in reports),"valid_triplets":sum(x["valid_triplets"] for x in reports)}}; write_json_atomic(OUT/"manifest.json",manifest); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
