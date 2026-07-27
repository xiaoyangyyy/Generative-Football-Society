#!/usr/bin/env python3
"""Build pressure-onset geometry without changing sealed action datasets."""
from __future__ import annotations
import gc,hashlib,json
from pathlib import Path
import numpy as np
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_frame_actions_v82 import entity_mapping
from src.data_engine.dataset_registry import write_json_atomic
OUT=ROOT/"data/frame_world/v84_pressure"


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(frame,index,actor,target):
    positions=frame["positions"]; velocities=frame["velocities"]
    if actor<0 or target<0 or not frame["visible"][index,actor] or not frame["visible"][index,target]: return np.zeros(4,np.float32),False
    delta=positions[index,target]-positions[index,actor]; metric=np.array([delta[0]*105,delta[1]*68],np.float32); distance=float(np.linalg.norm(metric)); unit=metric/max(distance,1e-6)
    relative=velocities[index,target]-velocities[index,actor]; relative_metric=np.array([relative[0]*105,relative[1]*68],np.float32); closing=-float(relative_metric@unit)
    return np.array([min(distance/10,2),np.clip(closing/10,-1,1),unit[0],unit[1]],np.float32),True


def main():
    OUT.mkdir(parents=True,exist_ok=True); frames=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text()); actions=json.loads((ROOT/"data/frame_world/v82_actions/manifest.json").read_text()); lookup={(x["provider"],x["match_id"]):x for x in actions["matches"]}; reports=[]
    for item in frames["matches"]:
        if item["provider"]!="skillcorner": continue
        action=lookup[(item["provider"],item["match_id"])]
        with np.load(ROOT/action["labels_file"],allow_pickle=False) as source: label_arrays={key:source[key] for key in source.files}
        with np.load(ROOT/item["file"],allow_pickle=False) as source: frame={key:source[key] for key in ("timestamps","positions","velocities","visible")}
        labels=label_arrays; onset=np.zeros(len(frame["timestamps"]),np.uint8); features=np.zeros((len(onset),4),np.float32); actors=np.full(len(onset),-1,np.int8); targets=np.full(len(onset),-1,np.int8); valid=np.zeros(len(onset),np.uint8)
        pressure=labels["labels"][:,2].astype(bool); starts=np.flatnonzero(pressure&~np.r_[False,pressure[:-1]])
        for index in starts:
            actor=int(labels["actor_indices"][index,2]); target=int(labels["target_indices"][index,2]); onset[index]=1; actors[index]=actor; targets[index]=target; features[index],ok=geometry(frame,index,actor,target); valid[index]=ok
        mapping=entity_mapping(item["provider"],item["match_id"]); events=[json.loads(line) for line in (ROOT/action["events_file"]).read_text().splitlines() if line]; pressure=[x for x in events if x["action"]=="pressure" and x["supervision"]=="strong"]
        event_frames=np.array([x["frame_index"] for x in pressure],np.int32); event_actors=np.array([mapping.get(str(x["actor_id"]),-1) for x in pressure],np.int8); event_targets=np.array([mapping.get(str(x["target_id"]),-1) for x in pressure],np.int8); event_features=np.zeros((len(pressure),4),np.float32); event_valid=np.zeros(len(pressure),np.uint8)
        for n,(index,actor,target) in enumerate(zip(event_frames,event_actors,event_targets)): event_features[n],ok=geometry(frame,int(index),int(actor),int(target)); event_valid[n]=ok
        path=OUT/f'{item["provider"]}_{item["match_id"]}.npz'; np.savez_compressed(path,onset=onset,features=features,actor_indices=actors,target_indices=targets,geometry_valid=valid,event_frames=event_frames,event_actor_indices=event_actors,event_target_indices=event_targets,event_features=event_features,event_geometry_valid=event_valid)
        reports.append({"provider":item["provider"],"match_id":item["match_id"],"merged_onsets":int(onset.sum()),"event_onsets":len(pressure),"geometry_valid":int(event_valid.sum()),"geometry_rate":float(event_valid.sum()/max(1,len(pressure))),"file":path.relative_to(ROOT).as_posix(),"sha256":sha256(path)})
        del frame,label_arrays,labels,onset,features,actors,targets,valid,events,pressure,event_frames,event_actors,event_targets,event_features,event_valid; gc.collect()
    manifest={"version":"8.4.0-candidate","definition":"event-level start of each strong pressure interval; concurrent actors remain separate","features":["distance_10m","closing_speed_10mps","relative_unit_x","relative_unit_y"],"matches":reports,"totals":{"event_onsets":sum(x["event_onsets"] for x in reports),"merged_onsets":sum(x["merged_onsets"] for x in reports),"geometry_valid":sum(x["geometry_valid"] for x in reports)}}; write_json_atomic(OUT/"manifest.json",manifest); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
