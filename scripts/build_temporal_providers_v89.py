#!/usr/bin/env python3
"""Build explicit two-provider post-reception temporal supervision."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_temporal_mark_v88 import TemporalEvents,data_items
from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.frame_world.temporal import TemporalMarkConfig
from scripts.train_pass_triplet_v85 import feature_vector

OUT=ROOT/"data/frame_world/v89_temporal_providers"; CFG=TemporalMarkConfig()


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def seconds(value):
    h,m,s=value.split(":"); return int(h)*3600+int(m)*60+float(s)
def metric(point): return np.asarray(point,np.float32)*np.array([105,68],np.float32)
def lane_distance(defender,start,end):
    a=metric(start); b=metric(end); p=metric(defender); delta=b-a; alpha=np.clip(float((p-a)@delta)/max(float(delta@delta),1e-6),0,1); return float(np.linalg.norm(p-(a+alpha*delta)))
def add_static_context(vector,pos,lane):
    ball,receiver,defender=pos; value=vector.copy(); value[7:19]=np.array([abs(ball[0]-.5),abs(ball[1]-.5),abs(receiver[0]-.5),abs(receiver[1]-.5),abs(defender[0]-.5),abs(defender[1]-.5),abs(receiver[0]-.5),abs(ball[0]-.5),np.linalg.norm(metric(defender-receiver))/20,lane/10,np.linalg.norm(metric(receiver-ball))/50,np.linalg.norm(metric(defender-ball))/50],np.float32); return value
def label(action,delay):
    observed=action in {"pass","shot","loss","stoppage"} and delay is not None and delay<=5
    event_bin=min(CFG.bins-1,max(0,int(np.ceil(float(delay)/CFG.bin_seconds))-1)) if observed else CFG.bins-1
    continuation=1 if action in {"pass","shot"} else 0 if action in {"loss","stoppage"} else -1
    subtype=1 if action=="shot" else 0 if action=="pass" else -1
    return event_bin,observed,continuation,subtype,min(float(delay if delay is not None else 5),5)
def save(provider,match,features,labels,source):
    path=OUT/f"{provider}_{match}.npz"; np.savez_compressed(path,features=np.asarray(features,np.float32),event_bin=np.asarray([x[0] for x in labels],np.int64),observed=np.asarray([x[1] for x in labels],bool),continuation=np.asarray([x[2] for x in labels],np.int64),subtype=np.asarray([x[3] for x in labels],np.int64),delay=np.asarray([x[4] for x in labels],np.float32)); return {"provider":provider,"match_id":str(match),"events":len(labels),"observed":sum(x[1] for x in labels),"file":path.relative_to(ROOT).as_posix(),"sha256":sha256(path),"source":source}


def skillcorner():
    reports=[]
    for item in data_items():
        dataset=TemporalEvents([item],CFG); features=[]; labels=[]
        with np.load(ROOT/item["triplet_file"],allow_pickle=False) as source: triplets={key:source[key] for key in source.files}
        with np.load(ROOT/item["file"],allow_pickle=False) as source: positions=source["positions"]
        chains=[json.loads(line) for line in (ROOT/item["chain_file"]).read_text().splitlines()]
        geometry=[]
        for n in range(len(triplets["valid"])):
            index=int(triplets["event_frames"][n])
            if not triplets["valid"][n] or index<1: continue
            slots=np.array([32,int(triplets["receiver_indices"][n]),int(triplets["defender_indices"][n])]); geometry.append((positions[index,slots],float(triplets["lane_distance_m"][n]),chains[n]["outcome"]))
        for row,(pos,lane,outcome) in zip(dataset.rows,geometry):
            if outcome!="complete": continue
            features.append(add_static_context(row[0],pos,lane)); labels.append((int(row[1]),bool(row[2]),int(row[3]),int(row[4]),float(row[5])))
        reports.append(save("skillcorner",item["match_id"],features,labels,"explicit pass outcome and player-possession frame timestamps"))
    return reports


def action_after(events,start_index,receipt):
    player=(receipt.get("player") or {}).get("id"); start=seconds(receipt["timestamp"]); period=receipt["period"]
    for event in events[start_index+1:]:
        if event.get("period")!=period: break
        delay=seconds(event["timestamp"])-start
        if delay>5: break
        kind=(event.get("type") or {}).get("name"); actor=(event.get("player") or {}).get("id")
        if actor!=player: continue
        if kind=="Pass": return "pass",delay
        if kind=="Shot": return "shot",delay
        if kind in {"Dispossessed","Miscontrol"}: return "loss",delay
        if kind=="Dribble" and ((event.get("dribble") or {}).get("outcome") or {}).get("name")=="Incomplete": return "loss",delay
        if kind in {"Foul Won","Clearance","Offside"}: return "stoppage",delay
    return "unknown",None


def loss_after(events,start_index,receipt):
    start=seconds(receipt["timestamp"]); period=receipt["period"]
    for event in events[start_index+1:]:
        if event.get("period")!=period: break
        delay=seconds(event["timestamp"])-start
        if delay>5: break
        if (event.get("type") or {}).get("name") in {"Pass","Shot","Carry","Ball Recovery","Interception","Clearance","Miscontrol","Dispossessed","Dribble"} and event.get("player"):
            return "loss",max(delay,.01)
    return "unknown",None
def statsbomb():
    reports=[]; event_dir=ROOT/"data/external/statsbomb360/raw/events"; frame_dir=ROOT/"data/external/statsbomb360/raw/three-sixty"
    for event_path in sorted(event_dir.glob("*.json")):
        frame_path=frame_dir/event_path.name
        if not frame_path.exists(): continue
        try:
            events=json.loads(event_path.read_text(encoding="utf-8")); frame_rows=json.loads(frame_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        frames={str(x["event_uuid"]):x for x in frame_rows}; by_id={str(x["id"]):(i,x) for i,x in enumerate(events)}; features=[]; labels=[]
        for event in events:
            if (event.get("type") or {}).get("name")!="Pass" or str(event["id"]) not in frames: continue
            payload=event.get("pass") or {}; recipient=(payload.get("recipient") or {}).get("id"); end=payload.get("end_location"); related=event.get("related_events") or []
            receipt_pair=next((by_id.get(str(x)) for x in related if by_id.get(str(x)) and (by_id[str(x)][1].get("type") or {}).get("name")=="Ball Receipt*"),None)
            if recipient is None or end is None or receipt_pair is None: continue
            receipt_index,receipt=receipt_pair
            if (receipt.get("player") or {}).get("id")!=recipient: continue
            freeze=frames[str(event["id"])].get("freeze_frame") or []; teammates=[x for x in freeze if x.get("teammate") and not x.get("actor") and x.get("location")]; defenders=[x for x in freeze if not x.get("teammate") and x.get("location")]
            if not teammates or not defenders or not event.get("location"): continue
            start=np.asarray(event["location"][:2],np.float32)/np.array([120,80],np.float32); target=np.asarray(end[:2],np.float32)/np.array([120,80],np.float32); receiver=np.asarray(min(teammates,key=lambda x:np.linalg.norm(np.asarray(x["location"][:2])-np.asarray(end[:2])))["location"][:2],np.float32)/np.array([120,80],np.float32); defender=np.asarray(min(defenders,key=lambda x:np.linalg.norm(np.asarray(x["location"][:2])-np.asarray(receiver)*np.array([120,80])))["location"][:2],np.float32)/np.array([120,80],np.float32)
            pos=np.stack([start,receiver,defender]); zero=np.zeros_like(pos); lane=lane_distance(defender,start,target); vector=add_static_context(feature_vector(pos,zero,zero,lane),pos,lane)
            incomplete=((receipt.get("ball_receipt") or {}).get("outcome") or {}).get("name")=="Incomplete" or bool(payload.get("outcome"))
            if incomplete: continue
            action,delay=action_after(events,receipt_index,receipt)
            features.append(vector); labels.append(label(action,delay))
        if labels: reports.append(save("statsbomb360",event_path.stem,features,labels,"explicit Pass-related Ball Receipt event and recipient action timestamps"))
    return reports


def main():
    OUT.mkdir(parents=True,exist_ok=True); matches=skillcorner()+statsbomb(); providers={p:{"matches":sum(x["provider"]==p for x in matches),"events":sum(x["events"] for x in matches if x["provider"]==p),"observed":sum(x["observed"] for x in matches if x["provider"]==p)} for p in ("skillcorner","statsbomb360")}; manifest={"version":"8.9.0-candidate","contract":"provider-neutral static pass geometry; explicit successful reception identity and timestamps; right censor at 5s","kinematic_policy":"velocity and acceleration masked for both providers","providers":providers,"matches":matches}; write_json_atomic(OUT/"manifest.json",manifest); print(json.dumps(providers,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
