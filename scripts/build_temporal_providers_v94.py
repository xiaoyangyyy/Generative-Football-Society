#!/usr/bin/env python3
"""Extend explicit-clock temporal supervision with Metrica EPTS Sample Game 3."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
from kloppy import metrica
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.build_temporal_providers_v89 import add_static_context,label,lane_distance
from scripts.train_pass_triplet_v85 import feature_vector
from src.data_engine.dataset_registry import write_json_atomic
OUT=ROOT/"data/frame_world/v94_temporal_providers"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def next_action(events,index,receiver,period,reception_s):
    for event in events[index+1:]:
        if int(event["period"])!=period: break
        start=float(event["start"]["time"]); delay=start-reception_s
        if delay<0: continue
        if delay>5: break
        actor=(event.get("from") or {}).get("id")
        if actor!=receiver: continue
        kind=(event.get("type") or {}).get("name")
        if kind=="PASS": return "pass",delay
        if kind=="SHOT": return "shot",delay
        if kind in {"BALL LOST","CHALLENGE"}: return "loss",delay
        if kind in {"BALL OUT","FAULT RECEIVED","SET PIECE"}: return "stoppage",delay
    return "unknown",None
def game3():
    base=ROOT/"data/external/metrica/raw/Sample_Game_3"; event_path=base/"Sample_Game_3_events.json"; meta_path=base/"Sample_Game_3_metadata.xml"; tracking_path=base/"Sample_Game_3_tracking.txt"; events=json.loads(event_path.read_text())["data"]; events=sorted(events,key=lambda x:(x["period"],x["start"]["time"],x["index"])); tracking=metrica.load_tracking_epts(meta_path,tracking_path); frames={int(x.frame_id):x for x in tracking.frames}; features=[]; labels=[]; flights=[]; dropped={"missing_tracking":0,"invalid_clock":0}
    for index,event in enumerate(events):
        if (event.get("type") or {}).get("name")!="PASS" or not event.get("to"): continue
        start=event.get("start") or {}; end=event.get("end") or {}; frame=frames.get(int(start.get("frame",-1))); receiver=str(event["to"]["id"]); passer=str((event.get("from") or {}).get("id")); team=(event.get("team") or {}).get("id")
        if frame is None or start.get("time") is None or end.get("time") is None: dropped["missing_tracking"]+=1; continue
        players={p.player_id:(p.team.team_id,np.asarray([value.coordinates.x,value.coordinates.y],np.float32)) for p,value in frame.players_data.items() if value.coordinates is not None}; opponents=[value[1] for value in players.values() if value[0]!=team]
        if receiver not in players or passer not in players or not opponents: dropped["missing_tracking"]+=1; continue
        start_s=float(start["time"]); reception_s=float(end["time"])
        if reception_s<start_s or reception_s-start_s>5: dropped["invalid_clock"]+=1; continue
        ball=np.asarray([float(start["x"]),float(start["y"])],np.float32); receive=players[receiver][1]; defender=min(opponents,key=lambda p:np.linalg.norm(p-receive)); position=np.stack([ball,receive,defender]); zero=np.zeros_like(position); lane=lane_distance(defender,ball,receive); features.append(add_static_context(feature_vector(position,zero,zero,lane),position,lane)); action,delay=next_action(events,index,receiver,int(event["period"]),reception_s); labels.append(label(action,delay)); flights.append(reception_s-start_s)
    path=OUT/"metrica_game3.npz"; np.savez_compressed(path,features=np.asarray(features,np.float32),event_bin=np.asarray([x[0] for x in labels],np.int64),observed=np.asarray([x[1] for x in labels],bool),continuation=np.asarray([x[2] for x in labels],np.int64),subtype=np.asarray([x[3] for x in labels],np.int64),delay=np.asarray([x[4] for x in labels],np.float32)); return {"provider":"metrica","match_id":"game3","events":len(labels),"observed":sum(x[1] for x in labels),"file":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"source":"explicit EPTS PASS end frame/time and To receiver to receiver next marked action","source_sha256":{x.name:sha(x) for x in (event_path,meta_path,tracking_path)},"dropped":dropped,"pass_flight_s":{"median":float(np.median(flights)),"p95":float(np.quantile(flights,.95)),"max":float(np.max(flights))}}
def main():
    OUT.mkdir(parents=True,exist_ok=True); parent=json.loads((ROOT/"data/frame_world/v92_temporal_providers/manifest.json").read_text()); added=game3(); matches=list(parent["matches"])+[added]; providers={p:{"matches":sum(x["provider"]==p for x in matches),"events":sum(x["events"] for x in matches if x["provider"]==p),"observed":sum(x["observed"] for x in matches if x["provider"]==p)} for p in ("skillcorner","statsbomb360","metrica")}; manifest={"version":"9.4.0-candidate","contract":parent["contract"],"provider_clocks":dict(parent["provider_clocks"],metrica="CSV PASS End Time or EPTS PASS end.time to receiver next event start time"),"parent":"data/frame_world/v92_temporal_providers/manifest.json","providers":providers,"matches":matches}; write_json_atomic(OUT/"manifest.json",manifest); print(json.dumps({"providers":providers,"game3":added},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())