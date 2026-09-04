#!/usr/bin/env python3
"""Build v9.1 reception-to-next-action clocks without mutating frozen v8.9."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_temporal_mark_v88 import data_items
from src.data_engine.dataset_registry import write_json_atomic
OUT=ROOT/"data/frame_world/v91_temporal_providers"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def encode(action,delay):
    observed=action in {"pass","shot","loss","stoppage"} and delay is not None and delay<=5; continuation=1 if action in {"pass","shot"} else 0 if action in {"loss","stoppage"} else -1; subtype=1 if action=="shot" else 0 if action=="pass" else -1; return observed,continuation,subtype,min(float(delay if delay is not None else 5),5)
def corrected_durations(match):
    path=ROOT/f"data/external/skillcorner/raw/{match}/{match}_dynamic_events.csv"; data=pd.read_csv(path,low_memory=False); possessions=data[data.event_type.eq("player_possession")].sort_values(["period","frame_start"]); values=list(possessions.itertuples(index=False)); result={}
    for i,event in enumerate(values):
        if str(event.end_type)!="pass": continue
        nxt=next((x for x in values[i+1:] if x.period==event.period and x.frame_start>=event.frame_end),None); result[str(event.event_id)]=None if nxt is None else max(0,(int(nxt.frame_end)-int(nxt.frame_start))/10)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True); frozen=json.loads((ROOT/"data/frame_world/v89_temporal_providers/manifest.json").read_text()); old={(x["provider"],x["match_id"]):x for x in frozen["matches"]}; matches=[]
    for item in data_items():
        match=str(item["match_id"]); source=old[("skillcorner",match)]; values=np.load(ROOT/source["file"],allow_pickle=False); chains=[json.loads(line) for line in (ROOT/item["chain_file"]).read_text().splitlines()]; labels=np.load(ROOT/item["triplet_file"],allow_pickle=False); selected=[chains[n] for n in range(len(chains)) if labels["valid"][n] and int(labels["event_frames"][n])>=1 and chains[n]["outcome"]=="complete"]; durations=corrected_durations(match); rows=[encode(x["next_action"],durations.get(str(x["source_event_id"]))) for x in selected]; assert len(rows)==len(values["features"]); target=OUT/f"skillcorner_{match}.npz"; np.savez_compressed(target,features=values["features"],observed=np.asarray([x[0] for x in rows],bool),continuation=np.asarray([x[1] for x in rows],np.int64),subtype=np.asarray([x[2] for x in rows],np.int64),delay=np.asarray([x[3] for x in rows],np.float32),event_bin=np.minimum(9,np.maximum(0,np.ceil(np.asarray([x[3] for x in rows])/.5).astype(int)-1))); matches.append({"provider":"skillcorner","match_id":match,"events":len(rows),"observed":sum(x[0] for x in rows),"file":target.relative_to(ROOT).as_posix(),"sha256":sha(target),"source":"explicit receiver possession frame_start to frame_end duration"})
    for item in frozen["matches"]:
        if item["provider"]=="statsbomb360": matches.append(item)
    providers={p:{"matches":sum(x["provider"]==p for x in matches),"events":sum(x["events"] for x in matches if x["provider"]==p),"observed":sum(x["observed"] for x in matches if x["provider"]==p)} for p in ("skillcorner","statsbomb360")}; manifest={"version":"9.1.0-candidate","contract":"successful reception timestamp to receiver next marked action timestamp","correction":"SkillCorner uses receiver possession frame_end-frame_start; StatsBomb uses Ball Receipt timestamp to recipient action","parent":"data/frame_world/v89_temporal_providers/manifest.json","providers":providers,"matches":matches}; write_json_atomic(OUT/"manifest.json",manifest); print(json.dumps(providers,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
