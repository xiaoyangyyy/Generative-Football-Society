#!/usr/bin/env python3
"""Build v9.2 three-provider temporal data with explicit Metrica clocks."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.build_metrica_pass_dataset import _tracking_at_frames
from scripts.build_temporal_providers_v89 import add_static_context,label,lane_distance
from scripts.train_pass_triplet_v85 import feature_vector
from src.data_engine.dataset_registry import write_json_atomic
OUT=ROOT/"data/frame_world/v92_temporal_providers"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save_match(match,features,labels,sources):
    path=OUT/f"metrica_{match}.npz"; np.savez_compressed(path,features=np.asarray(features,np.float32),event_bin=np.asarray([x[0] for x in labels],np.int64),observed=np.asarray([x[1] for x in labels],bool),continuation=np.asarray([x[2] for x in labels],np.int64),subtype=np.asarray([x[3] for x in labels],np.int64),delay=np.asarray([x[4] for x in labels],np.float32)); return {"provider":"metrica","match_id":match,"events":len(labels),"observed":sum(x[1] for x in labels),"file":path.relative_to(ROOT).as_posix(),"sha256":sha(path),"source":"explicit PASS End Time/To receiver to receiver next event Start Time/From actor","source_sha256":{p.name:sha(p) for p in sources}}
def next_action(events,index,receiver,period,reception_s):
    for _,event in events.iloc[index+1:].iterrows():
        if int(event["Period"])!=period: break
        start=float(event["Start Time [s]"])
        if start<reception_s-1e-6: continue
        delay=start-reception_s
        if delay>5: break
        if str(event.get("From"))!=receiver: continue
        kind=str(event["Type"])
        if kind=="PASS": return "pass",delay
        if kind=="SHOT": return "shot",delay
        if kind in {"BALL LOST","CHALLENGE"}: return "loss",delay
        if kind in {"BALL OUT","FAULT RECEIVED","SET PIECE"}: return "stoppage",delay
    return "unknown",None
def metrica(game):
    raw=ROOT/"data/external/metrica/raw"; prefix=f"Sample_Game_{game}"; event_path=raw/f"{prefix}_RawEventsData.csv"; home_path=raw/f"{prefix}_RawTrackingData_Home_Team.csv"; away_path=raw/f"{prefix}_RawTrackingData_Away_Team.csv"; events=pd.read_csv(event_path).sort_values(["Period","Start Time [s]","Start Frame"],kind="stable").reset_index(drop=True); passes=events[(events.Type=="PASS")&events.From.notna()&events.To.notna()&events["End Time [s]"].notna()&events["Start Frame"].notna()]; wanted=set(passes["Start Frame"].astype(int)); tracking={"Home":_tracking_at_frames(home_path,wanted),"Away":_tracking_at_frames(away_path,wanted)}; features=[]; labels=[]; flights=[]; dropped={"missing_tracking":0,"invalid_clock":0}
    for index,event in passes.iterrows():
        team=str(event.Team); opponent="Away" if team=="Home" else "Home"; frame=int(event["Start Frame"]); own=tracking[team].get(frame,{}); guards=tracking[opponent].get(frame,{}); passer=str(event.From); receiver=str(event.To)
        if passer not in own or receiver not in own or not guards: dropped["missing_tracking"]+=1; continue
        start_s=float(event["Start Time [s]"]); reception_s=float(event["End Time [s]"])
        if reception_s<start_s or reception_s-start_s>5: dropped["invalid_clock"]+=1; continue
        start=np.asarray(own[passer],np.float32); receive=np.asarray(own[receiver],np.float32); defender=np.asarray(min(guards.values(),key=lambda p:np.linalg.norm(np.asarray(p)-receive)),np.float32); position=np.stack([start,receive,defender]); zero=np.zeros_like(position); lane=lane_distance(defender,start,receive); features.append(add_static_context(feature_vector(position,zero,zero,lane),position,lane)); action,delay=next_action(events,index,receiver,int(event.Period),reception_s); labels.append(label(action,delay)); flights.append(reception_s-start_s)
    report=save_match(f"game{game}",features,labels,[event_path,home_path,away_path]); report["dropped"]=dropped; report["pass_flight_s"]={"median":float(np.median(flights)),"p95":float(np.quantile(flights,.95)),"max":float(np.max(flights))}; return report
def main():
    OUT.mkdir(parents=True,exist_ok=True); parent=json.loads((ROOT/"data/frame_world/v91_temporal_providers/manifest.json").read_text()); added=[metrica(1),metrica(2)]; matches=list(parent["matches"])+added; names=("skillcorner","statsbomb360","metrica"); providers={p:{"matches":sum(x["provider"]==p for x in matches),"events":sum(x["events"] for x in matches if x["provider"]==p),"observed":sum(x["observed"] for x in matches if x["provider"]==p)} for p in names}; manifest={"version":"9.2.0-candidate","contract":"successful reception timestamp to receiver next marked action timestamp; right censor at 5s","provider_clocks":{"skillcorner":"receiver possession frame_start to frame_end at 10Hz","statsbomb360":"Ball Receipt timestamp to recipient action timestamp","metrica":"PASS End Time to receiver next event Start Time"},"rejected_provider":{"sportec":"PASS end_timestamp is absent in the available IDSSE event files"},"parent":"data/frame_world/v91_temporal_providers/manifest.json","providers":providers,"matches":matches}; write_json_atomic(OUT/"manifest.json",manifest); print(json.dumps(providers,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())