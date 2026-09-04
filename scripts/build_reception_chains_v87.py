#!/usr/bin/env python3
"""Build provider-honest pass reception and next-action event chains."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic

OUT=ROOT/"data/frame_world/v87_reception_chains"


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def next_class(end_type):
    value=str(end_type).lower()
    if value=="pass": return "pass"
    if value=="shot": return "shot"
    if value in {"possession_loss","direct_disruption","indirect_disruption"}: return "loss"
    if value in {"foul_suffered","foul_committed","clearance"}: return "stoppage"
    return "hold"


def skillcorner(base):
    data=pd.read_csv(base/f"{base.name}_dynamic_events.csv",low_memory=False); possessions=data[data.event_type.eq("player_possession")].sort_values(["period","frame_start"]); values=list(possessions.itertuples(index=False)); aligned=ROOT/f"data/frame_world/v82_actions/skillcorner_{base.name}.jsonl"; frame_lookup={str(x["source_event_id"]):int(x["frame_index"]) for x in (json.loads(line) for line in aligned.read_text().splitlines()) if x["action"]=="pass"}; rows=[]
    for i,event in enumerate(values):
        if str(event.end_type)!="pass": continue
        outcome=str(event.pass_outcome).lower(); outcome="offside" if outcome=="offside" else "complete" if outcome=="successful" else "turnover" if outcome=="unsuccessful" else "unknown"; nxt=next((x for x in values[i+1:] if x.period==event.period and x.frame_start>=event.frame_end),None); delay=None if nxt is None else (int(nxt.frame_start)-int(event.frame_end))/10; linked=nxt is not None and delay<=5
        rows.append({"provider":"skillcorner","match_id":base.name,"source_event_id":str(event.event_id),"frame_index":frame_lookup.get(str(event.event_id),-1),"actor_id":None if pd.isna(event.player_id) else str(int(event.player_id)),"receiver_id":None if pd.isna(event.player_targeted_id) else str(int(event.player_targeted_id)),"outcome":outcome,"next_action":next_class(nxt.end_type) if linked else "unknown","next_actor_id":None if not linked or pd.isna(nxt.player_id) else str(int(nxt.player_id)),"delay_s":delay,"supervision":"strong"})
    return rows


def metrica(game):
    path=ROOT/f"data/external/metrica/raw/Sample_Game_{game}_RawEventsData.csv"; events=list(csv.DictReader(open(path,encoding="utf-8-sig"))); rows=[]
    for i,event in enumerate(events):
        if event["Type"]!="PASS": continue
        nxt=next((x for x in events[i+1:] if x["Period"]==event["Period"] and float(x["Start Time [s]"])>=float(event["End Time [s]"])-.01),None); outcome="unknown"
        if nxt is not None and nxt["Team"]==event["Team"] and nxt["From"]==event["To"]: outcome="complete"
        elif nxt is not None and nxt["Team"]!=event["Team"]: outcome="turnover"
        delay=None if nxt is None else float(nxt["Start Time [s]"])-float(event["End Time [s]"]); rows.append({"provider":"metrica","match_id":f"game{game}","source_event_id":f"{game}:{i}","frame_index":int(round(float(event["Start Time [s]"])*10)),"actor_id":event["From"] or None,"receiver_id":event["To"] or None,"outcome":outcome,"next_action":next_class(nxt["Type"]) if nxt is not None and delay<=5 else "unknown","next_actor_id":None if nxt is None else nxt["From"] or None,"delay_s":delay,"supervision":"inferred_next_event"})
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True); groups=[]
    for game in (1,2): groups.append(("metrica",f"game{game}",metrica(game)))
    for base in sorted((ROOT/"data/external/skillcorner/raw").iterdir()): groups.append(("skillcorner",base.name,skillcorner(base)))
    reports=[]
    for provider,match,rows in groups:
        path=OUT/f"{provider}_{match}.jsonl"; path.write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in rows),encoding="utf-8"); outcomes={name:sum(x["outcome"]==name for x in rows) for name in ("complete","turnover","offside","unknown")}; actions={name:sum(x["next_action"]==name for x in rows) for name in ("pass","shot","loss","stoppage","hold","unknown")}; reports.append({"provider":provider,"match_id":match,"supervision":rows[0]["supervision"] if rows else "unknown","events":len(rows),"outcomes":outcomes,"next_actions":actions,"linked_within_5s":sum(x["next_action"]!="unknown" for x in rows),"file":path.relative_to(ROOT).as_posix(),"sha256":sha256(path)})
    manifest={"version":"8.7.0-candidate","outcomes":["complete","turnover","offside","unknown"],"next_actions":["pass","shot","loss","stoppage","hold","unknown"],"policy":"SkillCorner strong; Metrica inferred_next_event evaluation only; Sportec unavailable","matches":reports,"totals":{"events":sum(x["events"] for x in reports),"strong":sum(x["events"] for x in reports if x["supervision"]=="strong"),"inferred":sum(x["events"] for x in reports if x["supervision"]!="strong")}}; write_json_atomic(OUT/"manifest.json",manifest); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
