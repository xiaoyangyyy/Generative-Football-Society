#!/usr/bin/env python3
"""Align real pass, shot, and pressure events to v8's unified 10 Hz frames."""
from __future__ import annotations

import csv,hashlib,json,re,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.frame_world.actions import ACTION_NAMES,TimedAction,write_actions
from src.match_engine.frame_world.schema import MAX_TEAM_PLAYERS

OUT=ROOT/"data/frame_world/v82_actions"; PRESSURE={"pressure","pressing","recovery_press","counter_press"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def clock(text):
    parts=[float(x) for x in str(text).split(":")]; return sum(x*m for x,m in zip(reversed(parts),(1,60,3600)))
def unit(dx,dy):
    length=max((dx*dx+dy*dy)**.5,1e-6); return dx/length,dy/length


def metrica(game):
    path=ROOT/f"data/external/metrica/raw/Sample_Game_{game}_RawEventsData.csv"; actions=[]
    with open(path,encoding="utf-8-sig",newline="") as f:
        for index,row in enumerate(csv.DictReader(f)):
            action=str(row["Type"]).lower()
            if action not in {"pass","shot"}: continue
            period=int(row["Period"]); start=float(row["Start Time [s]"])+(period-1)*10000; end=float(row["End Time [s]"])+(period-1)*10000
            x,y=float(row["Start X"]),float(row["Start Y"]); dx,dy=unit(float(row["End X"])-x,float(row["End Y"])-y)
            actions.append(TimedAction("metrica",f"game{game}",action,start,end,row.get("From") or None,row.get("Team") or None,x,y,dx,dy,target_id=row.get("To") or None,source_event_id=f"{game}:{index}"))
    return actions,{"pass":True,"shot":True,"pressure":False}


def skillcorner(base):
    path=base/f"{base.name}_dynamic_events.csv"; use=["event_id","frame_start","frame_end","time_start","time_end","period","attacking_side","event_type","event_subtype","player_id","team_id","x_start","y_start","end_type","player_targeted_id","player_in_possession_id","player_targeted_x_pass","player_targeted_y_pass"]
    data=pd.read_csv(path,low_memory=False,usecols=use); actions=[]
    for row in data.itertuples(index=False):
        subtype=str(row.event_subtype).lower(); end_type=str(row.end_type).lower(); action=None; at_end=False
        if row.event_type=="player_possession" and end_type=="pass": action="pass"; at_end=True
        elif row.event_type=="player_possession" and end_type=="shot": action="shot"; at_end=True
        elif row.event_type=="on_ball_engagement" and subtype in PRESSURE: action="pressure"
        if action is None: continue
        offset=(int(row.period)-1)*10000; start=clock(row.time_end if at_end else row.time_start)+offset; end=clock(row.time_end)+offset
        x=None if pd.isna(row.x_start) else float(row.x_start)/105+.5; y=None if pd.isna(row.y_start) else float(row.y_start)/68+.5
        if action=="pass" and not pd.isna(row.player_targeted_x_pass): dx,dy=unit(float(row.player_targeted_x_pass)/105+.5-x,float(row.player_targeted_y_pass)/68+.5-y)
        elif action=="shot": dx,dy=(1.,0.) if row.attacking_side=="left_to_right" else (-1.,0.)
        else: dx,dy=(0.,0.)
        if action=="pass" and row.attacking_side=="right_to_left": dx,dy=-dx,-dy
        target=row.player_in_possession_id if action=="pressure" else row.player_targeted_id
        actions.append(TimedAction("skillcorner",base.name,action,start,max(start,end),None if pd.isna(row.player_id) else str(int(row.player_id)),None if pd.isna(row.team_id) else str(int(row.team_id)),x,y,dx,dy,target_id=None if pd.isna(target) else str(int(target)),source_event_id=str(row.event_id)))
    return actions,{"pass":True,"shot":True,"pressure":True}


def sportec(event_path):
    match=re.search(r"MAT-(J\w+)\.xml$",event_path.name).group(1); position=next((ROOT/"data/external/sportec/raw").glob(f"*positions*MAT-{match}.xml")); first=None
    with open(position,encoding="utf-8") as f:
        in_ball=False
        for line in f:
            if line.startswith("<FrameSet"): in_ball='TeamId="BALL"' in line
            elif in_ball and line.startswith("<Frame "): first=datetime.fromisoformat(re.search(r'T="([^"]+)',line).group(1)).timestamp(); break
    actions=[]; weak=[]; root=ET.parse(event_path).getroot()
    for event in root.findall("Event"):
        t=datetime.fromisoformat(event.attrib["EventTime"]).timestamp()-first; play=event.find(".//Play")
        action="shot" if event.find(".//ShotAtGoal") is not None else "pass" if event.find(".//Pass") is not None else None
        if action:
            node=play if play is not None else event; x=event.attrib.get("X-Position"); y=event.attrib.get("Y-Position")
            actions.append(TimedAction("sportec",match,action,t,t,node.attrib.get("Player"),node.attrib.get("Team"),None if x is None else float(x)/105,None if y is None else float(y)/68,None,None,target_id=node.attrib.get("Recipient"),source_event_id=event.attrib.get("EventId")))
        elif event.find(".//TacklingGame") is not None:
            node=event.find(".//TacklingGame"); weak.append(TimedAction("sportec",match,"pressure",t,t,node.attrib.get("Winner"),node.attrib.get("WinnerTeam"),dx=0,dy=0,confidence=.35,supervision="weak_proxy",target_id=node.attrib.get("Loser"),source_event_id=event.attrib.get("EventId")))
    return actions+weak,{"pass":True,"shot":True,"pressure":False}


def entity_mapping(provider,match_id):
    if provider=="metrica":
        game=match_id[-1]; mapping={}
        for team,name in enumerate(("Home","Away")):
            with open(ROOT/f"data/external/metrica/raw/Sample_Game_{game}_RawTrackingData_{name}_Team.csv",encoding="utf-8-sig",newline="") as f: reader=csv.reader(f); next(reader); next(reader); labels=next(reader)
            players=[labels[i] for i in range(3,len(labels)-1,2) if labels[i] and labels[i].strip().lower()!="ball"][:MAX_TEAM_PLAYERS]; mapping.update({p:team*MAX_TEAM_PLAYERS+i for i,p in enumerate(players)})
        return mapping
    if provider=="skillcorner":
        base=ROOT/f"data/external/skillcorner/raw/{match_id}"; meta=json.loads((base/f"{match_id}_match.json").read_text()); home=int(meta["home_team"]["id"]); mapping={}
        for team,group in enumerate(([p for p in meta["players"] if int(p["team_id"])==home],[p for p in meta["players"] if int(p["team_id"])!=home])):
            played=[p for p in group if (((p.get("playing_time") or {}).get("total") or {}).get("minutes_played",0) or 0)>0]
            for slot,p in enumerate(sorted(played,key=lambda p:((((p.get("playing_time") or {}).get("total") or {}).get("start_frame",10**9)),p["id"]))[:MAX_TEAM_PLAYERS]): mapping[str(p["id"])]=team*MAX_TEAM_PLAYERS+slot
        return mapping
    position=next((ROOT/"data/external/sportec/raw").glob(f"*positions*MAT-{match_id}.xml")); players=[]; current_team=None
    with open(position,encoding="utf-8") as f:
        for line in f:
            if line.startswith("<FrameSet"):
                attrs=dict(re.findall(r'(\w+)="([^"]*)"',line)); current_team=attrs.get("TeamId"); person=attrs.get("PersonId")
                if person and current_team not in {"BALL","referee"}: players.append((current_team,person))
    teams=sorted({x for x,_ in players}); unique={x:[] for x in teams}
    for team,person in players:
        if person not in unique[team]: unique[team].append(person)
    return {person:i*MAX_TEAM_PLAYERS+j for i,team in enumerate(teams[:2]) for j,person in enumerate(unique[team][:MAX_TEAM_PLAYERS])}

def align(actions,frame_file,mapping):
    raw=np.load(frame_file,allow_pickle=False); times=raw["timestamps"].astype(float); labels=np.zeros((len(times),len(ACTION_NAMES)),np.uint8); directions=np.zeros((len(times),2),np.float32); actor_indices=np.full((len(times),3),-1,np.int8); target_indices=np.full((len(times),3),-1,np.int8); known=np.zeros(len(ACTION_NAMES),np.uint8); errors=[]; aligned=[]
    for action in actions:
        index=int(np.searchsorted(times,action.start_s)); choices=[i for i in (index-1,index) if 0<=i<len(times)]; nearest=min(choices,key=lambda i:abs(times[i]-action.start_s)); error=abs(times[nearest]-action.start_s)
        if error>.061: continue
        item={**action.__dict__,"frame_index":nearest,"alignment_error_s":error}; aligned.append(item); errors.append(error)
        if action.supervision=="strong":
            end=nearest
            if action.action=="pressure": end=min(len(times)-1,int(np.searchsorted(times,action.end_s)))
            kind=ACTION_NAMES.index(action.action); labels[nearest:end+1,kind]=1; actor_indices[nearest:end+1,kind]=mapping.get(str(action.actor_id),-1); target_indices[nearest:end+1,kind]=mapping.get(str(action.target_id),-1)
            if action.action in {"pass","shot"} and action.dx is not None: directions[nearest:end+1]=[action.dx,action.dy]
    return labels,directions,actor_indices,target_indices,aligned,errors,raw["visible"]


def main():
    OUT.mkdir(parents=True,exist_ok=True); manifest=json.loads((ROOT/"data/frame_world/v8/manifest.json").read_text()); sources={}
    for game in (1,2): sources[("metrica",f"game{game}")]=metrica(game)
    for base in sorted((ROOT/"data/external/skillcorner/raw").iterdir()): sources[("skillcorner",base.name)]=skillcorner(base)
    for path in sorted((ROOT/"data/external/sportec/raw").glob("*events*.xml")):
        match=re.search(r"MAT-(J\w+)\.xml$",path.name).group(1); sources[("sportec",match)]=sportec(path)
    reports=[]
    for item in manifest["matches"]:
        key=(item["provider"],item["match_id"]); actions,availability=sources[key]; mapping=entity_mapping(*key); labels,directions,actors,targets,aligned,errors,visible=align(actions,ROOT/item["file"],mapping); event_path=OUT/f"{key[0]}_{key[1]}.jsonl"; label_path=OUT/f"{key[0]}_{key[1]}.npz"
        event_path.write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in aligned),encoding="utf-8"); np.savez_compressed(label_path,labels=labels,directions=directions,actor_indices=actors,target_indices=targets,known=np.array([availability[x] for x in ACTION_NAMES],np.uint8))
        positive=labels.astype(bool); actor_mask=(actors>=0)&positive; target_mask=(targets>=0)&positive; actor_visible=sum(int(visible[:,i][actor_mask[:,i]].sum()) for i in range(3)); actor_total=int(actor_mask.sum())
        counts=Counter(x["action"] for x in aligned if x["supervision"]=="strong"); reports.append({"provider":key[0],"match_id":key[1],"availability":availability,"strong_counts":dict(counts),"positive_frames":dict(zip(ACTION_NAMES,labels.sum(0).astype(int).tolist())),"actor_mapping_rate":float(actor_mask.sum()/max(1,positive.sum())),"target_mapping_rate":float(target_mask.sum()/max(1,positive.sum())),"mapped_actor_visible_rate":float(actor_visible/max(1,actor_total)),"weak_count":sum(x["supervision"]!="strong" for x in aligned),"aligned":len(aligned),"alignment_p95_ms":float(np.quantile(errors,.95)*1000) if errors else None,"events_file":event_path.relative_to(ROOT).as_posix(),"labels_file":label_path.relative_to(ROOT).as_posix(),"labels_sha256":sha(label_path)})
    output={"version":"8.2.0-candidate","classes":ACTION_NAMES,"tolerance_ms":61,"matches":reports,"totals":{name:sum(x["strong_counts"].get(name,0) for x in reports) for name in ACTION_NAMES},"pressure_policy":"SkillCorner strong labels only; Sportec tackles retained as weak_proxy and excluded from frame targets; Metrica pressure unavailable"}; write_json_atomic(OUT/"manifest.json",output); print(json.dumps(output,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
