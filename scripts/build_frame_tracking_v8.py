#!/usr/bin/env python3
"""Convert Metrica, Sportec, and SkillCorner tracking to the v8 frame contract."""
from __future__ import annotations
import csv,hashlib,json,re,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.frame_world.schema import BALL_INDEX,ENTITY_COUNT,MAX_TEAM_PLAYERS,FrameSequence,save_sequence
OUT=ROOT/"data/frame_world/v8"
ATTRIBUTE=re.compile(r'(\w+)="([^"]*)"')
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def finalize(provider,match,times,positions,visible,teams):
    times=np.asarray(times,np.float32); positions=np.asarray(positions,np.float32); visible=np.asarray(visible,bool)
    positions=np.where(visible[...,None],positions,0.0)
    velocities=np.zeros_like(positions)
    if len(times)>1:
        dt=np.maximum(times[1:]-times[:-1],1e-3)[:,None,None]; both=visible[1:]&visible[:-1]
        delta=(positions[1:]-positions[:-1])/dt; velocities[1:]=np.where(both[...,None],delta,0); velocities[0]=velocities[1]
    velocities=np.clip(velocities,-2,2)
    possession=np.full(len(times),-1,np.int8)
    for i in range(len(times)):
        if not visible[i,BALL_INDEX]: continue
        candidates=np.flatnonzero(visible[i,:BALL_INDEX])
        if len(candidates):
            nearest=candidates[np.argmin(np.linalg.norm(positions[i,candidates]-positions[i,BALL_INDEX],axis=1))]
            if np.linalg.norm(positions[i,nearest]-positions[i,BALL_INDEX])<=.06: possession[i]=teams[nearest]
    sequence=FrameSequence(provider,match,times,positions,velocities,visible,teams,possession)
    path=OUT/f"{provider}_{match}.npz"; save_sequence(path,sequence)
    return {"provider":provider,"match_id":match,"frames":len(times),"visible_rate":float(visible.mean()),"complete_outfield_rate":float((visible[:,:BALL_INDEX].sum(1)>=14).mean()),"ball_rate":float(visible[:,BALL_INDEX].mean()),"file":path.relative_to(ROOT).as_posix(),"sha256":sha(path)}
def metrica(game):
    raw=ROOT/"data/external/metrica/raw"; records={}; slots={}; teams=np.r_[np.zeros(MAX_TEAM_PLAYERS),np.ones(MAX_TEAM_PLAYERS),-1].astype(np.int8)
    for team,name in enumerate(("Home","Away")):
        path=raw/f"Sample_Game_{game}_RawTrackingData_{name}_Team.csv"
        with open(path,encoding="utf-8-sig",newline="") as f:
            reader=csv.reader(f); next(reader); next(reader); labels=next(reader)
            columns=[(i,labels[i]) for i in range(3,len(labels)-1,2) if labels[i] and labels[i].strip().lower()!="ball"][:MAX_TEAM_PLAYERS]
            for slot,(_,player) in enumerate(columns): slots[(team,player)]=team*MAX_TEAM_PLAYERS+slot
            last_bin=-1
            for row in reader:
                try: t=float(row[2]); period=int(row[0]); frame=int(row[1])
                except (ValueError,IndexError): continue
                key=(period,round(t*10))
                if key in records and team==0: continue
                rec=records.setdefault(key,{"t":t+(period-1)*10000,"p":np.zeros((ENTITY_COUNT,2),np.float32),"v":np.zeros(ENTITY_COUNT,bool)})
                for col,player in columns:
                    try: x,y=float(row[col]),float(row[col+1])
                    except (ValueError,IndexError): continue
                    if np.isfinite(x+y): idx=slots[(team,player)]; rec["p"][idx]=[x,y]; rec["v"][idx]=1
                if team==0:
                    try: x,y=float(row[-2]),float(row[-1]); rec["p"][BALL_INDEX]=[x,y]; rec["v"][BALL_INDEX]=np.isfinite(x+y)
                    except ValueError: pass
    ordered=[records[k] for k in sorted(records)]; return finalize("metrica",f"game{game}",[r["t"] for r in ordered],[r["p"] for r in ordered],[r["v"] for r in ordered],teams)
def skillcorner(base):
    meta=json.loads((base/f"{base.name}_match.json").read_text()); home=int(meta["home_team"]["id"]); teams=np.r_[np.zeros(MAX_TEAM_PLAYERS),np.ones(MAX_TEAM_PLAYERS),-1].astype(np.int8); mapping={}
    for team_id,offset in ((home,0),): pass
    for team,group in enumerate(([p for p in meta["players"] if int(p["team_id"])==home],[p for p in meta["players"] if int(p["team_id"])!=home])):
        played=[p for p in group if (((p.get("playing_time") or {}).get("total") or {}).get("minutes_played",0) or 0)>0]
        for slot,p in enumerate(sorted(played,key=lambda p:((((p.get("playing_time") or {}).get("total") or {}).get("start_frame",10**9)),p["id"]))[:MAX_TEAM_PLAYERS]): mapping[str(p["id"])]=team*MAX_TEAM_PLAYERS+slot
    times=[]; positions=[]; visible=[]
    with open(base/f"{base.name}_tracking_extrapolated.jsonl") as f:
        for line in f:
            row=json.loads(line)
            if row.get("period") is None or row.get("timestamp") is None: continue
            parts=[float(x) for x in row["timestamp"].split(":")]; t=sum(x*m for x,m in zip(reversed(parts),(1,60,3600)))+(int(row["period"])-1)*10000
            p=np.zeros((ENTITY_COUNT,2),np.float32); v=np.zeros(ENTITY_COUNT,bool)
            for player in row.get("player_data",[]):
                idx=mapping.get(str(player.get("player_id")))
                if idx is not None and player.get("x") is not None: p[idx]=[player["x"]/105+.5,player["y"]/68+.5]; v[idx]=bool(player.get("is_detected"))
            ball=row.get("ball_data") or {}
            if ball.get("x") is not None: p[BALL_INDEX]=[ball["x"]/105+.5,ball["y"]/68+.5]; v[BALL_INDEX]=bool(ball.get("is_detected"))
            times.append(t); positions.append(p); visible.append(v)
    return finalize("skillcorner",base.name,times,positions,visible,teams)
def sportec(path):
    ball={}; frame_team={}; players=[]; current=None; current_team=None
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.startswith("<FrameSet"):
                a=dict(ATTRIBUTE.findall(line)); current=a.get("PersonId"); current_team=a.get("TeamId")
                if current and current_team not in {"BALL","referee"}: players.append((current_team,current))
            elif current_team=="BALL" and line.startswith("<Frame "):
                a=dict(ATTRIBUTE.findall(line)); n=int(a["N"]); t=datetime.fromisoformat(a["T"]).timestamp(); ball[n]=(t,float(a["X"])/105+.5,float(a["Y"])/68+.5)
    team_ids=sorted({t for t,_ in players}); unique={t:[] for t in team_ids}
    for t,p in players:
        if p not in unique[t]: unique[t].append(p)
    mapping={p:i*MAX_TEAM_PLAYERS+j for i,t in enumerate(team_ids[:2]) for j,p in enumerate(unique[t][:MAX_TEAM_PLAYERS])}
    first=min(t for t,_,_ in ball.values())
    bins={}
    for n,(t,_,_) in ball.items():
        sample=round((t-first)*10); error=abs((t-first)-sample/10)
        if sample not in bins or error<bins[sample][0]: bins[sample]=(error,n)
    selected={value[1] for value in bins.values()}
    records={n:{"p":np.zeros((ENTITY_COUNT,2),np.float32),"v":np.zeros(ENTITY_COUNT,bool)} for n in selected}; current=None
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.startswith("<FrameSet"): current=dict(ATTRIBUTE.findall(line)).get("PersonId")
            elif current in mapping and line.startswith("<Frame "):
                a=dict(ATTRIBUTE.findall(line)); n=int(a["N"])
                if n in records: idx=mapping[current]; records[n]["p"][idx]=[float(a["X"])/105+.5,float(a["Y"])/68+.5]; records[n]["v"][idx]=True
    ordered=[]
    for n in sorted(selected):
        r=records[n]; t,x,y=ball[n]; r["p"][BALL_INDEX]=[x,y]; r["v"][BALL_INDEX]=True; r["t"]=t-first; ordered.append(r)
    teams=np.r_[np.zeros(MAX_TEAM_PLAYERS),np.ones(MAX_TEAM_PLAYERS),-1].astype(np.int8); match=re.search(r"MAT-(J\w+)\.xml$",path.name).group(1)
    return finalize("sportec",match,[r["t"] for r in ordered],[r["p"] for r in ordered],[r["v"] for r in ordered],teams)
def main():
    OUT.mkdir(parents=True,exist_ok=True); reports=[metrica(1),metrica(2)]
    reports += [skillcorner(p) for p in sorted((ROOT/"data/external/skillcorner/raw").iterdir())]
    reports += [sportec(p) for p in sorted((ROOT/"data/external/sportec/raw").glob("*positions*"))]
    manifest={"schema_version":1,"candidate":"8.0.0","sample_rate_hz":10,"entity_count":ENTITY_COUNT,"matches":reports,"providers":sorted({r["provider"] for r in reports}),"total_frames":sum(r["frames"] for r in reports),"split_policy":"match and provider holdout only"}; write_json_atomic(OUT/"manifest.json",manifest); print(json.dumps(manifest,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
