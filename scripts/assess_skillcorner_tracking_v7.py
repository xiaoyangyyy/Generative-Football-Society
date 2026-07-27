#!/usr/bin/env python3
"""Validate imported SkillCorner tracking through the provider-neutral adapter."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.skillcorner_adapter import load_skillcorner_frames
from src.data_engine.tracking_contract import tracking_quality
def main():
    manifest=json.loads((ROOT/"data/external/skillcorner/derived/manifest.json").read_text())
    summary=pd.read_csv(ROOT/"data/external/skillcorner/derived/tracking_1hz.csv")
    frames=[]
    for base in sorted((ROOT/"data/external/skillcorner/raw").iterdir()):
        frames.extend(load_skillcorner_frames(base/f"{base.name}_match.json",base/f"{base.name}_tracking_extrapolated.jsonl",sample_every=1000))
    quality=tracking_quality(frames)
    report={"provider":"skillcorner","raw_matches":len(manifest["matches"]),"raw_frames":sum(m["frames"] for m in manifest["matches"]),"adapter_sample_frames":len(frames),"adapter_quality":quality,"mean_players":float(summary.players.mean()),"mean_detected_players":float(summary.detected_players.mean()),"ball_available_rate":float(summary.ball_x.notna().mean()),"candidate_ready":bool(manifest["tracking_import_complete"] and quality["matches"]==10 and quality["timestamps_monotonic"] and quality["complete_frame_rate"]>=.70)}
    target=ROOT/"reports/acceptance/skillcorner_tracking_v7.json"; target.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2)); return 0 if report["candidate_ready"] else 1
if __name__=="__main__": raise SystemExit(main())
