#!/usr/bin/env python3
"""Non-promotional Metrica audit against inferred next-event reception labels."""
from __future__ import annotations
import json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_reception_chain_v87 import Chains,evaluate
from src.match_engine.frame_world.reception import load_reception_chain


def main():
    frames=json.loads((ROOT/"data/frame_world/v85_tracking/manifest.json").read_text())["matches"]; triplets=json.loads((ROOT/"data/frame_world/v85_pass_triplets/manifest.json").read_text())["matches"]; chains=json.loads((ROOT/"data/frame_world/v87_reception_chains/manifest.json").read_text())["matches"]; tri={(x["provider"],x["match_id"]):x for x in triplets}; chain={(x["provider"],x["match_id"]):x for x in chains}; items=[{**x,"triplet_file":tri[("metrica",x["match_id"])]["file"],"chain_file":chain[("metrica",x["match_id"])]["file"]} for x in frames]; model,cfg,_=load_reception_chain(ROOT/"data/frame_world/reception_chain_v87_skillcorner.pt"); metrics=evaluate(model,DataLoader(Chains(items,cfg),batch_size=256),torch.device("cpu")); report={"version":"8.7.0-candidate","scope":"metrica inferred_next_event external audit; not promotion evidence","metrics":metrics}; target=ROOT/"reports/acceptance/reception_chain_v87_external.json"; target.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
