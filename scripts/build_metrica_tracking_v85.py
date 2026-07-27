#!/usr/bin/env python3
"""Build corrected Metrica frames without overwriting registered v8 evidence."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import scripts.build_frame_tracking_v8 as builder
from src.data_engine.dataset_registry import write_json_atomic


def main():
    output=ROOT/"data/frame_world/v85_tracking"; output.mkdir(parents=True,exist_ok=True); builder.OUT=output; reports=[builder.metrica(1),builder.metrica(2)]; manifest={"version":"8.5.0-candidate","correction":"Metrica Ball header excluded from player slots","matches":reports}; write_json_atomic(output/"manifest.json",manifest); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
