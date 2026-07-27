from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scripts.build_temporal_providers_v92 import next_action


def test_metrica_next_action_delay_starts_at_reception_end():
    events=pd.DataFrame([
        {"Period":1,"Start Time [s]":10.0,"From":"P1","Type":"PASS"},
        {"Period":1,"Start Time [s]":10.8,"From":"P2","Type":"PASS"},
        {"Period":1,"Start Time [s]":11.5,"From":"P1","Type":"SHOT"},
    ])
    action,delay=next_action(events,0,"P1",1,10.6); assert action=="shot" and abs(delay-.9)<1e-9


def test_v92_manifest_has_three_explicit_clock_providers():
    root=Path(__file__).resolve().parents[1]; manifest=json.loads((root/"data/frame_world/v92_temporal_providers/manifest.json").read_text()); assert set(manifest["provider_clocks"])=={"metrica","skillcorner","statsbomb360"}; assert manifest["rejected_provider"]["sportec"]

def test_v94_metrica_game3_has_explicit_epts_clock_and_pinned_sources():
    root=Path(__file__).resolve().parents[1]; manifest=json.loads((root/"data/frame_world/v94_temporal_providers/manifest.json").read_text()); game=next(x for x in manifest["matches"] if x["provider"]=="metrica" and x["match_id"]=="game3"); assert manifest["providers"]["metrica"]["matches"]==3; assert game["observed"]>=1000 and len(game["source_sha256"])==3