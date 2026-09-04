from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.match_engine.frame_world.pass_triplet import PassTripletConfig,PassTripletTransition
from src.match_engine.frame_world.schema import BALL_INDEX

ROOT=Path(__file__).resolve().parents[1]


def test_triplet_disabled_is_exact_fallback_and_residuals_are_bounded():
    cfg=PassTripletConfig(); model=PassTripletTransition(cfg); current=torch.rand(4,3,2); velocity=torch.rand(4,3,2)*.01; features=torch.rand(4,cfg.feature_dim); baseline=current+velocity*.5
    torch.testing.assert_close(model.predict(current,velocity,features,False),baseline); model.net[-1].bias.data.fill_(100); residual=model.predict(current,velocity,features)-baseline
    assert residual[:,0].abs().max()<=cfg.ball_cap+1e-7; assert residual[:,1:].abs().max()<=cfg.player_cap+1e-7


def test_corrected_metrica_has_no_player_slot_duplicating_ball():
    manifest=json.loads((ROOT/"data/frame_world/v85_tracking/manifest.json").read_text())
    for item in manifest["matches"]:
        with np.load(ROOT/item["file"],allow_pickle=False) as frame:
            visible=frame["visible"].astype(bool); same=np.all(frame["positions"][:,:BALL_INDEX]==frame["positions"][:,BALL_INDEX,None],axis=-1)&visible[:,:BALL_INDEX]&visible[:,BALL_INDEX,None]
            assert float(same.mean())<.001


def test_triplet_manifest_is_hash_pinned_and_geometrically_plausible():
    manifest=json.loads((ROOT/"data/frame_world/v85_pass_triplets/manifest.json").read_text()); assert manifest["totals"]=={"passes":16170,"valid_triplets":14088}
    for item in manifest["matches"]:
        path=ROOT/item["file"]; assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]; assert item["median_lane_distance_m"]>.5
