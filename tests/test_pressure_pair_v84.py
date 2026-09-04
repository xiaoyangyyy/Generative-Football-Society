from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from src.match_engine.frame_world.pressure import PressurePairConfig,PressurePairTransition

ROOT=Path(__file__).resolve().parents[1]


def test_disabled_pair_transition_is_exact_kinematic_fallback():
    cfg=PressurePairConfig(); model=PressurePairTransition(cfg); current=torch.rand(3,2,2); velocity=torch.rand(3,2,2)*.01; features=torch.rand(3,cfg.feature_dim)
    predicted=model.predict(current,velocity,features,enabled=False)
    torch.testing.assert_close(predicted,current+velocity*(cfg.horizon_steps/cfg.sample_rate_hz))


def test_pair_transition_is_bounded():
    cfg=PressurePairConfig(); model=PressurePairTransition(cfg); model.net[-1].bias.data.fill_(100); current=torch.zeros(2,2,2); velocity=torch.zeros_like(current); features=torch.zeros(2,cfg.feature_dim)
    residual=model.predict(current,velocity,features)-current
    assert residual.abs().max()<=cfg.residual_cap+1e-7


def test_v84_manifest_is_event_complete_and_hash_pinned():
    manifest=json.loads((ROOT/"data/frame_world/v84_pressure/manifest.json").read_text()); assert manifest["totals"]["event_onsets"]==7063; assert manifest["totals"]["geometry_valid"]==6375; assert len(manifest["matches"])==10
    for item in manifest["matches"]:
        path=ROOT/item["file"]; assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
