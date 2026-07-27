from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from src.match_engine.frame_world.reception import ReceptionChainModel,ReceptionConfig,apply_reception_state
from src.match_engine.frame_world.router import ActionRequest,ActionTransitionRouter,FrameContext
from src.match_engine.frame_world.schema import ENTITY_COUNT

ROOT=Path(__file__).resolve().parents[1]


def test_reception_state_changes_possession_by_outcome():
    teams=np.r_[np.zeros(16),np.ones(16),-1]; complete=apply_reception_state(True,3,20,teams,"pass"); turnover=apply_reception_state(False,3,20,teams,"terminal"); assert (complete.possessor_index,complete.possession_team)==(3,0); assert (turnover.possessor_index,turnover.possession_team)==(20,1)


def test_router_reception_is_provider_gated():
    model=ReceptionChainModel(ReceptionConfig()); model.completion.bias.data.fill_(10); model.next_action.bias.data[0]=10; positions=np.zeros((5,ENTITY_COUNT,2),np.float32); velocities=np.zeros_like(positions); visible=np.ones((5,ENTITY_COUNT),bool); teams=np.r_[np.zeros(16),np.ones(16),-1].astype(np.int8); context=FrameContext(positions,velocities,visible,teams); router=ActionTransitionRouter(reception_models={"skillcorner":model}); action=ActionRequest("pass","skillcorner",1,2,20)
    state=router.resolve_reception(context,action); assert state.outcome=="complete" and state.possessor_index==2 and state.next_action=="pass"; assert router.resolve_reception(context,ActionRequest("pass","sportec",1,2,20)) is None


def test_chain_manifest_is_hash_pinned():
    manifest=json.loads((ROOT/"data/frame_world/v87_reception_chains/manifest.json").read_text()); assert manifest["totals"]=={"events":10348,"inferred":1763,"strong":8585}
    for item in manifest["matches"]:
        path=ROOT/item["file"]; assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
