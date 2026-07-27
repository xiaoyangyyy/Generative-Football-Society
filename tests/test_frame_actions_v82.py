from __future__ import annotations
import hashlib,json
from pathlib import Path
import pytest,torch
from src.match_engine.frame_world.actions import TimedAction
from src.match_engine.frame_world.schema import BALL_INDEX,ENTITY_COUNT
from src.match_engine.frame_world.semantic import SemanticActionFrameWorld,SemanticFrameConfig

ROOT=Path(__file__).resolve().parents[1]

def model_inputs():
    cfg=SemanticFrameConfig(hidden_dim=16,graph_layers=1); shape=(2,cfg.history,ENTITY_COUNT); pos=torch.full((*shape,2),.5); vel=torch.zeros_like(pos); vis=torch.ones(shape,dtype=torch.bool); teams=torch.tensor([0]*16+[1]*16+[-1]).repeat(2,1); return cfg,pos,vel,vis,teams

def test_timed_action_validates_interval_and_class():
    TimedAction("test","m","pass",1,2,dx=1,dy=0)
    with pytest.raises(ValueError): TimedAction("test","m","carry",1,2)
    with pytest.raises(ValueError): TimedAction("test","m","pass",2,1)

def test_unknown_direction_falls_back_to_kinematics():
    cfg,pos,vel,vis,teams=model_inputs(); model=SemanticActionFrameWorld(cfg).eval(); model.delta[-1].bias.data.fill_(1); action=torch.zeros((2,5)); action[:,0]=1
    predicted,_=model.predict(pos,vel,vis,teams,action)
    torch.testing.assert_close(predicted,pos[:,-1])

def test_directed_pass_residual_is_scoped_to_ball():
    cfg,pos,vel,vis,teams=model_inputs(); model=SemanticActionFrameWorld(cfg).eval(); model.delta[-1].bias.data.fill_(1); action=torch.zeros((2,5)); action[:,0]=1; action[:,3]=1
    predicted,_=model.predict(pos,vel,vis,teams,action)
    torch.testing.assert_close(predicted[:,:BALL_INDEX],pos[:,-1,:BALL_INDEX]); assert torch.all(predicted[:,BALL_INDEX]>pos[:,-1,BALL_INDEX])

def test_action_manifest_hashes_and_supervision_boundary():
    manifest=json.loads((ROOT/"data/frame_world/v82_actions/manifest.json").read_text()); assert manifest["totals"]=={"pass":16170,"shot":445,"pressure":7063}; assert len(manifest["matches"])==19
    for item in manifest["matches"]:
        path=ROOT/item["labels_file"]; assert hashlib.sha256(path.read_bytes()).hexdigest()==item["labels_sha256"]
        assert item["availability"]["pressure"]==(item["provider"]=="skillcorner")

def test_pressure_residual_is_scoped_to_actor_and_target():
    cfg,pos,vel,vis,teams=model_inputs(); cfg=SemanticFrameConfig(hidden_dim=16,graph_layers=1,pressure_enabled=True); model=SemanticActionFrameWorld(cfg).eval(); model.delta[-1].bias.data.fill_(1)
    action=torch.zeros((2,5)); action[:,2]=1; actors=torch.full((2,3),-1); targets=torch.full((2,3),-1); actors[:,2]=torch.tensor([3,5]); targets[:,2]=torch.tensor([8,11])
    residual,_=model(pos,vel,vis,teams,action,actors,targets); active=residual.abs().sum(-1)>0; expected=torch.zeros_like(active); expected[0,[3,8]]=True; expected[1,[5,11]]=True
    assert torch.equal(active,expected)

def test_pressure_without_training_evidence_is_exact_fallback():
    cfg,pos,vel,vis,teams=model_inputs(); model=SemanticActionFrameWorld(cfg).eval(); model.delta[-1].bias.data.fill_(1); action=torch.zeros((2,5)); action[:,2]=1; actors=torch.tensor([[-1,-1,3],[-1,-1,5]]); targets=torch.tensor([[-1,-1,8],[-1,-1,11]])
    residual,_=model(pos,vel,vis,teams,action,actors,targets)
    assert torch.count_nonzero(residual)==0
