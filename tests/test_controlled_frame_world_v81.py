from __future__ import annotations

import torch

from src.match_engine.frame_world.controlled import ActionConditionedFrameWorld,ControlledFrameConfig,rollout,transition
from src.match_engine.frame_world.schema import BALL_INDEX,ENTITY_COUNT


def inputs():
    cfg=ControlledFrameConfig(hidden_dim=16,graph_layers=1); shape=(2,cfg.history,ENTITY_COUNT)
    positions=torch.full((*shape,2),.5); velocities=torch.zeros_like(positions); visible=torch.ones(shape,dtype=torch.bool)
    teams=torch.tensor([0]*16+[1]*16+[-1]).repeat(2,1); return cfg,positions,velocities,visible,teams


def test_observed_context_does_not_apply_unmasked_control():
    cfg,pos,vel,vis,teams=inputs(); model=ActionConditionedFrameWorld(cfg).eval(); action=torch.zeros((2,ENTITY_COUNT,3)); action[...,0]=cfg.max_acceleration
    predicted,next_velocity,_=transition(model,pos,vel,vis,teams,action)
    torch.testing.assert_close(predicted,pos[:,-1]); torch.testing.assert_close(next_velocity,vel[:,-1])


def test_masked_action_moves_controlled_entity_forward():
    cfg,pos,vel,vis,teams=inputs(); model=ActionConditionedFrameWorld(cfg).eval(); action=torch.zeros((2,ENTITY_COUNT,3)); action[:,BALL_INDEX,0]=cfg.max_acceleration; action[:,BALL_INDEX,2]=1
    predicted,_,_=transition(model,pos,vel,vis,teams,action)
    assert torch.all(predicted[:,BALL_INDEX,0]>pos[:,-1,BALL_INDEX,0])
    torch.testing.assert_close(predicted[:,:BALL_INDEX],pos[:,-1,:BALL_INDEX])


def test_closed_loop_is_finite_bounded_and_deterministic():
    cfg,pos,vel,vis,teams=inputs(); model=ActionConditionedFrameWorld(cfg).eval()
    first=rollout(model,pos,vel,vis,teams,20); second=rollout(model,pos,vel,vis,teams,20)
    torch.testing.assert_close(first,second); assert torch.isfinite(first).all(); assert first.min()>=-.15 and first.max()<=1.15
