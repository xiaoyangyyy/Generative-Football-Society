from __future__ import annotations
import numpy as np
import torch
from torch import nn
from src.match_engine.frame_world.router import ActionRequest,ActionTransitionRouter,FrameContext
from src.match_engine.frame_world.schema import BALL_INDEX,ENTITY_COUNT


class PairStub(nn.Module):
    def __init__(self,count,amount): super().__init__(); self.anchor=nn.Parameter(torch.zeros(())); self.count=count; self.amount=amount
    def predict(self,current,velocities,features,enabled=True):
        baseline=current+velocities*.5; residual=torch.zeros_like(current); residual[...,0]=self.amount; return baseline+residual if enabled else baseline


def context():
    positions=np.zeros((5,ENTITY_COUNT,2),np.float32); velocities=np.zeros_like(positions); visible=np.ones((5,ENTITY_COUNT),bool); teams=np.r_[np.zeros(16),np.ones(16),-1].astype(np.int8); positions[:,:,0]=np.linspace(.1,.9,ENTITY_COUNT); positions[:,:,1]=.5; return FrameContext(positions,velocities,visible,teams)


def test_simultaneous_actions_share_one_baseline_and_stay_local():
    router=ActionTransitionRouter(pass_models={"skillcorner":PairStub(3,.01)},pressure_models={"skillcorner":PairStub(2,.002)}); state=context(); actions=[ActionRequest("pass","skillcorner",actor_index=1,target_index=2,defender_index=20),ActionRequest("pressure","skillcorner",actor_index=21,target_index=2)]
    result=router.transition(state,actions); changed=np.flatnonzero(np.any(result.positions!=state.positions[-1],axis=1)); assert set(changed)=={2,20,21,BALL_INDEX}; assert all(x.enabled for x in result.decisions)


def test_missing_identity_and_unsupported_provider_are_exact_fallback():
    router=ActionTransitionRouter(pass_models={"skillcorner":PairStub(3,.01)}); state=context(); result=router.transition(state,[ActionRequest("pass","sportec",1,2,20),ActionRequest("pressure","skillcorner",-1,2)]); np.testing.assert_allclose(result.positions,state.positions[-1]); assert not any(x.enabled for x in result.decisions)


def test_closed_loop_is_finite_bounded_and_deterministic():
    router=ActionTransitionRouter(pass_models={"skillcorner":PairStub(3,.01)}); state=context(); schedule={0:[ActionRequest("pass","skillcorner",1,2,20)]}; first,_=router.rollout(state,10,schedule); second,_=router.rollout(state,10,schedule); np.testing.assert_allclose(first,second); assert np.isfinite(first).all(); assert first.min()>=-.15 and first.max()<=1.15
