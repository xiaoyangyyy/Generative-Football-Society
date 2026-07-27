from __future__ import annotations
import numpy as np
import torch
from src.match_engine.frame_world.router import ActionRequest,ActionTransitionRouter,FrameContext
from src.match_engine.frame_world.schema import ENTITY_COUNT
from src.match_engine.frame_world.temporal import TemporalMarkConfig,TemporalMarkModel,survival_loss


def context():
    positions=np.zeros((5,ENTITY_COUNT,2),np.float32); velocities=np.zeros_like(positions); visible=np.ones((5,ENTITY_COUNT),bool); teams=np.r_[np.zeros(16),np.ones(16),-1].astype(np.int8); positions[:,:,0]=np.linspace(.1,.9,ENTITY_COUNT); return FrameContext(positions,velocities,visible,teams)


def test_survival_loss_is_finite_for_events_and_censoring():
    logits=torch.zeros(3,10); bins=torch.tensor([0,4,9]); observed=torch.tensor([True,True,False]); assert torch.isfinite(survival_loss(logits,bins,observed))


def test_router_plans_mark_and_gates_missing_provider():
    model=TemporalMarkModel(TemporalMarkConfig()); model.hazard.bias.data.fill_(-10); model.hazard.bias.data[2]=10; model.continue_head.bias.data.fill_(10); model.subtype_head.bias.data.fill_(10); router=ActionTransitionRouter(temporal_models={"skillcorner":model}); action=ActionRequest("pass","skillcorner",1,2,20); plan=router.plan_next_mark(context(),action); assert plan.delay_steps==3 and plan.kind=="shot" and plan.actor_index==2; assert router.plan_next_mark(context(),ActionRequest("pass","sportec",1,2,20)) is None


def test_event_driven_rollout_is_finite_without_transition_model():
    model=TemporalMarkModel(TemporalMarkConfig()); model.hazard.bias.data.fill_(10); router=ActionTransitionRouter(temporal_models={"skillcorner":model}); output,plan,_=router.rollout_event_driven(context(),ActionRequest("pass","skillcorner",1,2,20),10); assert plan is not None; assert output.shape==(10,ENTITY_COUNT,2) and np.isfinite(output).all()


def test_calibrated_thresholds_are_used_by_shared_decoder():
    model=TemporalMarkModel(TemporalMarkConfig()); model.hazard.bias.data.fill_(10); model.continue_head.weight.data.zero_(); model.continue_head.bias.data.zero_(); model.subtype_head.weight.data.zero_(); model.subtype_head.bias.data.zero_()
    features=torch.zeros(1,22); _,default_kind=model.predict_mark(features); _,calibrated_kind=model.predict_mark(features,.6,.4)
    assert int(default_kind[0])==1 and int(calibrated_kind[0])==2


def test_v88_manifest_pins_dependencies():
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]; manifest=json.loads((root/"data/frame_world/temporal_router_v88.json").read_text())
    assert manifest["ready"] and manifest["provider_scope"]=="skillcorner"
    for dependency in manifest["dependencies"]:
        assert hashlib.sha256((root/dependency["artifact"]).read_bytes()).hexdigest()==dependency["sha256"]



def test_v89_static_context_and_monotonic_shot_score_are_mirror_invariant():
    from src.match_engine.frame_world.router import temporal_static_features
    state=context(); mirrored=FrameContext(state.positions.copy(),state.velocities.copy(),state.visible,state.teams); mirrored.positions[:,:,0]=1-mirrored.positions[:,:,0]
    left=temporal_static_features(state,2,20); right=temporal_static_features(mirrored,2,20); assert np.allclose(left[7:19],right[7:19])
    model=TemporalMarkModel(TemporalMarkConfig(direct_subtype=True)); _,_,a=model(torch.from_numpy(left[None])); _,_,b=model(torch.from_numpy(right[None])); assert torch.allclose(a,b)
