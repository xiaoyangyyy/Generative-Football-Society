from __future__ import annotations
import torch
import numpy as np
from src.match_engine.frame_world.router import ActionRequest,ActionTransitionRouter,FrameContext
from src.match_engine.frame_world.schema import ENTITY_COUNT
from src.match_engine.frame_world.continuous import ContinuousTimeMarkModel,mixture_survival_loss


def test_continuous_survival_loss_is_finite_for_events_and_censoring():
    model=ContinuousTimeMarkModel(); values=model(torch.zeros(4,22)); loss=mixture_survival_loss(*values[:3],torch.tensor([.05,.8,2.,5.]),torch.tensor([True,True,True,False])); assert torch.isfinite(loss)


def test_continuous_median_is_bounded_and_differentiable_outputs_exist():
    model=ContinuousTimeMarkModel(); features=torch.randn(7,22); median=model.median_time(features); assert median.shape==(7,) and bool(((median>=.025)&(median<=5)).all())


def test_monotonic_shot_score_uses_mirror_invariant_goal_proximity():
    model=ContinuousTimeMarkModel(); far=torch.zeros(1,22); near=torch.zeros(1,22); far[:,9]=.1; near[:,9]=.4; assert model(near)[4].item()>model(far)[4].item()


def test_router_preserves_continuous_seconds_and_quantizes_only_execution_step():
    from src.match_engine.frame_world.continuous import ContinuousTimeMarkModel
    positions=np.zeros((5,ENTITY_COUNT,2),np.float32); visible=np.ones((5,ENTITY_COUNT),bool); teams=np.r_[np.zeros(16),np.ones(16),-1].astype(np.int8); state=FrameContext(positions,positions.copy(),visible,teams); model=ContinuousTimeMarkModel(); model.median_time=lambda features: torch.tensor([1.26]); router=ActionTransitionRouter(temporal_models={"skillcorner":model}); plan=router.plan_next_mark(state,ActionRequest("pass","skillcorner",1,2,20)); assert abs(plan.delay_seconds-1.26)<1e-5 and plan.delay_steps==3



def test_continuous_cdf_and_quantiles_are_ordered():
    model=ContinuousTimeMarkModel(); features=torch.randn(8,22); q05=model.quantile_time(features,.05); q50=model.quantile_time(features,.5); q95=model.quantile_time(features,.95); assert bool(((q05<=q50)&(q50<=q95)).all()); assert bool((model.distribution_cdf(features,q05)<=model.distribution_cdf(features,q95)).all())


def test_router_exposes_only_configured_calibrated_interval():
    positions=np.zeros((5,ENTITY_COUNT,2),np.float32); visible=np.ones((5,ENTITY_COUNT),bool); teams=np.r_[np.zeros(16),np.ones(16),-1].astype(np.int8); state=FrameContext(positions,positions.copy(),visible,teams); model=ContinuousTimeMarkModel(); model.median_time=lambda features: torch.tensor([1.2]); action=ActionRequest("pass","metrica",1,2,20); plain=ActionTransitionRouter(temporal_models={"metrica":model}); assert plain.plan_next_mark_interval(state,action) is None; router=ActionTransitionRouter(temporal_models={"metrica":model},temporal_intervals={"metrica":{"0.9":2.1}}); interval=router.plan_next_mark_interval(state,action); assert interval.lower_seconds==0 and abs(interval.upper_seconds-3.3)<1e-6; assert interval.lower_steps==0 and interval.delay_steps==3 and interval.upper_steps==7

def test_verified_calibration_artifact_builds_router():
    from pathlib import Path
    from src.match_engine.frame_world.calibration import load_calibrated_temporal_router
    root=Path(__file__).resolve().parents[1]; router=load_calibrated_temporal_router(root/"data/frame_world/continuous_intervals_v92.json"); assert set(router.temporal_models)=={"metrica","skillcorner","statsbomb360"}; assert float(router.temporal_intervals["metrica"]["0.9"])>2

def test_nested_kind_interval_and_bound_rollouts_are_stable():
    positions=np.zeros((5,ENTITY_COUNT,2),np.float32); visible=np.ones((5,ENTITY_COUNT),bool); teams=np.r_[np.zeros(16),np.ones(16),-1].astype(np.int8); state=FrameContext(positions,positions.copy(),visible,teams); model=ContinuousTimeMarkModel(); model.median_time=lambda features: torch.tensor([1.2]); action=ActionRequest("pass","metrica",1,2,20); router=ActionTransitionRouter(temporal_models={"metrica":model},temporal_intervals={"metrica":{"terminal":{"0.9":2.1},"global":{"0.9":1.0}}}); outputs=[]
    for bound in ("lower","median","upper"):
        trajectory,interval,_=router.rollout_event_driven_interval(state,action,8,bound); assert trajectory.shape==(8,ENTITY_COUNT,2) and np.isfinite(trajectory).all(); outputs.append(interval)
    assert outputs[0].lower_steps<=outputs[0].delay_steps<=outputs[0].upper_steps

def test_operational_nested_calibration_loads_and_rejects_tampered_hash(tmp_path):
    import json
    from pathlib import Path
    import pytest
    from src.match_engine.frame_world.calibration import load_calibrated_temporal_router
    root=Path(__file__).resolve().parents[1]; source=root/"data/frame_world/continuous_intervals_v93.json"; router=load_calibrated_temporal_router(source); assert "global" in router.temporal_intervals["skillcorner"] and "pass" in router.temporal_intervals["skillcorner"]; payload=json.loads(source.read_text()); payload["providers"]["metrica"]["model"]=(root/payload["providers"]["metrica"]["model"]).as_posix(); payload["providers"]["metrica"]["model_sha256"]="0"*64; target=tmp_path/"tampered.json"; target.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError): load_calibrated_temporal_router(target)