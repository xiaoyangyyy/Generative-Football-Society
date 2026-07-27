"""Action-conditioned one-step frame model and closed-loop rollout."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ControlledFrameConfig:
    hidden_dim: int = 64
    graph_heads: int = 4
    graph_layers: int = 2
    history: int = 5
    sample_rate_hz: int = 10
    max_acceleration: float = 0.04
    max_velocity: float = 0.35


class ActionConditionedFrameWorld(nn.Module):
    def __init__(self, cfg: ControlledFrameConfig = ControlledFrameConfig()):
        super().__init__(); self.cfg=cfg; d=cfg.hidden_dim
        self.input=nn.Sequential(nn.Linear(8,d),nn.LayerNorm(d),nn.SiLU(),nn.Linear(d,d))
        self.temporal=nn.GRU(d,d,batch_first=True)
        layer=nn.TransformerEncoderLayer(d,cfg.graph_heads,d*3,dropout=.05,batch_first=True,activation="gelu",norm_first=True)
        self.graph=nn.TransformerEncoder(layer,cfg.graph_layers)
        self.action=nn.Sequential(nn.Linear(3,d),nn.SiLU(),nn.Linear(d,d))
        self.delta=nn.Sequential(nn.Linear(d*2,d),nn.SiLU(),nn.Linear(d,2))
        self.logvar=nn.Sequential(nn.Linear(d*2,d),nn.SiLU(),nn.Linear(d,2))
        nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias); nn.init.constant_(self.logvar[-1].bias,-7)

    def forward(self, positions, velocities, visible, teams, actions):
        batch,history,entities,_=positions.shape
        onehot=torch.nn.functional.one_hot((teams+1).long(),3).float()[:,None].expand(-1,history,-1,-1)
        features=torch.cat([positions,velocities,visible[...,None].float(),onehot],-1)
        tokens=self.input(features).permute(0,2,1,3).reshape(batch*entities,history,-1)
        _,state=self.temporal(tokens); state=state[-1].reshape(batch,entities,-1)
        action_tokens=self.action(actions)
        encoded=self.graph(state+action_tokens,src_key_padding_mask=~visible[:,-1].bool())
        fused=torch.cat([encoded,action_tokens],-1)
        return .003*torch.tanh(self.delta(fused)),self.logvar(fused).clamp(-10,0)

    def checkpoint(self):
        return {"version":"8.1","config":asdict(self.cfg),"state_dict":self.state_dict()}


def observed_actions(velocities: torch.Tensor, visible: torch.Tensor, cfg: ControlledFrameConfig) -> torch.Tensor:
    acceleration=(velocities[:,-1]-velocities[:,-2])*cfg.sample_rate_hz
    valid=(visible[:,-1]&visible[:,-2])[...,None]
    acceleration=torch.where(valid,acceleration,torch.zeros_like(acceleration)).clamp(-cfg.max_acceleration,cfg.max_acceleration)
    return torch.cat([acceleration,torch.zeros_like(acceleration[...,:1])],-1)


def transition(model, positions, velocities, visible, teams, actions):
    dt=1/model.cfg.sample_rate_hz
    acceleration=actions[...,:2].clamp(-model.cfg.max_acceleration,model.cfg.max_acceleration)*actions[...,2:3].clamp(0,1)
    residual,logvar=model(positions,velocities,visible,teams,actions)
    next_position=positions[:,-1]+velocities[:,-1]*dt+.5*acceleration*dt*dt+residual
    next_velocity=(velocities[:,-1]+acceleration*dt+residual/dt).clamp(-model.cfg.max_velocity,model.cfg.max_velocity)
    return next_position.clamp(-.15,1.15),next_velocity,logvar


def rollout(model, positions, velocities, visible, teams, steps, policy=None):
    pos=positions.clone(); vel=velocities.clone(); vis=visible.clone(); predictions=[]; initial=observed_actions(vel,vis,model.cfg)
    for step in range(steps):
        actions=initial*(.65**step) if policy is None else policy(step,pos,vel,vis,teams)
        next_pos,next_vel,_=transition(model,pos,vel,vis,teams,actions); predictions.append(next_pos)
        pos=torch.cat([pos[:,1:],next_pos[:,None]],1); vel=torch.cat([vel[:,1:],next_vel[:,None]],1); vis=torch.cat([vis[:,1:],vis[:,-1:,]],1)
    return torch.stack(predictions,1)


def load_controlled_frame_world(path):
    payload=torch.load(path,map_location="cpu",weights_only=False); cfg=ControlledFrameConfig(**payload["config"])
    model=ActionConditionedFrameWorld(cfg); model.load_state_dict(payload["state_dict"]); return model,cfg,payload.get("meta",{})
