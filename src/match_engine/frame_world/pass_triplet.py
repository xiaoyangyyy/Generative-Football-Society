"""Joint ball, receiver, and interceptor transition for aligned passes."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn


@dataclass(frozen=True)
class PassTripletConfig:
    feature_dim:int=22
    hidden_dim:int=96
    horizon_steps:int=5
    sample_rate_hz:int=10
    ball_cap:float=.05
    player_cap:float=.001


class PassTripletTransition(nn.Module):
    def __init__(self,cfg=PassTripletConfig()):
        super().__init__(); self.cfg=cfg; d=cfg.hidden_dim; self.net=nn.Sequential(nn.Linear(cfg.feature_dim,d),nn.LayerNorm(d),nn.SiLU(),nn.Linear(d,d),nn.SiLU(),nn.Linear(d,6)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,features):
        raw=torch.tanh(self.net(features)).reshape(-1,3,2); caps=raw.new_tensor([self.cfg.ball_cap,self.cfg.player_cap,self.cfg.player_cap])[None,:,None]; return raw*caps
    def predict(self,current,velocities,features,enabled=True):
        baseline=current+velocities*(self.cfg.horizon_steps/self.cfg.sample_rate_hz); return baseline+self(features) if enabled else baseline
    def checkpoint(self): return {"version":"8.5","config":asdict(self.cfg),"state_dict":self.state_dict()}


@lru_cache(maxsize=8)
def load_pass_triplet(path):
    payload=torch.load(path,map_location="cpu",weights_only=False); cfg=PassTripletConfig(**payload["config"]); model=PassTripletTransition(cfg); model.load_state_dict(payload["state_dict"]); return model,cfg,payload.get("meta",{})
