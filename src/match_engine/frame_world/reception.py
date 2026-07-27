"""Reception outcome, next-action prediction, and possession state transition."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import torch
from torch import nn

NEXT_ACTIONS=("pass","shot","terminal")


@dataclass(frozen=True)
class ReceptionConfig:
    feature_dim:int=22
    hidden_dim:int=64


class ReceptionChainModel(nn.Module):
    def __init__(self,cfg=ReceptionConfig()):
        super().__init__(); self.cfg=cfg; d=cfg.hidden_dim; self.encoder=nn.Sequential(nn.Linear(cfg.feature_dim,d),nn.LayerNorm(d),nn.SiLU(),nn.Linear(d,d),nn.SiLU()); self.completion=nn.Linear(d,1); self.next_action=nn.Linear(d,len(NEXT_ACTIONS))
    def forward(self,features):
        state=self.encoder(features); return self.completion(state).squeeze(-1),self.next_action(state)
    def checkpoint(self): return {"version":"8.7","config":asdict(self.cfg),"state_dict":self.state_dict()}


@dataclass(frozen=True)
class ReceptionState:
    outcome:str
    possessor_index:int
    possession_team:int
    next_action:str


def apply_reception_state(complete,receiver,defender,teams,next_action):
    index=receiver if complete else defender; return ReceptionState("complete" if complete else "turnover",int(index),int(teams[index]) if 0<=index<len(teams) else -1,next_action)


def load_reception_chain(path):
    payload=torch.load(path,map_location="cpu",weights_only=False); cfg=ReceptionConfig(**payload["config"]); model=ReceptionChainModel(cfg); model.load_state_dict(payload["state_dict"]); return model,cfg,payload.get("meta",{})
