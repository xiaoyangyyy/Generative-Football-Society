"""Graph-temporal probabilistic frame world model."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import torch
from torch import nn
from .schema import ENTITY_COUNT

@dataclass(frozen=True)
class FrameWorldConfig:
    hidden_dim:int=64
    graph_heads:int=4
    graph_layers:int=2
    history:int=5
    horizons:tuple[int,...]=(1,5,10,20)
    sample_rate_hz:int=10

class GraphTemporalFrameWorldModel(nn.Module):
    def __init__(self,cfg:FrameWorldConfig=FrameWorldConfig()):
        super().__init__(); self.cfg=cfg; d=cfg.hidden_dim
        self.input=nn.Sequential(nn.Linear(8,d),nn.LayerNorm(d),nn.SiLU(),nn.Linear(d,d))
        self.temporal=nn.GRU(d,d,batch_first=True)
        layer=nn.TransformerEncoderLayer(d,cfg.graph_heads,d*3,dropout=.05,batch_first=True,activation="gelu",norm_first=True)
        self.graph=nn.TransformerEncoder(layer,cfg.graph_layers)
        self.horizon=nn.Embedding(len(cfg.horizons),d)
        self.mean_head=nn.Sequential(nn.Linear(d*2,d),nn.SiLU(),nn.Linear(d,2))
        self.logvar_head=nn.Sequential(nn.Linear(d*2,d),nn.SiLU(),nn.Linear(d,2))
        self.register_buffer("variance_log_offset",torch.zeros(len(cfg.horizons)))
        self.possession_head=nn.Linear(d,3)
        nn.init.zeros_(self.mean_head[-1].weight); nn.init.zeros_(self.mean_head[-1].bias)
        nn.init.constant_(self.logvar_head[-1].bias,-6.0)
    def forward(self,positions,velocities,visible,teams):
        batch,history,entities,_=positions.shape
        team_onehot=torch.nn.functional.one_hot((teams+1).long(),3).float()[:,None].expand(-1,history,-1,-1)
        features=torch.cat([positions,velocities,visible[...,None].float(),team_onehot],dim=-1)
        tokens=self.input(features).permute(0,2,1,3).reshape(batch*entities,history,-1)
        _,state=self.temporal(tokens); state=state[-1].reshape(batch,entities,-1)
        encoded=self.graph(state,src_key_padding_mask=~visible[:,-1].bool())
        outputs=[]; logvars=[]
        for index in range(len(self.cfg.horizons)):
            h=self.horizon.weight[index][None,None].expand(batch,entities,-1); fused=torch.cat([encoded,h],dim=-1)
            outputs.append(.20*torch.tanh(self.mean_head(fused))); logvars.append(self.logvar_head(fused).clamp(-9,1))
        pooled=(encoded*visible[:,-1,:,None]).sum(1)/visible[:,-1].sum(1,keepdim=True).clamp_min(1)
        calibrated=torch.stack(logvars,1)+self.variance_log_offset[None,:,None,None]
        return torch.stack(outputs,1),calibrated,self.possession_head(pooled)
    def checkpoint(self): return {"version":8,"config":asdict(self.cfg),"state_dict":self.state_dict()}

def load_frame_world(path):
    payload=torch.load(path,map_location="cpu",weights_only=False); cfg=FrameWorldConfig(**payload["config"])
    model=GraphTemporalFrameWorldModel(cfg); model.load_state_dict(payload["state_dict"]); return model,cfg,payload.get("meta",{})
