"""Hierarchical marked discrete-time survival model for post-reception actions."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class TemporalMarkConfig:
    feature_dim: int = 22
    hidden_dim: int = 64
    bins: int = 10
    bin_seconds: float = 0.5
    direct_subtype: bool = False


class TemporalMarkModel(nn.Module):
    def __init__(self, cfg=TemporalMarkConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(cfg.feature_dim, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
        )
        self.hazard = nn.Linear(d, cfg.bins)
        self.continue_head = nn.Linear(d, 1)
        self.subtype_head = nn.Linear(7 if cfg.direct_subtype else d, 1)

    def forward(self, features):
        state = self.encoder(features)
        subtype = (
            (features[:, 9] - 0.25) * 10
            if self.cfg.direct_subtype
            else self.subtype_head(state).squeeze(-1)
        )
        return self.hazard(state), self.continue_head(state).squeeze(-1), subtype

    def predict_mark(self, features, continue_threshold=0.5, subtype_threshold=0.5):
        hazard, continuation, subtype = self(features)
        probability = torch.sigmoid(hazard)
        survival = torch.cumprod(1 - probability, dim=1)
        cumulative = 1 - survival
        reached = cumulative >= 0.5
        first = reached.float().argmax(1)
        first = torch.where(
            reached.any(1), first, torch.full_like(first, self.cfg.bins - 1)
        )
        kind = torch.where(
            torch.sigmoid(continuation) >= continue_threshold,
            torch.where(torch.sigmoid(subtype) >= subtype_threshold, 1, 0),
            torch.full_like(first, 2),
        )
        return first + 1, kind

    def checkpoint(self):
        return {
            "version": "8.8",
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }


def survival_loss(logits, bins, observed):
    log_h = torch.nn.functional.logsigmoid(logits)
    log_s = torch.nn.functional.logsigmoid(-logits)
    index = torch.arange(logits.shape[1], device=logits.device)[None]
    before = (index < bins[:, None]).float()
    at = (index == bins[:, None]).float() * observed[:, None].float()
    return (
        -((log_s * before) + (log_h * at) + (log_s * at * (~observed)[:, None].float()))
        .sum(1)
        .mean()
    )


def load_temporal_mark(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TemporalMarkConfig(**payload["config"])
    model = TemporalMarkModel(cfg)
    model.load_state_dict(payload["state_dict"])
    return model, cfg, payload.get("meta", {})
