"""Local pair transition for pressure-onset interventions."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class PressurePairConfig:
    feature_dim: int = 12
    hidden_dim: int = 64
    residual_cap: float = 0.005
    horizon_steps: int = 5
    sample_rate_hz: int = 10


class PressurePairTransition(nn.Module):
    def __init__(self, cfg=PressurePairConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(cfg.feature_dim, d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.cfg.residual_cap * torch.tanh(self.net(features)).reshape(-1, 2, 2)

    def predict(self, current, velocities, features, enabled=True):
        dt = self.cfg.horizon_steps / self.cfg.sample_rate_hz
        baseline = current + velocities * dt
        return baseline + self(features) if enabled else baseline

    def checkpoint(self):
        return {
            "version": "8.4",
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }


def load_pressure_pair(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PressurePairConfig(**payload["config"])
    model = PressurePairTransition(cfg)
    model.load_state_dict(payload["state_dict"])
    return model, cfg, payload.get("meta", {})
