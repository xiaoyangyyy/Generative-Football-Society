"""PyTorch latent world model: encoder → GRU/Transformer transition → decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.match_engine.world_model.action_codec import ACTION_DIM
from src.match_engine.world_model.config import WorldModelConfig
from src.match_engine.world_model.observation import OBS_DIM

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except ImportError:
    _TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class WorldModelOutput:
    next_obs: np.ndarray
    pass_success: float
    xg_delta_attacking: float
    shot_goal_prob: float
    latent: np.ndarray


if _TORCH:

    class LatentWorldModel(nn.Module):
        def __init__(self, cfg: WorldModelConfig):
            super().__init__()
            self.cfg = cfg
            d = cfg.latent_dim
            h = cfg.hidden_dim
            self.encoder = nn.Sequential(
                nn.Linear(OBS_DIM, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Linear(h, d),
            )
            self.ln_latent = nn.LayerNorm(d)
            self.action_embed = nn.Linear(ACTION_DIM, d)
            kind = (cfg.transition_type or "gru").lower()
            if kind == "transformer":
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=max(1, cfg.transformer_heads),
                    dim_feedforward=h,
                    batch_first=True,
                    dropout=0.05,
                )
                self.sequence_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.transformer_layers)
                self.transition_kind = "transformer"
            else:
                self.gru = nn.GRU(d + ACTION_DIM, d, batch_first=True)
                self.transition_kind = "gru"

            self.decoder = nn.Sequential(
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Linear(h, OBS_DIM),
            )
            self.head_pass = nn.Linear(d, 1)
            self.head_xg = nn.Linear(d, 1)
            self.head_shot = nn.Linear(d, 1)

        def encode(self, obs: torch.Tensor) -> torch.Tensor:
            return self.ln_latent(self.encoder(obs))

        def _transition_step(
            self,
            z: torch.Tensor,
            action: torch.Tensor,
            h: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if self.transition_kind == "transformer":
                za = z + self.action_embed(action)
                seq = torch.stack([z, za], dim=1)
                if h is not None and h.dim() == 3:
                    seq = torch.cat([h, seq], dim=1)
                out = self.sequence_encoder(seq)
                z_next = out[:, -1, :]
                h_new = out[:, -2:, :]
                return z_next, h_new

            inp = torch.cat([z, action], dim=-1).unsqueeze(1)
            if h is None:
                out, h_new = self.gru(inp)
            else:
                out, h_new = self.gru(inp, h)
            return out.squeeze(1), h_new

        def forward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            h: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            z = self.encode(obs)
            z_next, h_new = self._transition_step(z, action, h)
            next_obs = self.decoder(z_next)
            pass_logit = self.head_pass(z_next)
            xg_delta = self.head_xg(z_next)
            shot_logit = self.head_shot(z_next)
            return z_next, next_obs, pass_logit, xg_delta, shot_logit, h_new

        def imagine(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            *,
            steps: int = 1,
            h: Optional[np.ndarray] = None,
        ) -> WorldModelOutput:
            self.eval()
            with torch.no_grad():
                z = self.encode(torch.from_numpy(obs).float().unsqueeze(0))
                act = torch.from_numpy(action).float().unsqueeze(0)
                h_t = None
                if h is not None:
                    h_t = torch.from_numpy(h).float()
                    if h_t.dim() == 2:
                        h_t = h_t.unsqueeze(0)
                for _ in range(max(1, steps)):
                    z, h_t = self._transition_step(z, act, h_t)
                next_obs_t = self.decoder(z)
                ps = torch.sigmoid(self.head_pass(z)).item()
                xg = self.head_xg(z).item()
                sg = torch.sigmoid(self.head_shot(z)).item()
                h_out = h_t.squeeze(0).numpy() if h_t is not None else z.squeeze(0).numpy()
            return WorldModelOutput(
                next_obs=next_obs_t.squeeze(0).numpy().astype(np.float32),
                pass_success=float(ps),
                xg_delta_attacking=float(xg),
                shot_goal_prob=float(sg),
                latent=h_out.astype(np.float32),
            )

else:

    class LatentWorldModel:  # type: ignore
        def __init__(self, cfg: WorldModelConfig):
            raise RuntimeError(
                "PyTorch is required for the latent world model. Install: pip install torch"
            )


def build_model(cfg: WorldModelConfig | None = None) -> "LatentWorldModel":
    if not _TORCH:
        raise RuntimeError("PyTorch is required. pip install torch")
    return LatentWorldModel(cfg or WorldModelConfig.from_env())


def save_checkpoint(
    path: str,
    model: "LatentWorldModel",
    cfg: WorldModelConfig,
    meta: Dict | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "meta": meta or {},
        "version": 2,
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> Tuple["LatentWorldModel", WorldModelConfig, Dict]:
    if not _TORCH:
        raise RuntimeError("PyTorch is required. pip install torch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = WorldModelConfig(**payload.get("cfg", {}))
    model = LatentWorldModel(cfg)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint incompatible ({exc}). Retrain: python scripts/ensure_world_model.py"
        ) from exc
    model.eval()
    return model, cfg, payload.get("meta", {})
