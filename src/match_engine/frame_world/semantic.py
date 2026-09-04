"""Semantic event-conditioned graph-temporal frame transition model."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class SemanticFrameConfig:
    hidden_dim: int = 64
    graph_heads: int = 4
    graph_layers: int = 2
    history: int = 5
    sample_rate_hz: int = 10
    action_dim: int = 5
    horizon_steps: int = 5
    pressure_enabled: bool = False
    actor_roles_enabled: bool = False


class SemanticActionFrameWorld(nn.Module):
    def __init__(self, cfg: SemanticFrameConfig = SemanticFrameConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.input = nn.Sequential(
            nn.Linear(8, d), nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d)
        )
        self.temporal = nn.GRU(d, d, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d,
            cfg.graph_heads,
            d * 3,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # norm_first disables the nested-tensor fast path; opt out explicitly.
        self.graph = nn.TransformerEncoder(
            layer, cfg.graph_layers, enable_nested_tensor=False
        )
        self.semantic = nn.Sequential(
            nn.Linear(cfg.action_dim, d, bias=False),
            nn.SiLU(),
            nn.Linear(d, d, bias=False),
        )
        self.delta = nn.Sequential(nn.Linear(d * 2, d), nn.SiLU(), nn.Linear(d, 2))
        self.action_head = nn.Linear(d, 3)
        if cfg.actor_roles_enabled:
            self.actor_role = nn.Parameter(torch.zeros(d))
            self.target_role = nn.Parameter(torch.zeros(d))
            nn.init.normal_(self.actor_role, std=0.02)
            nn.init.normal_(self.target_role, std=0.02)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self,
        positions,
        velocities,
        visible,
        teams,
        actions,
        actor_indices=None,
        target_indices=None,
    ):
        batch, history, entities, _ = positions.shape
        onehot = (
            torch.nn.functional.one_hot((teams + 1).long(), 3)
            .float()[:, None]
            .expand(-1, history, -1, -1)
        )
        features = torch.cat(
            [positions, velocities, visible[..., None].float(), onehot], -1
        )
        tokens = (
            self.input(features)
            .permute(0, 2, 1, 3)
            .reshape(batch * entities, history, -1)
        )
        _, state = self.temporal(tokens)
        state = state[-1].reshape(batch, entities, -1)
        mask = visible[:, -1].bool()
        pooled = (state * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(
            1
        )
        semantic = self.semantic(actions)[:, None].expand(-1, entities, -1).clone()
        if self.cfg.actor_roles_enabled and actor_indices is not None:
            pressure = actions[:, 2] > 0
            role_sets = ((actor_indices[:, 2], self.actor_role),)
            if target_indices is not None:
                role_sets += ((target_indices[:, 2], self.target_role),)
            for indices, role in role_sets:
                valid = pressure & (indices >= 0) & (indices < entities)
                rows = torch.arange(batch, device=actions.device)[valid]
                semantic[rows, indices[valid].long()] += role
        encoded = self.graph(state + semantic, src_key_padding_mask=~mask)
        residual = 0.05 * torch.tanh(self.delta(torch.cat([encoded, semantic], -1)))
        scope = torch.zeros(
            (batch, entities, 1), device=actions.device, dtype=actions.dtype
        )
        direction_valid = (
            torch.linalg.vector_norm(actions[:, 3:], dim=1, keepdim=True) > 0.5
        ).to(actions.dtype)
        scope[:, -1] = actions[:, :2].amax(1, keepdim=True) * direction_valid
        if self.cfg.pressure_enabled and actor_indices is not None:
            pressure = actions[:, 2] > 0
            index_sets = (actor_indices[:, 2],)
            if target_indices is not None:
                index_sets += (target_indices[:, 2],)
            for indices in index_sets:
                valid = pressure & (indices >= 0) & (indices < entities)
                rows = torch.arange(batch, device=actions.device)[valid]
                scope[rows, indices[valid].long(), 0] = 0.1
        residual = residual * scope
        return residual, self.action_head(pooled)

    def predict(
        self,
        positions,
        velocities,
        visible,
        teams,
        actions,
        actor_indices=None,
        target_indices=None,
    ):
        residual, logits = self(
            positions,
            velocities,
            visible,
            teams,
            actions,
            actor_indices,
            target_indices,
        )
        dt = self.cfg.horizon_steps / self.cfg.sample_rate_hz
        return (positions[:, -1] + velocities[:, -1] * dt + residual).clamp(
            -0.15, 1.15
        ), logits

    def checkpoint(self):
        return {
            "version": "8.3" if self.cfg.pressure_enabled else "8.2",
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }


def load_semantic_frame_world(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SemanticFrameConfig(**payload["config"])
    model = SemanticActionFrameWorld(cfg)
    model.load_state_dict(payload["state_dict"])
    return model, cfg, payload.get("meta", {})
